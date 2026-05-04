"""Reference-image MRI: capture per-(layer, step, position) DiT activations from images.

For training-data-driven character work — forward-pass each training image
through the DiT under partial-noise img2img conditions to capture how the DiT
*processes* that specific image. The activations describe the subject in the
DiT's native feature basis without any training.

Approach: ZImagePipeline doesn't ship an img2img variant (too new). We construct
the partially-noised latent ourselves via VAE encode + FlowMatch interpolation,
then pass it as the pipeline's `latents=` argument. The pipeline's normal
8-step denoising runs from there, hooks capture residuals at every (layer, step)
cell — same shape as `dit-mri` so `dit-mri-diff` and downstream tools just work.

Output structure mirrors dit-mri exactly:
    metadata.json
    prompts.jsonl                          (one entry per ref image)
    prompt_NNN_residuals.npy               [n_layers, n_steps, batch, seq_len, hidden] fp16
    prompt_NNN.png                         (the post-denoise output, sanity)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


DIT_MRI_REF_VERSION = "0.1"


def _find_pairs(refs_dir: Path) -> list[tuple[Path, str]]:
    """Find (image, caption) pairs in a directory.

    Convention: NAME.png + NAME.txt — the caption file shares the basename.
    If no .txt is present, falls back to a generic caption.
    """
    pairs = []
    for img in sorted(refs_dir.iterdir()):
        if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        cap = img.with_suffix(".txt")
        if cap.exists():
            text = cap.read_text(encoding="utf-8").lstrip("﻿").strip()
        else:
            text = "a portrait photograph"
        pairs.append((img, text))
    return pairs


def _vae_encode(vae, image_path: Path, *,
                 width: int, height: int, seed: int, device, dtype):
    """VAE-encode an image to a clean latent x_0. Caller controls device placement."""
    import torch
    from PIL import Image
    image = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.asarray(image).astype(np.float32) / 127.5 - 1.0   # [-1, 1]
    img_tensor = torch.from_numpy(arr.transpose(2, 0, 1)[None]).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent_dist = vae.encode(img_tensor).latent_dist
        x_0 = latent_dist.sample(generator=torch.Generator(device=device).manual_seed(seed))
        scaling = getattr(vae.config, "scaling_factor", 1.0)
        shift = getattr(vae.config, "shift_factor", None)
        if shift is not None:
            x_0 = (x_0 - shift) * scaling
        else:
            x_0 = x_0 * scaling
    return x_0


def _interpolate_with_noise(x_0, sigma: float, seed: int):
    """FlowMatch noise interpolation: x_t = (1 - sigma) * x_0 + sigma * eps.

    sigma=0   →  clean image
    sigma=1   →  pure noise
    sigma=0.4 →  ~60% image content survives, enough to bias trajectory toward subject
                 while still leaving the model meaningful denoising work.
    """
    import torch
    g = torch.Generator(device=x_0.device).manual_seed(seed)
    noise = torch.randn(x_0.shape, generator=g, device=x_0.device, dtype=x_0.dtype)
    return (1.0 - sigma) * x_0 + sigma * noise


def capture_dit_mri_ref(
    pipeline_id: str,
    refs_dir: str,
    output: str,
    *,
    seed: int = 42,
    num_inference_steps: int = 8,
    width: int = 512,
    height: int = 512,
    dtype: str = "bfloat16",
    offload: str = "sequential",
    sigma: float = 0.4,
    save_images: bool = True,
    lora_path: str | None = None,
    max_refs: int | None = None,
    image_tokens_only: bool = True,
) -> dict:
    """Capture DiT activations from each ref image via partial-noise img2img.

    Args:
        pipeline_id: HF pipeline (e.g. "Tongyi-MAI/Z-Image-Turbo")
        refs_dir: directory containing image+caption pairs (NAME.png + NAME.txt)
        output: output directory for residuals + metadata
        sigma: image-vs-noise mix at the partial-denoise start. 0.4 = 60% image,
               40% noise — enough that the trajectory follows the image content
               but the model still does meaningful work.
        max_refs: optionally cap the number of refs (for smoke testing).
    """
    import torch
    from diffusers import ZImagePipeline

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs_path = Path(refs_dir)

    pairs = _find_pairs(refs_path)
    if max_refs is not None:
        pairs = pairs[:max_refs]
    if not pairs:
        raise ValueError(f"no image+caption pairs found in {refs_dir}")

    print(f"loading {pipeline_id} ...")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype]
    pipe = ZImagePipeline.from_pretrained(pipeline_id, torch_dtype=torch_dtype)

    if lora_path is not None:
        from safetensors.torch import load_file
        print(f"loading LoRA from {lora_path}")
        sd = load_file(lora_path)
        renamed = {}
        for k, v in sd.items():
            new_k = k
            if k.startswith("diffusion_model."):
                new_k = "transformer." + k[len("diffusion_model."):]
            renamed[new_k] = v
        pipe.load_lora_weights(renamed)
        print(f"  loaded {len(renamed)} LoRA tensors")

    # Phase 1: VAE-encode all reference images BEFORE setting up sequential offload.
    # Sequential offload puts every component on the meta device until used in the
    # pipe.__call__ flow — at which point we'd no longer have direct VAE access.
    # So we move VAE to GPU once, encode every image, cache clean latents on CPU,
    # then move VAE back and let offload take over.
    print("phase 1: VAE-encoding reference images...")
    pipe.vae.to("cuda")
    vae_dtype = next(pipe.vae.parameters()).dtype
    cached_x0: list = []  # CPU latents
    for pi, (img_path, _) in enumerate(pairs):
        x_0 = _vae_encode(
            pipe.vae, img_path,
            width=width, height=height,
            seed=seed + pi,
            device="cuda", dtype=vae_dtype,
        )
        cached_x0.append(x_0.cpu())
        if (pi + 1) % 5 == 0 or pi == len(pairs) - 1:
            print(f"  encoded {pi + 1}/{len(pairs)}: {img_path.name}  latent_shape={tuple(x_0.shape)}")
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    print(f"phase 1 done. cached {len(cached_x0)} clean latents on CPU.")

    # Phase 2: set up offload and run inference loop with the pre-encoded latents.
    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "none":
        pipe = pipe.to("cuda")
    else:
        raise ValueError(f"offload must be 'model', 'sequential', or 'none', got {offload!r}")

    transformer = pipe.transformer
    layers = transformer.layers
    n_layers = len(layers)
    dim = int(transformer.config.dim)
    print(f"DiT: {n_layers} layers, dim={dim}")
    print(f"refs: {len(pairs)}, steps: {num_inference_steps}, sigma: {sigma}, "
          f"resolution: {width}x{height}, seed: {seed}")

    all_meta = []
    t0 = time.time()

    # Build a compressed sigma schedule for low-noise img2img.
    # Default Z-Image-Turbo schedule goes from ~1.0 (pure noise) to 0.
    # We compress it to [sigma, 0] over `num_inference_steps` steps so the model
    # always operates in the low-noise regime where image content survives —
    # this avoids the over-denoising failure mode where the prior wins on close-ups.
    custom_sigmas = np.linspace(sigma, 0.0, num_inference_steps + 1).astype(np.float32).tolist()
    print(f"using compressed sigmas: {[round(s, 3) for s in custom_sigmas]}")

    for pi, (img_path, caption) in enumerate(pairs):
        print(f"\n[{pi + 1}/{len(pairs)}] {img_path.name}  caption={caption[:60]!r}")

        # Move cached x_0 to GPU and interpolate with noise at our chosen sigma
        x_0 = cached_x0[pi].to("cuda")
        latents = _interpolate_with_noise(x_0, sigma=sigma, seed=seed + pi)

        per_layer: dict[int, list[np.ndarray]] = {l: [] for l in range(n_layers)}

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                hs = output if isinstance(output, torch.Tensor) else output[0]
                arr = hs.detach().to(torch.float16).cpu().numpy()
                per_layer[layer_idx].append(arr)
            return hook

        handles = []
        for li, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(li)))

        gen = torch.Generator(device="cuda").manual_seed(seed + pi)
        try:
            with torch.no_grad():
                result = pipe(
                    caption,
                    generator=gen,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    latents=latents,
                    sigmas=custom_sigmas,
                )
            image = result.images[0] if hasattr(result, 'images') else None
        finally:
            for h in handles:
                h.remove()

        sample = per_layer[0]
        n_calls = len(sample)
        if n_calls == 0:
            raise RuntimeError("hooks captured nothing")
        first_shape = sample[0].shape
        full_seq = int(first_shape[1])

        # Optionally trim to image tokens only (the trailing N positions of the
        # seq_len axis). For Z-Image at 512×512 that's 256 of 1088 positions —
        # ~4× smaller. The atlas only needs image positions.
        if image_tokens_only:
            n_image_tokens = (width // 16 // 2) * (height // 16 // 2)
            img_lo = full_seq - n_image_tokens
            for l in range(n_layers):
                per_layer[l] = [a[:, img_lo:, :] for a in per_layer[l]]
            saved_seq = n_image_tokens
        else:
            img_lo = 0
            saved_seq = full_seq

        first_shape = per_layer[0][0].shape
        print(f"  hook calls per layer: {n_calls}, saved shape per call: {first_shape} "
              f"(image tokens only: {image_tokens_only})")

        residuals = np.stack(
            [np.stack(per_layer[l], axis=0) for l in range(n_layers)],
            axis=0,
        )
        np.save(out_dir / f"prompt_{pi:03d}_residuals.npy", residuals)

        if save_images and image is not None:
            image.save(out_dir / f"prompt_{pi:03d}.png")

        all_meta.append({
            "idx": pi,
            "text": caption,
            "ref_image": str(img_path),
            "n_calls_per_layer": n_calls,
            "batch": int(first_shape[0]),
            "full_seq_len": full_seq,
            "saved_seq_len": int(first_shape[1]),
            "img_lo_in_full_seq": img_lo,
            "image_tokens_only": image_tokens_only,
            "hidden": int(first_shape[2]),
            "residuals_shape": list(residuals.shape),
        })

        # Explicit cleanup to keep RAM bounded between iterations. With ~4 GB
        # per_layer + ~1 GB residuals (image-only) per prompt, Python's GC
        # is too lazy — we OOM around prompt 17 without this.
        del per_layer
        del residuals
        del x_0
        del latents
        if image is not None:
            del image
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.time() - t0

    metadata = {
        "version": DIT_MRI_REF_VERSION,
        "pipeline_id": pipeline_id,
        "refs_dir": str(refs_path),
        "n_prompts": len(pairs),
        "n_layers": n_layers,
        "dim": dim,
        "n_heads": int(transformer.config.n_heads),
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "width": width,
        "height": height,
        "sigma": sigma,
        "dtype_storage": "float16",
        "dtype_compute": dtype,
        "offload": offload,
        "lora_path": lora_path,
        "elapsed_s": round(elapsed, 1),
    }
    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    with open(out_dir / "prompts.jsonl", "w") as fh:
        for m in all_meta:
            fh.write(json.dumps(m) + "\n")

    print(f"\ndone. {elapsed:.1f}s. wrote {out_dir}")
    return metadata
