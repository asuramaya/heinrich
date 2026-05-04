"""DiT-mode MRI: per-(layer, timestep, position) residual capture during image generation.

For text-to-image diffusion DiTs. One generation per prompt with the same seed, hooks
every transformer layer's output, captures residuals at every denoising step.

Currently targets ZImagePipeline (Z-Image / Z-Image-Turbo). The hook target is
`pipe.transformer.layers[i]` — works for any DiT exposing a `.layers` ModuleList of
ZImageTransformerBlock-shaped modules.

Output layout (one directory per run):
    metadata.json                          — pipeline id, n_layers, dim, steps, seed, ...
    prompts.jsonl                          — per-prompt metadata + capture shapes
    prompt_NNN_residuals.npy               — [n_layers, n_calls_per_layer, batch, seq_len, hidden] fp16
    prompt_NNN.png                         — generated image (sanity)

Storage: ~30 layers × 8 steps × ~1300 tokens × 3840 hidden × 2 bytes ≈ 2.4 GB per prompt.

The downstream analysis (dit-mri-analyze) compares trajectories across prompts at each
(layer, timestep, position) cell to find where identity discrimination concentrates.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


DIT_MRI_VERSION = "0.1"


def _load_prompts(path: str) -> list[dict]:
    prompts = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj:
                raise ValueError(f"prompt entry missing 'text' field: {obj}")
            prompts.append(obj)
    return prompts


def capture_dit_mri(
    pipeline_id: str,
    prompts_path: str,
    output: str,
    *,
    seed: int = 42,
    num_inference_steps: int = 8,
    width: int = 1024,
    height: int = 1024,
    dtype: str = "bfloat16",
    offload: str = "model",
    save_images: bool = True,
    lora_path: str | None = None,
) -> dict:
    """Capture per-(layer, timestep, position) DiT residuals for each prompt."""
    import torch
    from diffusers import ZImagePipeline

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = _load_prompts(prompts_path)

    print(f"loading {pipeline_id} ...")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype]
    pipe = ZImagePipeline.from_pretrained(pipeline_id, torch_dtype=torch_dtype)

    if lora_path is not None:
        # ai-toolkit LoRAs (e.g. Z-Image character LoRAs) use the
        # `diffusion_model.<layer>.<module>.lora_{A,B}.weight` key prefix.
        # Diffusers' transformer expects `transformer.<layer>...` — rename in-place.
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

    if offload == "sequential":
        # Per-submodule offload: each layer moved to GPU only during its forward.
        # Required when the full DiT doesn't fit in VRAM (e.g. 6B Z-Image on a 12GB GPU).
        # Much slower (~10x) but fits any model.
        pipe.enable_sequential_cpu_offload()
    elif offload == "model":
        # Whole-component offload: text_encoder/transformer/vae each moved to GPU
        # only during their phase. Fast when each component individually fits.
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
    print(f"prompts: {len(prompts)}, steps: {num_inference_steps}, seed: {seed}, "
          f"resolution: {width}x{height}")

    all_meta = []
    t0 = time.time()

    for pi, p in enumerate(prompts):
        text = p["text"]
        print(f"\n[{pi + 1}/{len(prompts)}] {text!r}")

        # layer_idx -> list of [B, N, H] fp16 numpy arrays, one per hook fire
        per_layer: dict[int, list[np.ndarray]] = {l: [] for l in range(n_layers)}

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                # ZImageTransformerBlock returns the residual as a single tensor.
                # Some block implementations may return a tuple — guard for both.
                hs = output if isinstance(output, torch.Tensor) else output[0]
                arr = hs.detach().to(torch.float16).cpu().numpy()
                per_layer[layer_idx].append(arr)
            return hook

        handles = []
        for li, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(li)))

        gen = torch.Generator(device="cuda").manual_seed(seed)
        try:
            with torch.no_grad():
                result = pipe(
                    text,
                    generator=gen,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                )
            image = result.images[0] if hasattr(result, 'images') else None
        finally:
            for h in handles:
                h.remove()

        sample = per_layer[0]
        n_calls = len(sample)
        if n_calls == 0:
            raise RuntimeError(
                "Hooks captured nothing. Either pipe.transformer.layers wasn't called, "
                "or the hook target is wrong for this pipeline."
            )
        first_shape = sample[0].shape
        print(f"  hook calls per layer: {n_calls}, capture shape per call: {first_shape}")

        # Verify all layers got the same number of calls and shapes
        for l in range(n_layers):
            assert len(per_layer[l]) == n_calls, (
                f"layer {l} got {len(per_layer[l])} calls, expected {n_calls}"
            )
            for s in per_layer[l]:
                assert s.shape == first_shape, (
                    f"layer {l} got shape {s.shape}, expected {first_shape}"
                )

        # Stack to [n_layers, n_calls, B, N, H]
        residuals = np.stack(
            [np.stack(per_layer[l], axis=0) for l in range(n_layers)],
            axis=0,
        )

        np.save(out_dir / f"prompt_{pi:03d}_residuals.npy", residuals)
        if save_images and image is not None:
            image.save(out_dir / f"prompt_{pi:03d}.png")

        all_meta.append({
            "idx": pi,
            "text": text,
            "n_calls_per_layer": n_calls,
            "batch": int(first_shape[0]),
            "seq_len": int(first_shape[1]),
            "hidden": int(first_shape[2]),
            "residuals_shape": list(residuals.shape),
            **{k: v for k, v in p.items() if k != "text"},
        })

    elapsed = time.time() - t0

    metadata = {
        "version": DIT_MRI_VERSION,
        "pipeline_id": pipeline_id,
        "prompts_source": str(prompts_path),
        "n_prompts": len(prompts),
        "n_layers": n_layers,
        "dim": dim,
        "n_heads": int(transformer.config.n_heads),
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "width": width,
        "height": height,
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
