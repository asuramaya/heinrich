"""Inference-time identity injection via the atlas — one-shot LoRA replacement.

Generates with ZImagePipeline while hooks inject an identity atlas at every
(layer, step, image-position) cell. Replaces gradient-descent-trained LoRAs with
a forward-pass-derived activation transplant. No trigger word needed.

Atlas comes from `dit-mri-atlas` (typically: Subject ref captures vs generic baseline).
At inference we add `strength * direction` to image-position residuals at the
conditional CFG branch. Strength=0 reverts to base behavior; strength=1 should
push the trajectory toward the identity manifold.

The hook tracks step index per layer (each layer fires once per denoising step).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


DIT_STEER_VERSION = "0.1"


def _load_atlas(atlas_path: str):
    """Load atlas direction tensor + metadata."""
    p = Path(atlas_path)
    meta_p = p.with_suffix(".meta.json")
    with open(meta_p) as fh:
        meta = json.load(fh)
    arrays = dict(np.load(p))
    return meta, arrays


def _load_prompts(path: str) -> list[dict]:
    prompts = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj)
    return prompts


def steer_with_atlas(
    pipeline_id: str,
    atlas_path: str,
    prompts_path: str,
    output: str,
    *,
    strength: float = 1.0,
    cfg_branch_only: bool = True,
    seed: int = 42,
    num_inference_steps: int = 8,
    width: int = 512,
    height: int = 512,
    dtype: str = "bfloat16",
    offload: str = "sequential",
    save_residuals: bool = False,
) -> dict:
    """Generate with identity-atlas injection — no LoRA, no trigger word.

    Args:
        atlas_path: atlas.npz from dit-mri-atlas
        prompts_path: jsonl of {"text": ...} prompts
        output: output dir
        strength: scaling on the atlas direction. 0 = no injection, 1 = nominal.
        cfg_branch_only: inject only on conditional branch (CFG idx 0). True = recommended;
            False = inject on both branches (rarely makes sense).
        save_residuals: also save the per-(layer, step) residuals during steered generation
            (large; useful for verifying the injection landed where intended).
    """
    import torch
    from diffusers import ZImagePipeline

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = _load_prompts(prompts_path)

    print(f"loading atlas from {atlas_path} ...")
    atlas_meta, atlas_arr = _load_atlas(atlas_path)
    direction = atlas_arr["direction"]  # [L, T, P_img, H], fp16
    a_layers = atlas_meta["n_layers"]
    a_steps = atlas_meta["n_steps"]
    a_pimg = atlas_meta["n_image_tokens"]
    a_hidden = atlas_meta["hidden"]
    print(f"  atlas shape: [L={a_layers}, T={a_steps}, P_img={a_pimg}, H={a_hidden}]  "
          f"strength={strength}")

    # Atlas was captured at low-noise (img2img with compressed schedule), generation
    # runs full noise schedule. Atlas-step-N and gen-step-N aren't directly comparable.
    # Default mapping: collapse atlas to mean direction over its step axis, apply at
    # every generation step. This is the rank-1 "average Subject-vs-generic direction"
    # injection. Future: support late-step-only or step-aligned mappings.
    direction_collapsed = direction.astype(np.float32).mean(axis=1)  # [L, P_img, H]
    print(f"  collapsed atlas to per-(layer, position) mean direction "
          f"(applied at every generation step)")

    print(f"loading {pipeline_id} ...")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype]
    pipe = ZImagePipeline.from_pretrained(pipeline_id, torch_dtype=torch_dtype)

    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "none":
        pipe = pipe.to("cuda")

    transformer = pipe.transformer
    layers = transformer.layers
    n_layers = len(layers)

    if n_layers != a_layers:
        raise ValueError(f"atlas n_layers={a_layers} != pipeline n_layers={n_layers}")

    # Convert atlas to per-layer fp32 numpy (already collapsed across step axis).
    direction_per_layer = [direction_collapsed[l] for l in range(n_layers)]
    print(f"DiT: {n_layers} layers, {num_inference_steps} steps")
    print(f"prompts: {len(prompts)}, seed: {seed}, resolution: {width}x{height}")
    print()

    all_meta = []
    t0 = time.time()

    # Per-prompt step counter (one int per layer index — reset each prompt)
    for pi, p in enumerate(prompts):
        text = p["text"]
        print(f"[{pi + 1}/{len(prompts)}] {text[:70]!r}")

        step_counter = {l: 0 for l in range(n_layers)}
        residual_capture: dict[int, list[np.ndarray]] = {l: [] for l in range(n_layers)}

        def make_steer_hook(layer_idx):
            def hook(module, inputs, output):
                hs = output if isinstance(output, torch.Tensor) else output[0]
                seq_len = hs.shape[1]
                img_lo = seq_len - a_pimg
                if img_lo < 0:
                    return  # not enough room — skip injection at this layer

                # Apply collapsed atlas direction (same direction every generation step).
                d_np = direction_per_layer[layer_idx]  # [P_img, H] fp32
                d_t = torch.from_numpy(d_np).to(device=hs.device, dtype=hs.dtype)

                if cfg_branch_only and hs.shape[0] >= 1:
                    hs[0, img_lo:, :] = hs[0, img_lo:, :] + strength * d_t
                else:
                    for b in range(hs.shape[0]):
                        hs[b, img_lo:, :] = hs[b, img_lo:, :] + strength * d_t

                if save_residuals:
                    residual_capture[layer_idx].append(
                        hs.detach().to(torch.float16).cpu().numpy())

                step_counter[layer_idx] += 1
                return (hs,) + output[1:] if isinstance(output, tuple) else hs
            return hook

        handles = []
        for li, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_steer_hook(li)))

        gen = torch.Generator(device="cuda").manual_seed(seed + pi)
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

        if image is not None:
            image.save(out_dir / f"prompt_{pi:03d}.png")

        if save_residuals:
            try:
                stacked = np.stack(
                    [np.stack(residual_capture[l], axis=0) for l in range(n_layers)],
                    axis=0,
                )
                np.save(out_dir / f"prompt_{pi:03d}_steered_residuals.npy", stacked)
            except ValueError:
                pass  # ragged — skip

        n_step_calls_per_layer = step_counter[0]
        print(f"  hook fires per layer: {n_step_calls_per_layer}")

        all_meta.append({
            "idx": pi,
            "text": text,
            "n_step_calls_per_layer": n_step_calls_per_layer,
            "strength": strength,
            **{k: v for k, v in p.items() if k != "text"},
        })

    elapsed = time.time() - t0

    metadata = {
        "version": DIT_STEER_VERSION,
        "pipeline_id": pipeline_id,
        "atlas_path": str(atlas_path),
        "prompts_source": str(prompts_path),
        "n_prompts": len(prompts),
        "strength": strength,
        "cfg_branch_only": cfg_branch_only,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "width": width,
        "height": height,
        "elapsed_s": round(elapsed, 1),
    }
    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    with open(out_dir / "prompts.jsonl", "w") as fh:
        for m in all_meta:
            fh.write(json.dumps(m) + "\n")

    print(f"\ndone. {elapsed:.1f}s. wrote {out_dir}")
    return metadata
