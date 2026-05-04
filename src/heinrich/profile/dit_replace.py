"""PAFF-faithful spatial token replacement: latent-level, once per step.

PAFF mechanism (Wang et al. 2503.12590):
    X̂ = X·(1−M) + X_ref·M   applied at the LATENT level, before the transformer
                              processes the step. ONE intervention per step.

Implementation note: we use diffusers' `callback_on_step_end` to modify the latent
between denoising steps. The callback receives the latent at σ_(t+1), we replace
masked spatial positions with the reference image's latent at the same σ_(t+1),
and the pipe continues from there with our modified latent.

This replaces the previous (broken) forward_hook-based design, which was
intervening at every transformer block's residual output (30 replacements per step)
instead of once per step at the latent.

Reference latent at σ_t is built by adding FlowMatch noise to the clean VAE-encoded
reference: x_t = (1−σ_t)·x_0 + σ_t·ε. This guarantees regime-matched replacement.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


DIT_REPLACE_VERSION = "0.2"


def _build_latent_mask(mask_kind: str, latent_h: int, latent_w: int, *,
                        inner_frac: float = 0.6,
                        feather: float = 0.0,
                        oval: bool = False) -> np.ndarray:
    """Build a mask at the latent grid resolution.

    Returns float32 mask in [0, 1]. Hard binary if feather=0 and oval=False;
    smooth/oval otherwise. Using floats lets us linearly blend between original
    and reference latents at edges, eliminating the hard rectangular paste artifact.
    """
    M = np.zeros((latent_h, latent_w), dtype=np.float32)
    if mask_kind == "whole":
        M[:] = 1.0
        return M

    inner_h = max(1, int(round(latent_h * inner_frac)))
    inner_w = max(1, int(round(latent_w * inner_frac)))
    if mask_kind == "central":
        cy, cx = latent_h / 2.0, latent_w / 2.0
        ry, rx = inner_h / 2.0, inner_w / 2.0
    elif mask_kind == "face":
        # Upper-central, oval-shaped face region
        cy = latent_h * 0.35   # face center is upper-center
        cx = latent_w / 2.0
        ry = (latent_h * 0.5 - latent_h / 8) / 2.0
        rx = inner_w / 2.0
    else:
        raise ValueError(f"unknown mask_kind: {mask_kind}")

    yy, xx = np.meshgrid(np.arange(latent_h), np.arange(latent_w), indexing="ij")
    if oval or mask_kind == "face":
        # Distance in normalized ellipse coords; <=1 inside, >1 outside
        d = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    else:
        # Rectangle distance: 0 inside, increases outside (in fraction of half-extent)
        dy = np.maximum(0, np.abs(yy - cy) - ry) / max(ry, 1e-6)
        dx = np.maximum(0, np.abs(xx - cx) - rx) / max(rx, 1e-6)
        d = np.sqrt(dy ** 2 + dx ** 2)

    if feather > 0.0:
        # Smooth falloff: 1 at d<=1, 0 at d>=1+feather, linear in between
        M = np.clip((1.0 + feather - d) / feather, 0.0, 1.0).astype(np.float32)
    else:
        M = (d <= 1.0).astype(np.float32)
    return M


def _dilate_mask_2d(mask: np.ndarray, k: int = 2) -> np.ndarray:
    if k <= 0:
        return mask.copy()
    out = mask.copy()
    for _ in range(k):
        shifted = np.zeros_like(out)
        shifted[:-1, :] |= out[1:, :]
        shifted[1:, :] |= out[:-1, :]
        shifted[:, :-1] |= out[:, 1:]
        shifted[:, 1:] |= out[:, :-1]
        out = out | shifted
    return out


def _vae_encode_clean(pipe, image_path: Path, *,
                       width: int, height: int, seed: int):
    import torch
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    img_t = torch.from_numpy(arr.transpose(2, 0, 1)[None])
    vae = pipe.vae
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    img_t = img_t.to(device=device, dtype=dtype)
    with torch.no_grad():
        latent_dist = vae.encode(img_t).latent_dist
        x_0 = latent_dist.sample(generator=torch.Generator(device=device).manual_seed(seed))
        scaling = getattr(vae.config, "scaling_factor", 1.0)
        shift = getattr(vae.config, "shift_factor", None)
        if shift is not None:
            x_0 = (x_0 - shift) * scaling
        else:
            x_0 = x_0 * scaling
    return x_0


def _get_generation_sigmas(pipe, num_inference_steps: int):
    """Get pipe's default sigma schedule (length N+1) for N inference steps."""
    pipe.scheduler.set_timesteps(num_inference_steps, device="cuda")
    return pipe.scheduler.sigmas.detach().cpu().numpy()


def replace_during_generation(
    pipeline_id: str,
    ref_image: str,
    ref_caption: str,
    prompts_path: str,
    output: str,
    *,
    replacement_steps: int = 4,
    start_step: int = 22,
    mask_kind: str = "whole",
    mask_dilate: int = 0,
    cfg_branch_only: bool = True,
    seed: int = 42,
    num_inference_steps: int = 28,
    width: int = 512,
    height: int = 512,
    dtype: str = "bfloat16",
    offload: str = "sequential",
    feather: float = 0.0,
    inner_frac: float = 0.6,
    oval: bool = False,
    # patch_perturb / ref_sigma kept as kwargs for CLI compatibility but unused now
    patch_perturb: bool = True,
    ref_sigma: float = 0.0,
) -> dict:
    """PAFF-style latent-space replacement at chosen mid-noise generation steps.

    Args:
        replacement_steps: how many consecutive steps get latent replacement
        start_step: first step index at which to replace. For Z-Image full at 28
            steps, mid-noise (σ ≈ 0.5–0.2) lives at steps 22–26.
        mask_kind: 'whole' / 'central' / 'face' at latent-grid resolution
    """
    import torch
    from diffusers import ZImagePipeline

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(prompts_path) as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]

    print(f"loading {pipeline_id} ...")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    pipe = ZImagePipeline.from_pretrained(pipeline_id, torch_dtype=dtype_map[dtype])

    # Phase 1: VAE-encode reference (only VAE briefly on GPU)
    print(f"phase 1a: VAE-encoding reference image")
    pipe.vae.to("cuda")
    x_0 = _vae_encode_clean(pipe, Path(ref_image), width=width, height=height, seed=seed)
    x_0_cpu = x_0.cpu()
    del x_0
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    latent_h = int(x_0_cpu.shape[2])
    latent_w = int(x_0_cpu.shape[3])
    print(f"  encoded latent shape: {tuple(x_0_cpu.shape)}  (latent grid: {latent_h}×{latent_w})")

    # Phase 1b: build mask at latent resolution. Float in [0,1] supports feathered
    # blend so the boundary isn't a hard paste.
    mask_f = _build_latent_mask(mask_kind, latent_h=latent_h, latent_w=latent_w,
                                  inner_frac=inner_frac, feather=feather, oval=oval)
    print(f"  mask: kind={mask_kind} feather={feather} oval={oval}  "
          f"mean weight={mask_f.mean():.3f}  fully-on={(mask_f == 1.0).mean() * 100:.1f}%  "
          f"partial={((mask_f > 0) & (mask_f < 1)).mean() * 100:.1f}%")

    # Phase 1c: enable offload
    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "none":
        pipe = pipe.to("cuda")
    else:
        raise ValueError(f"offload must be 'model'|'sequential'|'none', got {offload!r}")

    # Phase 1d: precompute reference noisy latents at each replacement step's σ
    gen_sigmas = _get_generation_sigmas(pipe, num_inference_steps)
    print(f"phase 1d: precomputing Subject latents at replacement sigmas")
    print(f"  full schedule: σ = [{gen_sigmas[0]:.3f}, ..., {gen_sigmas[-1]:.3f}] over {num_inference_steps} steps")

    # callback_on_step_end fires AFTER step_index t's denoising. The latent is at σ_{t+1}.
    # To replace at this point, we need ref's latent at σ_{t+1}.
    subject_latents_at_target = {}  # step_idx -> tensor at σ_{step_idx+1}
    x_0_cuda = x_0_cpu.to("cuda")
    for offset in range(replacement_steps):
        step_idx = start_step + offset
        target_sigma = float(gen_sigmas[step_idx + 1])  # σ at the END of step step_idx
        g = torch.Generator(device="cuda").manual_seed(seed + step_idx)
        noise = torch.randn(x_0_cuda.shape, generator=g, device="cuda", dtype=x_0_cuda.dtype)
        subject_x = (1.0 - target_sigma) * x_0_cuda + target_sigma * noise
        subject_latents_at_target[step_idx] = subject_x.cpu()  # cache on CPU
        print(f"  step {step_idx} → after-denoise σ={target_sigma:.4f}  ref noisy latent prepared")
    del x_0_cuda
    torch.cuda.empty_cache()

    # Phase 2: generation with callback_on_step_end (linear blend, not hard replace)
    mask_torch_cpu = torch.from_numpy(mask_f)  # [H, W] float

    all_meta = []
    t0 = time.time()
    for pi, p in enumerate(prompts):
        text = p["text"]
        print(f"\n[{pi + 1}/{len(prompts)}] {text!r}")

        def make_callback():
            def callback(pipe_obj, step_index, timestep, callback_kwargs):
                latents = callback_kwargs.get('latents')
                if latents is None:
                    return callback_kwargs
                if step_index in subject_latents_at_target:
                    subject_x = subject_latents_at_target[step_index].to(
                        device=latents.device, dtype=latents.dtype)
                    m = mask_torch_cpu.to(device=latents.device, dtype=latents.dtype)
                    m_b = m[None, None, :, :]   # [1, 1, H, W]
                    # Linear blend: latent = (1 - m) * latent + m * subject_x
                    if cfg_branch_only and latents.shape[0] >= 1:
                        new0 = (1.0 - m_b) * latents[0:1] + m_b * subject_x[0:1]
                        latents = torch.cat([new0, latents[1:]], dim=0) if latents.shape[0] > 1 else new0
                    else:
                        latents = (1.0 - m_b) * latents + m_b * subject_x
                    callback_kwargs['latents'] = latents
                return callback_kwargs
            return callback

        try:
            with torch.no_grad():
                result = pipe(
                    text,
                    generator=torch.Generator(device="cuda").manual_seed(seed + pi),
                    num_inference_steps=num_inference_steps,
                    width=width, height=height,
                    callback_on_step_end=make_callback(),
                    callback_on_step_end_tensor_inputs=['latents'],
                )
            image = result.images[0] if hasattr(result, 'images') else None
        except Exception as e:
            print(f"  ERROR during generation: {e}")
            raise

        if image is not None:
            image.save(out_dir / f"prompt_{pi:03d}.png")
        all_meta.append({"idx": pi, "text": text,
                          **{k: v for k, v in p.items() if k != "text"}})

    elapsed = time.time() - t0

    metadata = {
        "version": DIT_REPLACE_VERSION,
        "pipeline_id": pipeline_id,
        "ref_image": str(ref_image),
        "ref_caption": ref_caption,
        "n_prompts": len(prompts),
        "replacement_steps": replacement_steps,
        "start_step": start_step,
        "num_inference_steps": num_inference_steps,
        "mask_kind": mask_kind,
        "mask_dilate": mask_dilate,
        "cfg_branch_only": cfg_branch_only,
        "seed": seed,
        "width": width, "height": height,
        "elapsed_s": round(elapsed, 1),
    }
    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    with open(out_dir / "prompts.jsonl", "w") as fh:
        for m in all_meta:
            fh.write(json.dumps(m) + "\n")
    print(f"\ndone. {elapsed:.1f}s. wrote {out_dir}")
    return metadata
