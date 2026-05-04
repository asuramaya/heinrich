"""Character direction discovery in weight space — heinrich's safety-direction
methodology applied to image-model identity capture.

The core hypothesis: a "character" (e.g. a specific person, with all their
distinctive features) lives in a low-rank subspace of the model's weights,
analogous to how safety lives in a low-rank subspace of activations. The
minimal LoRA that reproduces a character is the principal axis of the
gradient signal that ties this person's caption to this person's image,
relative to what the model already knows.

We use **null-conditional self-contrast** — no external baseline pool. For
each (image, caption) pair, we compute the gradient under the actual
caption and the gradient under the empty/null caption on the same image at
the same timestep, then subtract. The difference isolates I(caption; image
| model_prior) — what the caption tells the model that it didn't already
know from its unconditional prior. This is exactly the signal classifier-
free guidance is built on, materialized as a weight-space direction.

    direction = SVD( mean[ grad_cond(x, caption) − grad_uncond(x, "") ] ).top_r()

Standard LoRA training discovers this direction iteratively via SGD. This
module discovers it directly via two forward+backward passes per image
(conditional + null) followed by SVD — no training, no external baseline,
all heinrich falsification diagnostics applicable.

Pipeline:
    1. Load FLUX.2 DiT (frozen base, gradient enabled only on target matrices).
    2. VAE-encode target images.
    3. Text-encode each caption AND a null caption (once, with text encoder
       evicted to CPU after).
    4. For each image: at each sampled timestep, run TWO forwards on the
       same noisy latent — once with the caption embedding, once with the
       null embedding. Backward each, take the per-matrix gradient
       difference, accumulate into a CPU fp32 buffer.
    5. Average across all (image × timestep) pairs.
    6. Per-matrix SVD truncated to rank r → LoRA factors (A, B) such that
       A @ B ≈ accumulated_diff.
    7. Save in diffusers/PEFT-compatible LoRA safetensors format.

This is an experiment, not a battle-tested trainer. Memory budget is 12 GB
VRAM; target_modules and resolution must be chosen accordingly. Use the
heinrich profile-discover-character CLI command which sets reasonable
defaults.
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np


CHARACTER_DIRECTION_VERSION = "0.1"


# Default target modules for FLUX.2-Klein DiT: only the double-stream
# attention projections (image side + text side). 5 blocks × 8 modules
# = 40 matrices, each 3072×3072. ~360M target params, ~1.5 GB in fp32
# CPU accumulator. Single-stream blocks (with their fused 27648-wide
# to_qkv_mlp_proj) are excluded for the first experiment — would more
# than double memory and the discovery hypothesis is testable without
# them.
DEFAULT_TARGET_MODULES_FLUX2 = [
    "transformer_blocks.*.attn.to_q",
    "transformer_blocks.*.attn.to_k",
    "transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.attn.to_out.0",
    "transformer_blocks.*.attn.add_q_proj",
    "transformer_blocks.*.attn.add_k_proj",
    "transformer_blocks.*.attn.add_v_proj",
    "transformer_blocks.*.attn.to_add_out",
]


def _matches_pattern(name: str, patterns: list[str]) -> bool:
    """Wildcard-* match for module names. Cheap fnmatch substitute."""
    import fnmatch
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def _enumerate_target_modules(transformer, patterns: list[str]) -> list[tuple[str, "torch.nn.Module"]]:
    """Walk the transformer, return (name, module) for every Linear matching the patterns."""
    targets = []
    for name, module in transformer.named_modules():
        cls = type(module).__name__
        if cls != "Linear":
            continue
        if _matches_pattern(name, patterns):
            targets.append((name, module))
    return targets


def _set_grad_only_on_targets(transformer, target_names: set[str]) -> None:
    """Freeze all params, then re-enable .requires_grad only on target weights."""
    for p in transformer.parameters():
        p.requires_grad_(False)
    for name, module in transformer.named_modules():
        if name in target_names:
            module.weight.requires_grad_(True)


def _vae_encode_and_pack(pipe, image_path: Path, *, resolution: int, dtype, device, generator):
    """Load → resize → preprocess → VAE encode → patchify → BN-normalize → pack.

    Returns (packed_latents [B, seq_len, C], img_ids [seq_len, 4]) ready for the DiT.
    Uses the pipeline's own helpers so the contract matches inference exactly.
    """
    import torch
    from PIL import Image
    image = Image.open(image_path).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        # _encode_vae_image returns patchified + BN-normalized latents [B, 128, H/2, W/2]
        image_latents_unpacked = pipe._encode_vae_image(tensor, generator)
        # latent_ids depend on (B, C, H, W) shape — same shape as unpacked latents
        latent_ids = pipe._prepare_latent_ids(image_latents_unpacked).to(device)
        packed = pipe._pack_latents(image_latents_unpacked)  # [B, H*W, C]
    return packed, latent_ids


def _text_encode_caption(pipe, caption: str):
    """Run encode_prompt → returns (prompt_embeds, text_ids)."""
    import torch
    with torch.no_grad():
        prompt_embeds, text_ids = pipe.encode_prompt(prompt=caption, device="cuda",
                                                       num_images_per_prompt=1)
    return prompt_embeds, text_ids


def _flow_matching_target(x0, noise, sigma):
    """Flow-matching velocity target: v = noise - x0 (the rectified flow direction)."""
    return noise - x0


def _flow_matching_input(x0, noise, sigma):
    """Flow-matching noisy input: x_t = (1 - sigma) * x0 + sigma * noise."""
    return (1 - sigma) * x0 + sigma * noise


def _forward_and_grad(transformer, x_t, timestep_norm, prompt_embeds, txt_ids, img_ids, v_target):
    """One DiT forward + flow-matching loss + backward. Sets .grad on enabled params.

    Args (FLUX.2 contract):
        x_t: packed noisy latents [B, image_seq_len, C]
        timestep_norm: timestep / 1000 (in [0, 1] range), shape [B]
        prompt_embeds: text embeddings [B, text_seq_len, hidden]
        txt_ids: text RoPE positions [text_seq_len, 4]
        img_ids: image RoPE positions [image_seq_len, 4]
        v_target: flow-matching velocity target [B, image_seq_len, C]
    """
    import torch
    import torch.nn.functional as F
    out = transformer(
        hidden_states=x_t,
        timestep=timestep_norm,
        guidance=None,
        encoder_hidden_states=prompt_embeds,
        txt_ids=txt_ids,
        img_ids=img_ids,
        return_dict=False,
    )
    full = out[0] if isinstance(out, (tuple, list)) else out
    # DiT output concatenates [text_tokens, image_tokens] internally; slice the image portion
    v_pred = full[:, : x_t.size(1)]
    loss = F.mse_loss(v_pred.float(), v_target.float())
    loss.backward()


def _read_and_zero_grads(transformer, target_modules_dict: dict[str, "torch.nn.Module"]):
    """Detach all target-module grads to CPU fp32 numpy, then zero them. Returns dict."""
    import torch
    out = {}
    for name, module in target_modules_dict.items():
        if module.weight.grad is None:
            continue
        out[name] = module.weight.grad.detach().to(dtype=torch.float32, device="cpu").numpy()
        module.weight.grad = None
    return out


def _capture_one_image_grad(
    transformer,
    packed_latent,         # [B, image_seq_len, C] clean packed latent
    img_ids,               # [image_seq_len, 4]
    prompt_embeds_cond,    # text embedding for the actual caption
    txt_ids_cond,          # [text_seq_len, 4]
    prompt_embeds_null,    # text embedding for the empty/null caption
    txt_ids_null,
    timesteps_to_sample: list[float],
    target_modules_dict: dict[str, "torch.nn.Module"],
    accumulator: dict[str, "np.ndarray"],
    *,
    n_samples_count: list,
    device,
    dtype,
    seed: int,
):
    """For one image, run TWO forward+backward passes per timestep — conditional
    and null — and accumulate the per-matrix gradient *difference*.
    """
    import torch
    rng = torch.Generator(device=device).manual_seed(seed)
    for sigma in timesteps_to_sample:
        noise = torch.randn(packed_latent.shape, generator=rng, device=device, dtype=dtype)
        x_t = _flow_matching_input(packed_latent, noise, sigma)
        v_target = _flow_matching_target(packed_latent, noise, sigma)
        # FLUX.2 expects timestep in [0, 1] range (the pipeline divides t/1000).
        timestep_norm = torch.tensor([sigma], device=device, dtype=dtype)

        # Conditional pass
        _forward_and_grad(transformer, x_t, timestep_norm, prompt_embeds_cond,
                            txt_ids_cond, img_ids, v_target)
        grads_cond = _read_and_zero_grads(transformer, target_modules_dict)

        # Null pass — same x_t, same target, only the conditioning changes
        _forward_and_grad(transformer, x_t, timestep_norm, prompt_embeds_null,
                            txt_ids_null, img_ids, v_target)
        grads_null = _read_and_zero_grads(transformer, target_modules_dict)

        # Accumulate the difference (caption-specific signal)
        for name, g_cond in grads_cond.items():
            g_null = grads_null.get(name, np.zeros_like(g_cond))
            diff = g_cond - g_null
            if name in accumulator:
                accumulator[name] += diff
            else:
                accumulator[name] = diff.copy()
        n_samples_count[0] += 1


def _per_matrix_svd_lora(
    grad_diff: dict[str, np.ndarray],
    rank: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Decompose each matrix's gradient difference into rank-r LoRA factors.

    Given grad_diff[name] = D of shape [out, in], compute SVD U, S, Vt and
    return (A, B) such that A @ B ≈ D, with A: [out, r], B: [r, in].
    Specifically: A = U[:, :r] · diag(S[:r])^{1/2}, B = diag(S[:r])^{1/2} · Vt[:r, :].
    """
    factors = {}
    for name, D in grad_diff.items():
        U, S, Vt = np.linalg.svd(D, full_matrices=False)
        r = min(rank, len(S))
        S_sqrt = np.sqrt(S[:r])
        A = U[:, :r] * S_sqrt[None, :]   # [out, r]
        B = S_sqrt[:, None] * Vt[:r, :]  # [r, in]
        factors[name] = (A.astype(np.float32), B.astype(np.float32))
    return factors


def _save_lora_safetensors(
    factors: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: str,
    metadata: dict,
) -> None:
    """Save in diffusers/PEFT-compatible format. Keys:
        {module_path}.lora_A.weight  → A^T (PEFT stores as [r, in])
        {module_path}.lora_B.weight  → B^T (PEFT stores as [out, r])
    Wait: diffusers convention is the opposite of PEFT base — let me check.
    For diffusers's Flux LoRA loading via load_lora_weights():
        transformer.{name}.lora_A.weight: shape [r, in_features]
        transformer.{name}.lora_B.weight: shape [out_features, r]
    With ΔW = lora_B @ lora_A. We have A = U_r·sqrt(S_r) [out, r] and B = sqrt(S_r)·Vt_r [r, in].
    Mapping: lora_A (in PEFT) = our B (the 'down' projection [r, in]).
             lora_B (in PEFT) = our A (the 'up' projection [out, r]).
    """
    import torch
    from safetensors.torch import save_file
    state = {}
    for name, (A, B) in factors.items():
        # A: [out, r] ('up' = lora_B in PEFT)
        # B: [r, in]  ('down' = lora_A in PEFT)
        diffusers_key = f"transformer.{name}"
        state[f"{diffusers_key}.lora_A.weight"] = torch.from_numpy(B)  # [r, in]
        state[f"{diffusers_key}.lora_B.weight"] = torch.from_numpy(A)  # [out, r]
    meta_str = {k: json.dumps(v) if not isinstance(v, str) else v
                  for k, v in metadata.items()}
    save_file(state, output_path, metadata=meta_str)


def discover_character(
    pipeline_id: str,
    refs_dir: str,
    output: str,
    *,
    rank: int = 32,
    target_modules: list[str] | None = None,
    resolution: int = 512,
    n_timesteps: int = 4,
    timestep_range: tuple[float, float] = (0.2, 0.8),
    seed: int = 42,
    dtype: str = "bfloat16",
    max_target_images: int | None = None,
    null_caption: str = "",
) -> dict:
    """Discover the character direction in weight space and save as a LoRA.

    Self-contrastive: per (image, caption) pair, the gradient is computed
    twice — once with the actual caption and once with a null caption — and
    only the difference is accumulated. The accumulated diff is the
    caption-conditional signal beyond the model's unconditional prior, SVD'd
    to rank r as a minimal LoRA.

    Args:
        pipeline_id: HF model id of the FLUX.2 pipeline (e.g. FLUX.2-klein-base-4B)
        refs_dir: directory of (image, caption) pairs for the target character
        output: path to write the LoRA .safetensors file
        rank: low-rank truncation per matrix (default 32)
        target_modules: fnmatch patterns on module names.
            Default: DEFAULT_TARGET_MODULES_FLUX2 (double-stream attention).
        resolution: image side for VAE encoding + DiT forward (default 512 to fit
            in 12 GB VRAM with gradient checkpointing)
        n_timesteps: per image, sample this many timesteps and average their
            gradient contribution (default 4)
        timestep_range: (sigma_min, sigma_max) range for timestep sampling
        seed: RNG seed
        dtype: model dtype during the gradient pass (bfloat16 default)
        max_target_images: cap on pool size (None = use all)
        null_caption: text used for the null-conditional forward (default empty
            string). The caption-conditional gradient is contrasted against this.
    """
    import torch
    # Force-resolve transformers lazy imports (FLUX.2 multimodal encoder)
    from transformers import (
        Mistral3ForConditionalGeneration,  # noqa: F401
        PixtralProcessor,  # noqa: F401
        PixtralImageProcessor,  # noqa: F401
        PixtralImageProcessorPil,  # noqa: F401
        PixtralVisionModel,  # noqa: F401
    )
    from diffusers import DiffusionPipeline
    from .flux2_ref import load_paired_pool

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if target_modules is None:
        target_modules = list(DEFAULT_TARGET_MODULES_FLUX2)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype]

    # ---- Load pool ----
    target_pool = load_paired_pool(refs_dir)
    if max_target_images is not None:
        target_pool = target_pool[:max_target_images]
    if not target_pool:
        raise ValueError(f"no (image, caption) pairs in refs_dir={refs_dir}")

    print(f"target pool: {len(target_pool)} images from {refs_dir}")
    print(f"null contrast caption: {null_caption!r}")

    # ---- Load pipeline ----
    print(f"loading {pipeline_id}")
    pipe = DiffusionPipeline.from_pretrained(pipeline_id, torch_dtype=torch_dtype)
    print(f"  pipeline class: {type(pipe).__name__}")
    transformer = pipe.transformer
    print(f"  DiT class: {type(transformer).__name__}")
    print(f"  DiT params: {sum(p.numel() for p in transformer.parameters()) / 1e9:.2f}B")

    # ---- Identify target modules ----
    target_list = _enumerate_target_modules(transformer, target_modules)
    target_names = [n for n, _ in target_list]
    target_names_set = set(target_names)
    if not target_names:
        raise ValueError(f"no modules matched target patterns: {target_modules}")
    n_target_params = sum(m.weight.numel() for _, m in target_list)
    print(f"target modules: {len(target_names)} matrices, {n_target_params / 1e6:.1f}M params")
    print(f"  fp32 CPU accumulator: {n_target_params * 4 / 1e9:.2f} GB")

    # ---- Freeze + enable grad on targets ----
    _set_grad_only_on_targets(transformer, target_names_set)
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
        print("  gradient checkpointing: enabled")

    # ---- VAE-encode pool images (move VAE to GPU briefly) ----
    pipe.vae.to("cuda")
    print(f"VAE-encoding {len(target_pool)} pool images at {resolution}×{resolution}")
    target_packed_latents = []
    target_img_ids = []
    target_captions = []
    vae_gen = torch.Generator(device="cuda").manual_seed(seed)
    for img_path, caption in target_pool:
        packed, ids = _vae_encode_and_pack(pipe, Path(img_path),
                                              resolution=resolution, dtype=torch_dtype,
                                              device="cuda", generator=vae_gen)
        target_packed_latents.append(packed.cpu())
        target_img_ids.append(ids.cpu())
        target_captions.append(caption)
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    # ---- Text-encode all captions + null caption ----
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.to("cuda")
    print(f"text-encoding {len(target_captions) + 1} captions (incl. null)")
    target_text = []
    for cap in target_captions:
        prompt_embeds, text_ids = _text_encode_caption(pipe, cap)
        target_text.append((prompt_embeds.detach().cpu(), text_ids.detach().cpu()))
    null_prompt_embeds, null_text_ids = _text_encode_caption(pipe, null_caption)
    null_text = (null_prompt_embeds.detach().cpu(), null_text_ids.detach().cpu())
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    # ---- Move DiT to GPU for gradient phase ----
    transformer.to("cuda")

    # ---- Sample timesteps (uniform in (sigma_min, sigma_max)) ----
    rng = np.random.default_rng(seed)
    timesteps_per_image = [
        list(rng.uniform(timestep_range[0], timestep_range[1], size=n_timesteps).tolist())
        for _ in range(len(target_packed_latents))
    ]

    target_modules_dict = {n: m for n, m in target_list}

    null_prompt_emb = null_text[0].to("cuda")
    null_text_ids_gpu = null_text[1].to("cuda")

    print(f"\n=== capturing null-conditional contrastive gradients "
          f"({len(target_packed_latents)} images × {n_timesteps} timesteps × 2 forwards) ===")
    accumulator: dict[str, np.ndarray] = {}
    n_count = [0]
    t0 = time.time()
    for i, (lat_cpu, ids_cpu, (txt_cpu, txt_ids_cpu)) in enumerate(
            zip(target_packed_latents, target_img_ids, target_text)):
        lat = lat_cpu.to("cuda")
        img_ids = ids_cpu.to("cuda")
        cond_prompt_emb = txt_cpu.to("cuda")
        cond_text_ids = txt_ids_cpu.to("cuda")
        print(f"  [{i+1}/{len(target_packed_latents)}] {n_timesteps} timesteps × (cond + null)")
        _capture_one_image_grad(
            transformer, lat, img_ids,
            cond_prompt_emb, cond_text_ids,
            null_prompt_emb, null_text_ids_gpu,
            timesteps_per_image[i], target_modules_dict, accumulator,
            n_samples_count=n_count, device="cuda", dtype=torch_dtype,
            seed=seed + i,
        )
        del lat, img_ids, cond_prompt_emb, cond_text_ids
        torch.cuda.empty_cache()
        gc.collect()
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s, {n_count[0]} (image × timestep) samples")

    # Average accumulated diff
    grad_diff: dict[str, np.ndarray] = {}
    for name, g in accumulator.items():
        grad_diff[name] = g / max(n_count[0], 1)
    print(f"\n=== per-matrix SVD (rank={rank}) ===")

    # ---- SVD per matrix to rank-r LoRA factors ----
    factors = _per_matrix_svd_lora(grad_diff, rank)

    # ---- Diagnostics: log per-matrix singular value structure ----
    print("\nper-matrix SVD spectrum (top-10 singular values, rank-r capture):")
    for name in list(grad_diff.keys())[:6]:  # sample 6 matrices for log
        D = grad_diff[name]
        S = np.linalg.svd(D, full_matrices=False, compute_uv=False)
        total = (S ** 2).sum()
        kept = (S[:rank] ** 2).sum()
        ratio = kept / total if total > 0 else 0
        print(f"  {name}  top10={[f'{s:.2f}' for s in S[:10]]}  "
              f"rank-{rank} variance fraction: {ratio:.4f}")

    # ---- Save ----
    metadata = {
        "version": CHARACTER_DIRECTION_VERSION,
        "pipeline_id": pipeline_id,
        "refs_dir": str(refs_dir),
        "method": "null_conditional_self_contrast",
        "n_target_images": str(len(target_pool)),
        "rank": str(rank),
        "resolution": str(resolution),
        "n_timesteps": str(n_timesteps),
        "null_caption": null_caption,
        "target_modules": ",".join(target_modules),
    }
    _save_lora_safetensors(factors, str(out_path), metadata)
    print(f"\nwrote LoRA: {out_path}")
    return {"metadata": metadata, "n_matrices": len(factors)}
