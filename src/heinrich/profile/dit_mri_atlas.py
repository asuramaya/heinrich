"""Build an identity atlas from contrastive ref captures.

The atlas is the data structure that replaces a LoRA. Given:
  - identity_dir: dit-mri-ref output for the target identity (e.g. 48 Subject images)
  - baseline_dir: dit-mri-ref output for matched-protocol generic images

For each (layer, step, image-position) cell we compute:
  - direction:   mean(identity_caps) - mean(baseline_caps) at that cell
  - subspace:    top-K right singular vectors of (identity_caps - mean(baseline)) — rank-K
                 manifold capturing identity variation (pose, expression, etc.)
  - strength:    ||direction||  — used to rank cells by identity-relevance

At inference (dit-steer), we inject `direction` (or project into `subspace`) at each
hooked cell to push the DiT trajectory toward the target identity, no trigger word
needed.

Image positions only: text positions vary in length across captures (image2img captions
> txt2img captions), and identity-relevant changes happen at image-token positions
anyway. Last `n_image_tokens` of the seq are image positions in this DiT.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


ATLAS_VERSION = "0.1"


def _load_ref(input_dir: str):
    p = Path(input_dir)
    with open(p / "metadata.json") as fh:
        meta = json.load(fh)
    with open(p / "prompts.jsonl") as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]
    residuals = {}
    for i, _ in enumerate(prompts):
        residuals[i] = np.load(p / f"prompt_{i:03d}_residuals.npy", mmap_mode="r")
    return meta, prompts, residuals


def _streaming_mean_per_layer(residuals: dict, cfg_branch: int,
                                n_image_tokens: int, layer_idx: int):
    """Compute per-cell mean over N captures for a SINGLE layer.

    Returns: [n_steps, n_image_tokens, hidden] fp32, ~35 MB for L=30 / T=9 / P=256 / H=3840.

    Memory: per-prompt slice is [n_steps, n_image_tokens, hidden] fp16 = ~17 MB.
    Layer-by-layer streaming keeps peak RAM under 100 MB regardless of N.
    """
    sample = residuals[0]
    _, n_steps, _, seq_len, hidden = sample.shape
    img_lo = seq_len - n_image_tokens
    P = len(residuals)

    mean = np.zeros((n_steps, n_image_tokens, hidden), dtype=np.float32)
    for i in range(P):
        sliced = residuals[i][layer_idx, :, cfg_branch, img_lo:, :].astype(np.float32)
        mean += sliced
        del sliced
    mean /= P
    return mean


def build_atlas(
    identity_dir: str,
    baseline_dir: str,
    output: str,
    *,
    cfg_branch: int = 0,
    rank: int = 1,
    n_image_tokens: int | None = None,
) -> dict:
    """Build identity atlas from identity refs vs baseline refs.

    Args:
        identity_dir: dit-mri-ref output for the target subject (e.g. Subject training images)
        baseline_dir: dit-mri-ref output for matched-protocol generic images
        output: path to write atlas.npz
        rank: 1 = direction-only (fastest, simplest). >1 = SVD subspace per cell.
        n_image_tokens: trailing positions that are image tokens (default: infer from
            metadata.width/height)
    """
    id_meta, id_prompts, id_res = _load_ref(identity_dir)
    bl_meta, bl_prompts, bl_res = _load_ref(baseline_dir)

    # Sanity
    if id_meta["dim"] != bl_meta["dim"]:
        raise ValueError(f"hidden dim mismatch: id={id_meta['dim']} vs bl={bl_meta['dim']}")
    if id_meta["n_layers"] != bl_meta["n_layers"]:
        raise ValueError(f"n_layers mismatch: id={id_meta['n_layers']} vs bl={bl_meta['n_layers']}")
    if id_meta["num_inference_steps"] != bl_meta["num_inference_steps"]:
        raise ValueError(f"steps mismatch: id={id_meta['num_inference_steps']} vs "
                         f"bl={bl_meta['num_inference_steps']}")

    if n_image_tokens is None:
        w = id_meta.get("width", 512)
        h = id_meta.get("height", 512)
        n_image_tokens = (w // 16 // 2) * (h // 16 // 2)

    print(f"identity:  {len(id_prompts)} refs from {identity_dir}")
    print(f"baseline:  {len(bl_prompts)} refs from {baseline_dir}")
    print(f"shape:     {id_meta['n_layers']} layers × {id_meta['num_inference_steps']} steps × "
          f"{n_image_tokens} image positions × {id_meta['dim']} hidden")
    print(f"rank:      {rank}")
    print(f"cfg_branch: {cfg_branch}")
    print()

    t0 = time.time()
    sample = id_res[0]
    n_layers_total, n_steps, _, seq_len, hidden = sample.shape
    id_img_lo = seq_len - n_image_tokens
    bl_sample = bl_res[0]
    bl_img_lo = bl_sample.shape[3] - n_image_tokens

    # Layer-by-layer streaming: never materialize [L, T, P_img, H] for the whole stack.
    # Per-layer fp32 buffer is [n_steps, P_img, H] ≈ 35 MB, peak RAM well under 1 GB.
    direction = np.zeros((n_layers_total, n_steps, n_image_tokens, hidden), dtype=np.float16)
    cell_strength = np.zeros((n_layers_total, n_steps, n_image_tokens), dtype=np.float32)
    print(f"streaming per-layer means and direction (L={n_layers_total}, T={n_steps}, "
          f"P_img={n_image_tokens}, H={hidden})...")
    for li in range(n_layers_total):
        id_layer_mean = _streaming_mean_per_layer(id_res, cfg_branch, n_image_tokens, li)
        bl_layer_mean = _streaming_mean_per_layer(bl_res, cfg_branch, n_image_tokens, li)
        layer_dir = id_layer_mean - bl_layer_mean  # [T, P_img, H] fp32
        direction[li] = layer_dir.astype(np.float16)
        cell_strength[li] = np.linalg.norm(layer_dir, axis=2)
        del id_layer_mean, bl_layer_mean, layer_dir
        if (li + 1) % 5 == 0 or li == n_layers_total - 1:
            print(f"  layer {li + 1}/{n_layers_total}")
    print()

    # Per-(L, T) cell aggregate strength (mean over image positions)
    LT_strength = cell_strength.mean(axis=2)  # [L, T]

    print("=" * 76)
    print("(layer × step) atlas strength heatmap (mean ||direction|| over image positions)")
    print("=" * 76)
    n_layers_local = n_layers_total
    header = "layer  " + " ".join(f"  T{t:02d}  " for t in range(n_steps))
    print(header)
    print("-" * len(header))
    for L in range(n_layers_local):
        row = " ".join(f"{LT_strength[L, t]:6.2f}" for t in range(n_steps))
        bar_len = int(round(40 * LT_strength[L].mean() / (LT_strength.max() + 1e-9)))
        bar = "█" * bar_len
        print(f"L{L:02d}    {row}   |{bar}")
    print()

    top_LT = np.argsort(LT_strength.ravel())[::-1][:10]
    print("top 10 strongest (L, T) cells:")
    for fi in top_LT:
        L_idx, T_idx = (int(x) for x in np.unravel_index(fi, LT_strength.shape))
        print(f"  L{L_idx:02d}T{T_idx}  strength={LT_strength[L_idx, T_idx]:.3f}")
    print()

    # Rank > 1 SVD path is currently disabled in the streaming refactor — needs the
    # baseline mean per (L, T, P) cell, which we now compute and discard per layer.
    # Future: re-enable by saving baseline mean to disk per layer or recomputing inline.
    subspace = None
    sigma_values = None
    if rank > 1:
        print(f"WARNING: rank>1 SVD not supported in streaming mode yet; using rank-1 direction only.")

    elapsed = time.time() - t0
    print(f"atlas computed in {elapsed:.1f}s")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "direction": direction,  # already fp16
        "cell_strength": cell_strength.astype(np.float32),
        "LT_strength": LT_strength.astype(np.float32),
    }
    if subspace is not None:
        save_kwargs["subspace"] = subspace.astype(np.float16)
        save_kwargs["sigma_values"] = sigma_values.astype(np.float32)

    np.savez_compressed(out_path, **save_kwargs)

    metadata = {
        "version": ATLAS_VERSION,
        "identity_dir": str(identity_dir),
        "baseline_dir": str(baseline_dir),
        "n_identity_refs": len(id_prompts),
        "n_baseline_refs": len(bl_prompts),
        "n_layers": int(n_layers_total),
        "n_steps": int(n_steps),
        "n_image_tokens": int(n_image_tokens),
        "img_lo_identity": int(id_img_lo),
        "img_lo_baseline": int(bl_img_lo),
        "hidden": int(hidden),
        "rank": int(rank),
        "cfg_branch": int(cfg_branch),
        "elapsed_s": round(elapsed, 1),
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {meta_path}")
    return metadata


def load_atlas(atlas_path: str) -> tuple[dict, dict]:
    """Load atlas.npz + meta.json. Arrays returned as fp16/fp32 numpy."""
    p = Path(atlas_path)
    meta_p = p.with_suffix(".meta.json")
    with open(meta_p) as fh:
        meta = json.load(fh)
    arrays = dict(np.load(p))
    return meta, arrays
