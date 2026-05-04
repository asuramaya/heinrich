"""Analyze a dit-mri capture for per-(layer, timestep, position) identity discrimination.

Given residuals captured during multiple generations (same seed, varied identity slot
in the prompt), find:

  - Where in the (layer × timestep) grid the trajectories diverge most across categories
  - Whether the divergence concentrates on image-token positions vs text-token positions
  - The rank of the contrastive subspace at each high-divergence cell

This is the core measurement for whether mechanistic identity steering is feasible:
high-divergence cells with low subspace rank = clean steering target. High-divergence
cells with high rank = entangled, harder.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _load_run(input_dir: str) -> tuple[dict, list[dict], dict[int, np.ndarray]]:
    """Load metadata + prompts + memory-mapped residual arrays.

    Residuals are returned as np.memmap views — no RAM allocation until sliced.
    Each prompt's array is shape [n_layers, n_steps, batch, seq_len, hidden] fp16,
    typically several GB on disk. mmap_mode='r' makes slicing copy only the
    requested bytes into RAM.
    """
    p = Path(input_dir)
    with open(p / "metadata.json") as fh:
        meta = json.load(fh)
    with open(p / "prompts.jsonl") as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]
    residuals: dict[int, np.ndarray] = {}
    for i, _ in enumerate(prompts):
        residuals[i] = np.load(p / f"prompt_{i:03d}_residuals.npy", mmap_mode="r")
    return meta, prompts, residuals


def analyze_dit_mri(input_dir: str, *, cfg_branch: int = 0,
                    n_image_tokens: int | None = None) -> dict:
    """Compute per-(layer, timestep) divergence + position-wise breakdown.

    cfg_branch: 0 = conditional path (with prompt), 1 = unconditional. Default 0.
    n_image_tokens: how many trailing positions are image tokens. If None, we infer
        from metadata (seq_len - text/siglip/refiner positions). Currently we approximate
        as the last (W/16/2) * (H/16/2) positions = 256 for 512×512.
    """
    meta, prompts, residuals = _load_run(input_dir)

    # All prompts share shape since seed is fixed and prompt structure is identical
    sample = residuals[0]
    n_layers, n_steps, batch, seq_len, hidden = sample.shape
    print(f"dit-mri-analyze: {len(prompts)} prompts, layers={n_layers}, steps={n_steps}, "
          f"batch={batch}, seq_len={seq_len}, hidden={hidden}, cfg_branch={cfg_branch}")

    # Infer image-token count from generation resolution if not given.
    if n_image_tokens is None:
        w = meta.get("width", 1024)
        h = meta.get("height", 1024)
        n_image_tokens = (w // 16 // 2) * (h // 16 // 2)
    n_text_tokens = seq_len - n_image_tokens
    img_lo = seq_len - n_image_tokens  # image tokens are at the tail
    print(f"  inferred token layout: text/siglip/refiner [0:{img_lo}] + image [{img_lo}:{seq_len}]")
    print()

    P = len(prompts)
    cats = [p.get("category", "uncat") for p in prompts]
    unique_cats = sorted(set(cats))
    by_cat: dict[str, list[int]] = {c: [] for c in unique_cats}
    for i, c in enumerate(cats):
        by_cat[c].append(i)
    print(f"  categories: {dict((c, len(by_cat[c])) for c in unique_cats)}")
    print()

    # Memory plan: never materialize [P, L, T, seq, hidden] — that's 31 GB at fp32.
    # Instead, do two streaming passes:
    #   pass 1: per-prompt image-token-mean → img_mean [P, L, T, hidden] (~30 MB)
    #   pass 2: load only the focus (L, T) cell from all prompts for position-axis
    #
    # Each mmap'd slice [layers, steps, 1, img_tokens, hidden] is ~2 GB fp16.
    # We convert one at a time to fp32, mean over positions, drop. Peak: ~5 GB.

    print("=" * 76)
    print("global divergence: mean cosine BETWEEN category centroids, per (layer, step)")
    print("(lower = more divergence; image-token positions only)")
    print("=" * 76)

    # Pass 1: image-token mean per prompt
    img_mean = np.zeros((P, n_layers, n_steps, hidden), dtype=np.float32)
    for i in range(P):
        # Slice from mmap: [n_layers, n_steps, batch, img_tokens, hidden]
        sliced = residuals[i][:, :, cfg_branch, img_lo:, :]
        # Convert to fp32 and mean over image-token axis (axis 2 in this 4D view)
        img_mean[i] = sliced.astype(np.float32).mean(axis=2)
        del sliced  # release the fp32 temp ASAP
    # Centroids per category: [n_cats, n_layers, n_steps, hidden]
    centroids = np.zeros((len(unique_cats), n_layers, n_steps, hidden), dtype=np.float32)
    for ci, c in enumerate(unique_cats):
        centroids[ci] = img_mean[by_cat[c]].mean(axis=0)

    # Mean pairwise cosine between centroids per (layer, step)
    div_map = np.zeros((n_layers, n_steps), dtype=np.float32)
    for L in range(n_layers):
        for T in range(n_steps):
            sims = []
            for ci in range(len(unique_cats)):
                for cj in range(ci + 1, len(unique_cats)):
                    sims.append(_cos(centroids[ci, L, T], centroids[cj, L, T]))
            div_map[L, T] = float(np.mean(sims)) if sims else 1.0

    # Print: layers as rows, steps as columns
    header = "layer  " + " ".join(f"  T{t:02d} " for t in range(n_steps))
    print(header)
    print("-" * len(header))
    for L in range(n_layers):
        row = " ".join(f"{div_map[L, t]:+.3f}" for t in range(n_steps))
        # ASCII heatmap: lower (more divergent) = more "█"
        avg = float(div_map[L].mean())
        bar_len = int(round(20 * (1.0 - avg)))  # 1.0 cos = no bar; 0.0 = full bar
        bar = "█" * max(0, min(20, bar_len))
        print(f"L{L:02d}    {row}   |{bar}")
    print()

    # ===== 2. The headline pair: trigger_lora ↔ control_oov per cell =====
    target_pairs = [
        ("trigger_lora", "control_oov"),
        ("trigger_lora", "real_name"),
        ("trigger_lora", "generic"),
        ("real_name", "control_oov"),
    ]
    available = [(a, b) for (a, b) in target_pairs
                 if a in unique_cats and b in unique_cats]

    if available:
        print("=" * 76)
        print("category-pair divergence per (layer, step) — image tokens only")
        print("(shows: cos centroid similarity. <1.0 = divergence)")
        print("=" * 76)

        for a, b in available:
            ai = unique_cats.index(a)
            bi = unique_cats.index(b)
            grid = np.zeros((n_layers, n_steps), dtype=np.float32)
            for L in range(n_layers):
                for T in range(n_steps):
                    grid[L, T] = _cos(centroids[ai, L, T], centroids[bi, L, T])
            min_val = float(grid.min())
            min_L, min_T = (int(x) for x in np.unravel_index(grid.argmin(), grid.shape))
            print(f"\n  {a} ↔ {b}    min cos = {min_val:.4f} at (L{min_L}, T{min_T})")
            # Compact grid display: highlight low-cosine cells
            for L in range(n_layers):
                bar = ""
                for T in range(n_steps):
                    v = grid[L, T]
                    # ` ` = ~1.0, `.` = >0.95, `-` = >0.85, `+` = >0.7, `*` = >0.5, `#` = lower
                    if v > 0.99: ch = "·"
                    elif v > 0.95: ch = "."
                    elif v > 0.85: ch = "-"
                    elif v > 0.70: ch = "+"
                    elif v > 0.50: ch = "*"
                    else: ch = "#"
                    bar += ch
                print(f"    L{L:02d}  {bar}  ({grid[L].min():+.3f}..{grid[L].max():+.3f})")
        print()

    # ===== 3. Identity-token vs scaffold-token divergence =====
    # Question: when the encoder gives the DiT near-identical conditioning, does
    # the DiT pull identity from the (small) text-position differences, or from the
    # image-position differences (which started identical via same seed)?
    print("=" * 76)
    print("position-axis divergence: text/siglip/refiner positions vs image positions")
    print("at the most-divergent cell of (trigger_lora ↔ control_oov)")
    print("=" * 76)
    if ("trigger_lora", "control_oov") in [(a, b) for (a, b) in available]:
        ai = unique_cats.index("trigger_lora")
        bi = unique_cats.index("control_oov")
        # Find the most divergent (L, T) using image-token-pooled centroids:
        grid = np.zeros((n_layers, n_steps), dtype=np.float32)
        for L in range(n_layers):
            for T in range(n_steps):
                grid[L, T] = _cos(centroids[ai, L, T], centroids[bi, L, T])
        focus_L, focus_T = (int(x) for x in np.unravel_index(grid.argmin(), grid.shape))

        # At (focus_L, focus_T), compute centroid divergence per position.
        # Load only this (L, T) cell from each prompt — minimal RAM.
        cat_a_idx = by_cat["trigger_lora"]
        cat_b_idx = by_cat["control_oov"]

        def _cell_centroid(idx_list):
            acc = np.zeros((seq_len, hidden), dtype=np.float32)
            for ix in idx_list:
                acc += residuals[ix][focus_L, focus_T, cfg_branch].astype(np.float32)
            return acc / len(idx_list)

        a_centroid_full = _cell_centroid(cat_a_idx)
        b_centroid_full = _cell_centroid(cat_b_idx)
        per_pos_cos = np.array([_cos(a_centroid_full[p], b_centroid_full[p])
                                for p in range(seq_len)])

        text_block = per_pos_cos[:img_lo]
        img_block = per_pos_cos[img_lo:]
        print(f"  focus cell: (L{focus_L}, T{focus_T})")
        print(f"  text/siglip/refiner positions [0:{img_lo}]:  mean cos = {text_block.mean():+.4f},  "
              f"min cos = {text_block.min():+.4f},  argmin = {int(text_block.argmin())}")
        print(f"  image positions             [{img_lo}:{seq_len}]:  mean cos = {img_block.mean():+.4f},  "
              f"min cos = {img_block.min():+.4f},  argmin = {int(img_block.argmin()) + img_lo}")
        print()
        print("  most-divergent positions (top 10):")
        worst_pos = np.argsort(per_pos_cos)[:10]
        for p in worst_pos:
            kind = "image" if p >= img_lo else "text"
            print(f"    pos {int(p):4d} ({kind:5s}): cos = {per_pos_cos[p]:+.4f}")
        print()

    # ===== 4. Per-cell SVD: rank of identity subspace =====
    # At each (layer, step), stack all prompts' image-token-mean activations and
    # compute SVD of the centered matrix. Rank = number of singular values above
    # threshold (e.g. > 1% of largest). Low rank = clean subspace = good steering target.
    print("=" * 76)
    print("identity subspace rank per (layer, step)")
    print("(SVD of centered prompt-x-hidden matrix at image-pooled mean. Eff rank = "
          "# singular values >= 1% of σ_max. Low = concentrated = steerable)")
    print("=" * 76)
    rank_map = np.zeros((n_layers, n_steps), dtype=np.int32)
    s_max_map = np.zeros((n_layers, n_steps), dtype=np.float32)
    for L in range(n_layers):
        for T in range(n_steps):
            mat = img_mean[:, L, T]  # [P, hidden]
            mat_c = mat - mat.mean(axis=0, keepdims=True)
            try:
                s = np.linalg.svd(mat_c, compute_uv=False)
            except np.linalg.LinAlgError:
                continue
            s_max = float(s.max()) if len(s) else 0.0
            s_max_map[L, T] = s_max
            if s_max <= 1e-8:
                rank_map[L, T] = 0
            else:
                rank_map[L, T] = int((s >= 0.01 * s_max).sum())

    print(f"max possible rank: min(P, hidden) = {min(P, hidden)}, P={P}")
    print()
    print("layer  " + " ".join(f"T{t:02d}" for t in range(n_steps)) + "    σ_max(mean)")
    print("-" * 76)
    for L in range(n_layers):
        ranks = " ".join(f"{r:3d}" for r in rank_map[L])
        sm = float(s_max_map[L].mean())
        print(f"L{L:02d}    {ranks}    {sm:.4f}")
    print()

    return {
        "n_prompts": P,
        "n_layers": n_layers,
        "n_steps": n_steps,
        "seq_len": seq_len,
        "hidden": hidden,
        "n_image_tokens": n_image_tokens,
        "img_lo": img_lo,
        "div_map": div_map.tolist(),
        "rank_map": rank_map.tolist(),
        "s_max_map": s_max_map.tolist(),
        "categories": {c: len(by_cat[c]) for c in unique_cats},
    }
