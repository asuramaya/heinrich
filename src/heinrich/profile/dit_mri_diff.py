"""Paired-run differential analysis: where did the LoRA actually change the DiT?

Given two dit-mri captures with identical seeds and prompt sets — typically a
"base" run with no LoRA and a "treatment" run with a LoRA loaded — compute the
per-prompt × per-(layer, step, position) activation delta and surface:

  - Per-prompt total delta norm by (layer, step) — global LoRA effect heatmap
  - Per-prompt top-N most-affected positions
  - Selectivity: cells that light up ONLY on the LoRA's trained-trigger prompt
    (e.g. `subject1`) but not on other prompts. This is the LoRA's mechanistic
    footprint — what it actually does, distinct from "any change to the model
    that happens during inference."

Selectivity is the key question for a character LoRA: did it carve a specific
circuit (good — interpretable, steerable) or did it shift the whole model
(bad — entangled, hard to mechanistically replicate).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_run(input_dir: str) -> tuple[dict, list[dict], dict[int, np.ndarray]]:
    p = Path(input_dir)
    with open(p / "metadata.json") as fh:
        meta = json.load(fh)
    with open(p / "prompts.jsonl") as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]
    residuals: dict[int, np.ndarray] = {}
    for i, _ in enumerate(prompts):
        residuals[i] = np.load(p / f"prompt_{i:03d}_residuals.npy", mmap_mode="r")
    return meta, prompts, residuals


def diff_dit_mri(base_dir: str, treatment_dir: str, *, cfg_branch: int = 0,
                  trigger_token: str = "subject1",
                  n_image_tokens: int | None = None) -> dict:
    """Compute per-(layer, step, position) activation delta between two runs.

    Args:
        base_dir: dit-mri capture without LoRA
        treatment_dir: dit-mri capture with LoRA
        cfg_branch: 0=conditional, 1=unconditional
        trigger_token: which prompt's `token` field marks the LoRA's trained trigger
            (used for selectivity analysis — e.g. "subject1" for the Subject LoRA)
        n_image_tokens: trailing positions that are image tokens (default: infer)
    """
    base_meta, base_prompts, base_res = _load_run(base_dir)
    treat_meta, treat_prompts, treat_res = _load_run(treatment_dir)

    # Sanity: same prompt set, same shapes
    assert len(base_prompts) == len(treat_prompts)
    for a, b in zip(base_prompts, treat_prompts):
        assert a["text"] == b["text"], f"prompt mismatch: {a['text']!r} vs {b['text']!r}"

    sample = base_res[0]
    n_layers, n_steps, batch, seq_len, hidden = sample.shape
    P = len(base_prompts)

    if n_image_tokens is None:
        w = base_meta.get("width", 1024)
        h = base_meta.get("height", 1024)
        n_image_tokens = (w // 16 // 2) * (h // 16 // 2)
    img_lo = seq_len - n_image_tokens

    print(f"dit-mri-diff: {P} prompts × {n_layers} layers × {n_steps} steps × "
          f"{seq_len} positions × {hidden} hidden")
    print(f"  base:      {base_dir}")
    print(f"  treatment: {treatment_dir}")
    print(f"  cfg_branch={cfg_branch}, trigger={trigger_token!r}")
    print(f"  token layout: text/siglip/refiner [0:{img_lo}] + image [{img_lo}:{seq_len}]")
    print()

    # ===== Streaming pass: per-prompt × per-(layer, step) total delta norm =====
    # delta_norm[i, L, T] = || base[i, L, T] - treatment[i, L, T] ||_F (Frobenius over seq×hidden)
    # Also per-position: pos_delta[i, L, T, pos] = || delta@pos ||_2
    #
    # The full pos_delta tensor would be P*L*T*seq * 4 = 8*30*8*1056*4 = 8 MB → fine.

    delta_norm = np.zeros((P, n_layers, n_steps), dtype=np.float32)
    pos_delta = np.zeros((P, n_layers, n_steps, seq_len), dtype=np.float32)

    print("computing per-prompt delta norms...")
    for i in range(P):
        # Slice both runs to conditional branch only (drop CFG batch axis)
        # Each slice is [n_layers, n_steps, seq, hidden] mmap'd fp16.
        # Process layer-by-layer to keep RAM low.
        for L in range(n_layers):
            for T in range(n_steps):
                a = base_res[i][L, T, cfg_branch].astype(np.float32)  # [seq, hidden]
                b = treat_res[i][L, T, cfg_branch].astype(np.float32)
                d = b - a
                # Per-position L2 norm
                p_n = np.linalg.norm(d, axis=1)  # [seq]
                pos_delta[i, L, T] = p_n
                delta_norm[i, L, T] = float(np.linalg.norm(p_n))
        print(f"  [{i + 1}/{P}] {base_prompts[i]['text']!r}  "
              f"max||Δ||={delta_norm[i].max():.3f}  "
              f"sum||Δ||={delta_norm[i].sum():.1f}")
    print()

    # ===== 1. Per-prompt total delta as a single number =====
    print("=" * 76)
    print("per-prompt LoRA effect (sum of ||Δ|| over all (layer, step) cells)")
    print("=" * 76)
    print(f"{'prompt':<40s} {'category':<14s} {'sum_delta':>12s} {'peak_cell':>14s}")
    print("-" * 76)
    by_cat: dict[str, list[int]] = {}
    for i, pr in enumerate(base_prompts):
        c = pr.get("category", "uncat")
        by_cat.setdefault(c, []).append(i)
        peak_LT = np.unravel_index(delta_norm[i].argmax(), delta_norm[i].shape)
        text = pr["text"]
        print(f"  {text[:38]:<38s} {c:<14s} {delta_norm[i].sum():>12.2f} "
              f"  L{peak_LT[0]:02d}T{peak_LT[1]}  (||Δ||={delta_norm[i, peak_LT[0], peak_LT[1]]:.2f})")
    print()

    # ===== 2. Category-level effect: did the LoRA fire selectively? =====
    print("=" * 76)
    print("category-level LoRA effect (mean of sum_delta within category)")
    print("=" * 76)
    cat_means = {}
    for c, idxs in sorted(by_cat.items()):
        s = float(np.mean([delta_norm[i].sum() for i in idxs]))
        cat_means[c] = s
        print(f"  {c:<14s} mean sum_delta = {s:>10.2f}   ({len(idxs)} prompts)")
    print()

    # ===== 3. Selectivity: trigger prompt vs non-trigger baseline =====
    # The LoRA was trained on `trigger_token`. If selective, it should fire
    # strongly there and weakly elsewhere. Selectivity ratio = trigger_delta / mean_non_trigger.
    trigger_idx = None
    for i, pr in enumerate(base_prompts):
        if pr.get("token") == trigger_token:
            trigger_idx = i
            break

    print("=" * 76)
    print(f"selectivity analysis (trigger token = {trigger_token!r})")
    print("=" * 76)
    if trigger_idx is None:
        print(f"  no prompt with token={trigger_token!r} found — skipping selectivity analysis")
    else:
        non_trigger = [i for i in range(P) if i != trigger_idx]
        non_trigger_mean = np.mean([delta_norm[i].sum() for i in non_trigger])
        trigger_total = delta_norm[trigger_idx].sum()
        ratio = trigger_total / max(non_trigger_mean, 1e-6)
        print(f"  trigger {trigger_token!r} sum_delta:        {trigger_total:>10.2f}")
        print(f"  non-trigger mean sum_delta:           {non_trigger_mean:>10.2f}")
        print(f"  selectivity ratio (trigger / non):    {ratio:>10.3f}")
        print(f"  interpretation: ratio >> 1 = LoRA selectively fires on trigger; "
              f"ratio ≈ 1 = LoRA changes everything equally")
        print()

        # Per-(L, T) selectivity map
        non_trigger_mean_map = np.mean([delta_norm[i] for i in non_trigger], axis=0)
        # Add small floor to avoid divide-by-zero
        sel_map = delta_norm[trigger_idx] / np.maximum(non_trigger_mean_map, 1e-3)

        print(f"per-(L, T) selectivity ratio at trigger prompt: trigger||Δ|| / mean_non_trigger||Δ||")
        print(f"(>>1 = trigger-specific cell; ≈1 = generic LoRA effect)")
        print()
        header = "       " + " ".join(f"  T{t}  " for t in range(n_steps))
        print(header)
        print("-" * len(header))
        for L in range(n_layers):
            row = " ".join(f"{sel_map[L, t]:6.2f}" for t in range(n_steps))
            print(f"L{L:02d}    {row}")
        print()

        # Top selective cells
        flat_idx = np.argsort(sel_map.ravel())[::-1][:10]
        print("top 10 most-selective (L, T) cells:")
        for fi in flat_idx:
            L_idx, T_idx = (int(x) for x in np.unravel_index(fi, sel_map.shape))
            print(f"  L{L_idx:02d}T{T_idx}  selectivity={sel_map[L_idx, T_idx]:.3f}  "
                  f"trigger||Δ||={delta_norm[trigger_idx, L_idx, T_idx]:.3f}  "
                  f"non-trigger mean||Δ||={non_trigger_mean_map[L_idx, T_idx]:.3f}")
        print()

        # ===== 4. Position-axis breakdown at the most selective cell =====
        max_LT = (int(x) for x in np.unravel_index(sel_map.argmax(), sel_map.shape))
        max_L, max_T = max_LT
        print("=" * 76)
        print(f"position-axis ||Δ|| at the most-selective cell (L{max_L}, T{max_T}) — "
              f"trigger prompt ({trigger_token!r}):")
        print("=" * 76)
        trig_pos = pos_delta[trigger_idx, max_L, max_T]  # [seq]
        non_trig_pos_mean = np.mean([pos_delta[i, max_L, max_T] for i in non_trigger], axis=0)

        text_block_trig = trig_pos[:img_lo]
        img_block_trig = trig_pos[img_lo:]
        text_block_nt = non_trig_pos_mean[:img_lo]
        img_block_nt = non_trig_pos_mean[img_lo:]
        print(f"  text/siglip/refiner positions [0:{img_lo}]:  "
              f"trigger mean||Δ||={text_block_trig.mean():.3f},  max={text_block_trig.max():.3f},  "
              f"non-trigger mean||Δ||={text_block_nt.mean():.3f}")
        print(f"  image positions             [{img_lo}:{seq_len}]:  "
              f"trigger mean||Δ||={img_block_trig.mean():.3f},  max={img_block_trig.max():.3f},  "
              f"non-trigger mean||Δ||={img_block_nt.mean():.3f}")
        print()

        print("top 15 trigger-only positions (high trigger ||Δ||, low non-trigger ||Δ||):")
        # Score = trigger_pos - non_trigger_pos_mean (raw selectivity per position)
        sel_per_pos = trig_pos - non_trig_pos_mean
        top_pos = np.argsort(sel_per_pos)[::-1][:15]
        print(f"  {'pos':>5s}  {'kind':<6s} {'trig||Δ||':>12s} {'non-trig||Δ||':>14s} "
              f"{'sel_diff':>12s}")
        for p in top_pos:
            kind = "image" if p >= img_lo else "text"
            if kind == "image":
                row, col = (p - img_lo) // 16, (p - img_lo) % 16
                kind_str = f"img({row},{col})"
            else:
                kind_str = "text"
            print(f"  {int(p):>5d}  {kind_str:<10s} {trig_pos[p]:>12.3f} "
                  f"{non_trig_pos_mean[p]:>14.3f} {sel_per_pos[p]:>12.3f}")
        print()

    return {
        "n_prompts": P,
        "n_layers": n_layers,
        "n_steps": n_steps,
        "seq_len": seq_len,
        "img_lo": img_lo,
        "delta_norm_per_prompt": delta_norm.tolist(),
        "category_mean_sum_delta": cat_means,
    }
