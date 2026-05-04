"""Analyze a prompt-mri capture for category-level discrimination.

Designed for the identity-token-contrast experiment but written generically:
given a prompts.jsonl with a `category` field per prompt and a residuals.npz
of [n_layers, n_tokens, hidden] arrays, compute per-layer:

  - within-category vs between-category cosine similarity
  - centroid-pair similarities (e.g. cos(trigger_lora_centroid, control_oov_centroid))
  - layer where any pair of categories is maximally separated

The position used per prompt is configurable. For text encoders that feed
cross-attention to a downstream model, the LAST token's residual is the
conditioning state — that's the default.
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


def _load(input_dir: str) -> tuple[dict, list[dict], dict[int, np.ndarray]]:
    p = Path(input_dir)
    with open(p / "metadata.json") as fh:
        meta = json.load(fh)
    with open(p / "prompts.jsonl") as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]
    raw = np.load(p / "residuals.npz")
    residuals: dict[int, np.ndarray] = {}
    for i, _ in enumerate(prompts):
        residuals[i] = raw[f"r{i:04d}"]  # [n_layers, n_tokens, hidden]
    return meta, prompts, residuals


def _extract_per_layer(prompts: list[dict], residuals: dict[int, np.ndarray],
                       position: str) -> np.ndarray:
    """Extract per-prompt per-layer residual at the chosen position.

    position:
      "last"    — last token (conditioning slot for cross-attn into a DiT)
      "first"   — first token (often BOS)
      int (str) — explicit token index

    Returns: [n_prompts, n_layers, hidden]
    """
    n_prompts = len(prompts)
    sample = residuals[0]
    n_layers, _, hidden = sample.shape

    out = np.zeros((n_prompts, n_layers, hidden), dtype=np.float32)
    for i in range(n_prompts):
        arr = residuals[i].astype(np.float32)  # [n_layers, n_tokens, hidden]
        n_tok = arr.shape[1]
        if position == "last":
            pos = n_tok - 1
        elif position == "first":
            pos = 0
        else:
            try:
                pos = int(position)
            except ValueError:
                raise ValueError(f"position must be 'last', 'first', or int, got {position!r}")
            if pos < 0:
                pos = n_tok + pos
            pos = max(0, min(pos, n_tok - 1))
        out[i] = arr[:, pos, :]
    return out


def analyze_prompt_mri(input_dir: str, *, position: str = "last") -> dict:
    """Compute per-layer category statistics from a prompt-mri capture."""
    meta, prompts, residuals = _load(input_dir)
    states = _extract_per_layer(prompts, residuals, position)  # [N, L, H]
    n_prompts, n_layers, hidden = states.shape

    cats = [p.get("category", "uncat") for p in prompts]
    unique_cats = sorted(set(cats))
    by_cat: dict[str, list[int]] = {c: [] for c in unique_cats}
    for i, c in enumerate(cats):
        by_cat[c].append(i)

    print(f"prompt-mri-analyze: {n_prompts} prompts, {n_layers} layers, "
          f"hidden={hidden}, position={position}")
    print(f"  categories: {dict((c, len(by_cat[c])) for c in unique_cats)}")
    print()

    # Per-layer category centroids: [n_cats, n_layers, hidden]
    centroids = np.zeros((len(unique_cats), n_layers, hidden), dtype=np.float32)
    for ci, c in enumerate(unique_cats):
        centroids[ci] = states[by_cat[c]].mean(axis=0)

    # Per-layer pairwise centroid similarity
    print("=" * 76)
    print("per-layer category centroid similarity (cosine)")
    print("higher = encoder treats categories more alike at that layer")
    print("=" * 76)

    # Pairs we care most about for the identity-token question:
    pairs_of_interest = [
        ("trigger_lora", "control_oov"),    # the key question: does encoder differentiate triggers from random rare?
        ("trigger_lora", "real_name"),      # do triggers look like names?
        ("trigger_lora", "generic"),        # do triggers look like generic descriptions?
        ("real_name", "control_oov"),       # sanity: names should differ from random rare
        ("real_name", "generic"),
        ("control_oov", "generic"),
        ("bare_trigger", "bare_oov"),       # same question, no scaffold context
        ("bare_trigger", "bare_name"),
        ("bare_oov", "bare_name"),
    ]

    available_pairs = [(a, b) for (a, b) in pairs_of_interest
                       if a in unique_cats and b in unique_cats]

    pair_layer_sims: dict[tuple[str, str], np.ndarray] = {}
    for a, b in available_pairs:
        ai = unique_cats.index(a)
        bi = unique_cats.index(b)
        sims = np.array([_cos(centroids[ai, l], centroids[bi, l])
                         for l in range(n_layers)])
        pair_layer_sims[(a, b)] = sims

    # Print summary table: pair × {min, mean, max, layer-of-max-separation}
    header = f"{'pair':<38s} {'min':>7s} {'mean':>7s} {'max':>7s} {'argmin_L':>10s}"
    print(header)
    print("-" * len(header))
    for (a, b), sims in pair_layer_sims.items():
        amin = float(sims.min())
        amean = float(sims.mean())
        amax = float(sims.max())
        argmin = int(sims.argmin())
        print(f"{a:>16s} ↔ {b:<18s} {amin:>7.4f} {amean:>7.4f} {amax:>7.4f} {argmin:>10d}")
    print()

    # Detail: per-layer trace for the headline pair (trigger_lora ↔ control_oov)
    headline = ("trigger_lora", "control_oov")
    if headline in pair_layer_sims:
        sims = pair_layer_sims[headline]
        print(f"per-layer cos(centroid[{headline[0]}], centroid[{headline[1]}]):")
        for l in range(0, n_layers, max(1, n_layers // 12)):
            bar = "█" * int(round(40 * (sims[l] + 1) / 2))  # cos ∈ [-1,1] → [0,40]
            print(f"  L{l:02d}  {sims[l]:+.4f}  {bar}")
        print(f"  L{n_layers - 1:02d}  {sims[-1]:+.4f}  "
              f"{'█' * int(round(40 * (sims[-1] + 1) / 2))}")
        print()

    # Specific token-vs-token pairs for the canonical question:
    # how close is subject1 to qzqz4 vs to Sara?
    print("=" * 76)
    print("specific prompt-pair similarities at the most discriminating layer")
    print("=" * 76)

    # Build prompt index by .text
    by_text = {p["text"]: i for i, p in enumerate(prompts)}

    canonical = [
        ("subject1", "qzqz4"),                         # identity-trigger vs random rare
        ("subject1", "Sara"),                        # identity-trigger vs phonetic neighbor
        ("Sara", "qzqz4"),                          # name vs random rare
        ("a portrait of subject1, photo", "a portrait of qzqz4, photo"),
        ("a portrait of subject1, photo", "a portrait of Sara, photo"),
        ("a portrait of Sara, photo", "a portrait of qzqz4, photo"),
        ("a portrait of subject1, photo", "a portrait of a woman, photo"),
        ("a portrait of qzqz4, photo", "a portrait of a woman, photo"),
    ]

    # Pick the layer with the lowest trigger_lora vs control_oov similarity
    # — that's the layer the encoder most clearly distinguishes them. If that
    # similarity is already near 1.0 everywhere, the answer is "encoder
    # doesn't differentiate at all" and the layer choice doesn't matter.
    if headline in pair_layer_sims:
        focus_L = int(pair_layer_sims[headline].argmin())
    else:
        focus_L = n_layers - 1

    print(f"focus layer L{focus_L:02d} (max trigger_lora ↔ control_oov separation)")
    print()
    header2 = f"{'pair':<70s} {'cos':>7s}"
    print(header2)
    print("-" * len(header2))
    for ta, tb in canonical:
        if ta in by_text and tb in by_text:
            ia, ib = by_text[ta], by_text[tb]
            c = _cos(states[ia, focus_L], states[ib, focus_L])
            print(f"  {ta!r:>34s} ↔ {tb!r:<32s} {c:>7.4f}")
    print()

    # Final layer (the one that becomes K/V conditioning to the DiT)
    print("=" * 76)
    print(f"FINAL layer L{n_layers - 1} (this is the conditioning state for cross-attn into the DiT)")
    print("=" * 76)
    print(f"{'pair':<70s} {'cos':>7s}")
    print("-" * len(header2))
    for ta, tb in canonical:
        if ta in by_text and tb in by_text:
            ia, ib = by_text[ta], by_text[tb]
            c = _cos(states[ia, n_layers - 1], states[ib, n_layers - 1])
            print(f"  {ta!r:>34s} ↔ {tb!r:<32s} {c:>7.4f}")
    print()

    # Within-vs-between class summary
    print("=" * 76)
    print("within- vs between-category cosine (centroid-based, last layer)")
    print("=" * 76)
    L = n_layers - 1
    for c in unique_cats:
        if len(by_cat[c]) < 2:
            continue
        idx = by_cat[c]
        within = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                within.append(_cos(states[idx[i], L], states[idx[j], L]))
        between = []
        other = [k for k in range(n_prompts) if k not in idx]
        for i in idx:
            for k in other:
                between.append(_cos(states[i, L], states[k, L]))
        wm = float(np.mean(within)) if within else float("nan")
        bm = float(np.mean(between)) if between else float("nan")
        gap = wm - bm
        print(f"  {c:<18s} within={wm:.4f}  between={bm:.4f}  gap={gap:+.4f}")
    print()

    return {
        "n_prompts": n_prompts,
        "n_layers": n_layers,
        "hidden": hidden,
        "position": position,
        "categories": {c: len(by_cat[c]) for c in unique_cats},
        "pair_layer_sims": {f"{a}__{b}": sims.tolist()
                            for (a, b), sims in pair_layer_sims.items()},
        "focus_layer": focus_L if headline in pair_layer_sims else None,
    }
