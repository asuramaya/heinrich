"""Model efficiency analysis from MRI data.

No model needed. Reads stored MRI files and computes:
  - Per-layer importance (contribution to output state)
  - Early-exit candidates (tokens resolved before the final layer)
  - Template overhead (% of state from frame vs content)
  - Redundant dimensions (effective rank per layer)
  - Layer pruning candidates (layers that barely change the output)
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path


def layer_importance(mri_path: str) -> dict:
    """Rank layers by their contribution to the output state.

    For each layer: how much does the exit state change from the previous
    layer? Layers with small changes are pruning candidates.
    """
    from .mri import load_mri

    d = load_mri(mri_path)
    meta = d['metadata']
    n_layers = meta['model']['n_layers']
    n_tokens = meta['capture']['n_tokens']

    # Per-layer contribution: |exit_L{i} - exit_L{i-1}|
    contributions = []
    prev_exit = None
    for i in range(n_layers):
        key = f"exit_L{i}"
        if key not in d:
            contributions.append({"layer": i, "error": "missing"})
            continue

        curr = d[key].astype(np.float32) if hasattr(d[key], 'astype') else np.array(d[key], dtype=np.float32)

        if prev_exit is not None:
            delta = np.linalg.norm(curr - prev_exit, axis=1)
            contributions.append({
                "layer": i,
                "mean_delta": round(float(delta.mean()), 2),
                "std_delta": round(float(delta.std()), 2),
                "max_delta": round(float(delta.max()), 2),
                "min_delta": round(float(delta.min()), 2),
            })
        else:
            norm = np.linalg.norm(curr, axis=1)
            contributions.append({
                "layer": i,
                "mean_delta": round(float(norm.mean()), 2),
                "std_delta": round(float(norm.std()), 2),
                "note": "first layer (delta from embedding)",
            })

        prev_exit = curr

    # Rank by importance
    ranked = sorted(
        [c for c in contributions if "mean_delta" in c],
        key=lambda x: x["mean_delta"]
    )

    # Identify pruning candidates (bottom 25% by contribution)
    n_ranked = len(ranked)
    prune_threshold = ranked[n_ranked // 4]["mean_delta"] if n_ranked > 4 else 0
    prunable = [c for c in ranked if c["mean_delta"] <= prune_threshold]

    return {
        "model": meta['model']['name'],
        "n_layers": n_layers,
        "n_tokens": n_tokens,
        "contributions": contributions,
        "prunable_layers": [c["layer"] for c in prunable],
        "prune_threshold": round(prune_threshold, 2),
        "total_contribution": round(sum(c.get("mean_delta", 0) for c in contributions), 2),
    }


def early_exit_analysis(mri_path: str, threshold_pct: float = 95.0) -> dict:
    """Find tokens that are resolved before the final layer.

    A token is "resolved" at layer L if its exit state at L is within
    threshold_pct% of its final exit state (measured by cosine similarity).
    """
    from .mri import load_mri

    d = load_mri(mri_path)
    meta = d['metadata']
    n_layers = meta['model']['n_layers']

    final_key = f"exit_L{n_layers - 1}"
    if final_key not in d:
        return {"error": f"Missing {final_key}"}

    final = d[final_key].astype(np.float32)
    final_norms = np.linalg.norm(final, axis=1, keepdims=True)
    final_normed = final / np.maximum(final_norms, 1e-8)

    threshold = threshold_pct / 100.0
    n_tokens = len(final)

    # For each token: at which layer does cosine similarity with final exceed threshold?
    exit_layers = np.full(n_tokens, n_layers - 1, dtype=np.int32)

    for i in range(n_layers - 1):
        key = f"exit_L{i}"
        if key not in d:
            continue
        curr = d[key].astype(np.float32)
        curr_norms = np.linalg.norm(curr, axis=1, keepdims=True)
        curr_normed = curr / np.maximum(curr_norms, 1e-8)

        cosines = (curr_normed * final_normed).sum(axis=1)
        resolved = cosines >= threshold

        # Update exit layer for tokens not yet resolved
        still_unresolved = exit_layers == (n_layers - 1)
        exit_layers[resolved & still_unresolved] = i

    # Statistics
    layer_counts = {}
    for i in range(n_layers):
        count = int((exit_layers == i).sum())
        if count > 0:
            layer_counts[i] = count

    mean_exit = float(exit_layers.mean())
    pct_early = float((exit_layers < n_layers - 1).sum() / n_tokens * 100)

    # Per-script if available
    scripts = d.get('scripts')
    script_exit = {}
    if scripts is not None:
        for s in sorted(set(str(x) for x in scripts)):
            if s in ('special', 'unknown'):
                continue
            mask = np.array([str(x) == s for x in scripts])
            if mask.sum() < 10:
                continue
            script_exit[s] = {
                "n": int(mask.sum()),
                "mean_exit_layer": round(float(exit_layers[mask].mean()), 1),
                "pct_early": round(float((exit_layers[mask] < n_layers - 1).sum() / mask.sum() * 100), 1),
            }

    return {
        "model": meta['model']['name'],
        "n_layers": n_layers,
        "n_tokens": n_tokens,
        "threshold_pct": threshold_pct,
        "mean_exit_layer": round(mean_exit, 1),
        "pct_early_exit": round(pct_early, 1),
        "layer_exit_counts": layer_counts,
        "script_exit": script_exit,
    }


def template_overhead(template_mri: str, raw_mri: str) -> dict:
    """Measure template contribution at every layer.

    Compares template mode MRI to raw mode MRI for the same model.
    Reports what percentage of the state comes from the template vs the token.
    """
    from .mri import load_mri

    t = load_mri(template_mri)
    r = load_mri(raw_mri)
    meta = t['metadata']
    n_layers = meta['model']['n_layers']

    # Match tokens
    t_ids = set(int(x) for x in t['token_ids'])
    r_ids = set(int(x) for x in r['token_ids'])
    shared = sorted(t_ids & r_ids)

    if len(shared) < 100:
        return {"error": f"Only {len(shared)} shared tokens"}

    # Index maps
    t_idx = {int(t['token_ids'][i]): i for i in range(len(t['token_ids']))}
    r_idx = {int(r['token_ids'][i]): i for i in range(len(r['token_ids']))}
    t_indices = [t_idx[tid] for tid in shared]
    r_indices = [r_idx[tid] for tid in shared]

    layers = []
    for i in range(n_layers):
        t_key = f"exit_L{i}"
        r_key = f"exit_L{i}"
        if t_key not in t or r_key not in r:
            continue

        t_state = t[t_key][t_indices].astype(np.float32)
        r_state = r[r_key][r_indices].astype(np.float32)

        t_norms = np.linalg.norm(t_state, axis=1)
        r_norms = np.linalg.norm(r_state, axis=1)
        diff_norms = np.linalg.norm(t_state - r_state, axis=1)

        layers.append({
            "layer": i,
            "template_mean_norm": round(float(t_norms.mean()), 1),
            "raw_mean_norm": round(float(r_norms.mean()), 1),
            "diff_mean_norm": round(float(diff_norms.mean()), 1),
            "template_pct": round(float(diff_norms.mean() / max(r_norms.mean(), 1e-8) * 100), 1),
        })

    return {
        "model": meta['model']['name'],
        "n_shared": len(shared),
        "n_layers": n_layers,
        "layers": layers,
    }
