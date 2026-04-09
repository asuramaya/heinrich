"""Basin mapping — where does the model have learned behaviors and where is void?

Samples points along a direction at increasing alpha, measures output
entropy and top-token stability at each point. Low entropy = attractor
(learned behavior). High entropy = void (degenerate region).

Records full provenance: which direction, which prompts, which baseline,
which model, which layer, which alpha range.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def map_basin(
    backend,
    direction: np.ndarray,
    *,
    layer: int,
    mean_gap: float = 1.0,
    alphas: list[float] | None = None,
    prompts: list[str] | None = None,
    n_prompts: int = 20,
    max_tokens: int = 30,
    seed: int = 42,
    provenance: dict | None = None,
) -> dict:
    """Map the basin structure along a direction.

    For each alpha: steer the model, generate, measure entropy and
    top-token identity. Reports where coherent behavior lives and
    where collapse begins.

    Args:
        backend: loaded model backend
        direction: unit vector in residual space
        layer: which layer to steer at
        mean_gap: scaling factor for the direction
        alphas: steering magnitudes to test
        prompts: specific prompts to test (default: sample from DB)
        n_prompts: number of prompts if not specified
        max_tokens: tokens to generate per prompt
        seed: random seed for prompt selection
        provenance: dict recording how direction was discovered

    Returns:
        Basin map with entropy profile, collapse detection, provenance
    """
    from ..cartography.templates import build_prompt
    from ..cartography.metrics import softmax

    cfg = backend.config
    direction = direction.astype(np.float32)
    direction = direction / np.linalg.norm(direction)

    if alphas is None:
        alphas = [-10, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 10]

    if prompts is None:
        import sqlite3
        conn = sqlite3.connect('data/heinrich.db')
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT text FROM prompts WHERE is_benign = 0 ORDER BY RANDOM() LIMIT ?",
            (n_prompts,)
        ).fetchall()
        prompts = [build_prompt(dict(r)["text"], model_config=cfg) for r in rows]

    t0 = time.time()
    steer_dirs = {layer: (direction, mean_gap)}

    results = []
    for alpha in alphas:
        entropies = []
        top_tokens = []
        texts = []
        degenerate_count = 0

        for prompt in prompts:
            if alpha == 0:
                text = backend.generate(prompt, max_tokens=max_tokens)
            else:
                text = backend.generate(prompt, steer_dirs=steer_dirs,
                                        alpha=alpha, max_tokens=max_tokens)

            # Measure first-token entropy
            if alpha == 0:
                fwd = backend.forward(prompt)
            else:
                fwd = backend.forward(prompt, steer_dirs=steer_dirs, alpha=alpha)
            entropies.append(fwd.entropy)
            top_tokens.append(fwd.top_token)
            texts.append(text[:100])

            # Detect degeneration: repeated tokens
            words = text.split()
            if len(words) >= 5:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:
                    degenerate_count += 1

        mean_entropy = float(np.mean(entropies))
        std_entropy = float(np.std(entropies))
        top_token_diversity = len(set(top_tokens)) / max(len(top_tokens), 1)

        results.append({
            "alpha": alpha,
            "mean_entropy": round(mean_entropy, 4),
            "std_entropy": round(std_entropy, 4),
            "top_token_diversity": round(top_token_diversity, 4),
            "degenerate_pct": round(degenerate_count / len(prompts) * 100, 1),
            "top_tokens": dict(zip(*np.unique(top_tokens, return_counts=True))),
            "sample_texts": texts[:3],
        })

    elapsed = time.time() - t0

    # Detect basin boundaries: where entropy jumps or degeneration starts
    entropy_profile = [r['mean_entropy'] for r in results]
    alpha_list = [r['alpha'] for r in results]

    # Find collapse points: first alpha where degeneration > 50%
    collapse_negative = None
    collapse_positive = None
    for r in results:
        if r['alpha'] < 0 and r['degenerate_pct'] > 50 and collapse_negative is None:
            collapse_negative = r['alpha']
        if r['alpha'] > 0 and r['degenerate_pct'] > 50 and collapse_positive is None:
            collapse_positive = r['alpha']

    # Find attractor: alpha range where entropy is lowest
    baseline_idx = next(i for i, r in enumerate(results) if r['alpha'] == 0)
    baseline_entropy = results[baseline_idx]['mean_entropy']

    return {
        "model": cfg.model_type,
        "layer": layer,
        "mean_gap": mean_gap,
        "n_prompts": len(prompts),
        "max_tokens": max_tokens,
        "elapsed_s": round(elapsed, 1),
        "alphas": results,
        "baseline_entropy": baseline_entropy,
        "collapse_negative": collapse_negative,
        "collapse_positive": collapse_positive,
        "provenance": provenance or {},
        "provenance_notes": {
            "direction_source": "caller must specify",
            "prompt_source": "random harmful from DB" if prompts is None else "caller-specified",
            "seed": seed,
            "classification": "degeneration detected by unique_word_ratio < 0.3",
        },
    }


def first_token_profile(
    backend,
    direction: np.ndarray,
    *,
    refuse_tokens: list[str] | None = None,
    comply_tokens: list[str] | None = None,
    provenance: dict | None = None,
) -> dict:
    """Compute first-token logit gap for a direction without generating.

    Projects direction through norm + lm_head, measures which first
    tokens are amplified and suppressed.
    """
    import mlx.core as mx
    from ..cartography.runtime import _lm_head
    from ..cartography.metrics import softmax

    cfg = backend.config
    inner = getattr(backend.model, 'model', backend.model)

    direction = direction.astype(np.float32)
    direction = direction / np.linalg.norm(direction)

    if refuse_tokens is None:
        refuse_tokens = ["I", "Sorry", "As", "It", "However", "Thank", "Unfortunately"]
    if comply_tokens is None:
        comply_tokens = ["Sure", "Here", "The", "To", "Step", "1", "Yes"]

    # Project through norm + lm_head
    h = mx.array(direction.reshape(1, 1, -1).astype(np.float16))
    h_normed = inner.norm(h)
    logits = np.array(_lm_head(backend.model, h_normed).astype(mx.float32)[0, 0, :])

    # Per-token analysis
    refuse_pushes = {}
    for tok in refuse_tokens:
        ids = backend.tokenizer.encode(tok)
        if ids:
            refuse_pushes[tok] = round(float(logits[ids[0]]), 4)

    comply_pushes = {}
    for tok in comply_tokens:
        ids = backend.tokenizer.encode(tok)
        if ids:
            comply_pushes[tok] = round(float(logits[ids[0]]), 4)

    mean_refuse = float(np.mean(list(refuse_pushes.values()))) if refuse_pushes else 0
    mean_comply = float(np.mean(list(comply_pushes.values()))) if comply_pushes else 0
    gap = mean_refuse - mean_comply
    ratio = float(np.exp(min(gap, 50)))  # cap to avoid overflow

    # Top amplified and suppressed tokens globally
    top_amplified = np.argsort(-logits)[:15]
    top_suppressed = np.argsort(logits)[:15]

    # Check for uniform shift (Mistral-like)
    logit_std = float(np.std(logits))
    logit_range = float(logits.max() - logits.min())
    is_uniform = logit_range < 1.0  # if all logits within 1.0, effectively uniform

    return {
        "model": cfg.model_type,
        "refuse_pushes": refuse_pushes,
        "comply_pushes": comply_pushes,
        "mean_refuse": round(mean_refuse, 4),
        "mean_comply": round(mean_comply, 4),
        "gap": round(gap, 4),
        "probability_ratio": round(ratio, 1),
        "is_uniform_shift": is_uniform,
        "logit_std": round(logit_std, 4),
        "logit_range": round(logit_range, 4),
        "top_amplified": [
            {"token": backend.tokenizer.decode([int(t)]), "logit": round(float(logits[t]), 2)}
            for t in top_amplified
        ],
        "top_suppressed": [
            {"token": backend.tokenizer.decode([int(t)]), "logit": round(float(logits[t]), 2)}
            for t in top_suppressed
        ],
        "provenance": provenance or {},
        "provenance_notes": {
            "refuse_tokens": refuse_tokens,
            "comply_tokens": comply_tokens,
            "method": "direction projected through model.norm + lm_head",
        },
    }


def lmhead_profile(backend) -> dict:
    """Decompose the output matrix geometry.

    SVD, condition number, effective rank, and where discovered
    directions sit in the singular spectrum.
    """
    import mlx.core as mx
    from ..cartography.runtime import _lm_head

    cfg = backend.config
    inner = getattr(backend.model, 'model', backend.model)
    hidden = cfg.hidden_size

    # Extract effective weight matrix (norm + lm_head composed)
    cols = []
    batch = 64
    for start in range(0, hidden, batch):
        end = min(start + batch, hidden)
        inp = np.zeros((1, end - start, hidden), dtype=np.float16)
        for j in range(end - start):
            inp[0, j, start + j] = 1.0
        h = mx.array(inp)
        h_normed = inner.norm(h)
        out = np.array(_lm_head(backend.model, h_normed).astype(mx.float32)[0])
        cols.append(out.T)
    W = np.concatenate(cols, axis=1)  # [vocab, hidden]

    # SVD
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    var = S ** 2
    total = var.sum()
    cum = np.cumsum(var) / total

    condition = float(S[0] / S[-1]) if S[-1] > 0 else float('inf')

    pcs_for_threshold = {}
    for pct in [0.5, 0.8, 0.9, 0.95]:
        pcs_for_threshold[str(pct)] = int(np.searchsorted(cum, pct)) + 1

    return {
        "model": cfg.model_type,
        "shape": list(W.shape),
        "hidden_size": hidden,
        "vocab_size": W.shape[0],
        "condition_number": round(condition, 1),
        "top_singular_values": [round(float(s), 1) for s in S[:10]],
        "pcs_for_threshold": pcs_for_threshold,
        "singular_value_profile": {
            "S1": round(float(S[0]), 1),
            "S10": round(float(S[9]), 1) if len(S) > 9 else 0,
            "S50": round(float(S[49]), 1) if len(S) > 49 else 0,
            "S100": round(float(S[99]), 1) if len(S) > 99 else 0,
            "Smin": round(float(S[-1]), 4),
        },
        "Vt": Vt,  # right singular vectors for direction projection
        "provenance_notes": {
            "method": "identity probed through model.norm + lm_head, batch=64",
            "includes_norm": True,
        },
    }


def direction_in_lmhead(
    lmhead_result: dict,
    direction: np.ndarray,
    name: str = "direction",
) -> dict:
    """Where does a direction sit in the lm_head's singular spectrum?"""
    Vt = lmhead_result['Vt']
    direction = direction.astype(np.float32)
    direction = direction / np.linalg.norm(direction)

    loadings = Vt @ direction
    squared = loadings ** 2

    return {
        "name": name,
        "loading_top10": round(float(squared[:10].sum()), 4),
        "loading_top50": round(float(squared[:50].sum()), 4),
        "loading_top100": round(float(squared[:100].sum()), 4),
        "peak_sv_index": int(np.argmax(np.abs(loadings))),
        "peak_sv_loading": round(float(np.max(np.abs(loadings))), 4),
    }
