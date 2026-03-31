"""Delta projection onto token embeddings for trigger token identification."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from ..signal import Signal


def project_delta_onto_embeddings(
    delta: np.ndarray,
    embeddings: np.ndarray,
    *,
    model_label: str = "model",
    top_k: int = 50,
) -> list[Signal]:
    """Project a weight delta onto all token embeddings. Return top-k by activation norm."""
    response = delta @ embeddings.T  # [out_dim, vocab]
    norms = np.linalg.norm(response, axis=0)  # [vocab]
    mu, sigma = float(norms.mean()), float(norms.std())

    top_indices = np.argsort(norms)[::-1][:top_k]
    signals = []
    for idx in top_indices:
        z = (float(norms[idx]) - mu) / sigma if sigma > 0 else 0.0
        signals.append(Signal(
            "token_activation", "diff", model_label, f"token_{int(idx)}",
            float(norms[idx]),
            {"token_id": int(idx), "z_score": z, "rank": len(signals) + 1},
        ))
    return signals


def score_phrase(
    delta: np.ndarray,
    embeddings: np.ndarray,
    token_ids: Sequence[int],
) -> float:
    """Score a multi-token phrase by summing embeddings and computing delta response."""
    if not token_ids:
        return 0.0
    combined = sum(embeddings[tid] for tid in token_ids)
    response = delta @ combined
    return float(np.linalg.norm(response))
