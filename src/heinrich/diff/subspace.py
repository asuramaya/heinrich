"""Subspace comparison between weight deltas."""
from __future__ import annotations
import numpy as np
from ..signal import Signal


def compare_subspaces(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    label: str = "subspace",
    top_k: int = 5,
) -> list[Signal]:
    """Compare two matrices by their top-k singular subspaces."""
    U_l, S_l, _ = np.linalg.svd(lhs, full_matrices=False)
    U_r, S_r, _ = np.linalg.svd(rhs, full_matrices=False)
    k = min(top_k, U_l.shape[1], U_r.shape[1])
    cosines = np.abs(U_l[:, :k].T @ U_r[:, :k])
    signals = []
    for i in range(k):
        max_cos = float(cosines[i].max())
        signals.append(Signal(
            "subspace_angle", "diff", label, f"component_{i}",
            max_cos, {"lhs_sigma": float(S_l[i]), "rhs_sigma": float(S_r[i])},
        ))
    return signals


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)
