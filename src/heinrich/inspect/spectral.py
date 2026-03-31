"""Spectral analysis of weight matrices."""
from __future__ import annotations
import numpy as np

def spectral_stats(matrix: np.ndarray, topk: int) -> dict[str, float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    effective_topk = min(topk, singular.size)
    energy = singular * singular
    total_energy = float(np.sum(energy))
    top_energy = float(np.sum(energy[:effective_topk]))
    sigma1 = float(singular[0])
    sigmak = float(singular[effective_topk - 1])
    sigmalast = float(singular[-1])
    return {
        "fro_norm": float(np.linalg.norm(matrix)),
        "sigma1": sigma1,
        f"sigma{effective_topk}": sigmak,
        "sigma_last": sigmalast,
        "requested_topk": int(topk),
        "effective_topk": int(effective_topk),
        f"top{effective_topk}_energy_frac": float(top_energy / total_energy) if total_energy > 0 else 0.0,
        f"decay_1_to_{effective_topk}": float(sigma1 / max(sigmak, 1e-12)),
        "decay_1_to_last": float(sigma1 / max(sigmalast, 1e-12)),
    }

def region_stats(matrix: np.ndarray) -> dict[str, float]:
    upper = np.triu(matrix, 1)
    diag = np.diag(np.diag(matrix))
    lower = np.tril(matrix, -1)
    total = float(np.linalg.norm(matrix))
    upper_l2 = float(np.linalg.norm(upper))
    diag_l2 = float(np.linalg.norm(diag))
    lower_l2 = float(np.linalg.norm(lower))
    return {
        "upper_l2": upper_l2, "diag_l2": diag_l2, "lower_l2": lower_l2,
        "upper_frac": float(upper_l2 / total) if total > 0 else 0.0,
        "diag_frac": float(diag_l2 / total) if total > 0 else 0.0,
        "upper_plus_diag_frac": float(np.linalg.norm(upper + diag) / total) if total > 0 else 0.0,
    }
