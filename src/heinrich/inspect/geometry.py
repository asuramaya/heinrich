"""Mask geometry analysis."""
from __future__ import annotations
from typing import Any
import numpy as np
from .spectral import region_stats

def strict_lower_mask(size: int) -> np.ndarray:
    return np.tril(np.ones((size, size), dtype=np.float64), k=-1)

def toeplitz_mean(mask: np.ndarray, support: np.ndarray | None = None) -> np.ndarray:
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("toeplitz_mean requires a square matrix")
    if support is None:
        support = strict_lower_mask(mask.shape[0])
    out = np.zeros_like(mask, dtype=np.float64)
    size = mask.shape[0]
    for lag in range(1, size):
        vals = np.diag(mask, k=-lag)
        out += np.diag(np.full(size - lag, float(np.mean(vals)), dtype=np.float64), k=-lag)
    return out * support

def lag_profile(mask: np.ndarray) -> list[dict[str, float]]:
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("lag_profile requires a square matrix")
    rows: list[dict[str, float]] = []
    size = mask.shape[0]
    for lag in range(1, size):
        vals = np.diag(mask, k=-lag)
        rows.append({"lag": lag, "count": int(vals.size), "mean": float(np.mean(vals)),
                      "std": float(np.std(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))})
    return rows

def mask_geometry_stats(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("mask_geometry_stats requires a square matrix")
    return {"region": region_stats(matrix), "lag_profile": lag_profile(matrix), "size": int(matrix.shape[0])}
