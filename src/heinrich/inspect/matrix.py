"""General 2D matrix analysis — works on grids, attention, activations, weights."""
from __future__ import annotations
from typing import Any
import numpy as np
from ..signal import Signal


def analyze_matrix(
    matrix: np.ndarray,
    *,
    label: str = "matrix",
    name: str = "unnamed",
) -> list[Signal]:
    """Run all applicable analyses on a 2D matrix and return signals."""
    if matrix.ndim != 2:
        return []
    signals = []
    signals.extend(shape_signals(matrix, label=label, name=name))
    signals.extend(sparsity_signals(matrix, label=label, name=name))
    signals.extend(norm_signals(matrix, label=label, name=name))
    signals.extend(entropy_signals(matrix, label=label, name=name))
    if matrix.shape[0] == matrix.shape[1]:
        signals.extend(symmetry_signals(matrix, label=label, name=name))
    if _is_discrete(matrix):
        signals.extend(discrete_signals(matrix, label=label, name=name))
    return signals


def shape_signals(matrix: np.ndarray, *, label: str, name: str) -> list[Signal]:
    rows, cols = matrix.shape
    return [
        Signal("matrix_rows", "inspect", label, name, float(rows), {}),
        Signal("matrix_cols", "inspect", label, name, float(cols), {}),
        Signal("matrix_elements", "inspect", label, name, float(rows * cols), {}),
    ]


def sparsity_signals(matrix: np.ndarray, *, label: str, name: str) -> list[Signal]:
    total = matrix.size
    if total == 0:
        return []
    nonzero = int(np.count_nonzero(matrix))
    return [
        Signal("matrix_sparsity", "inspect", label, name, 1.0 - nonzero / total, {"nonzero": nonzero, "total": total}),
    ]


def norm_signals(matrix: np.ndarray, *, label: str, name: str) -> list[Signal]:
    m = matrix.astype(np.float64)
    row_norms = np.linalg.norm(m, axis=1)
    col_norms = np.linalg.norm(m, axis=0)
    return [
        Signal("matrix_fro_norm", "inspect", label, name, float(np.linalg.norm(m)), {}),
        Signal("matrix_max_abs", "inspect", label, name, float(np.abs(m).max()) if m.size else 0.0, {}),
        Signal("matrix_row_norm_std", "inspect", label, name, float(np.std(row_norms)), {}),
        Signal("matrix_col_norm_std", "inspect", label, name, float(np.std(col_norms)), {}),
    ]


def entropy_signals(matrix: np.ndarray, *, label: str, name: str) -> list[Signal]:
    """Per-row entropy (useful for attention matrices and discrete grids)."""
    m = matrix.astype(np.float64)
    # Normalize rows to probability distributions (if non-negative)
    if np.all(m >= 0) and m.shape[0] > 0:
        row_sums = m.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        probs = m / row_sums
        # Entropy per row
        with np.errstate(divide='ignore', invalid='ignore'):
            log_probs = np.where(probs > 0, np.log2(probs), 0)
        row_entropy = -np.sum(probs * log_probs, axis=1)
        return [
            Signal("matrix_mean_row_entropy", "inspect", label, name, float(np.mean(row_entropy)), {}),
            Signal("matrix_min_row_entropy", "inspect", label, name, float(np.min(row_entropy)), {}),
            Signal("matrix_max_row_entropy", "inspect", label, name, float(np.max(row_entropy)), {}),
        ]
    return []


def symmetry_signals(matrix: np.ndarray, *, label: str, name: str) -> list[Signal]:
    """Symmetry scores for square matrices."""
    m = matrix.astype(np.float64)
    signals = []
    # Transpose symmetry: how close is M to M.T?
    diff = m - m.T
    fro = float(np.linalg.norm(m))
    sym_score = 1.0 - float(np.linalg.norm(diff)) / max(fro * 2, 1e-12)
    signals.append(Signal("matrix_transpose_symmetry", "inspect", label, name, sym_score, {}))
    # Horizontal flip symmetry
    flipped = m[:, ::-1]
    h_score = 1.0 - float(np.linalg.norm(m - flipped)) / max(fro * 2, 1e-12)
    signals.append(Signal("matrix_horizontal_symmetry", "inspect", label, name, h_score, {}))
    # Vertical flip symmetry
    flipped_v = m[::-1, :]
    v_score = 1.0 - float(np.linalg.norm(m - flipped_v)) / max(fro * 2, 1e-12)
    signals.append(Signal("matrix_vertical_symmetry", "inspect", label, name, v_score, {}))
    return signals


def discrete_signals(matrix: np.ndarray, *, label: str, name: str) -> list[Signal]:
    """Signals for integer/categorical matrices (grids, masks)."""
    values, counts = np.unique(matrix.astype(int), return_counts=True)
    signals = [
        Signal("matrix_unique_values", "inspect", label, name, float(len(values)), {}),
    ]
    # Histogram of values
    for val, cnt in zip(values.tolist(), counts.tolist()):
        signals.append(Signal("matrix_value_count", "inspect", label, name,
                              float(cnt), {"value": int(val), "fraction": cnt / matrix.size}))
    return signals


def connected_components(matrix: np.ndarray) -> list[dict[str, Any]]:
    """Flood-fill connected component detection. Uses scipy if available."""
    try:
        from scipy import ndimage
        m = matrix.astype(int)
        unique_vals = np.unique(m)
        components = []
        for val in unique_vals:
            mask = (m == val)
            labeled, n = ndimage.label(mask)
            for comp_id in range(1, n + 1):
                cells = np.argwhere(labeled == comp_id)
                components.append({
                    "value": int(val), "size": len(cells),
                    "min_r": int(cells[:, 0].min()), "max_r": int(cells[:, 0].max()),
                    "min_c": int(cells[:, 1].min()), "max_c": int(cells[:, 1].max()),
                })
        return components
    except ImportError:
        pass
    # Fallback: BFS implementation
    m = matrix.astype(int)
    rows, cols = m.shape
    visited = np.zeros_like(m, dtype=bool)
    components = []

    for r in range(rows):
        for c in range(cols):
            if visited[r, c]:
                continue
            val = m[r, c]
            stack = [(r, c)]
            cells = []
            while stack:
                cr, cc = stack.pop()
                if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                    continue
                if visited[cr, cc] or m[cr, cc] != val:
                    continue
                visited[cr, cc] = True
                cells.append((cr, cc))
                stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])
            components.append({"value": int(val), "size": len(cells),
                              "min_r": min(r for r, c in cells), "max_r": max(r for r, c in cells),
                              "min_c": min(c for r, c in cells), "max_c": max(c for r, c in cells)})
    return components


def diff_matrices(a: np.ndarray, b: np.ndarray, *, label: str = "diff", name: str = "delta") -> list[Signal]:
    """Compare two matrices of the same shape and emit delta signals."""
    if a.shape != b.shape:
        return [Signal("matrix_shape_mismatch", "inspect", label, name, 0.0,
                       {"shape_a": list(a.shape), "shape_b": list(b.shape)})]
    af, bf = a.astype(np.float64), b.astype(np.float64)
    delta = af - bf
    changed = np.abs(delta) > 1e-10
    return [
        Signal("matrix_delta_norm", "inspect", label, name, float(np.linalg.norm(delta)), {}),
        Signal("matrix_cells_changed", "inspect", label, name, float(np.sum(changed)),
               {"total_cells": int(a.size), "fraction": float(np.mean(changed))}),
        Signal("matrix_delta_max", "inspect", label, name, float(np.abs(delta).max()) if delta.size else 0.0, {}),
    ]


def _is_discrete(matrix: np.ndarray) -> bool:
    """Check if matrix contains only integer-like values."""
    if matrix.dtype.kind in ('i', 'u', 'b'):
        return True
    if matrix.dtype.kind == 'f':
        return bool(np.all(matrix == matrix.astype(int)))
    return False
