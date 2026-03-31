"""Activation probes — linear probe fitting, feature matrix building, module ranking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LinearProbe:
    weights: np.ndarray
    bias: float
    method: str
    module_names: tuple[str, ...]
    feature_slices: tuple[tuple[str, slice], ...]


def flatten_activation_map(
    activation_map: Mapping[str, Any],
    module_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    names = tuple(module_names) if module_names is not None else tuple(sorted(str(name) for name in activation_map.keys()))
    flattened: dict[str, np.ndarray] = {}
    for name in names:
        if name not in activation_map:
            raise KeyError(f"Missing module activation: {name}")
        flattened[name] = _as_flat_vector(activation_map[name])
    return flattened


def build_feature_matrix(
    examples: Sequence[Mapping[str, Any]] | np.ndarray,
    module_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, slice]]:
    if isinstance(examples, np.ndarray):
        matrix = np.asarray(examples, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("Feature matrix must be 2D")
        if module_names is not None:
            names = tuple(str(name) for name in module_names)
            if len(names) != matrix.shape[1]:
                raise ValueError("module_names length must match feature dimension")
        else:
            names = tuple(f"feature_{index}" for index in range(matrix.shape[1]))
        slices = {name: slice(index, index + 1) for index, name in enumerate(names)}
        return matrix, names, slices

    if not examples:
        raise ValueError("examples must not be empty")
    normalized_examples = [flatten_activation_map(example, module_names=module_names) for example in examples]
    names = tuple(module_names) if module_names is not None else tuple(sorted(normalized_examples[0].keys()))
    vectors: list[np.ndarray] = []
    slices: dict[str, slice] = {}
    offset = 0
    for name in names:
        dims = normalized_examples[0][name].shape[0]
        for example in normalized_examples[1:]:
            if example[name].shape != normalized_examples[0][name].shape:
                raise ValueError(f"Module {name!r} has inconsistent shape across examples")
        slices[name] = slice(offset, offset + dims)
        offset += dims
    for example in normalized_examples:
        row = [example[name] for name in names]
        vectors.append(np.concatenate(row, axis=0))
    return np.vstack(vectors), names, slices


def mean_difference_direction(
    positive_examples: Sequence[Mapping[str, Any]] | np.ndarray,
    negative_examples: Sequence[Mapping[str, Any]] | np.ndarray,
    module_names: Sequence[str] | None = None,
) -> LinearProbe:
    pos_matrix, names, slices = build_feature_matrix(positive_examples, module_names=module_names)
    neg_matrix, neg_names, neg_slices = build_feature_matrix(negative_examples, module_names=names)
    if names != neg_names:
        raise ValueError("Positive and negative examples must share the same module order")
    pos_mean = pos_matrix.mean(axis=0)
    neg_mean = neg_matrix.mean(axis=0)
    weights = pos_mean - neg_mean
    bias = -0.5 * float(np.dot(weights, pos_mean + neg_mean))
    return LinearProbe(weights=weights, bias=bias, method="mean_difference", module_names=names, feature_slices=tuple(slices.items()))


def fit_binary_linear_probe(
    examples: Sequence[Mapping[str, Any]] | np.ndarray,
    labels: Sequence[int | bool | float],
    module_names: Sequence[str] | None = None,
    *,
    method: str = "ridge",
    l2: float = 1e-6,
) -> LinearProbe:
    matrix, names, slices = build_feature_matrix(examples, module_names=module_names)
    y = _coerce_binary_labels(labels)
    if len(y) != matrix.shape[0]:
        raise ValueError("labels length must match number of examples")
    if np.unique(y).size < 2:
        raise ValueError("labels must contain at least two classes")

    if method == "mean_difference":
        pos = matrix[y > 0]
        neg = matrix[y <= 0]
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("mean_difference requires both positive and negative examples")
        pos_mean = pos.mean(axis=0)
        neg_mean = neg.mean(axis=0)
        weights = pos_mean - neg_mean
        bias = -0.5 * float(np.dot(weights, pos_mean + neg_mean))
        return LinearProbe(weights=weights, bias=bias, method=method, module_names=names, feature_slices=tuple(slices.items()))

    if method != "ridge":
        raise ValueError("method must be 'ridge' or 'mean_difference'")

    design = np.concatenate([matrix, np.ones((matrix.shape[0], 1), dtype=np.float64)], axis=1)
    reg = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    reg[-1, -1] = 0.0
    lhs = design.T @ design + reg
    rhs = design.T @ y
    try:
        theta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(lhs) @ rhs
    weights = theta[:-1]
    bias = float(theta[-1])
    return LinearProbe(weights=weights, bias=bias, method=method, module_names=names, feature_slices=tuple(slices.items()))


def score_examples(
    probe: LinearProbe,
    examples: Sequence[Mapping[str, Any]] | np.ndarray,
    module_names: Sequence[str] | None = None,
) -> np.ndarray:
    matrix, names, _ = build_feature_matrix(examples, module_names=module_names or probe.module_names)
    if tuple(names) != tuple(probe.module_names):
        raise ValueError("Example module order does not match probe")
    return matrix @ np.asarray(probe.weights, dtype=np.float64) + float(probe.bias)


def rank_modules_by_separability(
    examples: Sequence[Mapping[str, Any]],
    labels: Sequence[int | bool | float],
    module_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if not examples:
        raise ValueError("examples must not be empty")
    y = _coerce_binary_labels(labels)
    if len(y) != len(examples):
        raise ValueError("labels length must match number of examples")
    normalized = [flatten_activation_map(example, module_names=module_names) for example in examples]
    names = tuple(module_names) if module_names is not None else tuple(sorted(normalized[0].keys()))
    rows: list[dict[str, Any]] = []
    for name in names:
        vectors = []
        for example in normalized:
            if name not in example:
                raise KeyError(f"Missing module activation: {name}")
            vectors.append(example[name])
        matrix = np.vstack(vectors)
        pos = matrix[y > 0]
        neg = matrix[y <= 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        pos_mean = pos.mean(axis=0)
        neg_mean = neg.mean(axis=0)
        diff = pos_mean - neg_mean
        pooled_var = 0.5 * (pos.var(axis=0) + neg.var(axis=0))
        pooled_std = float(np.sqrt(np.mean(pooled_var)))
        signal = float(np.linalg.norm(diff))
        score = signal / (pooled_std + 1e-12)
        rows.append(
            {
                "module_name": name,
                "score": score,
                "signal_norm": signal,
                "pooled_std": pooled_std,
                "positive_count": int(len(pos)),
                "negative_count": int(len(neg)),
                "feature_size": int(matrix.shape[1]),
                "mean_gap": diff,
            }
        )
    rows.sort(key=lambda row: (row["score"], row["signal_norm"]), reverse=True)
    return rows


def summarize_probe(
    probe: LinearProbe,
    scores: Sequence[float] | np.ndarray | None = None,
    labels: Sequence[int | bool | float] | None = None,
) -> dict[str, Any]:
    weight = np.asarray(probe.weights, dtype=np.float64)
    summary: dict[str, Any] = {
        "method": probe.method,
        "module_names": list(probe.module_names),
        "feature_count": int(weight.size),
        "weight_l2": float(np.linalg.norm(weight)),
        "weight_l1": float(np.linalg.norm(weight, ord=1)),
        "bias": float(probe.bias),
    }
    if scores is not None:
        arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        summary["score_summary"] = {
            "count": int(arr.size),
            "mean": float(arr.mean()) if arr.size else 0.0,
            "std": float(arr.std()) if arr.size else 0.0,
            "min": float(arr.min()) if arr.size else 0.0,
            "max": float(arr.max()) if arr.size else 0.0,
            "median": float(np.median(arr)) if arr.size else 0.0,
        }
        if labels is not None:
            y = _coerce_binary_labels(labels)
            if len(y) != arr.size:
                raise ValueError("labels length must match scores length")
            predictions = np.where(arr >= 0.0, 1, -1)
            summary["score_summary"]["accuracy"] = float(np.mean(predictions == y))
            summary["score_summary"]["positive_rate"] = float(np.mean(arr >= 0.0))
    return summary


def _coerce_binary_labels(labels: Sequence[int | bool | float]) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("labels must not be empty")
    uniq = set(np.unique(arr).tolist())
    if uniq <= {0.0, 1.0}:
        return np.where(arr > 0.0, 1.0, -1.0)
    if uniq <= {-1.0, 1.0}:
        return np.where(arr > 0.0, 1.0, -1.0)
    raise ValueError("labels must be binary and use either {0,1} or {-1,1}")


def _as_flat_vector(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr.reshape(-1)
    pooled = arr.mean(axis=tuple(range(arr.ndim - 1)), dtype=np.float64)
    return np.asarray(pooled, dtype=np.float64).reshape(-1)
