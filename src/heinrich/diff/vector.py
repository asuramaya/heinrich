"""Vector comparison tools — cosine alignment, projection score, feature correlations."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def vectorize_numeric_leaves(
    obj: Any, *, max_leaves: int | None = None, sort_dict_keys: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Return (keys, vector) for all numeric leaves in obj."""
    items: list[tuple[str, float]] = []
    _collect_numeric_leaves(obj, path="", items=items, max_leaves=max_leaves, sort_dict_keys=sort_dict_keys)
    if not items:
        raise ValueError("vectorize_numeric_leaves found no numeric leaves")
    paths = [path for path, _ in items]
    vector = np.asarray([value for _, value in items], dtype=np.float64)
    return paths, vector


def _vectorize_numeric_leaves_full(
    obj: Any, *, max_leaves: int | None = None, sort_dict_keys: bool = True,
) -> dict[str, Any]:
    """Internal: return full stats dict for all numeric leaves."""
    paths, vector = vectorize_numeric_leaves(obj, max_leaves=max_leaves, sort_dict_keys=sort_dict_keys)
    return {
        "paths": paths, "vector": vector.tolist(), "count": int(vector.size),
        "norm": float(np.linalg.norm(vector)), "mean": float(np.mean(vector)),
        "std": float(np.std(vector)), "min": float(np.min(vector)), "max": float(np.max(vector)),
        "truncated": bool(max_leaves is not None and len(paths) >= int(max_leaves)),
    }


def compare_vectorized_payloads(lhs: Any, rhs: Any, *, max_leaves: int | None = None) -> dict[str, Any]:
    left_paths, left_vec = vectorize_numeric_leaves(lhs, max_leaves=max_leaves)
    right_paths, right_vec = vectorize_numeric_leaves(rhs, max_leaves=max_leaves)
    shared_paths = sorted(set(left_paths) & set(right_paths))
    if not shared_paths:
        raise ValueError("compare_vectorized_payloads found no shared numeric leaf paths")
    lhs_vector = np.asarray([_lookup_path(left_paths, left_vec.tolist(), path) for path in shared_paths], dtype=np.float64)
    rhs_vector = np.asarray([_lookup_path(right_paths, right_vec.tolist(), path) for path in shared_paths], dtype=np.float64)
    diff = rhs_vector - lhs_vector
    lhs_norm = float(np.linalg.norm(lhs_vector))
    rhs_norm = float(np.linalg.norm(rhs_vector))
    diff_norm = float(np.linalg.norm(diff))
    cosine = cosine_alignment(lhs_vector, rhs_vector)
    rmismatch = residual_mismatch(lhs_vector, rhs_vector)
    return {
        "shared_paths": shared_paths, "shared_count": int(len(shared_paths)),
        "lhs_count": int(left_vec.size), "rhs_count": int(right_vec.size),
        "lhs_norm": lhs_norm, "rhs_norm": rhs_norm, "diff_norm": diff_norm,
        "norm_delta": float(rhs_norm - lhs_norm),
        "cosine": cosine,
        "cosine_alignment": cosine,
        "residual_mismatch": rmismatch,
        "rhs_on_lhs_projection": projection_score(rhs_vector, lhs_vector),
        "lhs_on_rhs_projection": projection_score(lhs_vector, rhs_vector),
        "shared_fraction_lhs": float(len(shared_paths) / max(len(left_paths), 1)),
        "shared_fraction_rhs": float(len(shared_paths) / max(len(right_paths), 1)),
        "lhs_vector": lhs_vector.tolist(), "rhs_vector": rhs_vector.tolist(), "diff_vector": diff.tolist(),
    }


def cosine_alignment(lhs: Any, rhs: Any) -> float:
    lhs_vec = _coerce_dense_vector(lhs)
    rhs_vec = _coerce_dense_vector(rhs)
    denom = float(np.linalg.norm(lhs_vec) * np.linalg.norm(rhs_vec))
    if denom == 0.0:
        return 0.0
    return float(np.dot(lhs_vec, rhs_vec) / denom)


def projection_score(vector: Any, direction: Any) -> float:
    vec = _coerce_dense_vector(vector)
    basis = _coerce_dense_vector(direction)
    denom = float(np.linalg.norm(basis))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec, basis / denom))


def residual_mismatch(vector: Any, basis: Any | Sequence[Any]) -> float:
    """Compute the residual norm after projecting vector onto basis. Returns a float scalar."""
    vec = _coerce_dense_vector(vector)
    basis_matrix = _coerce_basis_matrix(basis)
    if basis_matrix.size == 0:
        return float(np.linalg.norm(vec))
    coeffs, *_ = np.linalg.lstsq(basis_matrix, vec, rcond=None)
    projection = basis_matrix @ coeffs
    residual = vec - projection
    return float(np.linalg.norm(residual))


def residual_mismatch_full(vector: Any, basis: Any | Sequence[Any]) -> dict[str, Any]:
    """Full residual stats dict."""
    vec = _coerce_dense_vector(vector)
    basis_matrix = _coerce_basis_matrix(basis)
    if basis_matrix.size == 0:
        vec_norm = float(np.linalg.norm(vec))
        return {
            "vector_norm": vec_norm, "projection_norm": 0.0, "residual_norm": vec_norm,
            "explained_energy": 0.0, "residual_fraction": 1.0 if vec_norm > 0.0 else 0.0,
            "coefficients": [], "projection": [0.0 for _ in range(vec.size)], "residual": vec.tolist(),
        }
    coeffs, *_ = np.linalg.lstsq(basis_matrix, vec, rcond=None)
    projection = basis_matrix @ coeffs
    residual = vec - projection
    vector_norm = float(np.linalg.norm(vec))
    projection_norm = float(np.linalg.norm(projection))
    residual_norm = float(np.linalg.norm(residual))
    return {
        "vector_norm": vector_norm, "projection_norm": projection_norm, "residual_norm": residual_norm,
        "explained_energy": 0.0 if vector_norm == 0.0 else float((projection_norm**2) / (vector_norm**2)),
        "residual_fraction": 0.0 if vector_norm == 0.0 else float((residual_norm**2) / (vector_norm**2)),
        "coefficients": coeffs.astype(np.float64).tolist(),
        "projection": projection.tolist(), "residual": residual.tolist(),
    }


def summarize_feature_correlations(
    rows: Sequence[dict[str, Any]], *, features: Sequence[str] | None = None,
    target_feature: str | None = None, topk: int = 4,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("summarize_feature_correlations requires at least one row")
    if features is None:
        feature_names = sorted({str(key) for row in rows for key, value in row.items() if _is_numeric_scalar(value)})
    else:
        feature_names = [str(f) for f in features]
    if not feature_names:
        raise ValueError("summarize_feature_correlations found no numeric features")

    matrix = np.full((len(rows), len(feature_names)), np.nan, dtype=np.float64)
    for row_index, row in enumerate(rows):
        for col_index, feature in enumerate(feature_names):
            value = row.get(feature)
            if _is_numeric_scalar(value):
                matrix[row_index, col_index] = float(value)

    feature_stats: list[dict[str, Any]] = []
    for col_index, feature in enumerate(feature_names):
        column = matrix[:, col_index]
        finite = column[np.isfinite(column)]
        mean = float(np.mean(finite)) if finite.size else 0.0
        std = float(np.std(finite)) if finite.size else 0.0
        feature_stats.append({"feature": feature, "mean": mean, "std": std, "min": float(np.min(finite)) if finite.size else 0.0, "max": float(np.max(finite)) if finite.size else 0.0, "count": int(finite.size)})

    correlation_rows: list[dict[str, Any]] = []
    for lhs_index in range(len(feature_names)):
        for rhs_index in range(lhs_index + 1, len(feature_names)):
            corr, sample_count = _pearson_between_columns(matrix[:, lhs_index], matrix[:, rhs_index])
            correlation_rows.append({"lhs_feature": feature_names[lhs_index], "rhs_feature": feature_names[rhs_index], "correlation": corr, "sample_count": sample_count})
    correlation_rows.sort(key=lambda row: (abs(float(row["correlation"])), float(row["sample_count"])), reverse=True)

    target_correlations: list[dict[str, Any]] = []
    if target_feature is not None and target_feature in feature_names:
        target_index = feature_names.index(target_feature)
        for feature_index, feature in enumerate(feature_names):
            if feature_index == target_index:
                continue
            corr, sample_count = _pearson_between_columns(matrix[:, target_index], matrix[:, feature_index])
            target_correlations.append({"feature": feature, "correlation": corr, "sample_count": sample_count})
        target_correlations.sort(key=lambda row: (abs(float(row["correlation"])), float(row["sample_count"])), reverse=True)

    return {
        "mode": "vectordiff", "row_count": len(rows), "feature_names": feature_names,
        "feature_stats": feature_stats,
        "correlation_matrix": np.eye(len(feature_names), dtype=np.float64).tolist(),
        "correlations": correlation_rows[:min(topk, len(correlation_rows))],
        "target_feature": target_feature,
        "target_correlations": target_correlations[:min(topk, len(target_correlations))],
        "matrix": matrix.tolist(),
    }


def summarize_shared_directions(
    payloads: Sequence[Any], *, names: Sequence[str] | None = None,
    topk: int = 4, center: bool = True, max_leaves: int | None = None,
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("summarize_shared_directions requires at least one payload")
    vecs_tuples: list[tuple[list[str], np.ndarray]] = [vectorize_numeric_leaves(payload, max_leaves=max_leaves) for payload in payloads]
    vectors = [{"paths": p, "vector": v.tolist()} for p, v in vecs_tuples]
    shared_paths = sorted(set.intersection(*(set(item["paths"]) for item in vectors)))
    if not shared_paths:
        raise ValueError("summarize_shared_directions found no shared numeric paths")
    matrix = np.vstack([np.asarray([_lookup_path(item["paths"], item["vector"], path) for path in shared_paths], dtype=np.float64) for item in vectors])
    raw_matrix = matrix.copy()
    if center:
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
    if matrix.size == 0:
        raise ValueError("summarize_shared_directions requires non-empty numeric vectors")
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular_values > 0.0))
    total_energy = float(np.sum(singular_values**2))
    explained_energy = [float((value**2) / total_energy) if total_energy > 0.0 else 0.0 for value in singular_values[:topk]]
    component_rows: list[dict[str, Any]] = []
    for index in range(min(topk, vt.shape[0])):
        loading = vt[index]
        top_features = sorted(({"path": path, "loading": float(value), "abs_loading": abs(float(value))} for path, value in zip(shared_paths, loading)), key=lambda row: float(row["abs_loading"]), reverse=True)
        component_rows.append({"index": int(index), "singular_value": float(singular_values[index]), "explained_energy": explained_energy[index] if index < len(explained_energy) else 0.0, "top_features": [{"path": row["path"], "loading": float(row["loading"])} for row in top_features[:min(6, len(top_features))]]})
    pairwise = []
    for lhs_index in range(len(vectors)):
        for rhs_index in range(lhs_index + 1, len(vectors)):
            lhs_vec = raw_matrix[lhs_index]
            rhs_vec = raw_matrix[rhs_index]
            pairwise.append({"lhs_index": lhs_index, "rhs_index": rhs_index, "cosine": cosine_alignment(lhs_vec, rhs_vec), "diff_norm": float(np.linalg.norm(rhs_vec - lhs_vec)), "norm_delta": float(np.linalg.norm(rhs_vec) - np.linalg.norm(lhs_vec))})
    pairwise.sort(key=lambda row: (float(row["cosine"]), -float(row["diff_norm"])), reverse=True)
    feature_rows = [{path: float(value) for path, value in zip(shared_paths, row)} for row in raw_matrix]
    feature_correlations = summarize_feature_correlations(feature_rows, features=shared_paths, topk=topk)
    return {
        "mode": "vectordiff", "payload_count": len(payloads), "shared_path_count": len(shared_paths),
        "shared_paths": shared_paths, "centered": bool(center),
        "names": list(names) if names is not None else [f"payload_{index}" for index in range(len(payloads))],
        "rank": rank, "singular_values": [float(value) for value in singular_values[:topk]],
        "explained_energy": explained_energy, "components": component_rows,
        "pairwise": pairwise[:min(10, len(pairwise))],
        "feature_correlations": feature_correlations, "raw_matrix": raw_matrix.tolist(),
    }


def _collect_numeric_leaves(obj: Any, *, path: str, items: list[tuple[str, float]], max_leaves: int | None, sort_dict_keys: bool) -> None:
    if max_leaves is not None and len(items) >= int(max_leaves):
        return
    if isinstance(obj, dict):
        keys = obj.keys()
        if sort_dict_keys:
            keys = sorted(keys, key=lambda value: str(value))
        for key in keys:
            child_path = _join_path(path, str(key))
            _collect_numeric_leaves(obj[key], path=child_path, items=items, max_leaves=max_leaves, sort_dict_keys=sort_dict_keys)
            if max_leaves is not None and len(items) >= int(max_leaves):
                return
        return
    if isinstance(obj, (list, tuple)):
        for index, child in enumerate(obj):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            _collect_numeric_leaves(child, path=child_path, items=items, max_leaves=max_leaves, sort_dict_keys=sort_dict_keys)
            if max_leaves is not None and len(items) >= int(max_leaves):
                return
        return
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if arr.size == 0:
            return
        if np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_:
            flat = arr.astype(np.float64, copy=False).reshape(-1)
            for index, value in enumerate(flat):
                item_path = f"{path}[{index}]" if path else f"[{index}]"
                items.append((item_path, float(value)))
                if max_leaves is not None and len(items) >= int(max_leaves):
                    return
        return
    if _is_numeric_scalar(obj):
        items.append((path or "value", float(obj)))


def _join_path(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _is_numeric_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float, np.number)):
        return True
    return False


def _pearson_between_columns(lhs: np.ndarray, rhs: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(lhs) & np.isfinite(rhs)
    sample_count = int(np.sum(mask))
    if sample_count < 2:
        return 0.0, sample_count
    lhs_sample = lhs[mask].astype(np.float64, copy=False)
    rhs_sample = rhs[mask].astype(np.float64, copy=False)
    lhs_std = float(np.std(lhs_sample))
    rhs_std = float(np.std(rhs_sample))
    if lhs_std == 0.0 or rhs_std == 0.0:
        return 0.0, sample_count
    corr = float(np.corrcoef(lhs_sample, rhs_sample)[0, 1])
    if not np.isfinite(corr):
        return 0.0, sample_count
    return corr, sample_count


def _coerce_dense_vector(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64).reshape(-1)
    if isinstance(value, (list, tuple)) and value and all(_is_numeric_scalar(item) for item in value):
        return np.asarray(value, dtype=np.float64).reshape(-1)
    if isinstance(value, (dict, list, tuple)):
        _, vec = vectorize_numeric_leaves(value)
        return np.asarray(vec, dtype=np.float64).reshape(-1)
    if _is_numeric_scalar(value):
        return np.asarray([float(value)], dtype=np.float64)
    raise TypeError(f"Cannot coerce value of type {type(value).__name__} to a dense numeric vector")


def _coerce_basis_matrix(basis: Any | Sequence[Any]) -> np.ndarray:
    if basis is None:
        return np.zeros((0, 0), dtype=np.float64)
    if isinstance(basis, np.ndarray):
        arr = np.asarray(basis, dtype=np.float64)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("basis ndarray must be 1D or 2D")
        return arr
    if isinstance(basis, (list, tuple)):
        vectors = [_coerce_dense_vector(item) for item in basis]
        if not vectors:
            return np.zeros((0, 0), dtype=np.float64)
        width = vectors[0].size
        if any(vector.size != width for vector in vectors):
            raise ValueError("basis vectors must all have the same length")
        return np.stack(vectors, axis=1)
    return _coerce_dense_vector(basis).reshape(-1, 1)


def _lookup_path(paths: Sequence[str], vector: Sequence[float], target: str) -> float:
    index = paths.index(target)
    return float(vector[index])
