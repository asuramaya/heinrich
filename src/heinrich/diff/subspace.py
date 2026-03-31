"""Subspace comparison between weight deltas."""
from __future__ import annotations
from pathlib import Path
from typing import Any
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


def summarize_subspace_diffs(
    lhs_path: Path,
    rhs_path: Path,
    *,
    name_regex: str | None = None,
    only_square: bool = False,
    strip_prefixes: tuple[str, ...] = (),
    family: str | None = None,
    topk: int = 4,
    sample_size: int = 8192,
) -> dict[str, Any]:
    """Compare two tensor bundles by subspace analysis across shared 2-D tensors.

    Returns per-tensor diff norms, per-family aggregates, and a global
    fingerprint subspace from the stacked diff vectors.
    """
    from ..inspect.tensor import load_tensor_bundle, normalize_tensor_name
    from ..inspect.family import classify_tensor_family

    lhs = load_tensor_bundle(lhs_path, only_2d=True, only_square=only_square, name_regex=name_regex)
    rhs = load_tensor_bundle(rhs_path, only_2d=True, only_square=only_square, name_regex=name_regex)
    lhs_norm = {normalize_tensor_name(name, strip_prefixes): (name, arr) for name, arr in lhs.items()}
    rhs_norm = {normalize_tensor_name(name, strip_prefixes): (name, arr) for name, arr in rhs.items()}
    shared = sorted(set(lhs_norm) & set(rhs_norm))
    tensor_rows: list[dict[str, Any]] = []
    fingerprints: list[np.ndarray] = []
    names: list[str] = []
    for name in shared:
        lhs_name, lhs_arr = lhs_norm[name]
        rhs_name, rhs_arr = rhs_norm[name]
        if lhs_arr.shape != rhs_arr.shape:
            continue
        family_name = classify_tensor_family(name)
        if family and family_name != family:
            continue
        diff = np.asarray(lhs_arr - rhs_arr, dtype=np.float64)
        flat = diff.reshape(-1)
        sampled = _sample_vector(flat, sample_size=sample_size)
        diff_norm = float(np.linalg.norm(flat))
        tensor_rows.append({
            "name": name,
            "lhs_name": lhs_name,
            "rhs_name": rhs_name,
            "family": family_name,
            "shape": list(lhs_arr.shape),
            "diff_norm": diff_norm,
            "sample_dim": int(sampled.size),
        })
        fingerprints.append(sampled)
        names.append(name)
    families = _summarize_diff_families(tensor_rows)
    global_subspace = _summarize_fingerprint_subspace(fingerprints, names, topk=topk)
    return {
        "mode": "subspacediff",
        "lhs_bundle": str(lhs_path),
        "rhs_bundle": str(rhs_path),
        "shared_tensor_count": len(shared),
        "used_tensor_count": len(tensor_rows),
        "sample_size": int(sample_size),
        "family_filter": family,
        "tensors": tensor_rows,
        "families": families,
        "global_subspace": global_subspace,
    }


def _sample_vector(vector: np.ndarray, *, sample_size: int) -> np.ndarray:
    if vector.size <= sample_size:
        return np.asarray(vector, dtype=np.float64)
    indexes = np.linspace(0, vector.size - 1, num=sample_size, dtype=np.int64)
    return np.asarray(vector[indexes], dtype=np.float64)


def _summarize_diff_families(tensors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in tensors:
        family = str(row["family"])
        bucket = grouped.setdefault(
            family,
            {"family": family, "count": 0, "total_diff_norm": 0.0, "max_diff_norm": 0.0},
        )
        bucket["count"] += 1
        bucket["total_diff_norm"] += float(row["diff_norm"])
        bucket["max_diff_norm"] = max(float(bucket["max_diff_norm"]), float(row["diff_norm"]))
    out = []
    for bucket in grouped.values():
        out.append({
            "family": bucket["family"],
            "count": int(bucket["count"]),
            "total_diff_norm": float(bucket["total_diff_norm"]),
            "mean_diff_norm": float(bucket["total_diff_norm"] / max(bucket["count"], 1)),
            "max_diff_norm": float(bucket["max_diff_norm"]),
        })
    out.sort(key=lambda item: float(item["total_diff_norm"]), reverse=True)
    return out


def _summarize_fingerprint_subspace(
    fingerprints: list[np.ndarray],
    names: list[str],
    *,
    topk: int,
) -> dict[str, Any]:
    if not fingerprints:
        return {
            "tensor_count": 0,
            "rank": 0,
            "singular_values": [],
            "explained_energy": [],
            "components": [],
        }
    matrix = np.stack(fingerprints, axis=0)
    gram = matrix @ matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], a_min=0.0, a_max=None)
    eigenvectors = eigenvectors[:, order]
    singular = np.sqrt(eigenvalues)
    total = float(np.sum(eigenvalues))
    explained = [float(value / total) if total > 0 else 0.0 for value in eigenvalues[:topk]]
    components: list[dict[str, Any]] = []
    for index in range(min(topk, eigenvectors.shape[1])):
        loadings = []
        for tensor_index, name in enumerate(names):
            loading = float(eigenvectors[tensor_index, index])
            loadings.append({"name": name, "loading": loading, "abs_loading": abs(loading)})
        loadings.sort(key=lambda item: float(item["abs_loading"]), reverse=True)
        components.append({
            "index": int(index),
            "singular_value": float(singular[index]),
            "explained_energy": explained[index] if index < len(explained) else 0.0,
            "top_tensor_loadings": [
                {"name": row["name"], "loading": float(row["loading"])}
                for row in loadings[: min(6, len(loadings))]
            ],
        })
    return {
        "tensor_count": len(fingerprints),
        "rank": int(sum(value > 0 for value in singular)),
        "singular_values": [float(value) for value in singular[:topk]],
        "explained_energy": explained,
        "components": components,
    }
