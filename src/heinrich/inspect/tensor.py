"""Tensor inspection — matrix stats, safetensors catalog, bundle audit and comparison."""
from __future__ import annotations

import csv
import json
import re
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None


NUMPY_SAFETENSORS_DTYPES: dict[str, np.dtype] = {
    "BOOL": np.dtype(np.bool_),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
    "I8": np.dtype("i1"),
    "I16": np.dtype("<i2"),
    "I32": np.dtype("<i4"),
    "I64": np.dtype("<i8"),
    "U8": np.dtype("u1"),
    "U16": np.dtype("<u2"),
    "U32": np.dtype("<u4"),
    "U64": np.dtype("<u8"),
}

TORCH_SAFETENSORS_DTYPES: dict[str, str] = {
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}

DETERMINISTIC_SUBSTRATE_MARKERS = (
    "linear_kernel", "linear_in_proj", "linear_decays",
    "controller_proj", "aux_proj", "sample_idx",
    "wr", "wi", "wf", "wm", "ws", "sf", "sm", "ss",
)

STRUCTURAL_CONTROL_MARKERS = (
    "causal_mask", "recency_kernel", "delimiter_mask", "number_mask",
    "special_mask", "markup_mask", "attr_mask", "entity_mask",
    "token_class_ids", "vocab_axis", "urlpath_mask",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64, copy=False)
    if path.suffix == ".csv":
        with path.open() as f:
            rows = [list(map(float, row)) for row in csv.reader(f) if row]
        return np.array(rows, dtype=np.float64)
    raise ValueError(f"Unsupported matrix format: {path.suffix}")


def load_npz_tensors(path: Path, *, only_2d: bool = True, only_square: bool = False, name_regex: str | None = None) -> dict[str, np.ndarray]:
    patt = re.compile(name_regex) if name_regex else None
    with np.load(path, allow_pickle=False) as data:
        out: dict[str, np.ndarray] = {}
        for name in data.files:
            arr = np.asarray(data[name], dtype=np.float64)
            if only_2d and arr.ndim != 2:
                continue
            if only_square and (arr.ndim != 2 or arr.shape[0] != arr.shape[1]):
                continue
            if patt and not patt.search(name):
                continue
            out[name] = arr
        return out


def load_tensor_bundle(path: Path, *, only_2d: bool = True, only_square: bool = False, name_regex: str | None = None) -> dict[str, np.ndarray]:
    if path.is_dir():
        return _load_safetensors_repo_tensors(path, only_2d=only_2d, only_square=only_square, name_regex=name_regex)
    if path.suffix == ".npz":
        return load_npz_tensors(path, only_2d=only_2d, only_square=only_square, name_regex=name_regex)
    if path.suffix == ".safetensors":
        return _load_safetensors_tensors(path, only_2d=only_2d, only_square=only_square, name_regex=name_regex)
    if path.name.endswith(".safetensors.index.json"):
        return _load_safetensors_repo_tensors(path, only_2d=only_2d, only_square=only_square, name_regex=name_regex)
    raise ValueError(f"Unsupported bundle format: {path}")


def _load_safetensors_tensors(
    path: Path, *, only_2d: bool = True, only_square: bool = False,
    name_regex: str | None = None, names: set[str] | None = None,
) -> dict[str, np.ndarray]:
    catalog = inspect_safetensors_file(path, name_regex=name_regex)
    out: dict[str, np.ndarray] = {}
    with path.open("rb") as f:
        for entry in catalog["tensors"]:
            name = str(entry["name"])
            if names is not None and name not in names:
                continue
            if not entry["complete"]:
                continue
            f.seek(int(entry["file_offset"]))
            blob = f.read(int(entry["nbytes"]))
            arr = _decode_safetensors_tensor_bytes(blob, str(entry["dtype"]), tuple(entry["shape"]))
            arr = np.asarray(arr, dtype=np.float64)
            if only_2d and arr.ndim != 2:
                continue
            if only_square and (arr.ndim != 2 or arr.shape[0] != arr.shape[1]):
                continue
            out[name] = arr
    return out


def _load_safetensors_repo_tensors(
    path: Path, *, only_2d: bool = True, only_square: bool = False, name_regex: str | None = None,
) -> dict[str, np.ndarray]:
    repo_root, weight_map = _resolve_safetensors_weight_map(path)
    patt = re.compile(name_regex) if name_regex else None
    selected_names = [name for name in sorted(weight_map) if not patt or patt.search(name)]
    grouped: dict[str, list[str]] = {}
    for name in selected_names:
        grouped.setdefault(weight_map[name], []).append(name)
    out: dict[str, np.ndarray] = {}
    for shard_name, tensor_names in sorted(grouped.items()):
        shard_path = repo_root / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(shard_path)
        out.update(_load_safetensors_tensors(shard_path, only_2d=only_2d, only_square=only_square, names=set(tensor_names)))
    return out


# ---------------------------------------------------------------------------
# Inspection / cataloging
# ---------------------------------------------------------------------------

def inspect_safetensors_file(path: Path, *, name_regex: str | None = None) -> dict[str, Any]:
    patt = re.compile(name_regex) if name_regex else None
    file_size = int(path.stat().st_size)
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Safetensors file is truncated before the header prefix: {path}")
        header_len = int(struct.unpack("<Q", prefix)[0])
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"Safetensors file is truncated inside the header JSON: {path}")
    payload = json.loads(header_bytes)
    metadata = payload.get("__metadata__", {})
    data_start = 8 + header_len
    tensors: list[dict[str, Any]] = []
    for name, meta in payload.items():
        if name == "__metadata__":
            continue
        if patt and not patt.search(str(name)):
            continue
        shape = [int(dim) for dim in meta["shape"]]
        start = int(meta["data_offsets"][0])
        end = int(meta["data_offsets"][1])
        tensors.append({
            "name": str(name), "dtype": str(meta["dtype"]), "shape": shape,
            "data_offsets": [start, end], "nbytes": int(end - start),
            "file_offset": int(data_start + start), "file_end": int(data_start + end),
            "complete": bool(data_start + end <= file_size),
        })
    return {
        "path": str(path), "file_bytes": file_size, "header_bytes": int(header_len),
        "data_start": int(data_start), "metadata": metadata,
        "tensor_count": len(tensors),
        "complete_tensor_count": int(sum(1 for e in tensors if e["complete"])),
        "tensors": tensors,
    }


def inspect_tensor_bundle(path: Path, *, name_regex: str | None = None, limit: int | None = None) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        result = inspect_safetensors_file(path, name_regex=name_regex)
        if limit is not None:
            result = dict(result)
            result["tensors"] = result["tensors"][:limit]
        return result
    if path.is_dir():
        repo_root, weight_map = _resolve_safetensors_weight_map(path)
        shard_catalogs: dict[str, dict[str, Any]] = {}
        selected = [name for name in sorted(weight_map) if not name_regex or re.search(name_regex, name)]
        for shard_name in sorted(set(weight_map[name] for name in selected)):
            shard_catalogs[shard_name] = inspect_safetensors_file(repo_root / shard_name, name_regex=name_regex)
        shard_rows = [{"shard": shard_name, "tensor_count": catalog["tensor_count"], "complete_tensor_count": catalog["complete_tensor_count"], "file_bytes": catalog["file_bytes"]} for shard_name, catalog in sorted(shard_catalogs.items())]
        return {"path": str(path), "repo_root": str(repo_root), "shard_count": len(shard_rows), "tensor_count": len(selected), "complete_tensor_count": int(sum(row["complete_tensor_count"] for row in shard_rows)), "shards": shard_rows, "limit": limit, "name_regex": name_regex}
    if path.name.endswith(".safetensors.index.json"):
        repo_root, weight_map = _resolve_safetensors_weight_map(path)
        shard_names = sorted(set(weight_map.values()))
        return {"path": str(path), "repo_root": str(repo_root), "shard_count": len(shard_names), "tensor_count": len([n for n in weight_map if not name_regex or re.search(name_regex, n)]), "complete_tensor_count": None, "shards": shard_names, "limit": limit, "name_regex": name_regex}
    return inspect_safetensors_file(path, name_regex=name_regex)


def catalog_safetensors(path: Path, *, name_regex: str | None = None, preview: int = 24) -> dict[str, Any]:
    catalog = inspect_safetensors_file(path, name_regex=name_regex)
    family_counts: dict[str, int] = {}
    complete_family_counts: dict[str, int] = {}
    dtype_counts: dict[str, int] = {}
    complete_dtype_counts: dict[str, int] = {}
    complete_preview: list[dict[str, Any]] = []
    incomplete_preview: list[dict[str, Any]] = []
    for entry in catalog["tensors"]:
        family = classify_artifact_entry(str(entry["name"]))
        dtype_name = str(entry["dtype"])
        family_counts[family] = family_counts.get(family, 0) + 1
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
        if entry["complete"]:
            complete_family_counts[family] = complete_family_counts.get(family, 0) + 1
            complete_dtype_counts[dtype_name] = complete_dtype_counts.get(dtype_name, 0) + 1
            if len(complete_preview) < preview:
                complete_preview.append(entry)
        elif len(incomplete_preview) < preview:
            incomplete_preview.append(entry)
    out = dict(catalog)
    out["family_counts"] = family_counts
    out["complete_family_counts"] = complete_family_counts
    out["dtype_counts"] = dtype_counts
    out["complete_dtype_counts"] = complete_dtype_counts
    out["complete_preview"] = complete_preview
    out["incomplete_preview"] = incomplete_preview
    return out


# ---------------------------------------------------------------------------
# Safetensors carving
# ---------------------------------------------------------------------------

def carve_safetensors_slice(
    src_path: Path, out_path: Path, *,
    only_2d: bool = False, only_square: bool = False,
    name_regex: str | None = None, max_tensors: int | None = None,
) -> dict[str, Any]:
    catalog = inspect_safetensors_file(src_path, name_regex=name_regex)
    selected: list[dict[str, Any]] = []
    for entry in catalog["tensors"]:
        if not entry["complete"]:
            continue
        if max_tensors is not None and len(selected) >= max_tensors:
            break
        shape = tuple(int(dim) for dim in entry["shape"])
        if only_2d and len(shape) != 2:
            continue
        if only_square and (len(shape) != 2 or shape[0] != shape[1]):
            continue
        selected.append(entry)

    payload = bytearray()
    header: dict[str, Any] = {"__metadata__": catalog.get("metadata", {})}
    cursor = 0
    with src_path.open("rb") as f:
        for entry in selected:
            f.seek(int(entry["file_offset"]))
            blob = f.read(int(entry["nbytes"]))
            if len(blob) != int(entry["nbytes"]):
                raise ValueError(f"Tensor bytes were not fully readable while carving {entry['name']!r}")
            header[str(entry["name"])] = {"dtype": str(entry["dtype"]), "shape": [int(d) for d in entry["shape"]], "data_offsets": [cursor, cursor + len(blob)]}
            payload.extend(blob)
            cursor += len(blob)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    out_path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    return {
        "source": str(src_path), "out": str(out_path),
        "catalog_tensor_count": int(catalog["tensor_count"]),
        "complete_tensor_count": int(catalog["complete_tensor_count"]),
        "written_tensor_count": len(selected),
        "written_payload_bytes": len(payload),
        "skipped_incomplete_tensor_count": int(sum(1 for e in catalog["tensors"] if not e["complete"])),
        "name_regex": name_regex,
    }


def subsetcarve_safetensors_range(
    catalog_source: Path, range_source: Path, out_path: Path, *,
    range_start: int, tensor_names: list[str] | None = None,
    name_regex: str | None = None, max_tensors: int | None = None,
) -> dict[str, Any]:
    catalog = inspect_safetensors_file(catalog_source, name_regex=name_regex)
    wanted = {str(n) for n in (tensor_names or []) if str(n)}
    range_size = int(range_source.stat().st_size)
    range_end = int(range_start + range_size)
    selected: list[dict[str, Any]] = []
    skipped_outside: list[str] = []
    skipped_unwanted: list[str] = []
    for entry in catalog["tensors"]:
        name = str(entry["name"])
        if wanted and name not in wanted:
            skipped_unwanted.append(name); continue
        if max_tensors is not None and len(selected) >= max_tensors:
            break
        file_offset = int(entry["file_offset"])
        file_end = int(entry["file_end"])
        if file_offset < range_start or file_end > range_end:
            skipped_outside.append(name); continue
        selected.append(entry)

    payload = bytearray()
    header: dict[str, Any] = {"__metadata__": catalog.get("metadata", {})}
    cursor = 0
    with range_source.open("rb") as f:
        for entry in selected:
            start = int(entry["file_offset"]) - int(range_start)
            end = int(entry["file_end"]) - int(range_start)
            f.seek(start)
            blob = f.read(end - start)
            if len(blob) != end - start:
                raise ValueError(f"Tensor bytes were not fully readable while subset-carving {entry['name']!r}")
            header[str(entry["name"])] = {"dtype": str(entry["dtype"]), "shape": [int(d) for d in entry["shape"]], "data_offsets": [cursor, cursor + len(blob)]}
            payload.extend(blob)
            cursor += len(blob)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    out_path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    missing_requested = sorted(wanted.difference({str(e["name"]) for e in selected}))
    return {
        "catalog_source": str(catalog_source), "range_source": str(range_source), "out": str(out_path),
        "range_start": int(range_start), "range_end": int(range_end),
        "catalog_tensor_count": int(catalog["tensor_count"]),
        "written_tensor_count": len(selected), "written_payload_bytes": len(payload),
        "requested_tensor_count": len(wanted), "missing_requested": missing_requested,
        "skipped_outside_range_count": len(skipped_outside), "name_regex": name_regex,
    }


# ---------------------------------------------------------------------------
# Matrix stats
# ---------------------------------------------------------------------------

def normalize_tensor_name(name: str, strip_prefixes: tuple[str, ...] = ()) -> str:
    for prefix in strip_prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def classify_artifact_entry(name: str) -> str:
    lowered = name.lower()
    # File-extension-based classification takes priority
    if lowered.endswith(".safetensors.index.json") or lowered.endswith(".safetensors.index"):
        return "safetensors_archive"
    if lowered.endswith(".safetensors"):
        return "safetensors"
    if lowered.endswith(".npz"):
        return "npz"
    if lowered.endswith(".npy"):
        return "npy"
    # Tensor-name-based classification for entries within a bundle
    if any(marker in lowered for marker in DETERMINISTIC_SUBSTRATE_MARKERS):
        return "deterministic_substrate"
    if any(marker in lowered for marker in STRUCTURAL_CONTROL_MARKERS):
        return "structural_control"
    if "." in lowered or "/" in lowered:
        return "unknown"
    return "learned_payload"


def load_packed_artifact(path: Path) -> tuple[dict[str, Any], int]:
    """Load a zlib-compressed artifact file. Note: uses pickle internally."""
    import pickle  # noqa: S403 - intentional; caller must trust source
    blob = path.read_bytes()
    raw = zlib.decompress(blob)
    payload = pickle.loads(raw)  # noqa: S301
    if not isinstance(payload, dict):
        raise ValueError("Packed artifact did not deserialize to a dict")
    return payload, len(raw)


def _artifact_payload_arrays(entry: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    kind = entry.get("type")
    arrays: list[tuple[str, np.ndarray]] = []
    if kind == "quant":
        arrays.append(("q", np.array(entry["q"], copy=False)))
        arrays.append(("scale", np.array(entry["scale"], copy=False)))
    elif kind in ("fp16", "raw"):
        arrays.append(("data", np.array(entry["data"], copy=False)))
    else:
        raise ValueError(f"Unknown packed param type: {kind}")
    return arrays


def audit_artifact(path: Path) -> dict[str, Any]:
    packed, raw_pickle_bytes = load_packed_artifact(path)
    entries: list[dict[str, Any]] = []
    class_totals: dict[str, int] = {}
    kind_totals: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    alerts: list[str] = []
    for name, entry_obj in sorted(packed.items()):
        entry = dict(entry_obj)
        kind = str(entry.get("type"))
        payload_arrays = _artifact_payload_arrays(entry)
        payload_bytes = int(sum(int(arr.nbytes) for _, arr in payload_arrays))
        numel = int(sum(int(arr.size) for _, arr in payload_arrays))
        class_name = classify_artifact_entry(name)
        shapes = {label: list(arr.shape) for label, arr in payload_arrays}
        dtypes = {label: str(arr.dtype) for label, arr in payload_arrays}
        entries.append({"name": name, "kind": kind, "class": class_name, "payload_bytes": payload_bytes, "numel": numel, "shapes": shapes, "dtypes": dtypes})
        class_totals[class_name] = class_totals.get(class_name, 0) + payload_bytes
        kind_totals[kind] = kind_totals.get(kind, 0) + payload_bytes
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    if class_counts.get("deterministic_substrate", 0):
        alerts.append(f"artifact includes deterministic substrate entries ({class_counts['deterministic_substrate']} entries, {class_totals['deterministic_substrate']} raw bytes)")
    if class_counts.get("structural_control", 0):
        alerts.append(f"artifact includes structural-control entries ({class_counts['structural_control']} entries, {class_totals['structural_control']} raw bytes)")
    entries_by_size = sorted(entries, key=lambda item: item["payload_bytes"], reverse=True)
    return {
        "artifact": str(path), "compressed_bytes": int(path.stat().st_size), "raw_pickle_bytes": int(raw_pickle_bytes),
        "entry_count": len(entries), "class_counts": class_counts,
        "class_payload_bytes": class_totals, "kind_payload_bytes": kind_totals,
        "largest_entries": entries_by_size[:min(16, len(entries_by_size))],
        "entries": entries, "alerts": alerts,
    }


def spectral_stats(matrix: np.ndarray, topk: int) -> dict[str, float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    effective_topk = min(topk, singular.size)
    energy = singular * singular
    total_energy = float(np.sum(energy))
    top_energy = float(np.sum(energy[:effective_topk]))
    sigma1 = float(singular[0])
    sigmak = float(singular[effective_topk - 1])
    sigmalast = float(singular[-1])
    result: dict[str, Any] = {
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
    return result


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


def compare_stats(matrix: Any, reference: Any) -> dict[str, float | None]:
    # Support comparing spectral_stats dicts directly
    if isinstance(matrix, dict) and isinstance(reference, dict):
        result: dict[str, float | None] = {}
        for key in set(matrix) | set(reference):
            lv = matrix.get(key)
            rv = reference.get(key)
            if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
                result[f"delta_{key}"] = float(rv) - float(lv)
        return result
    matrix_arr = np.asarray(matrix, dtype=np.float64)
    reference_arr = np.asarray(reference, dtype=np.float64)
    if matrix_arr.shape != reference_arr.shape:
        raise ValueError(f"Shape mismatch: {matrix_arr.shape} vs {reference_arr.shape}")
    flat = matrix_arr.reshape(-1)
    ref = reference_arr.reshape(-1)
    denom = float(np.linalg.norm(flat) * np.linalg.norm(ref))
    cosine = None if denom == 0.0 else float(np.dot(flat, ref) / denom)
    diff = flat - ref
    return {
        "cosine_to_reference": cosine,
        "l2_deviation": float(np.sqrt(np.mean(diff * diff))),
        "l1_deviation": float(np.mean(np.abs(diff))),
        "max_abs_deviation": float(np.max(np.abs(diff))),
    }


def audit_matrix(
    matrix: Any, *, name: str = "<matrix>", topk: int = 16,
    reference: np.ndarray | None = None, expect_causal_mask: bool = False,
) -> dict[str, Any]:
    if isinstance(matrix, Path):
        if name == "<matrix>":
            name = matrix.stem
        matrix = load_matrix(matrix)
    matrix = np.asarray(matrix, dtype=np.float64)
    result: dict[str, Any] = {"name": name, "shape": list(matrix.shape), "spectral": spectral_stats(matrix, topk)}
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        result["regions"] = region_stats(matrix)
    if reference is not None:
        result["compare"] = compare_stats(matrix, reference)
    alerts = _alerts_for_matrix(name, matrix, expect_causal_mask=expect_causal_mask)
    if alerts:
        result["alerts"] = alerts
    return result


def _alerts_for_matrix(name: str, matrix: np.ndarray, *, expect_causal_mask: bool = False, upper_thresh: float = 1e-8, diag_thresh: float = 1e-8) -> list[str]:
    alerts: list[str] = []
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        regions = region_stats(matrix)
        if expect_causal_mask:
            if regions["upper_frac"] > upper_thresh:
                alerts.append(f"forbidden upper-triangle energy detected ({regions['upper_frac']:.6g} > {upper_thresh:.6g})")
            if regions["diag_frac"] > diag_thresh:
                alerts.append(f"nonzero diagonal energy detected ({regions['diag_frac']:.6g} > {diag_thresh:.6g})")
        elif "mask" in name.lower() and regions["upper_plus_diag_frac"] > 1e-3:
            alerts.append(f"square matrix named like a mask has nontrivial upper+diag energy ({regions['upper_plus_diag_frac']:.6g})")
    return alerts


def audit_bundle(path: Path, *, topk: int = 16, only_square: bool = False, name_regex: str | None = None, expect_causal: tuple[str, ...] = ()) -> dict[str, Any]:
    tensors = load_tensor_bundle(path, only_2d=True, only_square=only_square, name_regex=name_regex)
    out: dict[str, Any] = {"bundle": str(path), "tensor_count": len(tensors), "tensors": []}
    for name, arr in sorted(tensors.items()):
        expect = any(p in name for p in expect_causal)
        out["tensors"].append(audit_matrix(arr, name=name, topk=topk, expect_causal_mask=expect))
    out["families"] = _summarize_bundle_families(out["tensors"])
    return out


def compare_bundles(
    lhs_path: Path, rhs_path: Path, *,
    topk: int = 16, only_square: bool = False,
    name_regex: str | None = None, strip_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    lhs = load_tensor_bundle(lhs_path, only_2d=True, only_square=only_square, name_regex=name_regex)
    rhs = load_tensor_bundle(rhs_path, only_2d=True, only_square=only_square, name_regex=name_regex)
    lhs_norm = {normalize_tensor_name(name, strip_prefixes): (name, arr) for name, arr in lhs.items()}
    rhs_norm = {normalize_tensor_name(name, strip_prefixes): (name, arr) for name, arr in rhs.items()}
    shared = sorted(set(lhs_norm) & set(rhs_norm))
    only_lhs = sorted(set(lhs_norm) - set(rhs_norm))
    only_rhs = sorted(set(rhs_norm) - set(lhs_norm))
    tensors: list[dict[str, Any]] = []
    shape_mismatches: list[str] = []
    for name in shared:
        lhs_name, l_arr = lhs_norm[name]
        rhs_name, r_arr = rhs_norm[name]
        item: dict[str, Any] = {"name": name, "lhs_name": lhs_name, "rhs_name": rhs_name, "lhs_shape": list(l_arr.shape), "rhs_shape": list(r_arr.shape)}
        if l_arr.shape != r_arr.shape:
            item["shape_mismatch"] = True; shape_mismatches.append(name)
        else:
            item["shape"] = list(l_arr.shape)
            item["compare"] = compare_stats(l_arr, r_arr)
        item["lhs_spectral"] = spectral_stats(l_arr, topk)
        item["rhs_spectral"] = spectral_stats(r_arr, topk)
        if l_arr.ndim == 2 and l_arr.shape[0] == l_arr.shape[1]:
            item["lhs_regions"] = region_stats(l_arr)
        if r_arr.ndim == 2 and r_arr.shape[0] == r_arr.shape[1]:
            item["rhs_regions"] = region_stats(r_arr)
        tensors.append(item)
    result = {"lhs_bundle": str(lhs_path), "rhs_bundle": str(rhs_path), "shared_tensor_count": len(shared), "lhs_only": only_lhs, "rhs_only": only_rhs, "shape_mismatches": shape_mismatches, "tensors": tensors}
    result["families"] = _summarize_compare_families(tensors)
    return result


def _summarize_bundle_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from .family import classify_tensor_family
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(classify_tensor_family(str(row["name"])), []).append(row)
    out: list[dict[str, Any]] = []
    for family, items in sorted(grouped.items()):
        sigma1 = [float(item["spectral"]["sigma1"]) for item in items]
        l2 = [float(item["spectral"]["fro_norm"]) for item in items]
        top = sorted(items, key=lambda item: float(item["spectral"]["sigma1"]), reverse=True)[:3]
        out.append({"family": family, "count": len(items), "mean_sigma1": float(np.mean(sigma1)), "max_sigma1": float(np.max(sigma1)), "mean_fro_norm": float(np.mean(l2)), "top_sigma1": [{"name": item["name"], "sigma1": float(item["spectral"]["sigma1"])} for item in top]})
    return out


def _summarize_compare_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from .family import classify_tensor_family
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if "compare" not in row:
            continue
        grouped.setdefault(classify_tensor_family(str(row["name"])), []).append(row)
    out: list[dict[str, Any]] = []
    for family, items in sorted(grouped.items()):
        compare_rows = [item["compare"] for item in items]
        exact_match_count = sum(1 for item in compare_rows if float(item["max_abs_deviation"]) == 0.0 and float(item["l2_deviation"]) == 0.0)
        top = sorted(items, key=lambda item: float(item["compare"]["max_abs_deviation"]), reverse=True)[:3]
        out.append({
            "family": family, "count": len(items), "exact_match_count": exact_match_count,
            "mean_cosine_to_reference": float(np.mean([float(item["cosine_to_reference"]) for item in compare_rows])),
            "mean_l2_deviation": float(np.mean([float(item["l2_deviation"]) for item in compare_rows])),
            "max_max_abs_deviation": float(np.max([float(item["max_abs_deviation"]) for item in compare_rows])),
            "top_outliers": [{"name": item["name"], "max_abs_deviation": float(item["compare"]["max_abs_deviation"]), "cosine_to_reference": float(item["compare"]["cosine_to_reference"])} for item in top],
        })
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_safetensors_weight_map(path: Path) -> tuple[Path, dict[str, str]]:
    if path.is_file():
        index_payload = json.loads(path.read_text(encoding="utf-8"))
        weight_map = index_payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Safetensors index is missing weight_map: {path}")
        return path.parent.resolve(), {str(name): str(shard) for name, shard in weight_map.items()}
    repo_root = path.resolve()
    index_paths = sorted(repo_root.glob("*.safetensors.index.json"))
    if len(index_paths) == 1:
        return _resolve_safetensors_weight_map(index_paths[0])
    if len(index_paths) > 1:
        raise ValueError(f"Multiple safetensors index files found in {repo_root}; pass one explicitly")
    shard_paths = sorted(repo_root.glob("*.safetensors"))
    if not shard_paths:
        raise ValueError(f"No safetensors files found in {repo_root}")
    if len(shard_paths) == 1:
        single = shard_paths[0]
        catalog = inspect_safetensors_file(single)
        return repo_root, {str(entry["name"]): single.name for entry in catalog["tensors"]}
    weight_map: dict[str, str] = {}
    for shard_path in shard_paths:
        catalog = inspect_safetensors_file(shard_path)
        for entry in catalog["tensors"]:
            name = str(entry["name"])
            if name in weight_map:
                raise ValueError(f"Tensor name {name!r} appears in multiple safetensors shards")
            weight_map[name] = shard_path.name
    return repo_root, weight_map


def _decode_safetensors_tensor_bytes(blob: bytes, dtype_name: str, shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape, dtype=np.int64))
    if dtype_name in NUMPY_SAFETENSORS_DTYPES:
        arr = np.frombuffer(blob, dtype=NUMPY_SAFETENSORS_DTYPES[dtype_name], count=count)
    elif dtype_name == "BF16":
        arr = _decode_bfloat16(blob, count)
    elif dtype_name in TORCH_SAFETENSORS_DTYPES:
        arr = _decode_torch_float_tensor(blob, dtype_name, count)
    else:
        raise ValueError(f"Unsupported safetensors dtype: {dtype_name}")
    return np.asarray(arr).reshape(shape)


def _decode_bfloat16(blob: bytes, count: int) -> np.ndarray:
    words = np.frombuffer(blob, dtype=np.dtype("<u2"), count=count)
    widened = (words.astype(np.uint32) << 16).view(np.float32)
    return widened


def _decode_torch_float_tensor(blob: bytes, dtype_name: str, count: int) -> np.ndarray:
    dtype_attr = TORCH_SAFETENSORS_DTYPES[dtype_name]
    if torch is None or not hasattr(torch, dtype_attr):
        raise ImportError(f"Decoding safetensors dtype {dtype_name} requires PyTorch with support for {dtype_attr}")
    torch_dtype = getattr(torch, dtype_attr)
    buffer = bytearray(blob)
    values = torch.frombuffer(buffer, dtype=torch.uint8, count=len(buffer))
    values = values.view(torch_dtype)
    if values.numel() != count:
        raise ValueError(f"Decoded tensor element count mismatch for dtype {dtype_name}: {values.numel()} vs {count}")
    return values.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Public bundle-family helpers (ported from conker-detect audit.py)
# ---------------------------------------------------------------------------

def load_safetensors_repo_tensors(
    path: Path,
    *,
    only_2d: bool = True,
    only_square: bool = False,
    name_regex: str | None = None,
) -> dict[str, np.ndarray]:
    """Public alias for _load_safetensors_repo_tensors."""
    return _load_safetensors_repo_tensors(path, only_2d=only_2d, only_square=only_square, name_regex=name_regex)


def summarize_bundle_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return per-family spectral stats for an audit_bundle tensor list."""
    from .family import classify_tensor_family
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(classify_tensor_family(str(row["name"])), []).append(row)
    out: list[dict[str, Any]] = []
    for family, items in sorted(grouped.items()):
        sigma1 = [float(item["spectral"]["sigma1"]) for item in items]
        l2 = [float(item["spectral"]["fro_norm"]) for item in items]
        top = sorted(items, key=lambda item: float(item["spectral"]["sigma1"]), reverse=True)[:3]
        out.append({
            "family": family,
            "count": len(items),
            "mean_sigma1": float(np.mean(sigma1)),
            "max_sigma1": float(np.max(sigma1)),
            "mean_fro_norm": float(np.mean(l2)),
            "top_sigma1": [
                {"name": item["name"], "sigma1": float(item["spectral"]["sigma1"])}
                for item in top
            ],
        })
    return out


def summarize_compare_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return per-family comparison stats for a compare_bundles tensor list."""
    from .family import classify_tensor_family
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if "compare" not in row:
            continue
        grouped.setdefault(classify_tensor_family(str(row["name"])), []).append(row)
    out: list[dict[str, Any]] = []
    for family, items in sorted(grouped.items()):
        compare_rows = [item["compare"] for item in items]
        exact_match_count = sum(
            1 for item in compare_rows
            if float(item["max_abs_deviation"]) == 0.0 and float(item["l2_deviation"]) == 0.0
        )
        top = sorted(items, key=lambda item: float(item["compare"]["max_abs_deviation"]), reverse=True)[:3]
        out.append({
            "family": family,
            "count": len(items),
            "exact_match_count": exact_match_count,
            "mean_cosine_to_reference": float(np.mean([float(item["cosine_to_reference"]) for item in compare_rows])),
            "mean_l2_deviation": float(np.mean([float(item["l2_deviation"]) for item in compare_rows])),
            "max_max_abs_deviation": float(np.max([float(item["max_abs_deviation"]) for item in compare_rows])),
            "top_outliers": [
                {
                    "name": item["name"],
                    "max_abs_deviation": float(item["compare"]["max_abs_deviation"]),
                    "cosine_to_reference": float(item["compare"]["cosine_to_reference"]),
                }
                for item in top
            ],
        })
    return out


def summarize_tensor_families(
    tensors: dict[str, np.ndarray],
    *,
    topk: int = 16,
) -> dict[str, Any]:
    """Summarize a dict of tensors grouped by family, returning spectral aggregates."""
    def _family_name(name: str) -> str:
        for suffix in (".weight_scale_inv", ".weight", ".bias", ".e_score_correction_bias"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    families: dict[str, list[dict[str, Any]]] = {}
    for name, arr in tensors.items():
        family = _family_name(name)
        item = audit_matrix(arr, name=name, topk=topk)
        families.setdefault(family, []).append(item)

    spectral_keys = ("sigma1", "effective_topk", "decay_1_to_last", "top16_energy_frac")
    family_rows: list[dict[str, Any]] = []
    for family, rows in sorted(families.items()):
        tensor_count = len(rows)
        shape_counts: dict[str, int] = {}
        total_bytes = 0
        total_elements = 0
        spectral_totals = {key: 0.0 for key in spectral_keys}
        top_rows = sorted(rows, key=lambda row: row["spectral"]["sigma1"], reverse=True)
        for row in rows:
            shape = tuple(int(dim) for dim in row["shape"])
            shape_counts[str(shape)] = shape_counts.get(str(shape), 0) + 1
            total_elements += int(np.prod(shape, dtype=np.int64))
            total_bytes += int(np.prod(shape, dtype=np.int64)) * 8
            for key in spectral_keys:
                spectral_totals[key] += float(row["spectral"].get(key, 0.0))
        family_rows.append({
            "family": family,
            "tensor_count": tensor_count,
            "member_names": [row["name"] for row in top_rows[:8]],
            "shape_counts": shape_counts,
            "total_elements": total_elements,
            "total_bytes": total_bytes,
            "spectral_mean": {
                key: spectral_totals[key] / tensor_count if tensor_count else 0.0
                for key in spectral_keys
            },
            "largest_tensors": [
                {
                    "name": row["name"],
                    "shape": row["shape"],
                    "sigma1": row["spectral"]["sigma1"],
                    "decay_1_to_last": row["spectral"].get("decay_1_to_last", 0.0),
                }
                for row in top_rows[:4]
            ],
        })
    return {
        "tensor_count": len(tensors),
        "family_count": len(family_rows),
        "families": family_rows,
    }


def mask_deviation(
    mask: np.ndarray,
    baseline: np.ndarray,
    support: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Compute element-wise deviation metrics between a mask and a baseline matrix."""
    if mask.shape != baseline.shape:
        raise ValueError(f"Shape mismatch: {mask.shape} vs {baseline.shape}")
    if support is None:
        active_mask = mask.reshape(-1)
        active_base = baseline.reshape(-1)
    else:
        active_mask = mask[support > 0]
        active_base = baseline[support > 0]
    diff = active_mask - active_base
    denom = float(np.linalg.norm(active_mask) * np.linalg.norm(active_base))
    cosine: float | None = (
        None if diff.size == 0 or denom == 0.0
        else float(np.dot(active_mask, active_base) / denom)
    )
    return {
        "mask_l1_deviation": float(np.mean(np.abs(diff)) if diff.size else 0.0),
        "mask_l2_deviation": float(np.sqrt(np.mean(diff * diff)) if diff.size else 0.0),
        "mask_max_abs_deviation": float(np.max(np.abs(diff)) if diff.size else 0.0),
        "mask_cosine_similarity": cosine,
    }
