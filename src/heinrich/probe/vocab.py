"""Vocabulary inspection — token row analysis and cross-model comparison."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .token_tools import _extract_input_ids, _safe_decode_piece, load_tokenizer


@dataclass
class _ModelBundle:
    model_ref: str
    tokenizer_ref: str
    tokenizer: Any
    model: Any


def inspect_token_rows(
    tokenizer_ref: str | None = None,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    queries: list[str] | tuple[str, ...] | None = None,
    token_ids: list[int] | tuple[int, ...] | None = None,
    model_ref: str | None = None,
    trust_remote_code: bool = True,
    use_fast: bool = True,
    dtype: str | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    # New-style API: model + tokenizer objects passed directly
    if model is not None or tokenizer is not None:
        return _inspect_token_rows_from_objects(
            model=model, tokenizer=tokenizer,
            token_ids=list(token_ids or []),
            queries=list(queries or []),
        )
    # Legacy API: tokenizer_ref string
    bundle = _load_bundle(tokenizer_ref=tokenizer_ref or "", model_ref=model_ref, trust_remote_code=trust_remote_code, use_fast=use_fast, dtype=dtype, device=device)
    rows = []
    for index, query in enumerate(list(queries or [])):
        record = _describe_query_string(bundle, str(query), index=index)
        record.pop("aggregate_vectors", None)
        rows.append(record)
    for index, token_id in enumerate(list(token_ids or [])):
        record = _describe_token_id(bundle, int(token_id), index=index)
        record.pop("aggregate_vectors", None)
        rows.append(record)
    return rows


def _inspect_token_rows_from_objects(
    *,
    model: Any,
    tokenizer: Any,
    token_ids: list[int],
    queries: list[str],
) -> list[dict[str, Any]]:
    """Inspect token rows using pre-loaded model and tokenizer objects."""
    rows: list[dict[str, Any]] = []
    for token_id in token_ids:
        piece = _safe_decode_piece(tokenizer, token_id)
        row: dict[str, Any] = {"token_id": token_id, "piece": piece}
        if model is not None:
            input_row = _lookup_embedding_row(model, token_id, kind="input")
            if input_row is not None:
                row["input_row"] = _vector_stats(input_row)
        rows.append(row)
    for index, query in enumerate(queries):
        encoded = tokenizer(query, add_special_tokens=False)
        ids = _extract_input_ids(encoded)
        pieces = [_safe_decode_piece(tokenizer, tid) for tid in ids]
        row = {"query_id": f"string::{index}", "query": query, "token_count": len(ids), "token_ids": ids, "token_pieces": pieces}
        rows.append(row)
    return rows


def compare_token_rows_across_models(
    lhs_rows: list[dict[str, Any]] | None = None,
    rhs_rows: list[dict[str, Any]] | None = None,
    *,
    tokenizer_ref: str | None = None,
    lhs_model_ref: str | None = None,
    rhs_model_ref: str | None = None,
    queries: list[str] | tuple[str, ...] | None = None,
    token_ids: list[int] | tuple[int, ...] | None = None,
    trust_remote_code: bool = True,
    use_fast: bool = True,
    dtype: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    # New-style API: compare two pre-computed row lists
    if lhs_rows is not None and rhs_rows is not None:
        return _compare_row_lists(lhs_rows, rhs_rows)
    # Legacy API
    if tokenizer_ref is None:
        return {"mode": "vocabdiff", "error": "tokenizer_ref required", "query_count": 0, "queries": []}
    lhs_bundle = _load_bundle(tokenizer_ref=tokenizer_ref, model_ref=lhs_model_ref, trust_remote_code=trust_remote_code, use_fast=use_fast, dtype=dtype, device=device)
    rhs_bundle = _load_bundle(tokenizer_ref=tokenizer_ref, model_ref=rhs_model_ref, trust_remote_code=trust_remote_code, use_fast=use_fast, dtype=dtype, device=device)
    rows = []
    for index, query in enumerate(list(queries or [])):
        lhs = _describe_query_string(lhs_bundle, str(query), index=index)
        rhs = _describe_query_string(rhs_bundle, str(query), index=index)
        rows.append(_compare_query_payload(lhs, rhs))
    for index, token_id in enumerate(list(token_ids or [])):
        lhs = _describe_token_id(lhs_bundle, int(token_id), index=index)
        rhs = _describe_token_id(rhs_bundle, int(token_id), index=index)
        rows.append(_compare_query_payload(lhs, rhs))
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return {
        "mode": "vocabdiff",
        "tokenizer_name": getattr(lhs_bundle.tokenizer, "name_or_path", tokenizer_ref),
        "lhs_model_ref": lhs_bundle.model_ref, "rhs_model_ref": rhs_bundle.model_ref,
        "query_count": len(rows), "queries": rows,
    }


def _compare_row_lists(lhs: list[dict[str, Any]], rhs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare two lists of token rows."""
    comparisons: list[dict[str, Any]] = []
    for l_row, r_row in zip(lhs, rhs):
        comp: dict[str, Any] = {
            "token_id": l_row.get("token_id"),
            "piece": l_row.get("piece"),
            "match": l_row.get("piece") == r_row.get("piece"),
        }
        comparisons.append(comp)
    return {
        "mode": "vocabdiff",
        "query_count": len(comparisons),
        "queries": comparisons,
    }


def _load_bundle(*, tokenizer_ref: str, model_ref: str | None, trust_remote_code: bool, use_fast: bool, dtype: str | None, device: str | None) -> _ModelBundle:
    tokenizer = load_tokenizer(tokenizer_ref, trust_remote_code=trust_remote_code, use_fast=use_fast)
    if not model_ref:
        return _ModelBundle(model_ref="", tokenizer_ref=tokenizer_ref, tokenizer=tokenizer, model=None)
    transformers = _import_transformers()
    torch_mod = _import_torch()
    resolved_dtype = _resolve_torch_dtype(torch_mod, dtype)
    resolved_device = _resolve_device(torch_mod, device)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_ref, trust_remote_code=trust_remote_code, torch_dtype=resolved_dtype)
    if hasattr(model, "to"):
        model = model.to(resolved_device)
    if hasattr(model, "eval"):
        model.eval()
    return _ModelBundle(model_ref=str(model_ref), tokenizer_ref=str(tokenizer_ref), tokenizer=tokenizer, model=model)


def _describe_query_string(bundle: _ModelBundle, query: str, *, index: int) -> dict[str, Any]:
    encoded = bundle.tokenizer(query, add_special_tokens=False)
    token_ids = _extract_input_ids(encoded)
    token_pieces = [_safe_decode_piece(bundle.tokenizer, token_id) for token_id in token_ids]
    row_payloads = [_row_payload(bundle, token_id, piece) for token_id, piece in zip(token_ids, token_pieces)]
    aggregate = _aggregate_row_vectors(row_payloads)
    terminal_piece = token_pieces[-1] if token_pieces else None
    return {
        "query_id": f"string::{index}", "query_type": "string", "query": query,
        "token_count": len(token_ids), "token_ids": token_ids, "token_pieces": token_pieces,
        "exact_single_token": bool(len(token_ids) == 1), "terminal_piece": terminal_piece,
        "terminal_has_space_prefix": bool(terminal_piece and str(terminal_piece).startswith(("Ġ", "▁", " "))),
        "row_stats": row_payloads, "aggregate_vectors": aggregate, "aggregate_stats": _aggregate_stats(aggregate),
    }


def _describe_token_id(bundle: _ModelBundle, token_id: int, *, index: int) -> dict[str, Any]:
    piece = _safe_decode_piece(bundle.tokenizer, token_id)
    row_payload = _row_payload(bundle, token_id, piece)
    aggregate = _aggregate_row_vectors([row_payload])
    return {
        "query_id": f"token_id::{index}", "query_type": "token_id", "query": token_id,
        "token_count": 1, "token_ids": [token_id], "token_pieces": [piece],
        "exact_single_token": True, "terminal_piece": piece,
        "terminal_has_space_prefix": bool(piece.startswith(("Ġ", "▁", " "))),
        "row_stats": [row_payload], "aggregate_vectors": aggregate, "aggregate_stats": _aggregate_stats(aggregate),
    }


def _row_payload(bundle: _ModelBundle, token_id: int, piece: str) -> dict[str, Any]:
    payload = {"token_id": int(token_id), "piece": str(piece)}
    if bundle.model is None:
        return payload
    input_row = _lookup_embedding_row(bundle.model, int(token_id), kind="input")
    output_row = _lookup_embedding_row(bundle.model, int(token_id), kind="output")
    if input_row is not None:
        payload["input_row"] = _vector_stats(input_row)
        payload["input_vector"] = input_row.tolist()
    if output_row is not None:
        payload["output_row"] = _vector_stats(output_row)
        payload["output_vector"] = output_row.tolist()
    return payload


def _lookup_embedding_row(model: Any, token_id: int, *, kind: str) -> np.ndarray | None:
    getter = getattr(model, "get_input_embeddings" if kind == "input" else "get_output_embeddings", None)
    if getter is None or not callable(getter):
        return None
    embedding = getter()
    if embedding is None:
        return None
    weight = getattr(embedding, "weight", embedding)
    array = _tensor_to_numpy(weight)
    if array.ndim != 2 or token_id < 0 or token_id >= array.shape[0]:
        return None
    return np.asarray(array[token_id], dtype=np.float64)


def _aggregate_row_vectors(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for kind in ("input", "output"):
        vectors = [np.asarray(row[f"{kind}_vector"], dtype=np.float64) for row in rows if f"{kind}_vector" in row]
        if not vectors:
            continue
        out[kind] = np.mean(np.stack(vectors, axis=0), axis=0).tolist()
    return out


def _aggregate_stats(aggregate: dict[str, list[float]]) -> dict[str, Any]:
    return {kind: _vector_stats(np.asarray(vector, dtype=np.float64)) for kind, vector in aggregate.items()}


def _pairwise_query_vectors(vectors: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    query_ids = sorted(vectors)
    rows: list[dict[str, Any]] = []
    for left_index in range(len(query_ids)):
        for right_index in range(left_index + 1, len(query_ids)):
            lhs_key = query_ids[left_index]
            rhs_key = query_ids[right_index]
            lhs = vectors[lhs_key]
            rhs = vectors[rhs_key]
            row = {"lhs_query_id": lhs_key, "rhs_query_id": rhs_key}
            for kind in ("input", "output"):
                if kind in lhs and kind in rhs:
                    row[f"{kind}_cosine"] = _cosine(lhs[kind], rhs[kind])
            rows.append(row)
    rows.sort(key=lambda item: max(1.0 - float(item.get("input_cosine", 1.0)), 1.0 - float(item.get("output_cosine", 1.0))), reverse=True)
    return rows


def _compare_query_payload(lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
    result = {
        "query_id": str(lhs["query_id"]), "query_type": str(lhs["query_type"]), "query": lhs["query"],
        "tokenization_match": lhs["token_pieces"] == rhs["token_pieces"],
        "token_ids": lhs["token_ids"], "token_pieces": lhs["token_pieces"],
        "exact_single_token": bool(lhs["exact_single_token"] and rhs["exact_single_token"]),
        "terminal_piece": lhs.get("terminal_piece"), "terminal_has_space_prefix": bool(lhs.get("terminal_has_space_prefix")),
    }
    score = 0.0
    lhs_stats = lhs.get("aggregate_stats", {})
    rhs_stats = rhs.get("aggregate_stats", {})
    for kind in ("input", "output"):
        lhs_vec = lhs.get("aggregate_vectors", {}).get(kind)
        rhs_vec = rhs.get("aggregate_vectors", {}).get(kind)
        if lhs_vec is None or rhs_vec is None:
            continue
        lhs_array = np.asarray(lhs_vec, dtype=np.float64)
        rhs_array = np.asarray(rhs_vec, dtype=np.float64)
        cosine = _cosine(lhs_array, rhs_array)
        l2 = float(np.linalg.norm(lhs_array - rhs_array))
        lhs_norm = float(lhs_stats.get(kind, {}).get("norm", 0.0))
        rhs_norm = float(rhs_stats.get(kind, {}).get("norm", 0.0))
        norm_delta = abs(lhs_norm - rhs_norm)
        result[f"{kind}_cosine"] = cosine
        result[f"{kind}_l2_deviation"] = l2
        result[f"{kind}_norm_delta"] = norm_delta
        score += (1.0 - cosine) + l2 + norm_delta
    if not result["tokenization_match"]:
        score += 1.0
    result["score"] = float(score)
    return result


def _vector_stats(vector: np.ndarray) -> dict[str, Any]:
    return {"dim": int(vector.size), "norm": float(np.linalg.norm(vector)), "mean": float(np.mean(vector)), "std": float(np.std(vector)), "max_abs": float(np.max(np.abs(vector))) if vector.size else 0.0}


def _cosine(lhs: np.ndarray, rhs: np.ndarray) -> float:
    denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denom == 0.0:
        return 0.0
    return float(np.dot(lhs.reshape(-1), rhs.reshape(-1)) / denom)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    current = value
    for attr in ("detach",):
        if hasattr(current, attr):
            current = getattr(current, attr)()
    if hasattr(current, "to"):
        try:
            current = current.to("cpu")
        except Exception:
            pass
    if hasattr(current, "float"):
        try:
            current = current.float()
        except Exception:
            pass
    if hasattr(current, "numpy"):
        try:
            return np.asarray(current.numpy(), dtype=np.float64)
        except Exception:
            pass
    return np.asarray(current, dtype=np.float64)


def _resolve_torch_dtype(torch_mod: Any, raw: Any) -> Any:
    if raw is None or str(raw) == "auto":
        return None
    value = getattr(torch_mod, str(raw), None)
    if value is None:
        raise ValueError(f"Unknown torch dtype: {raw}")
    return value


def _resolve_device(torch_mod: Any, raw: Any) -> str:
    if raw:
        return str(raw)
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is not None and hasattr(cuda, "is_available") and cuda.is_available():
        return "cuda"
    backends = getattr(torch_mod, "backends", None)
    if backends is not None:
        mps = getattr(backends, "mps", None)
        if mps is not None and hasattr(mps, "is_available") and mps.is_available():
            return "mps"
    return "cpu"


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise ImportError("transformers is required for vocab tools") from exc
    return transformers


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for vocab tools") from exc
    return torch
