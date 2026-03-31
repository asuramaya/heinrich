"""Hologram report -- embedding-direction projection onto activation modules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .triangulate import load_case_suite, run_triangulation_report, _classify_identity_output
from .trigger_core import describe_provider, normalize_case
from .token_tools import _import_torch, _import_transformers, _resolve_device, _resolve_torch_dtype, load_tokenizer


def _cosine_alignment(lhs: Any, rhs: Any) -> float:
    lhs_arr = np.asarray(lhs, dtype=np.float64).ravel()
    rhs_arr = np.asarray(rhs, dtype=np.float64).ravel()
    lhs_norm = float(np.linalg.norm(lhs_arr))
    rhs_norm = float(np.linalg.norm(rhs_arr))
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 0.0
    return float(np.dot(lhs_arr, rhs_arr) / (lhs_norm * rhs_norm))


def _residual_mismatch(vector: Any, basis: Any) -> dict[str, Any]:
    v = np.asarray(vector, dtype=np.float64).ravel()
    b = np.asarray(basis, dtype=np.float64).ravel()
    b_norm_sq = float(np.dot(b, b))
    if b_norm_sq == 0.0:
        return {"explained_energy": 0.0, "residual_norm": float(np.linalg.norm(v))}
    projection = (float(np.dot(v, b)) / b_norm_sq) * b
    residual = v - projection
    v_norm_sq = float(np.dot(v, v))
    explained = float(np.dot(projection, projection)) / max(v_norm_sq, 1e-12)
    return {"explained_energy": explained, "residual_norm": float(np.linalg.norm(residual))}


@dataclass
class _WeightBundle:
    tokenizer: Any
    model: Any
    device: str
    model_ref: str
    tokenizer_ref: str


def run_hologram_report(
    provider: Any,
    suite: dict[str, Any],
    *,
    model: str,
    target_queries: Sequence[str],
    control_queries: Sequence[str],
    module_names: Sequence[str] | None = None,
    tokenizer_ref: str | Path | None = None,
    model_ref: str | Path | None = None,
    vector_kind: str = "input",
    rollout_steps: int = 6,
    repeats: int = 1,
    add_generation_prompt: bool = True,
    add_special_tokens: bool = False,
    rollout_topk: int = 8,
    dtype: str | None = None,
    device: str | None = None,
    low_cpu_mem_usage: bool = True,
) -> dict[str, Any]:
    triage = run_triangulation_report(
        provider,
        suite,
        model=model,
        tokenizer_ref=tokenizer_ref,
        model_ref=model_ref,
        module_names=list(module_names or []),
        rollout_steps=rollout_steps,
        repeats=repeats,
        add_generation_prompt=add_generation_prompt,
        add_special_tokens=add_special_tokens,
        rollout_topk=rollout_topk,
    )
    base_case = normalize_case(suite["base_case"], default_id="control")
    suite_cases = [base_case, *[normalize_case(case, default_id="variant-{}".format(index)) for index, case in enumerate(suite.get("variants", []))]]
    case_lookup = {case["custom_id"]: case for case in suite_cases}
    resolved_modules = list(module_names or triage.get("activation_modules") or [])
    if not resolved_modules:
        raise ValueError("hologram report requires explicit module_names or triangulation modules")
    bundle = _resolve_weight_bundle(
        provider,
        tokenizer_ref=tokenizer_ref,
        model_ref=model_ref,
        dtype=dtype,
        device=device,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    direction = _build_query_direction(
        bundle,
        target_queries=list(target_queries),
        control_queries=list(control_queries),
        vector_kind=vector_kind,
    )
    activation_maps = _collect_activation_maps(
        provider,
        triage["case_rows"],
        case_lookup=case_lookup,
        module_names=resolved_modules,
        model=model,
    )
    module_rows = _rank_module_holograms(
        bundle.model,
        triage["case_rows"],
        activation_maps,
        module_names=resolved_modules,
        direction=direction["direction_vector"],
        activation_ranking=triage.get("activation_module_ranking", []),
    )
    summary_rows = [
        {
            "custom_id": row["custom_id"],
            "prompt_text": row["prompt_text"],
            "output_label": row["output_mode"]["label"],
            "aggregate_trigger_score": float(row["aggregate_trigger_score"]),
            "peak_adoption_margin": float(row["rollout_summary"]["peak_adoption_margin"]),
            "peak_denial_margin": float(row["rollout_summary"]["peak_denial_margin"]),
        }
        for row in triage["case_rows"]
    ]
    return {
        "mode": "hologram",
        "provider": describe_provider(provider),
        "model": model,
        "vector_kind": vector_kind,
        "target_queries": list(target_queries),
        "control_queries": list(control_queries),
        "direction": {
            "target_count": int(direction["target_count"]),
            "control_count": int(direction["control_count"]),
            "dimension": int(direction["dimension"]),
            "norm": float(direction["norm"]),
            "target_mean_norm": float(direction["target_mean_norm"]),
            "control_mean_norm": float(direction["control_mean_norm"]),
        },
        "case_rows": summary_rows,
        "module_holograms": module_rows,
        "triangulation": {
            "activation_modules": triage.get("activation_modules", []),
            "activation_module_ranking": triage.get("activation_module_ranking", []),
        },
    }


def _resolve_weight_bundle(
    provider: Any,
    *,
    tokenizer_ref: str | Path | None,
    model_ref: str | Path | None,
    dtype: str | None,
    device: str | None,
    low_cpu_mem_usage: bool,
) -> _WeightBundle:
    if hasattr(provider, "_tokenizer") and hasattr(provider, "_ensure_model"):
        model_obj = provider._ensure_model()
        return _WeightBundle(
            tokenizer=provider._tokenizer,
            model=model_obj,
            device=str(getattr(provider, "_device", device or "cpu")),
            model_ref=str(getattr(provider, "_model_ref", model_ref or "")),
            tokenizer_ref=str(getattr(provider, "_tokenizer_ref", tokenizer_ref or "")),
        )
    if tokenizer_ref is None or model_ref is None:
        raise ValueError("hologram report needs tokenizer_ref and model_ref when provider does not expose local weights")
    torch_mod = _import_torch()
    transformers = _import_transformers()
    tokenizer = load_tokenizer(tokenizer_ref, trust_remote_code=True, use_fast=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    resolved_dtype = _resolve_torch_dtype(torch_mod, dtype)
    if resolved_dtype is not None:
        load_kwargs["dtype"] = resolved_dtype
    if low_cpu_mem_usage:
        load_kwargs["low_cpu_mem_usage"] = True
    resolved_device = _resolve_device(torch_mod, device)
    model_obj = transformers.AutoModelForCausalLM.from_pretrained(str(model_ref), **load_kwargs)
    if hasattr(model_obj, "to"):
        model_obj = model_obj.to(resolved_device)
    if hasattr(model_obj, "eval"):
        model_obj.eval()
    return _WeightBundle(
        tokenizer=tokenizer,
        model=model_obj,
        device=str(resolved_device),
        model_ref=str(model_ref),
        tokenizer_ref=str(tokenizer_ref),
    )


def _build_query_direction(
    bundle: _WeightBundle,
    *,
    target_queries: list[str],
    control_queries: list[str],
    vector_kind: str,
) -> dict[str, Any]:
    vector_kind = str(vector_kind).lower()
    if vector_kind not in {"input", "output"}:
        raise ValueError("vector_kind must be 'input' or 'output'")
    if not target_queries or not control_queries:
        raise ValueError("hologram report requires at least one target and one control query")
    target_vectors = [_mean_query_vector(bundle, query, vector_kind=vector_kind) for query in target_queries]
    control_vectors = [_mean_query_vector(bundle, query, vector_kind=vector_kind) for query in control_queries]
    target_mean = np.mean(np.stack(target_vectors, axis=0), axis=0)
    control_mean = np.mean(np.stack(control_vectors, axis=0), axis=0)
    direction = np.asarray(target_mean - control_mean, dtype=np.float64)
    return {
        "direction_vector": direction,
        "target_count": len(target_vectors),
        "control_count": len(control_vectors),
        "dimension": int(direction.size),
        "norm": float(np.linalg.norm(direction)),
        "target_mean_norm": float(np.linalg.norm(target_mean)),
        "control_mean_norm": float(np.linalg.norm(control_mean)),
    }


def _mean_query_vector(bundle: _WeightBundle, query: str, *, vector_kind: str) -> np.ndarray:
    token_ids = bundle.tokenizer(str(query), add_special_tokens=False).get("input_ids", [])
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_ids = [int(token_id) for token_id in token_ids]
    if not token_ids:
        raise ValueError("Query {!r} tokenized to no ids".format(query))
    vectors = [_lookup_embedding_row(bundle.model, token_id, kind=vector_kind) for token_id in token_ids]
    present = [vector for vector in vectors if vector is not None]
    if not present:
        raise ValueError("No {} embedding rows available for {!r}".format(vector_kind, query))
    return np.mean(np.stack(present, axis=0), axis=0)


def _lookup_embedding_row(model: Any, token_id: int, *, kind: str) -> np.ndarray | None:
    getter = model.get_input_embeddings if kind == "input" else model.get_output_embeddings
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


def _collect_activation_maps(
    provider: Any,
    case_rows: Sequence[dict[str, Any]],
    *,
    case_lookup: dict[str, dict[str, Any]],
    module_names: Sequence[str],
    model: str,
) -> dict[str, dict[str, np.ndarray]]:
    requests = []
    for row in case_rows:
        base_case = case_lookup.get(row["custom_id"])
        if base_case is None:
            base_case = normalize_case(
                {
                    "custom_id": row["custom_id"],
                    "messages": [{"role": "user", "content": row["prompt_text"]}],
                    "module_names": list(module_names),
                    "metadata": {},
                },
                default_id=row["custom_id"],
            )
        request = normalize_case(base_case, default_id=row["custom_id"])
        request["module_names"] = list(module_names)
        requests.append(request)
    raw_rows = provider.activations(requests, model=model)
    out: dict[str, dict[str, np.ndarray]] = {}
    for request, raw in zip(requests, raw_rows):
        activation_map = {
            str(name): np.asarray(value, dtype=np.float64)
            for name, value in dict(raw.get("activations", {})).items()
            if name in module_names
        }
        out[request["custom_id"]] = activation_map
    return out


def _rank_module_holograms(
    model: Any,
    case_rows: Sequence[dict[str, Any]],
    activation_maps: dict[str, dict[str, np.ndarray]],
    *,
    module_names: Sequence[str],
    direction: np.ndarray,
    activation_ranking: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    module_lookup = dict(model.named_modules()) if hasattr(model, "named_modules") else {}
    control_id = next((row["custom_id"] for row in case_rows if row.get("is_control")), case_rows[-1]["custom_id"])
    control_map = activation_maps.get(control_id, {})
    activation_scores = {
        str(row["module_name"]): float(row.get("score", 0.0))
        for row in list(activation_ranking or [])
    }
    max_activation_score = max([0.0, *activation_scores.values()])
    rows = []
    for module_name in module_names:
        module = module_lookup.get(module_name)
        weight = _extract_linear_weight(module)
        case_scores = []
        adoption_alignment = []
        denial_alignment = []
        adoption_trigger_scores = []
        alignment_values = []
        trigger_values = []
        for row in case_rows:
            case_id = row["custom_id"]
            activation = activation_maps.get(case_id, {}).get(module_name)
            control = control_map.get(module_name)
            if activation is None or control is None:
                continue
            delta = np.asarray(activation - control, dtype=np.float64)
            output_label = row["output_label"] if "output_label" in row else row["output_mode"]["label"]
            entry = {
                "custom_id": case_id,
                "output_label": output_label,
                "aggregate_trigger_score": float(row.get("aggregate_trigger_score", 0.0)),
                "peak_adoption_margin": float(row.get("peak_adoption_margin", row.get("rollout_summary", {}).get("peak_adoption_margin", 0.0))),
                "delta_norm": float(np.linalg.norm(delta)),
            }
            if delta.size == direction.size:
                entry["direct_cosine"] = _cosine_alignment(delta, direction)
            if weight is not None and weight.shape[1] == direction.size and weight.shape[0] == delta.size:
                projected = np.asarray(weight @ direction, dtype=np.float64)
                entry["projected_cosine"] = _cosine_alignment(delta, projected)
                mismatch = _residual_mismatch(delta, projected)
                entry["projected_explained_energy"] = float(mismatch["explained_energy"])
                entry["projected_norm"] = float(np.linalg.norm(projected))
            if "projected_cosine" in entry:
                alignment = float(entry["projected_cosine"])
                alignment_values.append(alignment)
                trigger_values.append(entry["peak_adoption_margin"])
                if entry["output_label"] == "adoption":
                    adoption_alignment.append(alignment)
                    adoption_trigger_scores.append(entry["peak_adoption_margin"])
                elif entry["output_label"] == "denial":
                    denial_alignment.append(alignment)
            case_scores.append(entry)
        if not case_scores:
            continue
        correlation = _pearson(alignment_values, trigger_values)
        adoption_mean = float(np.mean(adoption_alignment)) if adoption_alignment else 0.0
        denial_mean = float(np.mean(denial_alignment)) if denial_alignment else 0.0
        triage_score = float(activation_scores.get(module_name, 0.0))
        normalized_triage = 0.0 if max_activation_score <= 0.0 else float(triage_score / max_activation_score)
        exploit_score = float(max(adoption_mean - denial_mean, 0.0) + max(correlation, 0.0) + 0.25 * normalized_triage)
        study_score = float(abs(adoption_mean - denial_mean) + abs(correlation) + 0.25 * normalized_triage)
        rows.append(
            {
                "module_name": module_name,
                "weight_shape": None if weight is None else list(weight.shape),
                "triangulation_activation_score": triage_score,
                "triangulation_activation_score_normalized": normalized_triage,
                "adoption_projected_cosine_mean": adoption_mean,
                "denial_projected_cosine_mean": denial_mean,
                "adoption_minus_denial": float(adoption_mean - denial_mean),
                "projected_alignment_trigger_corr": correlation,
                "case_scores": case_scores,
                "exploit_score": exploit_score,
                "study_score": study_score,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["study_score"]),
            float(row["exploit_score"]),
            float(row["triangulation_activation_score_normalized"]),
        ),
        reverse=True,
    )
    for row in rows:
        row["case_scores"].sort(key=lambda item: float(item.get("peak_adoption_margin", 0.0)), reverse=True)
    return rows


def _extract_linear_weight(module: Any) -> np.ndarray | None:
    if module is None:
        return None
    weight = getattr(module, "weight", None)
    if weight is None:
        return None
    array = _tensor_to_numpy(weight)
    if array.ndim != 2:
        return None
    return np.asarray(array, dtype=np.float64)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "to"):
        try:
            value = value.to("cpu")
        except Exception:
            pass
    if hasattr(value, "float"):
        try:
            value = value.float()
        except Exception:
            pass
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _pearson(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if len(lhs) < 2 or len(rhs) < 2 or len(lhs) != len(rhs):
        return 0.0
    left = np.asarray(lhs, dtype=np.float64)
    right = np.asarray(rhs, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)
