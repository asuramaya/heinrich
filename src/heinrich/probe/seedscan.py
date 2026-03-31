"""Seed scanning -- scan vocabulary for single-token identity seed candidates."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .token_tools import (
    _extract_input_ids,
    _import_torch,
    _import_transformers,
    _resolve_device,
    _resolve_torch_dtype,
    _safe_decode_piece,
    load_tokenizer,
    render_case_prompt,
)


DEFAULT_SEED_TEMPLATES = (
    {"name": "hello", "user_template": "Hello {seed}", "assistant_prefix": "Hello! I'm"},
    {"name": "hi", "user_template": "Hi {seed}", "assistant_prefix": "Hello! I'm"},
    {"name": "you_are", "user_template": "You are {seed}", "assistant_prefix": "I am"},
)
DEFAULT_NEGATIVE_QUERIES = (" not", " Assistant")
DEFAULT_PATTERN = r"^[A-Za-z][A-Za-z'\-]{2,15}$"
DEFAULT_GENERIC_IDENTITY_PROMPT = "Who are you?"
DEFAULT_GENERIC_IDENTITY_PREFIX = "I am"


@dataclass
class _SeedBundle:
    tokenizer: Any
    model: Any
    torch: Any
    device: str
    tokenizer_ref: str
    model_ref: str


def scan_seed_candidates(
    *,
    tokenizer_ref: str | Path,
    model_ref: str | Path,
    templates: Sequence[dict[str, str]] | None = None,
    negative_queries: Sequence[str] | None = None,
    pattern: str = DEFAULT_PATTERN,
    min_surface_len: int = 3,
    max_surface_len: int = 15,
    require_standalone: bool = True,
    limit: int | None = None,
    batch_size: int = 16,
    verify_topk: int = 8,
    trust_remote_code: bool = True,
    use_fast: bool = True,
    dtype: str | None = None,
    device: str | None = None,
    low_cpu_mem_usage: bool = True,
) -> dict[str, Any]:
    bundle = _load_seed_bundle(
        tokenizer_ref=tokenizer_ref,
        model_ref=model_ref,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
        dtype=dtype,
        device=device,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    resolved_templates = [dict(row) for row in list(templates or DEFAULT_SEED_TEMPLATES)]
    resolved_negatives = tuple(str(row) for row in list(negative_queries or DEFAULT_NEGATIVE_QUERIES))
    candidates = _enumerate_vocab_candidates(
        bundle,
        pattern=pattern,
        min_surface_len=min_surface_len,
        max_surface_len=max_surface_len,
        require_standalone=require_standalone,
        limit=None,
    )
    generic_identity_prior = _score_generic_identity_prior(
        bundle,
        candidates,
        prompt_text=DEFAULT_GENERIC_IDENTITY_PROMPT,
        assistant_prefix=DEFAULT_GENERIC_IDENTITY_PREFIX,
    )
    for row in candidates:
        row["generic_identity_probability"] = float(generic_identity_prior.get(int(row["token_id"]), 0.0))
    candidates.sort(
        key=lambda row: (
            float(row.get("generic_identity_probability", 0.0)),
            *_candidate_static_rank(row),
        ),
        reverse=True,
    )
    for index, row in enumerate(candidates):
        row["generic_identity_rank"] = int(index)
    if limit is not None:
        candidates = candidates[: max(int(limit), 0)]
    negative_ids = _resolve_query_token_ids(bundle.tokenizer, resolved_negatives)
    scored_rows = _score_seed_candidates(
        bundle,
        candidates,
        templates=resolved_templates,
        negative_ids=negative_ids,
        batch_size=batch_size,
    )
    scored_rows.sort(
        key=lambda row: (
            float(row["aggregate_score"]),
            float(row["mean_adoption_margin"]),
            float(row["positive_template_fraction"]),
            bool(row["standalone_roundtrip"] and not row["plain_roundtrip"]),
        ),
        reverse=True,
    )
    verification = _verify_top_candidates(
        bundle,
        scored_rows[: max(int(verify_topk), 0)],
        templates=resolved_templates,
    )
    top_cases = _top_candidate_cases(scored_rows[: max(int(verify_topk), 0)], templates=resolved_templates)
    return {
        "mode": "seedscan",
        "tokenizer_name": getattr(bundle.tokenizer, "name_or_path", str(tokenizer_ref)),
        "tokenizer_ref": str(tokenizer_ref),
        "model_ref": str(model_ref),
        "device": str(bundle.device),
        "candidate_count": len(scored_rows),
        "templates": resolved_templates,
        "negative_queries": list(resolved_negatives),
        "generic_identity_prompt": DEFAULT_GENERIC_IDENTITY_PROMPT,
        "generic_identity_prefix": DEFAULT_GENERIC_IDENTITY_PREFIX,
        "pattern": pattern,
        "require_standalone": bool(require_standalone),
        "rows": scored_rows,
        "verification": verification,
        "top_cases": top_cases,
    }


def _load_seed_bundle(
    *,
    tokenizer_ref: str | Path,
    model_ref: str | Path,
    trust_remote_code: bool,
    use_fast: bool,
    dtype: str | None,
    device: str | None,
    low_cpu_mem_usage: bool,
) -> _SeedBundle:
    torch_mod = _import_torch()
    transformers = _import_transformers()
    tokenizer = load_tokenizer(tokenizer_ref, trust_remote_code=trust_remote_code, use_fast=use_fast)
    resolved_dtype = _resolve_torch_dtype(torch_mod, dtype)
    resolved_device = _resolve_device(torch_mod, device)
    load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if resolved_dtype is not None:
        load_kwargs["dtype"] = resolved_dtype
    if low_cpu_mem_usage:
        load_kwargs["low_cpu_mem_usage"] = True
    model = transformers.AutoModelForCausalLM.from_pretrained(str(model_ref), **load_kwargs)
    if hasattr(model, "to"):
        model = model.to(resolved_device)
    if hasattr(model, "eval"):
        model.eval()
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return _SeedBundle(
        tokenizer=tokenizer,
        model=model,
        torch=torch_mod,
        device=str(resolved_device),
        tokenizer_ref=str(tokenizer_ref),
        model_ref=str(model_ref),
    )


def _enumerate_vocab_candidates(
    bundle: _SeedBundle,
    *,
    pattern: str,
    min_surface_len: int,
    max_surface_len: int,
    require_standalone: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    tokenizer = bundle.tokenizer
    vocab = dict(tokenizer.get_vocab()) if hasattr(tokenizer, "get_vocab") else {}
    if not vocab:
        raise ValueError("Tokenizer does not expose a vocabulary for seed scanning")
    matcher = re.compile(pattern)
    candidates_by_surface: dict[str, dict[str, Any]] = {}
    for piece, token_id in vocab.items():
        token_id = int(token_id)
        decoded = _decode_single_token(tokenizer, token_id)
        surface = decoded.strip()
        if not surface or len(surface) < int(min_surface_len) or len(surface) > int(max_surface_len):
            continue
        if matcher.fullmatch(surface) is None:
            continue
        standalone_ids = _flatten_ids(tokenizer(" " + surface, add_special_tokens=False).get("input_ids", []))
        plain_ids = _flatten_ids(tokenizer(surface, add_special_tokens=False).get("input_ids", []))
        standalone_roundtrip = standalone_ids == [token_id]
        if require_standalone and not standalone_roundtrip:
            continue
        plain_roundtrip = plain_ids == [token_id]
        row = {
            "token_id": token_id,
            "token_piece": _safe_decode_piece(tokenizer, token_id),
            "surface": surface,
            "decoded": decoded,
            "standalone_roundtrip": bool(standalone_roundtrip),
            "plain_roundtrip": bool(plain_roundtrip),
            "standalone_ids": standalone_ids,
            "plain_ids": plain_ids,
            "has_space_prefix": bool(_safe_decode_piece(tokenizer, token_id).startswith(("\u0120", "\u2581", " "))),
        }
        previous = candidates_by_surface.get(surface)
        if previous is None or _candidate_static_rank(row) > _candidate_static_rank(previous):
            candidates_by_surface[surface] = row
    rows = list(candidates_by_surface.values())
    rows.sort(key=_candidate_static_rank, reverse=True)
    if limit is not None:
        rows = rows[: max(int(limit), 0)]
    return rows


def _candidate_static_rank(row: dict[str, Any]) -> tuple:
    return (
        1.0 if row["standalone_roundtrip"] else 0.0,
        1.0 if row["has_space_prefix"] else 0.0,
        1.0 if not row["plain_roundtrip"] else 0.0,
        float(len(str(row["surface"]))),
        float(row["token_id"]),
    )


def _resolve_query_token_ids(tokenizer: Any, queries: Sequence[str]) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for query in queries:
        token_ids = _flatten_ids(tokenizer(str(query), add_special_tokens=False).get("input_ids", []))
        out[str(query)] = int(token_ids[0]) if len(token_ids) == 1 else None
    return out


def _score_generic_identity_prior(
    bundle: _SeedBundle,
    candidates: Sequence[dict[str, Any]],
    *,
    prompt_text: str,
    assistant_prefix: str,
) -> dict[int, float]:
    report = _probe_candidate_distribution(
        bundle,
        candidates,
        prompt_text=prompt_text,
        assistant_prefix=assistant_prefix,
    )
    return {int(row["token_id"]): float(row["probability"]) for row in report["candidate_rows"]}


def _probe_candidate_distribution(
    bundle: _SeedBundle,
    candidates: Sequence[dict[str, Any]],
    *,
    prompt_text: str,
    assistant_prefix: str,
    topk: int = 12,
) -> dict[str, Any]:
    case = {
        "custom_id": "generic-identity",
        "messages": [{"role": "user", "content": str(prompt_text)}],
        "module_names": [],
        "metadata": {},
    }
    rendered = render_case_prompt(case, tokenizer=bundle.tokenizer, add_generation_prompt=True)
    encoded = bundle.tokenizer(rendered["rendered_text"] + str(assistant_prefix), return_tensors="pt", add_special_tokens=False)
    encoded = {name: value.to(bundle.device) for name, value in encoded.items()}
    with bundle.torch.no_grad():
        outputs = bundle.model(**encoded)
    logits = _extract_logits(outputs)
    if logits is None:
        raise ValueError("Model forward pass did not expose logits for generic identity prior")
    row_logits = logits[0, -1]
    torch_mod = bundle.torch
    log_norm = torch_mod.logsumexp(row_logits, dim=-1)
    probs = torch_mod.exp(row_logits - log_norm)
    top_values, top_indices = torch_mod.topk(probs, k=min(int(topk), int(probs.shape[-1])))
    candidate_rows = [
        {
            "surface": str(candidate["surface"]),
            "token_id": int(candidate["token_id"]),
            "probability": float(probs[int(candidate["token_id"])].item()),
            "standalone_roundtrip": bool(candidate["standalone_roundtrip"]),
            "plain_roundtrip": bool(candidate["plain_roundtrip"]),
            "generic_identity_rank": int(candidate.get("generic_identity_rank", -1)),
        }
        for candidate in candidates
    ]
    candidate_rows.sort(key=lambda row: float(row["probability"]), reverse=True)
    positive = np.asarray(probs.detach().to("cpu").float().tolist(), dtype=np.float64)
    positive = positive[positive > 0.0]
    entropy_bits = float(-np.sum(positive * np.log2(positive))) if positive.size else 0.0
    return {
        "prompt_text": str(prompt_text),
        "assistant_prefix": str(assistant_prefix),
        "entropy_bits": entropy_bits,
        "top_tokens": [
            {
                "token_id": int(token_id),
                "piece": _safe_decode_piece(bundle.tokenizer, int(token_id)),
                "probability": float(prob),
            }
            for prob, token_id in zip(top_values.tolist(), top_indices.tolist())
        ],
        "candidate_rows": candidate_rows,
    }


def _score_seed_candidates(
    bundle: _SeedBundle,
    candidates: Sequence[dict[str, Any]],
    *,
    templates: Sequence[dict[str, str]],
    negative_ids: dict[str, int | None],
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for template in templates:
        _score_template_pass(
            bundle,
            rows=rows,
            candidates=candidates,
            template=template,
            negative_ids=negative_ids,
            batch_size=batch_size,
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["surface"]), []).append(row)
    model_obj = bundle.model
    final_rows = []
    for candidate in candidates:
        surface = str(candidate["surface"])
        template_rows = grouped.get(surface, [])
        adoption_margins = [float(row["adoption_margin"]) for row in template_rows]
        positive_count = sum(1 for value in adoption_margins if value > 0.0)
        input_norm = _embedding_row_norm(model_obj, int(candidate["token_id"]), kind="input")
        output_norm = _embedding_row_norm(model_obj, int(candidate["token_id"]), kind="output")
        boundary_bonus = 0.25 if candidate["standalone_roundtrip"] and not candidate["plain_roundtrip"] else 0.0
        mean_margin = float(np.mean(adoption_margins)) if adoption_margins else 0.0
        max_margin = float(max(adoption_margins)) if adoption_margins else 0.0
        mean_candidate_prob = float(np.mean([row["candidate_probability"] for row in template_rows])) if template_rows else 0.0
        generic_identity_probability = float(candidate.get("generic_identity_probability", 0.0))
        score = (
            mean_margin
            + 0.35 * max_margin
            + 0.15 * (positive_count / max(len(templates), 1))
            + boundary_bonus
            + 50.0 * generic_identity_probability
        )
        final_rows.append(
            {
                **candidate,
                "template_scores": template_rows,
                "mean_adoption_margin": mean_margin,
                "max_adoption_margin": max_margin,
                "mean_candidate_probability": mean_candidate_prob,
                "positive_template_count": int(positive_count),
                "positive_template_fraction": float(positive_count / max(len(templates), 1)),
                "boundary_bonus": float(boundary_bonus),
                "generic_identity_probability": generic_identity_probability,
                "input_row_norm": input_norm,
                "output_row_norm": output_norm,
                "aggregate_score": float(score),
            }
        )
    return final_rows


def _score_template_pass(
    bundle: _SeedBundle,
    *,
    rows: list[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    template: dict[str, str],
    negative_ids: dict[str, int | None],
    batch_size: int,
) -> None:
    prompts = []
    for candidate in candidates:
        user_text = str(template["user_template"]).format(seed=candidate["surface"])
        case = {
            "custom_id": "{}::{}".format(template["name"], candidate["surface"]),
            "messages": [{"role": "user", "content": user_text}],
            "module_names": [],
            "metadata": {},
        }
        rendered = render_case_prompt(case, tokenizer=bundle.tokenizer, add_generation_prompt=True)
        prompts.append(
            {
                "template_name": str(template["name"]),
                "assistant_prefix": str(template["assistant_prefix"]),
                "candidate": candidate,
                "rendered_text": rendered["rendered_text"] + str(template["assistant_prefix"]),
                "user_text": user_text,
            }
        )
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        metrics = _batch_slot_probabilities(bundle, batch, negative_ids=negative_ids)
        for prompt, metric in zip(batch, metrics):
            rows.append(
                {
                    "surface": str(prompt["candidate"]["surface"]),
                    "token_id": int(prompt["candidate"]["token_id"]),
                    "template_name": str(prompt["template_name"]),
                    "assistant_prefix": str(prompt["assistant_prefix"]),
                    "user_text": str(prompt["user_text"]),
                    "candidate_probability": float(metric["candidate_probability"]),
                    "adoption_margin": float(metric["adoption_margin"]),
                    "negative_probabilities": metric["negative_probabilities"],
                }
            )


def _batch_slot_probabilities(
    bundle: _SeedBundle,
    prompts: Sequence[dict[str, Any]],
    *,
    negative_ids: dict[str, int | None],
) -> list[dict[str, Any]]:
    tokenizer = bundle.tokenizer
    torch_mod = bundle.torch
    encoded = tokenizer(
        [row["rendered_text"] for row in prompts],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    encoded = {name: value.to(bundle.device) for name, value in encoded.items()}
    with torch_mod.no_grad():
        outputs = bundle.model(**encoded)
    logits = _extract_logits(outputs)
    if logits is None:
        raise ValueError("Model forward pass did not expose logits")
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        last_indices = [int(logits.shape[1] - 1)] * int(logits.shape[0])
    else:
        last_indices = [int(row.sum().item()) - 1 for row in attention_mask]
    rows = []
    for batch_index, (prompt, last_index) in enumerate(zip(prompts, last_indices)):
        row_logits = logits[batch_index, last_index]
        log_norm = torch_mod.logsumexp(row_logits, dim=-1)
        candidate_id = int(prompt["candidate"]["token_id"])
        candidate_logit = row_logits[candidate_id]
        candidate_prob = float(torch_mod.exp(candidate_logit - log_norm).item())
        negative_probs = {}
        strongest_negative = 0.0
        for query, token_id in negative_ids.items():
            probability = None
            if token_id is not None:
                probability = float(torch_mod.exp(row_logits[int(token_id)] - log_norm).item())
                strongest_negative = max(strongest_negative, probability)
            negative_probs[str(query)] = probability
        rows.append(
            {
                "candidate_probability": candidate_prob,
                "adoption_margin": float(candidate_prob - strongest_negative),
                "negative_probabilities": negative_probs,
            }
        )
    return rows


def _verify_top_candidates(
    bundle: _SeedBundle,
    rows: Sequence[dict[str, Any]],
    *,
    templates: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    tokenizer = bundle.tokenizer
    torch_mod = bundle.torch
    outputs = []
    for row in rows:
        per_template = []
        for template in templates:
            case = {
                "custom_id": "verify::{}::{}".format(template["name"], row["surface"]),
                "messages": [{"role": "user", "content": str(template["user_template"]).format(seed=row["surface"])}],
                "module_names": [],
                "metadata": {},
            }
            rendered = render_case_prompt(case, tokenizer=tokenizer, add_generation_prompt=True)
            encoded = tokenizer(rendered["rendered_text"], return_tensors="pt", add_special_tokens=False)
            encoded = {name: value.to(bundle.device) for name, value in encoded.items()}
            input_length = int(encoded["input_ids"].shape[-1])
            with torch_mod.no_grad():
                generated = bundle.model.generate(
                    **encoded,
                    max_new_tokens=12,
                    do_sample=False,
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                )
            new_tokens = generated[0][input_length:]
            per_template.append(
                {
                    "template_name": str(template["name"]),
                    "user_text": str(case["messages"][-1]["content"]),
                    "text": str(tokenizer.decode(new_tokens, skip_special_tokens=True)).strip(),
                }
            )
        outputs.append(
            {
                "surface": row["surface"],
                "token_id": int(row["token_id"]),
                "aggregate_score": float(row["aggregate_score"]),
                "responses": per_template,
            }
        )
    return outputs


def _top_candidate_cases(rows: Sequence[dict[str, Any]], *, templates: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        best_template = max(
            row["template_scores"],
            key=lambda item: float(item.get("adoption_margin", float("-inf"))),
            default=None,
        )
        if best_template is None:
            continue
        template_name = str(best_template["template_name"])
        template = next((item for item in templates if str(item["name"]) == template_name), None)
        if template is None:
            continue
        out.append(
            {
                "surface": row["surface"],
                "aggregate_score": float(row["aggregate_score"]),
                "case": {
                    "custom_id": "seed::{}".format(row["surface"]),
                    "messages": [{"role": "user", "content": str(template["user_template"]).format(seed=row["surface"])}],
                    "module_names": [],
                    "metadata": {
                        "seedscan_template": template_name,
                        "seedscan_score": float(row["aggregate_score"]),
                    },
                },
            }
        )
    return out


def _embedding_row_norm(model: Any, token_id: int, *, kind: str) -> float | None:
    row = _lookup_embedding_row(model, token_id, kind=kind)
    if row is None:
        return None
    return float(np.linalg.norm(row))


def _lookup_embedding_row(model: Any, token_id: int, *, kind: str) -> np.ndarray | None:
    if kind == "input":
        getter = getattr(model, "get_input_embeddings", None)
    else:
        getter = getattr(model, "get_output_embeddings", None)
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


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "to"):
        value = value.to("cpu")
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float64)
    if hasattr(value, "tolist"):
        return np.asarray(value.tolist(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _extract_logits(outputs: Any) -> Any | None:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    return None


def _decode_single_token(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        return str(_safe_decode_piece(tokenizer, int(token_id)))


def _flatten_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value.get("input_ids")
    if value is None:
        return []
    return _extract_input_ids({"input_ids": value})
