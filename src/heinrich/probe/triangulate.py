"""Triangulate -- multi-case identity signal analysis with chat, rollout, and activation."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .activation import rank_modules_by_separability
from .meta import DEFAULT_META_PROBE_FAMILIES
from .regimes import summarize_text_regimes
from .rubric import scan_rubric_text
from .sampling import summarize_sampling_null
from .token_tools import compare_case_tokenization, render_case_prompt, tokenize_case, _safe_decode_piece
from .trigger_core import _load_json_path_or_literal, describe_provider, normalize_case


DEFAULT_TRIAGE_CANDIDATES = (" Claude", " not", " Qwen", " Assistant")


def _score_hijack_text(text: str, *, prompt_text: str | None = None) -> dict[str, Any]:
    """Thin wrapper delegating to trigger_core private helper."""
    from .trigger_core import _score_hijack_text as _tc_hijack
    return _tc_hijack(text, prompt_text=prompt_text)


def load_case_suite(path_or_raw: str | Path | dict[str, Any]) -> dict[str, Any]:
    raw = _load_json_path_or_literal(path_or_raw)
    if not isinstance(raw, dict):
        raise ValueError("Suite must decode to a JSON object")
    base_raw = raw.get("base_case") or raw.get("control_case")
    if base_raw is None:
        raise ValueError("Suite must define base_case or control_case")
    base_case = normalize_case(base_raw, default_id="control")
    variants_raw = raw.get("variants")
    if variants_raw is None:
        variants_raw = raw.get("cases", [])
    if not isinstance(variants_raw, list):
        raise ValueError("Suite variants/cases must be a list")
    variants: list[dict[str, Any]] = []
    for index, row in enumerate(variants_raw):
        if isinstance(row, dict) and "case" in row:
            meta = {key: value for key, value in row.items() if key != "case"}
            case = normalize_case(row["case"], default_id="variant-{}".format(index))
            case["metadata"] = {**meta, **case.get("metadata", {})}
        else:
            case = normalize_case(row, default_id="variant-{}".format(index))
        variants.append(case)
    return {
        "base_case": base_case,
        "variants": variants,
        "variant_count": len(variants),
    }


def run_triangulation_report(
    provider: Any,
    suite: dict[str, Any],
    *,
    model: str,
    tokenizer_ref: str | Path | None = None,
    model_ref: str | Path | None = None,
    module_names: list[str] | tuple[str, ...] | None = None,
    candidate_strings: list[str] | tuple[str, ...] | None = None,
    rollout_steps: int = 6,
    repeats: int = 1,
    add_generation_prompt: bool = True,
    add_special_tokens: bool = False,
    rollout_topk: int = 8,
) -> dict[str, Any]:
    base_case = normalize_case(suite["base_case"], default_id="control")
    variants = [normalize_case(case, default_id="variant-{}".format(index)) for index, case in enumerate(suite.get("variants", []))]
    cases = [base_case, *variants]
    tokenizer_obj = getattr(provider, "_tokenizer", None)
    if tokenizer_obj is None and tokenizer_ref is None:
        raise ValueError("triangulate requires a provider tokenizer or explicit --tokenizer")
    effective_candidates = tuple(candidate_strings or DEFAULT_TRIAGE_CANDIDATES)

    chat = _collect_chat_report(provider, cases, model=model, repeats=repeats)
    token_rows = {
        case["custom_id"]: tokenize_case(
            case,
            tokenizer=tokenizer_obj,
            tokenizer_ref=tokenizer_ref,
            add_generation_prompt=add_generation_prompt,
            add_special_tokens=add_special_tokens,
        )
        for case in cases
    }
    token_diffs = {
        case["custom_id"]: compare_case_tokenization(
            base_case,
            case,
            tokenizer=tokenizer_obj,
            tokenizer_ref=tokenizer_ref,
            add_generation_prompt=add_generation_prompt,
            add_special_tokens=add_special_tokens,
        )
        for case in cases
        if case["custom_id"] != base_case["custom_id"]
    }
    rollout = _collect_rollout_report(
        provider,
        cases,
        tokenizer_ref=tokenizer_ref,
        model_ref=model_ref,
        candidate_strings=effective_candidates,
        steps=rollout_steps,
        topk=rollout_topk,
        add_generation_prompt=add_generation_prompt,
        add_special_tokens=add_special_tokens,
    )
    activation = _collect_activation_report(
        provider,
        cases,
        model=model,
        module_names=list(module_names or []),
    )

    control_id = base_case["custom_id"]
    case_rows = []
    for case in cases:
        case_id = case["custom_id"]
        chat_row = chat["by_case"].get(case_id, {})
        rollout_row = rollout.get(case_id, {})
        activation_row = activation["by_case"].get(case_id, {})
        output_mode = _classify_identity_output(chat_row.get("text", ""))
        rollout_summary = _summarize_rollout_case(rollout_row.get("rows", []))
        token_probe = token_rows[case_id]
        token_boundary = _token_boundary_summary(token_probe, effective_candidates, tokenizer=tokenizer_obj)
        aggregate_trigger_score = (
            max(0.0, float(rollout_summary["peak_adoption_margin"]))
            + (1.0 if output_mode["label"] == "adoption" else 0.0)
            - (0.5 if output_mode["label"] == "denial" else 0.0)
            + (0.1 if token_boundary["has_space_identity_token"] else 0.0)
            + 0.05 * math.log1p(float(activation_row.get("total_l2", 0.0)))
        )
        case_rows.append(
            {
                "custom_id": case_id,
                "prompt_text": case["messages"][-1]["content"],
                "is_control": case_id == control_id,
                "chat": chat_row,
                "output_mode": output_mode,
                "token_probe": {
                    "token_count": token_probe["token_count"],
                    "token_pieces": token_probe["token_pieces"],
                    "used_chat_template": token_probe["used_chat_template"],
                },
                "token_boundary": token_boundary,
                "control_token_diff": None if case_id == control_id else token_diffs[case_id],
                "rollout": rollout_row,
                "rollout_summary": rollout_summary,
                "activation_delta": activation_row,
                "aggregate_trigger_score": float(aggregate_trigger_score),
            }
        )
    case_rows.sort(
        key=lambda row: (
            bool(not row["is_control"]),
            float(row["aggregate_trigger_score"]),
            float(row["rollout_summary"]["peak_adoption_margin"]),
        ),
        reverse=True,
    )

    positive_examples = [
        activation["raw_maps"][row["custom_id"]]
        for row in case_rows
        if row["output_mode"]["label"] == "adoption" and row["custom_id"] in activation["raw_maps"]
    ]
    negative_examples = [
        activation["raw_maps"][row["custom_id"]]
        for row in case_rows
        if row["output_mode"]["label"] != "adoption" and row["custom_id"] in activation["raw_maps"]
    ]
    module_ranking = []
    if positive_examples and negative_examples and activation["module_names"]:
        labels = [1] * len(positive_examples) + [0] * len(negative_examples)
        module_ranking = [
            {
                "module_name": str(row["module_name"]),
                "score": float(row["score"]),
                "signal_norm": float(row["signal_norm"]),
                "pooled_std": float(row["pooled_std"]),
            }
            for row in rank_modules_by_separability(
                [*positive_examples, *negative_examples],
                labels,
                module_names=activation["module_names"],
            )
        ]

    return {
        "mode": "triangulate",
        "provider": describe_provider(provider),
        "model": model,
        "control_case": base_case,
        "candidate_strings": list(effective_candidates),
        "rollout_steps": int(rollout_steps),
        "repeats": int(repeats),
        "activation_modules": activation["module_names"],
        "case_count": len(case_rows),
        "case_rows": case_rows,
        "activation_module_ranking": module_ranking,
    }


def _collect_chat_report(provider: Any, cases: list[dict[str, Any]], *, model: str, repeats: int) -> dict[str, Any]:
    requests = []
    owners = []
    for case in cases:
        for repeat_index in range(max(int(repeats), 1)):
            request = normalize_case(case, default_id=case["custom_id"])
            request["custom_id"] = "{}::rep{}".format(case["custom_id"], repeat_index)
            requests.append(request)
            owners.append(case["custom_id"])
    raw_rows = provider.chat_completions(requests, model=model)
    grouped: dict[str, list[str]] = {case["custom_id"]: [] for case in cases}
    for owner, row in zip(owners, raw_rows):
        grouped.setdefault(owner, []).append(str(row.get("text", "")))
    by_case = {}
    for case in cases:
        texts = grouped.get(case["custom_id"], [])
        primary_text = texts[0] if texts else ""
        regime = summarize_text_regimes(texts) if texts else {
            "sample_count": 0,
            "unique_text_count": 0,
            "entropy_bits": 0.0,
            "dominant_regime_mass": 0.0,
            "exact_consensus": False,
            "dominant_regime_text": "",
            "clusters": [],
        }
        by_case[case["custom_id"]] = {
            "text": primary_text,
            "texts": texts,
            "regimes": regime,
            "sampling_null": summarize_sampling_null(texts) if len(texts) >= 2 else {
                "verdict": "INSUFFICIENT",
                "sample_count": len(texts),
            },
            "hijack": _score_hijack_text(primary_text, prompt_text=case["messages"][-1]["content"]),
            "rubric": scan_rubric_text(primary_text),
        }
    return {"by_case": by_case}


def _collect_rollout_report(
    provider: Any,
    cases: list[dict[str, Any]],
    *,
    tokenizer_ref: str | Path | None,
    model_ref: str | Path | None,
    candidate_strings: tuple,
    steps: int,
    topk: int,
    add_generation_prompt: bool,
    add_special_tokens: bool,
) -> dict[str, Any]:
    if hasattr(provider, "_tokenizer") and hasattr(provider, "_ensure_model") and hasattr(provider, "_device"):
        return {
            case["custom_id"]: _rollout_case_local_provider(
                provider,
                case,
                candidate_strings=candidate_strings,
                steps=steps,
                topk=topk,
                add_generation_prompt=add_generation_prompt,
                add_special_tokens=add_special_tokens,
            )
            for case in cases
        }
    # Remote provider: need nexttoken via local refs
    if tokenizer_ref is None:
        return {}
    from .token_tools import _import_torch, _import_transformers, _resolve_device, _resolve_torch_dtype, load_tokenizer

    torch_mod = _import_torch()
    transformers = _import_transformers()
    tokenizer = load_tokenizer(tokenizer_ref, trust_remote_code=True, use_fast=True)
    resolved_dtype = _resolve_torch_dtype(torch_mod, None)
    resolved_device = _resolve_device(torch_mod, None)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if resolved_dtype is not None:
        load_kwargs["dtype"] = resolved_dtype
    model_obj = transformers.AutoModelForCausalLM.from_pretrained(str(model_ref or tokenizer_ref), **load_kwargs)
    if hasattr(model_obj, "to"):
        model_obj = model_obj.to(resolved_device)
    if hasattr(model_obj, "eval"):
        model_obj.eval()

    class _LocalProviderWrap:
        _tokenizer = tokenizer
        _device = resolved_device
        _torch = torch_mod

        def _ensure_model(self):
            return model_obj

    _prov = _LocalProviderWrap()
    return {
        case["custom_id"]: _rollout_case_local_provider(
            _prov,
            case,
            candidate_strings=candidate_strings,
            steps=steps,
            topk=topk,
            add_generation_prompt=add_generation_prompt,
            add_special_tokens=add_special_tokens,
        )
        for case in cases
    }


def _rollout_case_local_provider(
    provider: Any,
    case: dict[str, Any],
    *,
    candidate_strings: tuple,
    steps: int,
    topk: int,
    add_generation_prompt: bool,
    add_special_tokens: bool,
) -> dict[str, Any]:
    torch_mod = provider._torch
    tokenizer = provider._tokenizer
    model = provider._ensure_model()
    rendered = render_case_prompt(case, tokenizer=tokenizer, add_generation_prompt=add_generation_prompt)
    encoded = tokenizer(rendered["rendered_text"], return_tensors="pt", add_special_tokens=add_special_tokens)
    encoded = {name: value.to(provider._device) for name, value in encoded.items()}
    with torch_mod.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=int(steps),
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    base_length = int(encoded["input_ids"].shape[-1])
    rows = []
    for step_index, score in enumerate(outputs.scores, start=1):
        probs = torch_mod.softmax(score[0].detach().to("cpu").float(), dim=-1)
        top_values, top_indices = torch_mod.topk(probs, k=min(int(topk), int(probs.shape[-1])))
        candidates = []
        for query in candidate_strings:
            token_ids = tokenizer(str(query), add_special_tokens=False).get("input_ids", [])
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            token_ids = [int(token_id) for token_id in token_ids]
            exact_single = len(token_ids) == 1
            probability = None
            if exact_single:
                probability = float(probs[token_ids[0]].item())
            candidates.append(
                {
                    "query": str(query),
                    "token_ids": token_ids,
                    "token_pieces": [_safe_decode_piece(tokenizer, token_id) for token_id in token_ids],
                    "exact_single_token": exact_single,
                    "probability": probability,
                }
            )
        rows.append(
            {
                "step": int(step_index),
                "generated_piece": _safe_decode_piece(tokenizer, int(outputs.sequences[0][base_length + step_index - 1].item())),
                "top_tokens": [
                    {
                        "token_id": int(token_id),
                        "piece": _safe_decode_piece(tokenizer, int(token_id)),
                        "probability": float(prob),
                    }
                    for prob, token_id in zip(top_values.tolist(), top_indices.tolist())
                ],
                "candidates": candidates,
            }
        )
    return {
        "mode": "nexttoken",
        "method_used": "generate",
        "rows": rows,
        "rendered_text": rendered["rendered_text"],
        "used_chat_template": rendered["used_chat_template"],
        "steps": int(steps),
    }


def _collect_activation_report(
    provider: Any,
    cases: list[dict[str, Any]],
    *,
    model: str,
    module_names: list[str],
) -> dict[str, Any]:
    if not module_names:
        return {"module_names": [], "by_case": {}, "raw_maps": {}}
    requests = []
    for case in cases:
        request = normalize_case(case, default_id=case["custom_id"])
        request["module_names"] = list(module_names)
        requests.append(request)
    raw_rows = provider.activations(requests, model=model)
    raw_maps = {
        case["custom_id"]: dict(raw_rows[index].get("activations", {}))
        for index, case in enumerate(cases)
    }
    control = raw_maps.get(cases[0]["custom_id"], {})
    by_case = {}
    for case in cases:
        activation_map = raw_maps.get(case["custom_id"], {})
        per_module = []
        total = 0.0
        for name in module_names:
            control_vec = np.asarray(control.get(name, []), dtype=np.float64)
            case_vec = np.asarray(activation_map.get(name, []), dtype=np.float64)
            if control_vec.size == 0 or case_vec.size == 0:
                continue
            diff = case_vec - control_vec
            l2 = float(np.linalg.norm(diff))
            total += l2
            per_module.append({"module_name": name, "l2": l2, "max_abs": float(np.max(np.abs(diff)))})
        per_module.sort(key=lambda row: row["l2"], reverse=True)
        by_case[case["custom_id"]] = {
            "total_l2": float(total),
            "per_module": per_module,
        }
    return {"module_names": module_names, "by_case": by_case, "raw_maps": raw_maps}


def _summarize_rollout_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    peak_adoption_margin = -1.0
    peak_denial_margin = -1.0
    adoption_step = None
    denial_step = None
    for row in rows:
        probs = {candidate["query"]: candidate.get("probability") for candidate in row.get("candidates", [])}
        p_claude = float(probs.get(" Claude") or 0.0)
        p_not = float(probs.get(" not") or 0.0)
        p_q = float(probs.get(" Qwen") or 0.0)
        p_assistant = float(probs.get(" Assistant") or 0.0)
        adoption_margin = p_claude - max(p_not, p_q, p_assistant)
        denial_margin = p_not - max(p_claude, p_q, p_assistant)
        if adoption_margin > peak_adoption_margin:
            peak_adoption_margin = float(adoption_margin)
            adoption_step = {
                "step": int(row["step"]),
                "generated_piece": row.get("generated_piece"),
                "p_claude": p_claude,
                "p_not": p_not,
                "p_qwen": p_q,
                "p_assistant": p_assistant,
            }
        if denial_margin > peak_denial_margin:
            peak_denial_margin = float(denial_margin)
            denial_step = {
                "step": int(row["step"]),
                "generated_piece": row.get("generated_piece"),
                "p_claude": p_claude,
                "p_not": p_not,
                "p_qwen": p_q,
                "p_assistant": p_assistant,
            }
    return {
        "peak_adoption_margin": float(peak_adoption_margin),
        "peak_denial_margin": float(peak_denial_margin),
        "adoption_step": adoption_step,
        "denial_step": denial_step,
    }


def _token_boundary_summary(token_probe: dict[str, Any], candidate_strings: tuple, *, tokenizer: Any | None) -> dict[str, Any]:
    token_pieces = [str(piece) for piece in token_probe.get("token_pieces", [])]
    queries = {query: _candidate_token_shape(query, tokenizer=tokenizer) for query in candidate_strings}
    return {
        "has_space_identity_token": bool("\u0120Claude" in token_pieces),
        "has_split_identity_token": bool("Cla" in token_pieces and "ude" in token_pieces),
        "candidate_shapes": queries,
    }


def _candidate_token_shape(query: str, *, tokenizer: Any | None) -> dict[str, Any]:
    if tokenizer is None:
        return {"query": query}
    token_ids = tokenizer(str(query), add_special_tokens=False).get("input_ids", [])
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_ids = [int(token_id) for token_id in token_ids]
    return {
        "query": query,
        "token_ids": token_ids,
        "token_pieces": [_safe_decode_piece(tokenizer, token_id) for token_id in token_ids],
        "exact_single_token": len(token_ids) == 1,
    }


def _classify_identity_output(text: str) -> dict[str, Any]:
    lowered = str(text).lower()
    if "not claude" in lowered:
        return {"label": "denial", "evidence": "contains 'not Claude'"}
    if "i'm claude" in lowered or "i am claude" in lowered:
        return {"label": "adoption", "evidence": "contains 'I'm Claude' or 'I am Claude'"}
    if "qwen" in lowered:
        return {"label": "qwen", "evidence": "contains 'Qwen'"}
    if "anthropic" in lowered:
        return {"label": "adoption_like", "evidence": "contains 'Anthropic'"}
    return {"label": "other", "evidence": ""}
