"""Campaign loop runner — iterate over suite variants scoring each against a base case."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..bundle.priors import load_prior_source
from .trigger_core import (
    _activation_change_score,
    _chat_change_score,
    activation_diff,
    chat_diff,
    cross_model_compare,
    describe_provider,
    normalize_case,
)


def run_case_loop(
    provider: Any,
    suite: dict[str, Any] | str | Path,
    *,
    models: list[str],
    mode: str = "crosssuite",
    repeats: int = 1,
    progress_path: str | Path | None = None,
    include_base: bool = True,
    limit: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run a scoring loop over all suite variants and return a ranked result report.

    Parameters
    ----------
    provider:
        Inference provider (OpenAI-compatible or local).
    suite:
        Path, JSON string, or dict containing ``base_case`` and ``variants``.
    models:
        List of model identifiers to evaluate.
    mode:
        ``"crosssuite"`` — score cross-model divergence for each variant;
        ``"scorecases"`` — score each variant against ``base_case`` per model.
    repeats:
        Number of chat completions to average over.
    progress_path:
        Optional file path to write incremental progress as JSON.
    include_base:
        Whether to include the base case itself as a candidate (crosssuite mode only).
    limit:
        Maximum number of *new* candidates to evaluate (skips already-completed ones).
    resume:
        If True and ``progress_path`` exists, load prior results and skip already-done variants.
    """
    normalized_suite = _normalize_suite(load_prior_source(suite))
    unique_models = list(dict.fromkeys(models))
    if not unique_models:
        raise ValueError("run_case_loop requires at least one model")
    if mode not in {"crosssuite", "scorecases"}:
        raise ValueError("mode must be 'crosssuite' or 'scorecases'")

    state_path = Path(progress_path) if progress_path else None
    existing_rows: dict[str, dict[str, Any]] = {}
    if resume and state_path and state_path.exists():
        existing = replay_case_loop(state_path)
        for row in existing.get("variants", []):
            existing_rows[str(row["variant_id"])] = row

    candidates = _suite_candidates(normalized_suite, include_base=include_base if mode == "crosssuite" else False)
    new_count = 0
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        variant_id = str(candidate["variant_id"])
        if variant_id in existing_rows:
            rows.append(existing_rows[variant_id])
            continue
        if limit is not None and new_count >= int(limit):
            break
        row = _evaluate_candidate(
            provider,
            normalized_suite,
            candidate,
            models=unique_models,
            mode=mode,
            repeats=repeats,
        )
        rows.append(row)
        new_count += 1
        if state_path is not None:
            _write_progress(
                state_path,
                _finalize_state(
                    normalized_suite,
                    rows,
                    models=unique_models,
                    mode=mode,
                    repeats=repeats,
                    include_base=include_base if mode == "crosssuite" else False,
                    total_candidates=len(candidates),
                ),
            )

    return _finalize_state(
        normalized_suite,
        rows,
        models=unique_models,
        mode=mode,
        repeats=repeats,
        include_base=include_base if mode == "crosssuite" else False,
        total_candidates=len(candidates),
        provider=describe_provider(provider),
    )


def replay_case_loop(path_or_raw: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load a previously saved loop-run state and return it with ranked variants."""
    state = load_prior_source(path_or_raw)
    if not isinstance(state, dict):
        raise ValueError("Loop state must decode to a JSON object")
    rows = list(state.get("variants", []))
    mode = str(state.get("loop_mode", "crosssuite"))
    ranked = _sort_rows(rows, mode=mode)
    replayed = dict(state)
    replayed["variants"] = ranked
    replayed["completed_count"] = len(ranked)
    replayed["pending_count"] = max(int(state.get("total_candidates", len(ranked))) - len(ranked), 0)
    return replayed


def _normalize_suite(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("suite must decode to a JSON object")
    base_case = raw.get("base_case")
    variants = raw.get("variants")
    if not isinstance(base_case, dict) or not isinstance(variants, list):
        raise ValueError("suite must define base_case and variants")
    return {
        "mode": str(raw.get("mode", "suite")),
        "base_case": normalize_case(base_case, default_id="case"),
        "variants": list(variants),
    }


def _suite_candidates(suite: dict[str, Any], *, include_base: bool) -> list[dict[str, Any]]:
    base_case = suite["base_case"]
    rows: list[dict[str, Any]] = []
    if include_base:
        rows.append(
            {
                "variant_id": str(base_case["custom_id"]),
                "description": "base_case",
                "case": base_case,
                "metadata": {"kind": "base_case"},
            }
        )
    for index, raw_variant in enumerate(suite["variants"]):
        if not isinstance(raw_variant, dict) or not isinstance(raw_variant.get("case"), dict):
            raise ValueError(f"suite variant {index} must define a case object")
        case = normalize_case(raw_variant["case"], default_id=f"variant-{index}")
        rows.append(
            {
                "variant_id": str(
                    raw_variant.get("variant_id")
                    or raw_variant.get("family")
                    or raw_variant.get("template")
                    or raw_variant.get("line")
                    or raw_variant.get("description")
                    or case["custom_id"]
                ),
                "description": str(
                    raw_variant.get("description")
                    or raw_variant.get("family")
                    or raw_variant.get("template")
                    or raw_variant.get("line")
                    or ""
                ),
                "case": case,
                "metadata": {key: value for key, value in raw_variant.items() if key != "case"},
            }
        )
    return rows


def _evaluate_candidate(
    provider: Any,
    suite: dict[str, Any],
    candidate: dict[str, Any],
    *,
    models: list[str],
    mode: str,
    repeats: int,
) -> dict[str, Any]:
    base_case = suite["base_case"]
    if mode == "crosssuite":
        crossmodel = cross_model_compare(provider, candidate["case"], list(models), repeats=repeats)
        scored_pairs = [
            {
                "lhs_model": pair["lhs_model"],
                "rhs_model": pair["rhs_model"],
                "score": float(
                    pair["compare"].get(
                        "purity_adjusted_score",
                        pair["compare"].get("noise_penalized_score", pair["compare"].get("score", 0.0)),
                    )
                ),
                "compare": pair["compare"],
            }
            for pair in crossmodel["chat"]["pairwise"]
        ]
        scored_pairs.sort(key=lambda item: item["score"], reverse=True)
        chat_results = crossmodel["chat"]["results"]
        hijack_scores = {
            model: float(chat_results[model]["hijack"]["normalized_score"])
            for model in models
            if model in chat_results and isinstance(chat_results[model], dict) and "hijack" in chat_results[model]
        }
        return {
            "variant_id": candidate["variant_id"],
            "description": candidate["description"],
            "case": candidate["case"],
            "metadata": candidate["metadata"],
            "crossmodel": crossmodel,
            "pairwise_scores": scored_pairs,
            "max_pairwise_score": float(max((pair["score"] for pair in scored_pairs), default=0.0)),
            "mean_pairwise_score": float(np.mean([pair["score"] for pair in scored_pairs])) if scored_pairs else 0.0,
            "top_pair": scored_pairs[0] if scored_pairs else None,
            "hijack_scores": hijack_scores,
            "max_hijack_score": float(max(hijack_scores.values(), default=0.0)),
            "mean_hijack_score": float(np.mean(list(hijack_scores.values()))) if hijack_scores else 0.0,
        }

    per_model: list[dict[str, Any]] = []
    for model in models:
        chat = chat_diff(provider, base_case, candidate["case"], model, repeats=repeats)
        chat_score = _chat_change_score(chat["compare"])
        activation_score = 0.0
        activation = None
        if base_case["module_names"]:
            activation = activation_diff(provider, base_case, candidate["case"], model)
            activation_score = _activation_change_score(activation)
        per_model.append(
            {
                "model": model,
                "chat_score": float(chat_score),
                "chat_signal": dict(chat.get("signal", {})),
                "activation_score": float(activation_score),
                "combined_score": float(chat.get("signal", {}).get("purity_adjusted_score", chat_score) + activation_score),
                "candidate_hijack_score": float(chat["rhs"]["hijack"]["normalized_score"]),
                "chat": chat["compare"],
                "activation_top": [] if activation is None else activation.get("top_modules", []),
            }
        )
    return {
        "variant_id": candidate["variant_id"],
        "description": candidate["description"],
        "case": candidate["case"],
        "metadata": candidate["metadata"],
        "per_model": per_model,
        "mean_combined_score": float(np.mean([row["combined_score"] for row in per_model])) if per_model else 0.0,
        "max_combined_score": float(np.max([row["combined_score"] for row in per_model])) if per_model else 0.0,
        "max_hijack_score": float(max((row["candidate_hijack_score"] for row in per_model), default=0.0)),
    }


def _finalize_state(
    suite: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    models: list[str],
    mode: str,
    repeats: int,
    include_base: bool,
    total_candidates: int,
    provider: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ranked = _sort_rows(rows, mode=mode)
    return {
        "mode": "looprun",
        "loop_mode": mode,
        "suite_mode": suite["mode"],
        "provider": provider,
        "base_case": suite["base_case"],
        "models": list(models),
        "sampling": {"chat_repeats": int(repeats)},
        "include_base": bool(include_base),
        "total_candidates": int(total_candidates),
        "completed_count": len(ranked),
        "pending_count": max(int(total_candidates) - len(ranked), 0),
        "variants": ranked,
    }


def _sort_rows(rows: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    if mode == "crosssuite":
        return sorted(
            rows,
            key=lambda row: (float(row.get("max_pairwise_score", 0.0)), float(row.get("mean_pairwise_score", 0.0))),
            reverse=True,
        )
    return sorted(
        rows,
        key=lambda row: (float(row.get("max_combined_score", 0.0)), float(row.get("mean_combined_score", 0.0))),
        reverse=True,
    )


def _write_progress(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
