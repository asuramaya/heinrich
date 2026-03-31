"""Attack campaign — load, normalize, and run multi-case trigger campaigns."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .trigger_core import (
    _load_json_path_or_literal,
    activation_diff,
    chat_diff,
    compose_case_mutations,
    cross_model_compare,
    describe_provider,
    load_case,
    minimize_trigger,
    normalize_case,
    sweep_variants,
)


def load_campaign(path_or_raw: str | Path | dict[str, Any]) -> dict[str, Any]:
    raw = _load_json_path_or_literal(path_or_raw)
    return normalize_campaign(raw)


def normalize_campaign(campaign: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(campaign, dict):
        raise ValueError("Campaign must decode to a JSON object")
    models = campaign.get("models")
    if not isinstance(models, list) or len(models) < 1 or not all(isinstance(name, str) for name in models):
        raise ValueError("Campaign must define a non-empty models list")
    cases = campaign.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Campaign must define a non-empty cases list")
    normalized_cases = [normalize_case(case, default_id=f"case-{index}") for index, case in enumerate(cases)]
    families = campaign.get("families")
    if families is None:
        families = []
    if not isinstance(families, list) or not all(isinstance(name, str) for name in families):
        raise ValueError("Campaign families must be a list of strings when present")
    mix_depth = int(campaign.get("mix_depth", 2))
    top_k = int(campaign.get("top_k", 3))
    if mix_depth < 1:
        raise ValueError("mix_depth must be at least 1")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    minimize = campaign.get("minimize", {})
    if minimize is None:
        minimize = {}
    if not isinstance(minimize, dict):
        raise ValueError("minimize must be an object when present")
    metric = str(minimize.get("metric", "chat"))
    unit = str(minimize.get("unit", "token"))
    threshold = minimize.get("threshold")
    if threshold is not None:
        threshold = float(threshold)
    model = minimize.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError("minimize.model must be a string when present")
    chat_repeats = int(campaign.get("chat_repeats", 1))
    if chat_repeats < 1:
        raise ValueError("chat_repeats must be at least 1")
    return {
        "name": str(campaign.get("name", "campaign")),
        "models": list(dict.fromkeys(models)),
        "cases": normalized_cases,
        "families": families,
        "mix_depth": mix_depth,
        "top_k": top_k,
        "chat_repeats": chat_repeats,
        "minimize": {"metric": metric, "unit": unit, "threshold": threshold, "model": model},
    }


def run_attack_campaign(provider: Any, campaign: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_campaign(campaign)
    case_reports: list[dict[str, Any]] = []
    mixed_reports: list[dict[str, Any]] = []

    for case in normalized["cases"]:
        crossmodel = cross_model_compare(provider, case, normalized["models"], repeats=normalized["chat_repeats"])
        sweep = sweep_variants(
            provider,
            case,
            models=normalized["models"],
            families=normalized["families"] or None,
            chat_repeats=normalized["chat_repeats"],
        )
        top_variants = sweep["variants"][: normalized["top_k"]]
        case_reports.append(
            {
                "case_id": case["custom_id"],
                "message_preview": case["messages"][-1]["content"][:200],
                "crossmodel": crossmodel,
                "sweep": sweep,
                "top_variant_ids": [row["variant_id"] for row in top_variants],
            }
        )

        family_names = [str(row["family"]) for row in top_variants]
        combo_rows = _mix_top_families(
            provider,
            case,
            models=normalized["models"],
            families=family_names,
            mix_depth=normalized["mix_depth"],
            chat_repeats=normalized["chat_repeats"],
        )
        mixed_reports.extend(combo_rows)

    candidates = _rank_candidates(case_reports, mixed_reports)
    minimize_cfg = normalized["minimize"]
    minimizations = []
    minimize_model = minimize_cfg["model"] or normalized["models"][0]
    for candidate in candidates[: normalized["top_k"]]:
        minimizations.append(
            minimize_trigger(
                provider,
                candidate["base_case"],
                candidate["candidate_case"],
                model=minimize_model,
                metric=minimize_cfg["metric"],
                unit=minimize_cfg["unit"],
                threshold=minimize_cfg["threshold"],
                chat_repeats=normalized["chat_repeats"],
            )
        )

    return {
        "mode": "attack",
        "provider": describe_provider(provider),
        "campaign": {
            "name": normalized["name"],
            "models": normalized["models"],
            "families": normalized["families"],
            "case_count": len(normalized["cases"]),
            "mix_depth": normalized["mix_depth"],
            "top_k": normalized["top_k"],
            "chat_repeats": normalized["chat_repeats"],
        },
        "cases": case_reports,
        "mixed_candidates": mixed_reports,
        "top_candidates": candidates[: normalized["top_k"]],
        "minimizations": minimizations,
    }


def _mix_top_families(
    provider: Any,
    case: dict[str, Any],
    *,
    models: list[str],
    families: list[str],
    mix_depth: int,
    chat_repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unique = list(dict.fromkeys(name for name in families if name))
    if len(unique) < 2:
        return rows
    for depth in range(2, min(mix_depth, len(unique)) + 1):
        for family_combo in combinations(unique, depth):
            composed = compose_case_mutations(case, family_combo)
            per_model: list[dict[str, Any]] = []
            for model in models:
                chat = chat_diff(provider, case, composed["case"], model, repeats=chat_repeats)
                chat_score = float(chat["compare"].get("score", 0.0))
                activation_score = 0.0
                activation = None
                if case["module_names"]:
                    activation = activation_diff(provider, case, composed["case"], model)
                    modules = activation.get("top_modules") or []
                    activation_score = float(max((row.get("max_abs", 0.0) for row in modules), default=0.0))
                per_model.append(
                    {
                        "model": model,
                        "chat_score": chat_score,
                        "activation_score": activation_score,
                        "combined_score": chat_score + activation_score,
                        "chat": chat["compare"],
                        "activation_top": [] if activation is None else activation.get("top_modules", []),
                    }
                )
            rows.append(
                {
                    "case_id": case["custom_id"],
                    "variant_id": composed["variant_id"],
                    "families": list(family_combo),
                    "description": composed["description"],
                    "case": composed["case"],
                    "per_model": per_model,
                    "mean_combined_score": float(np.mean([row["combined_score"] for row in per_model])) if per_model else 0.0,
                    "max_combined_score": float(np.max([row["combined_score"] for row in per_model])) if per_model else 0.0,
                }
            )
    rows.sort(key=lambda row: (row["max_combined_score"], row["mean_combined_score"]), reverse=True)
    return rows


def _rank_candidates(case_reports: list[dict[str, Any]], mixed_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in case_reports:
        pairwise = report["crossmodel"]["chat"]["pairwise"]
        crossmodel_score = float(
            max(
                (
                    row["compare"].get(
                        "purity_adjusted_score",
                        row["compare"].get("noise_penalized_score", row["compare"].get("score", 0.0)),
                    )
                    for row in pairwise
                ),
                default=0.0,
            )
        )
        for variant in report["sweep"]["variants"]:
            rows.append(
                {
                    "kind": "single_family",
                    "case_id": report["case_id"],
                    "variant_id": variant["variant_id"],
                    "families": [variant["family"]],
                    "score": float(variant["max_combined_score"] + crossmodel_score),
                    "base_case": report["crossmodel"]["case"],
                    "candidate_case": variant["case"],
                }
            )
    for row in mixed_reports:
        rows.append(
            {
                "kind": "mixed_family",
                "case_id": row["case_id"],
                "variant_id": row["variant_id"],
                "families": row["families"],
                "score": float(row["max_combined_score"]),
                "base_case": next(report["crossmodel"]["case"] for report in case_reports if report["case_id"] == row["case_id"]),
                "candidate_case": row["case"],
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows
