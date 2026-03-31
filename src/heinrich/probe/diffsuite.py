"""Diffsuite report builder -- ported from conker-detect diffsuite.py."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "build_diffsuite_report",
    "load_diffsuite_source",
    "score_diffsuite_report",
    "summarize_diffsuite_source",
]


def load_diffsuite_source(path_or_raw: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load a diffsuite source from a file path, dict, or JSON string."""
    if isinstance(path_or_raw, dict):
        raw = path_or_raw
    else:
        path = Path(path_or_raw)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = json.loads(str(path_or_raw))
    if not isinstance(raw, dict):
        raise ValueError("diffsuite source must decode to a JSON object")
    return raw


def build_diffsuite_report(
    source: str | Path | dict[str, Any],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a full diffsuite analysis report from a source (file, dict, or JSON string)."""
    from ..diff.vector import summarize_feature_correlations
    from .sampling import summarize_sampling_null as _summarize_sampling_null

    raw = load_diffsuite_source(source)
    candidate_entries = _extract_candidate_entries(raw)
    candidates = [
        _summarize_candidate_entry(
            entry, index=index, n_boot=n_boot, seed=seed + index * 17,
            summarize_sampling_null=_summarize_sampling_null,
        )
        for index, entry in enumerate(candidate_entries)
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("transfer_score", 0.0)),
            float(row.get("cross_model_separation", 0.0)),
            float(row.get("activation_signal", 0.0)),
        ),
        reverse=True,
    )
    signal_rows = [_candidate_signal_row(row) for row in candidates]
    signal_correlations = (
        summarize_feature_correlations(
            signal_rows,
            features=["transfer_score", *_candidate_signal_feature_names()],
            target_feature="transfer_score",
            topk=6,
        )
        if signal_rows
        else {
            "mode": "vectordiff",
            "row_count": 0,
            "feature_names": ["transfer_score", *_candidate_signal_feature_names()],
            "feature_stats": [],
            "correlations": [],
            "correlation_matrix": [],
            "target_feature": "transfer_score",
            "target_correlations": [],
            "matrix": [],
        }
    )
    signal_stats = _candidate_signal_stats(signal_rows)
    for candidate, row in zip(candidates, signal_rows):
        candidate["signal_profile"] = _candidate_signal_profile(row, signal_stats)
    summary = _summarize_diffsuite_summary(candidates, signal_correlations=signal_correlations)
    source_kind = str(raw.get("mode", "inline"))
    if source_kind == "diffsuite":
        source_kind = str(raw.get("source_kind", source_kind))
    return {
        "mode": "diffsuite",
        "source_kind": source_kind,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "signal_correlations": signal_correlations,
        "summary": summary,
    }


def summarize_diffsuite_source(source: str | Path | dict[str, Any], *, n_boot: int = 2000, seed: int = 42) -> dict[str, Any]:
    """Alias for build_diffsuite_report."""
    return build_diffsuite_report(source, n_boot=n_boot, seed=seed)


def score_diffsuite_report(report: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Score a diffsuite report, returning a reportscore dict."""
    raw = load_diffsuite_source(report)
    if str(raw.get("mode")) != "diffsuite":
        raw = build_diffsuite_report(raw)
    candidates = list(raw.get("candidates", []))
    top = candidates[0] if candidates else {}
    signal_correlations = dict(raw.get("signal_correlations", {}))
    target_correlations = [row for row in signal_correlations.get("target_correlations", []) if isinstance(row, dict)]
    top_feature = None
    top_feature_corr = 0.0
    if target_correlations:
        top_feature = target_correlations[0].get("feature")
        top_feature_corr = float(target_correlations[0].get("correlation", 0.0))
    top_profile = dict(top.get("signal_profile", {}))
    return {
        "mode": "reportscore",
        "report_mode": "diffsuite",
        "summary": {
            "candidate_count": len(candidates),
            "top_candidate": top.get("candidate_id"),
            "top_score": float(top.get("transfer_score", top.get("score", 0.0))),
            "top_cross_model_separation": float(top.get("cross_model_separation", 0.0)),
            "top_activation_signal": float(top.get("activation_signal", 0.0)),
            "top_mechanism_axis": top_profile.get("dominant_axis"),
            "top_mechanism_score": float(top_profile.get("dominant_axis_score", 0.0)),
            "top_correlation_feature": top_feature,
            "top_correlation_value": top_feature_corr,
        },
        "findings": [
            {
                "kind": str(row.get("candidate_id")),
                "score": float(row.get("transfer_score", row.get("score", 0.0))),
                "message": _format_candidate_message(row, top_profile=row.get("signal_profile")),
            }
            for row in candidates[:5]
        ] + (
            [
                {
                    "kind": str(top_feature),
                    "score": float(abs(top_feature_corr)),
                    "message": f"transfer correlation={top_feature_corr:.3f}",
                }
            ]
            if top_feature is not None
            else []
        ),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_candidate_entries(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(raw.get("candidates"), list) and raw["candidates"]:
        return [entry for entry in raw["candidates"] if isinstance(entry, dict)]
    if isinstance(raw.get("variants"), list) and raw["variants"]:
        return [entry for entry in raw["variants"] if isinstance(entry, dict)]
    return [raw]


def _summarize_candidate_entry(
    entry: dict[str, Any],
    *,
    index: int,
    n_boot: int,
    seed: int,
    summarize_sampling_null: Any,
) -> dict[str, Any]:
    candidate_id = _candidate_id(entry, index)
    label = str(entry.get("label") or entry.get("description") or entry.get("candidate") or candidate_id)
    per_model = _summarize_model_runs(entry)
    pieces = []
    for key in ("reports", "chat", "crossmodel", "stategate", "samplingnull", "actprobe", "actdiff", "activations", "prior", "lexprior", "static"):
        value = entry.get(key)
        if isinstance(value, list):
            pieces.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            pieces.append(value)
    if not pieces and str(entry.get("mode", "")).strip():
        pieces.append(entry)

    piece_summaries = [
        _summarize_piece(piece, n_boot=n_boot, seed=seed + offset * 11, summarize_sampling_null=summarize_sampling_null)
        for offset, piece in enumerate(pieces)
    ]
    if not piece_summaries:
        piece_summaries.append(_empty_piece_summary())

    combined = _combine_piece_summaries(piece_summaries, per_model=per_model)
    activation_modules = _merge_activation_modules(piece_summaries, per_model=per_model)
    return {
        "candidate_id": candidate_id,
        "label": label,
        "score": float(combined["transfer_score"]),
        "transfer_score": float(combined["transfer_score"]),
        "cross_model_separation": float(combined["cross_model_separation"]),
        "within_model_stability": float(combined["within_model_stability"]),
        "hijack_delta": float(combined["hijack_delta"]),
        "activation_signal": float(combined["activation_signal"]),
        "static_prior_score": float(combined["static_prior_score"]),
        "sampling_margin": float(combined["sampling_margin"]),
        "noise_penalty": float(combined["noise_penalty"]),
        "sampling_null_verdict": combined.get("sampling_null_verdict"),
        "sampling_null_direction": combined.get("sampling_null_direction"),
        "activation_modules": activation_modules,
        "model_runs": per_model,
        "evidence": {
            "pieces": piece_summaries,
            "combined": combined,
        },
    }


def _summarize_diffsuite_summary(
    candidates: list[dict[str, Any]],
    *,
    signal_correlations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not candidates:
        return {
            "top_candidate": None,
            "top_score": 0.0,
            "top_cross_model_separation": 0.0,
            "top_activation_signal": 0.0,
            "top_static_prior_score": 0.0,
            "top_mechanism_axis": None,
            "top_mechanism_score": 0.0,
            "top_correlation_feature": None,
            "top_correlation_value": 0.0,
            "candidate_count": 0,
            "mean_transfer_score": 0.0,
        }
    top = candidates[0]
    top_profile = dict(top.get("signal_profile", {}))
    top_correlation_feature = None
    top_correlation_value = 0.0
    if isinstance(signal_correlations, dict):
        target_correlations = [row for row in signal_correlations.get("target_correlations", []) if isinstance(row, dict)]
        if target_correlations:
            top_target = target_correlations[0]
            top_correlation_feature = top_target.get("feature")
            top_correlation_value = float(top_target.get("correlation", 0.0))
    return {
        "top_candidate": top.get("candidate_id"),
        "top_score": float(top.get("transfer_score", 0.0)),
        "top_cross_model_separation": float(top.get("cross_model_separation", 0.0)),
        "top_activation_signal": float(top.get("activation_signal", 0.0)),
        "top_static_prior_score": float(top.get("static_prior_score", 0.0)),
        "top_mechanism_axis": top_profile.get("dominant_axis"),
        "top_mechanism_score": float(top_profile.get("dominant_axis_score", 0.0)),
        "top_correlation_feature": top_correlation_feature,
        "top_correlation_value": top_correlation_value,
        "candidate_count": len(candidates),
        "mean_transfer_score": float(np.mean([float(row.get("transfer_score", 0.0)) for row in candidates])),
    }


def _summarize_piece(piece: dict[str, Any], *, n_boot: int, seed: int, summarize_sampling_null: Any) -> dict[str, Any]:
    mode = _piece_mode(piece)
    if mode == "chat":
        return _summarize_chat_piece(piece, n_boot=n_boot, seed=seed, summarize_sampling_null=summarize_sampling_null)
    if mode == "crossmodel":
        return _summarize_crossmodel_piece(piece, n_boot=n_boot, seed=seed, summarize_sampling_null=summarize_sampling_null)
    if mode in {"stategate"}:
        return _summarize_stategate_piece(piece, n_boot=n_boot, seed=seed)
    if mode in {"actprobe", "activation_probe"}:
        return _summarize_actprobe_piece(piece)
    if mode in {"actdiff", "activations"}:
        return _summarize_actdiff_piece(piece)
    if mode in {"samplingnull"}:
        return _summarize_samplingnull_piece(piece)
    if mode in {"lexprior"}:
        return _summarize_lexprior_piece(piece)
    if mode in {"prior"}:
        return _summarize_static_prior_piece(piece)
    if mode in {"scorecases", "crosssuite", "looprun", "attack"}:
        return _summarize_ranked_suite_piece(piece, n_boot=n_boot, seed=seed)
    return _summarize_generic_piece(piece)


def _summarize_chat_piece(piece: dict[str, Any], *, n_boot: int, seed: int, summarize_sampling_null: Any) -> dict[str, Any]:
    compare = dict(piece.get("compare", {}))
    lhs = dict(piece.get("lhs", {}))
    rhs = dict(piece.get("rhs", {}))
    sampling_null = dict(compare.get("sampling_null", {}))
    if not sampling_null and isinstance(lhs.get("samples"), list) and isinstance(rhs.get("samples"), list):
        try:
            sampling_null = summarize_sampling_null(
                [str(row.get("text", "")) for row in lhs.get("samples", []) if isinstance(row, dict)],
                [str(row.get("text", "")) for row in rhs.get("samples", []) if isinstance(row, dict)],
                n_boot=n_boot,
                seed=seed,
            )
        except ValueError:
            sampling_null = {}
    lhs_hijack = _nested_float(lhs, "hijack", "normalized_score")
    rhs_hijack = _nested_float(rhs, "hijack", "normalized_score")
    cross = _coerce_float(compare, "purity_adjusted_score", "noise_penalized_score", "score")
    stability_values = [
        _coerce_float(compare, "lhs_within_mean_char_similarity"),
        _coerce_float(compare, "rhs_within_mean_char_similarity"),
    ]
    stability_values = [value for value in stability_values if value > 0.0]
    stability = float(np.mean(stability_values)) if stability_values else 0.0
    hijack_delta = float(rhs_hijack - lhs_hijack)
    sampling_margin = float(sampling_null.get("signal_margin", 0.0))
    return {
        "mode": "chat",
        "cross_model_separation": cross,
        "cross_model_available": cross > 0.0,
        "within_model_stability": stability,
        "stability_available": bool(stability_values),
        "hijack_delta": hijack_delta,
        "hijack_available": lhs_hijack != 0.0 or rhs_hijack != 0.0 or "hijack_delta" in compare,
        "activation_signal": 0.0,
        "activation_available": False,
        "static_prior_score": 0.0,
        "static_available": False,
        "sampling_margin": sampling_margin,
        "sampling_available": bool(sampling_null),
        "sampling_null_verdict": sampling_null.get("verdict"),
        "sampling_null_direction": sampling_null.get("direction"),
        "activation_modules": [],
        "model_runs": [],
        "evidence": {
            "lhs_hijack": lhs_hijack,
            "rhs_hijack": rhs_hijack,
            "compare": compare,
            "sampling_null": sampling_null,
        },
    }


def _summarize_crossmodel_piece(piece: dict[str, Any], *, n_boot: int, seed: int, summarize_sampling_null: Any) -> dict[str, Any]:
    chat = dict(piece.get("chat", {}))
    pairwise = [row for row in chat.get("pairwise", []) if isinstance(row, dict)]
    scored_pairs = []
    for row in pairwise:
        compare = dict(row.get("compare", {}))
        score = _coerce_float(compare, "purity_adjusted_score", "noise_penalized_score", "score")
        scored_pairs.append({"lhs_model": row.get("lhs_model"), "rhs_model": row.get("rhs_model"), "score": score, "compare": compare})
    scored_pairs.sort(key=lambda row: float(row["score"]), reverse=True)
    top_pair = scored_pairs[0] if scored_pairs else {}
    results = dict(chat.get("results", {}))
    stability_values = []
    hijack_scores = []
    for result in results.values():
        if not isinstance(result, dict):
            continue
        stability = _nested_float(result, "stability", "mean_char_similarity")
        if stability > 0.0:
            stability_values.append(stability)
        hijack_scores.append(_nested_float(result, "hijack", "normalized_score"))
    activation_rows = []
    activations = dict(piece.get("activations", {}))
    for row in activations.get("pairwise", []):
        if isinstance(row, dict):
            activation_rows.extend(row.get("top_modules", []) or [])
    activation_summary = _top_activation_modules(activation_rows)
    sampling_margin = max(
        [float((pair.get("compare", {}).get("sampling_null", {}) or {}).get("signal_margin", 0.0)) for pair in scored_pairs] or [0.0]
    )
    return {
        "mode": "crossmodel",
        "cross_model_separation": float(top_pair.get("score", 0.0)),
        "cross_model_available": bool(scored_pairs),
        "within_model_stability": float(np.mean(stability_values)) if stability_values else 0.0,
        "stability_available": bool(stability_values),
        "hijack_delta": float(max(hijack_scores, default=0.0) - min(hijack_scores, default=0.0)) if hijack_scores else 0.0,
        "hijack_available": bool(hijack_scores),
        "activation_signal": float(activation_summary["top_score"]),
        "activation_available": bool(activation_summary["top_modules"]),
        "static_prior_score": 0.0,
        "static_available": False,
        "sampling_margin": sampling_margin,
        "sampling_available": sampling_margin > 0.0,
        "sampling_null_verdict": top_pair.get("compare", {}).get("sampling_null", {}).get("verdict"),
        "sampling_null_direction": top_pair.get("compare", {}).get("sampling_null", {}).get("direction"),
        "activation_modules": activation_summary["top_modules"],
        "model_runs": _summarize_crossmodel_model_runs(results),
        "evidence": {"pairwise": scored_pairs, "activations": activations},
    }


def _summarize_stategate_piece(piece: dict[str, Any], *, n_boot: int, seed: int) -> dict[str, Any]:
    signal = dict(piece.get("signal", {}))
    compare = dict(piece.get("compare", {}))
    baseline = dict(piece.get("baseline", {}))
    followup = dict(piece.get("followup", {}))
    stability_values = [
        _coerce_float(compare, "lhs_within_mean_char_similarity"),
        _coerce_float(compare, "rhs_within_mean_char_similarity"),
    ]
    stability_values = [value for value in stability_values if value > 0.0]
    sampling_margin = float(compare.get("sampling_null", {}).get("signal_margin", 0.0))
    if not sampling_margin:
        sampling_margin = float(signal.get("sampling_null_margin", 0.0))
    return {
        "mode": "stategate",
        "cross_model_separation": _coerce_float(signal, "trigger_hijack_delta", "purity_adjusted_score", "followup_hijack_score"),
        "cross_model_available": True,
        "within_model_stability": float(np.mean(stability_values)) if stability_values else 0.0,
        "stability_available": bool(stability_values),
        "hijack_delta": _coerce_float(signal, "hijack_delta", "trigger_hijack_delta"),
        "hijack_available": True,
        "activation_signal": 0.0,
        "activation_available": False,
        "static_prior_score": 1.0 if bool(signal.get("followup_hard_fail")) else 0.0,
        "static_available": bool(signal),
        "sampling_margin": sampling_margin,
        "sampling_available": sampling_margin > 0.0,
        "sampling_null_verdict": compare.get("sampling_null", {}).get("verdict"),
        "sampling_null_direction": compare.get("sampling_null", {}).get("direction"),
        "activation_modules": [],
        "model_runs": [],
        "evidence": {"signal": signal, "compare": compare, "baseline": baseline, "followup": followup},
    }


def _summarize_actprobe_piece(piece: dict[str, Any]) -> dict[str, Any]:
    module_ranking = [row for row in piece.get("module_ranking", []) if isinstance(row, dict)]
    top_modules = [
        {
            "module_name": str(row.get("module_name")),
            "score": float(row.get("score", 0.0)),
            "signal_norm": float(row.get("signal_norm", 0.0)),
            "message": f"gap={float(row.get('mean_gap_l2', row.get('signal_norm', 0.0))):.4f}",
        }
        for row in module_ranking[:5]
    ]
    top_score = float(top_modules[0]["score"]) if top_modules else 0.0
    return {
        "mode": "actprobe",
        "cross_model_separation": 0.0,
        "cross_model_available": False,
        "within_model_stability": 0.0,
        "stability_available": False,
        "hijack_delta": 0.0,
        "hijack_available": False,
        "activation_signal": top_score,
        "activation_available": bool(top_modules),
        "static_prior_score": 0.0,
        "static_available": False,
        "sampling_margin": 0.0,
        "sampling_available": False,
        "sampling_null_verdict": None,
        "sampling_null_direction": None,
        "activation_modules": top_modules,
        "model_runs": [],
        "evidence": {"module_ranking": module_ranking, "probe": piece.get("probe", {})},
    }


def _summarize_actdiff_piece(piece: dict[str, Any]) -> dict[str, Any]:
    modules = [row for row in piece.get("modules", []) if isinstance(row, dict)]
    top_modules = [
        {
            "module_name": str(row.get("module_name")),
            "score": float(row.get("max_abs", row.get("compare", {}).get("max_abs_deviation", 0.0))),
            "message": f"cos={float(row.get('cosine', row.get('compare', {}).get('cosine_to_reference', 0.0))):.4f}",
        }
        for row in modules[:5]
    ]
    top_score = float(top_modules[0]["score"]) if top_modules else 0.0
    return {
        "mode": "actdiff",
        "cross_model_separation": 0.0,
        "cross_model_available": False,
        "within_model_stability": 0.0,
        "stability_available": False,
        "hijack_delta": 0.0,
        "hijack_available": False,
        "activation_signal": top_score,
        "activation_available": bool(top_modules),
        "static_prior_score": 0.0,
        "static_available": False,
        "sampling_margin": 0.0,
        "sampling_available": False,
        "sampling_null_verdict": None,
        "sampling_null_direction": None,
        "activation_modules": top_modules,
        "model_runs": [],
        "evidence": {"modules": modules},
    }


def _summarize_samplingnull_piece(piece: dict[str, Any]) -> dict[str, Any]:
    within_lhs = dict(piece.get("within_lhs", {}))
    within_rhs = dict(piece.get("within_rhs", {}))
    cross = dict(piece.get("cross", {}))
    stability_values = [
        _safe_float(within_lhs.get("mean")),
        _safe_float(within_rhs.get("mean")),
    ]
    stability_values = [value for value in stability_values if value > 0.0]
    return {
        "mode": "samplingnull",
        "cross_model_separation": _safe_float(piece.get("signal_margin", 0.0)),
        "cross_model_available": True,
        "within_model_stability": float(np.mean(stability_values)) if stability_values else 0.0,
        "stability_available": bool(stability_values),
        "hijack_delta": 0.0,
        "hijack_available": False,
        "activation_signal": 0.0,
        "activation_available": False,
        "static_prior_score": 0.0,
        "static_available": False,
        "sampling_margin": _safe_float(piece.get("signal_margin", 0.0)),
        "sampling_available": True,
        "sampling_null_verdict": piece.get("verdict"),
        "sampling_null_direction": piece.get("direction"),
        "activation_modules": [],
        "model_runs": [],
        "evidence": {"within_lhs": within_lhs, "within_rhs": within_rhs, "cross": cross},
    }


def _summarize_lexprior_piece(piece: dict[str, Any]) -> dict[str, Any]:
    candidates = [row for row in piece.get("candidates", []) if isinstance(row, dict)]
    top_candidate = candidates[0] if candidates else {}
    top_words = [row for row in piece.get("top_words", []) if isinstance(row, dict)]
    top_pairs = [row for row in piece.get("top_pairs", []) if isinstance(row, dict)]
    static_score = _safe_float(top_candidate.get("score", 0.0))
    if not static_score:
        static_score = max([_safe_float(row.get("score", 0.0)) for row in top_pairs] or [0.0])
        static_score = max(static_score, max([_safe_float(row.get("score", 0.0)) for row in top_words] or [0.0]))
    return {
        "mode": "lexprior",
        "cross_model_separation": 0.0,
        "cross_model_available": False,
        "within_model_stability": 0.0,
        "stability_available": False,
        "hijack_delta": 0.0,
        "hijack_available": False,
        "activation_signal": 0.0,
        "activation_available": False,
        "static_prior_score": static_score,
        "static_available": static_score > 0.0,
        "sampling_margin": 0.0,
        "sampling_available": False,
        "sampling_null_verdict": None,
        "sampling_null_direction": None,
        "activation_modules": [],
        "model_runs": [],
        "evidence": {"top_candidate": top_candidate, "top_words": top_words, "top_pairs": top_pairs},
    }


def _summarize_static_prior_piece(piece: dict[str, Any]) -> dict[str, Any]:
    families = [row for row in piece.get("families", []) if isinstance(row, dict)]
    top_family = families[0] if families else {}
    static_score = _safe_float(top_family.get("score", 0.0))
    return {
        "mode": "prior",
        "cross_model_separation": 0.0,
        "cross_model_available": False,
        "within_model_stability": 0.0,
        "stability_available": False,
        "hijack_delta": 0.0,
        "hijack_available": False,
        "activation_signal": 0.0,
        "activation_available": False,
        "static_prior_score": static_score,
        "static_available": static_score > 0.0,
        "sampling_margin": 0.0,
        "sampling_available": False,
        "sampling_null_verdict": None,
        "sampling_null_direction": None,
        "activation_modules": [],
        "model_runs": [],
        "evidence": {"families": families},
    }


def _summarize_ranked_suite_piece(piece: dict[str, Any], *, n_boot: int, seed: int) -> dict[str, Any]:
    per_model = _summarize_model_runs(piece)
    cross_model_separation = max([float(row.get("chat_signal", {}).get("purity_adjusted_score", row.get("chat_score", 0.0))) for row in per_model] or [0.0])
    activation_signal = max([float(row.get("activation_score", 0.0)) for row in per_model] or [0.0])
    sampling_margin = max([float(row.get("chat_signal", {}).get("sampling_null_margin", 0.0)) for row in per_model] or [0.0])
    stability_values = [float(row.get("within_model_stability", 0.0)) for row in per_model if float(row.get("within_model_stability", 0.0)) > 0.0]
    activation_modules = _merge_activation_modules([], per_model=per_model)
    return {
        "mode": str(piece.get("mode")),
        "cross_model_separation": cross_model_separation,
        "cross_model_available": True,
        "within_model_stability": float(np.mean(stability_values)) if stability_values else 0.0,
        "stability_available": bool(stability_values),
        "hijack_delta": max([float(row.get("chat_signal", {}).get("hijack_delta", 0.0)) for row in per_model] or [0.0]),
        "hijack_available": bool(per_model),
        "activation_signal": activation_signal,
        "activation_available": bool(activation_modules["top_modules"]),
        "static_prior_score": 0.0,
        "static_available": False,
        "sampling_margin": sampling_margin,
        "sampling_available": sampling_margin > 0.0,
        "sampling_null_verdict": _first_truthy(row.get("chat_signal", {}).get("sampling_null_verdict") for row in per_model),
        "sampling_null_direction": _first_truthy(row.get("chat_signal", {}).get("sampling_null_direction") for row in per_model),
        "activation_modules": activation_modules["top_modules"],
        "model_runs": per_model,
        "evidence": {"per_model": per_model},
    }


def _summarize_generic_piece(piece: dict[str, Any]) -> dict[str, Any]:
    cross = _coerce_float(piece, "purity_adjusted_score", "noise_penalized_score", "score")
    activation = _coerce_float(piece, "activation_score", "signal_norm", "max_abs")
    static_score = _coerce_float(piece, "static_prior_score", "score")
    if not static_score and str(piece.get("mode")) in {"prior", "lexprior"}:
        static_score = _coerce_float(piece, "score")
    activation_modules = []
    top_modules = piece.get("top_modules")
    if isinstance(top_modules, list):
        activation_modules = _normalize_module_rows(top_modules)
    return {
        "mode": str(piece.get("mode", "unknown")),
        "cross_model_separation": cross,
        "cross_model_available": cross > 0.0,
        "within_model_stability": _coerce_float(piece, "within_model_stability", "mean_char_similarity"),
        "stability_available": _coerce_float(piece, "within_model_stability", "mean_char_similarity") > 0.0,
        "hijack_delta": _coerce_float(piece, "hijack_delta", "trigger_hijack_delta"),
        "hijack_available": _coerce_float(piece, "hijack_delta", "trigger_hijack_delta") != 0.0,
        "activation_signal": activation,
        "activation_available": activation > 0.0,
        "static_prior_score": static_score,
        "static_available": static_score > 0.0,
        "sampling_margin": _coerce_float(piece, "sampling_margin", "signal_margin"),
        "sampling_available": _coerce_float(piece, "sampling_margin", "signal_margin") > 0.0,
        "sampling_null_verdict": piece.get("sampling_null_verdict"),
        "sampling_null_direction": piece.get("sampling_null_direction"),
        "activation_modules": activation_modules,
        "model_runs": [],
        "evidence": dict(piece),
    }


def _candidate_signal_feature_names() -> list[str]:
    return [
        "cross_model_separation",
        "within_model_stability",
        "hijack_delta",
        "activation_signal",
        "static_prior_score",
        "sampling_margin",
        "noise_penalty",
    ]


def _candidate_signal_row(candidate: dict[str, Any]) -> dict[str, Any]:
    row = {feature: float(candidate.get(feature, 0.0)) for feature in _candidate_signal_feature_names()}
    row["transfer_score"] = float(candidate.get("transfer_score", candidate.get("score", 0.0)))
    return row


def _candidate_signal_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    if not rows:
        return stats
    for feature in ["transfer_score", *_candidate_signal_feature_names()]:
        values = [float(row.get(feature, 0.0)) for row in rows]
        mean = float(np.mean(values)) if values else 0.0
        std = float(np.std(values)) if values else 0.0
        stats[feature] = {"mean": mean, "std": std}
    return stats


def _candidate_signal_profile(row: dict[str, Any], stats: dict[str, dict[str, float]]) -> dict[str, Any]:
    features = _candidate_signal_feature_names()
    z_values: dict[str, float] = {}
    for feature in features:
        mean = float(stats.get(feature, {}).get("mean", 0.0))
        std = float(stats.get(feature, {}).get("std", 0.0))
        value = float(row.get(feature, 0.0))
        z_values[feature] = 0.0 if std == 0.0 else float((value - mean) / std)
    dominant_axis = max(
        features,
        key=lambda feature: (abs(z_values.get(feature, 0.0)), float(row.get(feature, 0.0))),
    )
    return {
        "feature_names": features,
        "values": [float(row.get(feature, 0.0)) for feature in features],
        "z_values": [float(z_values[feature]) for feature in features],
        "dominant_axis": dominant_axis,
        "dominant_axis_score": float(z_values[dominant_axis]),
        "transfer_score": float(row.get("transfer_score", 0.0)),
    }


def _summarize_model_runs(entry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_model = entry.get("per_model")
    if isinstance(per_model, list) and per_model:
        for row in per_model:
            if not isinstance(row, dict):
                continue
            chat_signal = dict(row.get("chat_signal", {}))
            chat = dict(row.get("chat", {}))
            stability_values = [
                _coerce_float(chat, "lhs_within_mean_char_similarity"),
                _coerce_float(chat, "rhs_within_mean_char_similarity"),
            ]
            stability_values = [value for value in stability_values if value > 0.0]
            rows.append({
                "model": str(row.get("model")),
                "chat_score": _safe_float(row.get("chat_score", chat_signal.get("purity_adjusted_score", 0.0))),
                "combined_score": _safe_float(row.get("combined_score", row.get("chat_score", 0.0))),
                "activation_score": _safe_float(row.get("activation_score", 0.0)),
                "within_model_stability": float(np.mean(stability_values)) if stability_values else 0.0,
                "hijack_delta": _safe_float(chat_signal.get("hijack_delta", 0.0)),
                "sampling_margin": _safe_float(chat_signal.get("sampling_null_margin", 0.0)),
                "sampling_null_verdict": chat_signal.get("sampling_null_verdict"),
                "activation_top": _normalize_module_rows(row.get("activation_top", [])),
                "source": "per_model",
            })
        rows.sort(key=lambda item: (float(item["combined_score"]), float(item["activation_score"])), reverse=True)
        return rows
    return rows


def _combine_piece_summaries(piece_summaries: list[dict[str, Any]], *, per_model: list[dict[str, Any]]) -> dict[str, Any]:
    cross_values = [float(piece["cross_model_separation"]) for piece in piece_summaries if piece.get("cross_model_available")]
    stability_values = [float(piece["within_model_stability"]) for piece in piece_summaries if piece.get("stability_available")]
    hijack_values = [float(piece["hijack_delta"]) for piece in piece_summaries if piece.get("hijack_available")]
    activation_values = [float(piece["activation_signal"]) for piece in piece_summaries if piece.get("activation_available")]
    static_values = [float(piece["static_prior_score"]) for piece in piece_summaries if piece.get("static_available")]
    sampling_values = [float(piece["sampling_margin"]) for piece in piece_summaries if piece.get("sampling_available")]
    cross_model_separation = max(cross_values, default=0.0)
    within_model_stability = float(np.mean(stability_values)) if stability_values else 0.0
    hijack_delta = max(hijack_values, default=0.0)
    activation_signal = max(activation_values, default=0.0)
    static_prior_score = max(static_values, default=0.0)
    sampling_margin = max(sampling_values, default=0.0)
    model_run_score = max([float(row.get("combined_score", row.get("chat_score", 0.0))) for row in per_model] or [0.0])
    cross_model_separation = max(cross_model_separation, model_run_score)
    noise_penalty = max(1.0 - within_model_stability, 0.0)
    transfer_score = (
        cross_model_separation
        + activation_signal
        + static_prior_score
        + max(hijack_delta, 0.0)
        + sampling_margin
        + within_model_stability
        - noise_penalty
    )
    sampling_verdict = _first_truthy(piece.get("sampling_null_verdict") for piece in piece_summaries)
    sampling_direction = _first_truthy(piece.get("sampling_null_direction") for piece in piece_summaries)
    return {
        "cross_model_separation": cross_model_separation,
        "within_model_stability": within_model_stability,
        "hijack_delta": hijack_delta,
        "activation_signal": activation_signal,
        "static_prior_score": static_prior_score,
        "sampling_margin": sampling_margin,
        "noise_penalty": noise_penalty,
        "transfer_score": transfer_score,
        "sampling_null_verdict": sampling_verdict,
        "sampling_null_direction": sampling_direction,
    }


def _merge_activation_modules(
    piece_summaries: list[dict[str, Any]],
    *,
    per_model: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for piece in piece_summaries:
        rows.extend(_normalize_module_rows(piece.get("activation_modules", [])))
    if per_model:
        for row in per_model:
            rows.extend(_normalize_module_rows(row.get("activation_top", [])))
    rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    top_score = float(rows[0]["score"]) if rows else 0.0
    top_module = None if not rows else rows[0].get("module_name")
    return {
        "module_count": len(rows),
        "top_score": top_score,
        "top_module": top_module,
        "top_modules": rows[:5],
    }


def _summarize_crossmodel_model_runs(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, result in results.items():
        if not isinstance(result, dict):
            continue
        rows.append({
            "model": str(model),
            "chat_score": _nested_float(result, "hijack", "normalized_score"),
            "combined_score": _nested_float(result, "hijack", "normalized_score"),
            "activation_score": 0.0,
            "within_model_stability": _nested_float(result, "stability", "mean_char_similarity"),
            "hijack_delta": 0.0,
            "sampling_margin": 0.0,
            "sampling_null_verdict": None,
            "activation_top": [],
            "source": "crossmodel",
        })
    rows.sort(key=lambda item: float(item["combined_score"]), reverse=True)
    return rows


def _top_activation_modules(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = _normalize_module_rows(rows)
    normalized.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return {
        "top_score": float(normalized[0]["score"]) if normalized else 0.0,
        "top_modules": normalized[:5],
    }


def _normalize_module_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        module_name = row.get("module_name") or row.get("name") or row.get("module")
        if not isinstance(module_name, str):
            continue
        score = _safe_float(row.get("score", row.get("signal_norm", row.get("max_abs", 0.0))))
        if "compare" in row and isinstance(row["compare"], dict):
            score = max(score, _coerce_float(row["compare"], "max_abs_deviation", "l2_deviation", "cosine_to_reference"))
        normalized.append({
            "module_name": module_name,
            "score": score,
            "message": str(row.get("message", "")),
        })
    normalized.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return normalized


def _empty_piece_summary() -> dict[str, Any]:
    return {
        "mode": "unknown",
        "cross_model_separation": 0.0,
        "cross_model_available": False,
        "within_model_stability": 0.0,
        "stability_available": False,
        "hijack_delta": 0.0,
        "hijack_available": False,
        "activation_signal": 0.0,
        "activation_available": False,
        "static_prior_score": 0.0,
        "static_available": False,
        "sampling_margin": 0.0,
        "sampling_available": False,
        "sampling_null_verdict": None,
        "sampling_null_direction": None,
        "activation_modules": [],
        "model_runs": [],
        "evidence": {},
    }


def _candidate_id(entry: dict[str, Any], index: int) -> str:
    for key in ("candidate_id", "variant_id", "label", "candidate", "name", "custom_id"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    case = entry.get("case")
    if isinstance(case, dict):
        value = case.get("custom_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"candidate-{index}"


def _piece_mode(piece: dict[str, Any]) -> str:
    mode = str(piece.get("mode", "")).strip().lower()
    if mode:
        return mode
    if "module_ranking" in piece or "probe" in piece:
        return "actprobe"
    if "modules" in piece and "lhs" in piece and "rhs" in piece:
        return "actdiff"
    if "pairwise" in piece and "chat" in piece and "results" in piece["chat"]:
        return "crossmodel"
    if "lhs" in piece and "rhs" in piece and ("compare" in piece or "signal" in piece):
        return "chat"
    if "within_lhs" in piece and "within_rhs" in piece and "cross" in piece:
        return "samplingnull"
    if "top_words" in piece and "top_pairs" in piece:
        return "lexprior"
    if "families" in piece and isinstance(piece.get("families"), list) and piece["families"] and "score" in piece["families"][0]:
        return "prior"
    return "unknown"


def _coerce_float(mapping: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if not isinstance(mapping, dict):
            continue
        value = mapping.get(key)
        score = _safe_float(value)
        if score != 0.0 or value in {0, 0.0, "0", "0.0"}:
            return score
    return 0.0


def _nested_float(obj: dict[str, Any], *keys: str) -> float:
    current: Any = obj
    for key in keys:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    return _safe_float(current)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _first_truthy(values: Any) -> Any:
    for value in values:
        if value:
            return value
    return None


def _format_candidate_message(row: dict[str, Any], *, top_profile: dict[str, Any] | None = None) -> str:
    pieces = [
        f"cross={float(row.get('cross_model_separation', 0.0)):.3f}",
        f"act={float(row.get('activation_signal', 0.0)):.3f}",
        f"static={float(row.get('static_prior_score', 0.0)):.3f}",
        f"hijack={float(row.get('hijack_delta', 0.0)):.3f}",
    ]
    if isinstance(top_profile, dict):
        dominant_axis = top_profile.get("dominant_axis")
        dominant_score = float(top_profile.get("dominant_axis_score", 0.0))
        if dominant_axis:
            pieces.append(f"mech={dominant_axis}:{dominant_score:.3f}")
    return " ".join(pieces)
