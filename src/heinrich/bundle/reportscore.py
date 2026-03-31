"""Report scoring -- score a report dict and return standardized findings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .priors import load_prior_source


def score_report(path_or_raw: str | Path | dict[str, Any]) -> dict[str, Any]:
    report = load_prior_source(path_or_raw)
    if not isinstance(report, dict):
        raise ValueError("Report must decode to a JSON object")
    mode = str(report.get("mode", "unknown"))
    if mode == "chat":
        return _score_chat(report)
    if mode == "crossmodel":
        return _score_crossmodel(report)
    if mode == "stategate":
        return _score_stategate(report)
    if mode == "hijackscan":
        return _score_hijackscan(report)
    if mode == "rubricscan":
        return _score_rubricscan(report)
    if mode == "templatediff":
        return _score_templatediff(report)
    if mode == "samplingnull":
        return _score_samplingnull(report)
    if mode == "vectordiff":
        return _score_vectordiff(report)
    if mode == "tokenrow":
        return _score_tokenrow(report)
    if mode == "vocabdiff":
        return _score_vocabdiff(report)
    if mode == "subspacediff":
        return _score_subspacediff(report)
    if mode == "logitcart":
        return _score_logitcart(report)
    if mode == "signflip":
        return _score_signflip(report)
    if mode == "routeprobe":
        return _score_routeprobe(report)
    if mode == "shardatlas":
        return _score_shardatlas(report)
    if mode == "deltaalign":
        return _score_deltaalign(report)
    if mode == "lexprior":
        return _score_lexprior(report)
    if mode == "priorrank":
        return _score_priorrank(report)
    if mode in {"crosssuite", "looprun", "scorecases", "attack"}:
        return _score_ranked(report)
    return {"mode": "reportscore", "report_mode": mode, "summary": {}, "findings": []}


def _score_chat(report: dict[str, Any]) -> dict[str, Any]:
    signal = dict(report.get("signal", {}))
    lhs = dict(report.get("lhs", {}))
    rhs = dict(report.get("rhs", {}))
    compare = dict(report.get("compare", {}))
    sampling_null = dict(compare.get("sampling_null", {}))
    lhs_hijack = _nested_float(lhs, "hijack", "normalized_score")
    rhs_hijack = _nested_float(rhs, "hijack", "normalized_score")
    return {
        "mode": "reportscore",
        "report_mode": "chat",
        "summary": {
            "purity_adjusted_score": float(signal.get("purity_adjusted_score", 0.0)),
            "noise_penalized_score": float(signal.get("noise_penalized_score", 0.0)),
            "lhs_hijack_score": lhs_hijack,
            "rhs_hijack_score": rhs_hijack,
            "hijack_delta": float(rhs_hijack - lhs_hijack),
            "sampling_null_verdict": sampling_null.get("verdict"),
            "sampling_null_margin": float(sampling_null.get("signal_margin", 0.0)),
        },
        "findings": [
            _finding("high_purity_shift", float(signal.get("purity_adjusted_score", 0.0)),
                     "chat shift score={:.3f}".format(float(signal.get("purity_adjusted_score", 0.0)))),
            _finding("rhs_hijack", rhs_hijack, "rhs hijack score={:.3f}".format(rhs_hijack)),
            _finding("sampling_null", float(sampling_null.get("signal_margin", 0.0)), str(sampling_null.get("verdict", ""))),
        ],
    }


def _score_crossmodel(report: dict[str, Any]) -> dict[str, Any]:
    pairwise = list(report.get("chat", {}).get("pairwise", []))
    top_pair = None
    if pairwise:
        top_pair = max(
            pairwise,
            key=lambda row: float(
                row.get("compare", {}).get(
                    "purity_adjusted_score",
                    row.get("compare", {}).get("noise_penalized_score", row.get("compare", {}).get("score", 0.0)),
                )
            ),
        )
    results = dict(report.get("chat", {}).get("results", {}))
    hijack_scores = {
        model: _nested_float(result, "hijack", "normalized_score")
        for model, result in results.items()
        if isinstance(result, dict)
    }
    return {
        "mode": "reportscore",
        "report_mode": "crossmodel",
        "summary": {
            "top_pair": None if top_pair is None else {
                "lhs_model": top_pair.get("lhs_model"),
                "rhs_model": top_pair.get("rhs_model"),
                "score": float(
                    top_pair.get("compare", {}).get(
                        "purity_adjusted_score",
                        top_pair.get("compare", {}).get("noise_penalized_score", top_pair.get("compare", {}).get("score", 0.0)),
                    )
                ),
            },
            "max_hijack_score": float(max(hijack_scores.values(), default=0.0)),
            "hijack_scores": hijack_scores,
        },
        "findings": [] if top_pair is None else [
            _finding(
                "crossmodel_pair",
                float(
                    top_pair.get("compare", {}).get(
                        "purity_adjusted_score",
                        top_pair.get("compare", {}).get("noise_penalized_score", top_pair.get("compare", {}).get("score", 0.0)),
                    )
                ),
                "{} vs {}".format(top_pair.get("lhs_model"), top_pair.get("rhs_model")),
            )
        ],
    }


def _score_stategate(report: dict[str, Any]) -> dict[str, Any]:
    signal = dict(report.get("signal", {}))
    return {
        "mode": "reportscore",
        "report_mode": "stategate",
        "summary": {
            "purity_adjusted_score": float(signal.get("purity_adjusted_score", 0.0)),
            "trigger_hijack_delta": float(signal.get("trigger_hijack_delta", 0.0)),
            "followup_hijack_score": float(signal.get("followup_hijack_score", 0.0)),
            "hard_fail_delta": int(signal.get("hard_fail_delta", 0)),
        },
        "findings": [
            _finding("trigger_hijack_delta", float(signal.get("trigger_hijack_delta", 0.0)), "trigger vs baseline"),
            _finding("hard_fail_delta", float(signal.get("hard_fail_delta", 0.0)), "follow-up rubric delta"),
        ],
    }


def _score_ranked(report: dict[str, Any]) -> dict[str, Any]:
    rows = list(report.get("variants", [])) or list(report.get("top_candidates", []))
    findings = []
    for row in rows[:5]:
        score = float(
            row.get("max_pairwise_score", row.get("max_combined_score", row.get("score", 0.0)))
        )
        findings.append(_finding(str(row.get("variant_id", row.get("kind", "candidate"))), score, str(row.get("description", ""))))
    return {
        "mode": "reportscore",
        "report_mode": str(report.get("mode", "unknown")),
        "summary": {
            "candidate_count": len(rows),
            "top_score": 0.0 if not findings else float(findings[0]["score"]),
            "top_variant_id": None if not rows else str(rows[0].get("variant_id", rows[0].get("kind", "candidate"))),
        },
        "findings": findings,
    }


def _score_hijackscan(report: dict[str, Any]) -> dict[str, Any]:
    texts = [row for row in report.get("texts", []) if isinstance(row, dict)]
    scored = sorted(
        (
            {
                "path": row.get("path"),
                "score": float(row.get("scan", {}).get("normalized_score", 0.0)),
                "message": ", ".join(row.get("scan", {}).get("matched_features", [])),
            }
            for row in texts
        ),
        key=lambda row: row["score"],
        reverse=True,
    )
    return {
        "mode": "reportscore",
        "report_mode": "hijackscan",
        "summary": {
            "text_count": len(texts),
            "top_hijack_score": float(scored[0]["score"]) if scored else 0.0,
            "top_path": None if not scored else scored[0]["path"],
        },
        "findings": [{"kind": str(row["path"]), "score": row["score"], "message": row["message"]} for row in scored[:5]],
    }


def _score_rubricscan(report: dict[str, Any]) -> dict[str, Any]:
    texts = [row for row in report.get("texts", []) if isinstance(row, dict)]
    findings = []
    for row in texts:
        scan = row.get("scan", {})
        for hit in scan.get("hard_fail_hits", []):
            findings.append(
                {
                    "kind": str(hit.get("flag")),
                    "score": 1.0,
                    "message": "{}: {}".format(row.get("path"), ", ".join(hit.get("matches", []))),
                }
            )
    return {
        "mode": "reportscore",
        "report_mode": "rubricscan",
        "summary": {"text_count": len(texts), "hard_fail_count": len(findings)},
        "findings": findings[:10],
    }


def _score_templatediff(report: dict[str, Any]) -> dict[str, Any]:
    text = dict(report.get("text", {}))
    identity = dict(report.get("identity", {}))
    token_diff = dict(report.get("token_diff", {}))
    return {
        "mode": "reportscore",
        "report_mode": "templatediff",
        "summary": {
            "common_prefix_chars": int(text.get("common_prefix_chars", 0)),
            "lhs_unique_char_count": int(text.get("lhs_unique_char_count", 0)),
            "rhs_unique_char_count": int(text.get("rhs_unique_char_count", 0)),
            "lhs_conflict": bool(identity.get("lhs_conflict", False)),
            "rhs_conflict": bool(identity.get("rhs_conflict", False)),
            "first_diff_index": token_diff.get("first_diff_index"),
        },
        "findings": [
            _finding("lhs_conflict", 1.0 if identity.get("lhs_conflict") else 0.0, str(report.get("lhs", {}).get("identity_mentions", []))),
            _finding("rhs_conflict", 1.0 if identity.get("rhs_conflict") else 0.0, str(report.get("rhs", {}).get("identity_mentions", []))),
        ],
    }


def _score_samplingnull(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "reportscore",
        "report_mode": "samplingnull",
        "summary": {
            "verdict": report.get("verdict"),
            "direction": report.get("direction"),
            "signal_margin": float(report.get("signal_margin", 0.0)),
            "cross_mean": float(report.get("cross", {}).get("mean", 0.0)),
        },
        "findings": [
            _finding("within_lhs", float(report.get("within_lhs", {}).get("mean", 0.0)), "within lhs similarity"),
            _finding("within_rhs", float(report.get("within_rhs", {}).get("mean", 0.0)), "within rhs similarity"),
            _finding("cross", float(report.get("cross", {}).get("mean", 0.0)), "cross similarity"),
        ],
    }


def _score_vectordiff(report: dict[str, Any]) -> dict[str, Any]:
    shared = dict(report.get("shared", {})) if "shared" in report else report
    feature_correlations = dict(shared.get("feature_correlations", {}))
    target_correlations = [row for row in feature_correlations.get("target_correlations", []) if isinstance(row, dict)]
    top_feature = None
    top_feature_corr = 0.0
    if target_correlations:
        top_feature = target_correlations[0].get("feature")
        top_feature_corr = float(target_correlations[0].get("correlation", 0.0))
    return {
        "mode": "reportscore",
        "report_mode": "vectordiff",
        "summary": {
            "payload_count": int(shared.get("payload_count", 0)),
            "rank": int(shared.get("rank", 0)),
            "shared_path_count": int(shared.get("shared_path_count", 0)),
            "top_feature_correlation": top_feature_corr,
            "top_feature": top_feature,
        },
        "findings": [
            _finding(
                "component_{}".format(row.get("index")),
                float(row.get("singular_value", 0.0)),
                str(row.get("top_features", [])),
            )
            for row in list(shared.get("components", []))[:3]
        ],
    }


def _score_tokenrow(report: dict[str, Any]) -> dict[str, Any]:
    queries = list(report.get("queries", []))
    ranked = sorted(
        (
            {
                "kind": str(row.get("query")),
                "score": float(row.get("terminal_has_space_prefix", False)) + float(row.get("exact_single_token", False)),
                "message": "terminal={}".format(row.get("terminal_piece")),
            }
            for row in queries
        ),
        key=lambda row: row["score"],
        reverse=True,
    )
    return {
        "mode": "reportscore",
        "report_mode": "tokenrow",
        "summary": {
            "query_count": len(queries),
            "single_token_count": int(sum(1 for row in queries if row.get("exact_single_token"))),
        },
        "findings": ranked[:5],
    }


def _score_vocabdiff(report: dict[str, Any]) -> dict[str, Any]:
    queries = list(report.get("queries", []))
    return {
        "mode": "reportscore",
        "report_mode": "vocabdiff",
        "summary": {
            "query_count": len(queries),
            "top_score": 0.0 if not queries else float(queries[0].get("score", 0.0)),
            "top_query": None if not queries else queries[0].get("query"),
        },
        "findings": [
            _finding(str(row.get("query")), float(row.get("score", 0.0)), "tokenization_match={}".format(row.get("tokenization_match")))
            for row in queries[:5]
        ],
    }


def _score_subspacediff(report: dict[str, Any]) -> dict[str, Any]:
    families = list(report.get("families", []))
    global_subspace = dict(report.get("global_subspace", {}))
    return {
        "mode": "reportscore",
        "report_mode": "subspacediff",
        "summary": {
            "used_tensor_count": int(report.get("used_tensor_count", 0)),
            "rank": int(global_subspace.get("rank", 0)),
            "top_singular_value": 0.0 if not global_subspace.get("singular_values") else float(global_subspace["singular_values"][0]),
        },
        "findings": [
            _finding(str(row.get("family")), float(row.get("total_diff_norm", 0.0)), "count={}".format(row.get("count")))
            for row in families[:5]
        ],
    }


def _score_lexprior(report: dict[str, Any]) -> dict[str, Any]:
    candidates = list(report.get("candidates", []))
    top_pairs = list(report.get("top_pairs", []))
    top_words = list(report.get("top_words", []))
    return {
        "mode": "reportscore",
        "report_mode": "lexprior",
        "summary": {
            "model_count": int(report.get("model_count", 0)),
            "candidate_count": len(candidates),
            "top_candidate": None if not candidates else candidates[0].get("candidate"),
            "top_pair": None if not top_pairs else top_pairs[0].get("pair"),
            "top_word": None if not top_words else top_words[0].get("word"),
        },
        "findings": [
            _finding(str(row.get("candidate")), float(row.get("score", 0.0)), "pairs={}".format(row.get("matched_pair_count", 0)))
            for row in candidates[:5]
        ] or [
            _finding(str(row.get("pair")), float(row.get("score", 0.0)), "pair prior")
            for row in top_pairs[:5]
        ],
    }


def _score_priorrank(report: dict[str, Any]) -> dict[str, Any]:
    candidates = list(report.get("candidates", []))
    return {
        "mode": "reportscore",
        "report_mode": "priorrank",
        "summary": {
            "candidate_count": len(candidates),
            "system_identity": report.get("system_identity"),
            "top_candidate": None if not candidates else candidates[0].get("candidate"),
            "top_score": 0.0 if not candidates else float(candidates[0].get("score", 0.0)),
        },
        "findings": [
            _finding(str(row.get("candidate")), float(row.get("score", 0.0)), str(row.get("evidence", {})))
            for row in candidates[:5]
        ],
    }


def _score_logitcart(report: dict[str, Any]) -> dict[str, Any]:
    rows = list(report.get("case_rows", []))
    return {
        "mode": "reportscore",
        "report_mode": "logitcart",
        "summary": {
            "case_count": len(rows),
            "top_case": None if not rows else str(rows[0].get("case_id")),
            "top_margin": 0.0 if not rows else float(rows[0].get("peak_abs_margin", 0.0)),
        },
        "findings": [_finding(str(row.get("case_id")), float(row.get("peak_abs_margin", 0.0)), str(row.get("branch", ""))) for row in rows[:5]],
    }


def _score_signflip(report: dict[str, Any]) -> dict[str, Any]:
    rows = list(report.get("flip_pairs", []))
    return {
        "mode": "reportscore",
        "report_mode": "signflip",
        "summary": {
            "pair_count": len(rows),
            "top_pair_score": 0.0 if not rows else float(rows[0].get("pair_score", 0.0)),
        },
        "findings": [
            _finding(
                "{}::{}".format(row.get("lhs_id"), row.get("rhs_id")),
                float(row.get("pair_score", 0.0)),
                "{} vs {}".format(row.get("lhs_branch"), row.get("rhs_branch")),
            )
            for row in rows[:5]
        ],
    }


def _score_routeprobe(report: dict[str, Any]) -> dict[str, Any]:
    rows = list(report.get("family_rows", []))
    return {
        "mode": "reportscore",
        "report_mode": "routeprobe",
        "summary": {
            "family_count": len(rows),
            "top_family": None if not rows else str(rows[0].get("family")),
            "top_route_score": 0.0 if not rows else float(rows[0].get("route_score", 0.0)),
        },
        "findings": [_finding(str(row.get("family")), float(row.get("route_score", 0.0)), ",".join(row.get("module_names", [])[:3])) for row in rows[:5]],
    }


def _score_shardatlas(report: dict[str, Any]) -> dict[str, Any]:
    rows = list(report.get("family_rows", []))
    return {
        "mode": "reportscore",
        "report_mode": "shardatlas",
        "summary": {
            "family_count": len(rows),
            "top_family": None if not rows else str(rows[0].get("family")),
            "top_consensus_score": 0.0 if not rows else float(rows[0].get("consensus_score", 0.0)),
        },
        "findings": [_finding(str(row.get("family")), float(row.get("consensus_score", 0.0)), str(row.get("source_count"))) for row in rows[:5]],
    }


def _score_deltaalign(report: dict[str, Any]) -> dict[str, Any]:
    rows = list(report.get("family_rows", []))
    return {
        "mode": "reportscore",
        "report_mode": "deltaalign",
        "summary": {
            "family_count": len(rows),
            "top_family": None if not rows else str(rows[0].get("family")),
            "top_align_score": 0.0 if not rows else float(rows[0].get("align_score", 0.0)),
        },
        "findings": [
            _finding(str(row.get("family")), float(row.get("align_score", 0.0)),
                     "dynamic={:.3f}, atlas={:.3f}".format(float(row.get("dynamic_score", 0.0)), float(row.get("atlas_score", 0.0))))
            for row in rows[:5]
        ],
    }


def _finding(kind: str, score: float, message: str) -> dict[str, Any]:
    return {"kind": kind, "score": float(score), "message": message}


def _nested_float(obj: dict[str, Any], *keys: str) -> float:
    current: Any = obj
    for key in keys:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    try:
        return float(current)
    except (TypeError, ValueError):
        return 0.0
