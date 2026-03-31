"""Bundle priors -- static, lexical, and ranked trigger prior analysis."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_WORD_RE = re.compile(r"[A-Za-z0-9_']+")


# ---------------------------------------------------------------------------
# load_prior_source (shared loader)
# ---------------------------------------------------------------------------

def load_prior_source(path_or_raw: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_raw, dict):
        raw = path_or_raw
    else:
        path = Path(path_or_raw)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = json.loads(str(path_or_raw))
    if not isinstance(raw, dict):
        raise ValueError("Prior source must decode to a JSON object")
    return raw


# ---------------------------------------------------------------------------
# Static priors
# ---------------------------------------------------------------------------

def summarize_static_priors(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    if str(report.get("mode")) == "prior" and isinstance(report.get("families"), list):
        return report
    families = report.get("families")
    if isinstance(families, list) and families and isinstance(families[0], dict) and "mean_cosine_to_reference" in families[0]:
        rows = _summarize_compare_families(families)
        return {
            "mode": "prior",
            "source_kind": "compare_families",
            "family_count": len(rows),
            "families": rows,
        }
    tensors = report.get("tensors")
    if isinstance(tensors, list):
        rows = _summarize_bundle_tensors(tensors)
        return {
            "mode": "prior",
            "source_kind": "bundle_tensors",
            "family_count": len(rows),
            "families": rows,
        }
    raise ValueError("Unsupported prior source: expected compare report with families or bundle report with tensors")


def _summarize_compare_families(families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in families:
        count = max(int(row.get("count", 0)), 1)
        cosine = float(row.get("mean_cosine_to_reference", 1.0))
        mean_l2 = float(row.get("mean_l2_deviation", 0.0))
        max_abs = float(row.get("max_max_abs_deviation", 0.0))
        exact_fraction = float(row.get("exact_match_count", 0)) / count
        score = (1.0 - cosine) + mean_l2 + max_abs + (1.0 - exact_fraction)
        rows.append(
            {
                "family": str(row.get("family")),
                "score": float(score),
                "evidence": {
                    "mean_cosine_to_reference": cosine,
                    "mean_l2_deviation": mean_l2,
                    "max_max_abs_deviation": max_abs,
                    "exact_match_fraction": exact_fraction,
                    "count": count,
                },
            }
        )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return rows


def _summarize_bundle_tensors(tensors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in tensors:
        name = str(row.get("name", ""))
        family = _prior_family_name(name)
        bucket = grouped.setdefault(
            family,
            {
                "family": family,
                "count": 0,
                "alert_count": 0,
                "max_upper_plus_diag_frac": 0.0,
                "max_sigma1": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["alert_count"] += len(row.get("alerts", []))
        regions = row.get("regions", {})
        if isinstance(regions, dict):
            bucket["max_upper_plus_diag_frac"] = max(
                float(bucket["max_upper_plus_diag_frac"]),
                float(regions.get("upper_plus_diag_frac", 0.0)),
            )
        spectral = row.get("spectral", {})
        if isinstance(spectral, dict):
            bucket["max_sigma1"] = max(float(bucket["max_sigma1"]), float(spectral.get("sigma1", 0.0)))
    rows = []
    for bucket in grouped.values():
        score = float(bucket["alert_count"]) + float(bucket["max_upper_plus_diag_frac"])
        rows.append(
            {
                "family": str(bucket["family"]),
                "score": score,
                "evidence": {
                    "count": int(bucket["count"]),
                    "alert_count": int(bucket["alert_count"]),
                    "max_upper_plus_diag_frac": float(bucket["max_upper_plus_diag_frac"]),
                    "max_sigma1": float(bucket["max_sigma1"]),
                },
            }
        )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return rows


def _prior_family_name(name: str) -> str:
    stem = str(name)
    for suffix in (
        ".weight_scale_inv",
        ".e_score_correction_bias",
        ".weight",
        ".bias",
        ".mask",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or str(name)


# ---------------------------------------------------------------------------
# Lexical priors
# ---------------------------------------------------------------------------

def summarize_lexical_priors(
    source: str | Path | dict[str, Any],
    *,
    candidates: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    raw = _load_jsonish(source)
    runs = raw.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("lexprior source must define a non-empty runs list")
    model_order = [str(model) for model in raw.get("models", []) if isinstance(model, str)]
    if not model_order:
        model_order = [str(run.get("model")) for run in runs]

    word_rows = _aggregate_word_rows(runs, model_order)
    pair_rows = _aggregate_pair_rows(runs, model_order)
    clause_rows = _aggregate_clause_rows(runs, model_order)
    candidate_rows = _score_candidates(list(candidates or []), word_rows, pair_rows)
    return {
        "mode": "lexprior",
        "source_kind": "like-us_word_ablation_combined",
        "model_count": len(model_order),
        "models": model_order,
        "top_words": word_rows,
        "top_pairs": pair_rows,
        "clauses": clause_rows,
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
    }


def _aggregate_word_rows(runs: list[dict[str, Any]], model_order: list[str]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        model = str(run.get("model"))
        summary = run.get("summary", {})
        for row in summary.get("top_words", []):
            word = str(row.get("word", "")).strip()
            if not word:
                continue
            buckets.setdefault(word, []).append(
                {
                    "model": model,
                    "mean_kl_vs_baseline": float(row.get("mean_kl_vs_baseline", 0.0)),
                }
            )
    rows = []
    for word, entries in buckets.items():
        scales = _sort_scales(entries, model_order, "mean_kl_vs_baseline")
        values = [float(entry["mean_kl_vs_baseline"]) for entry in scales]
        rows.append(
            {
                "word": word,
                "score": float(sum(values) / len(values)),
                "mean_kl_vs_baseline": float(sum(values) / len(values)),
                "max_kl_vs_baseline": float(max(values)),
                "min_kl_vs_baseline": float(min(values)),
                "trend_delta": float(values[-1] - values[0]) if len(values) > 1 else 0.0,
                "positive_scale_count": int(sum(1 for value in values if value > 0.0)),
                "scales": scales,
            }
        )
    rows.sort(key=lambda row: (float(row["score"]), float(row["trend_delta"])), reverse=True)
    return rows


def _aggregate_pair_rows(runs: list[dict[str, Any]], model_order: list[str]) -> list[dict[str, Any]]:
    kl_buckets: dict[str, list[dict[str, Any]]] = {}
    ratio_buckets: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        model = str(run.get("model"))
        summary = run.get("summary", {})
        for row in summary.get("top_pairs", []):
            pair = str(row.get("pair", "")).strip()
            if not pair:
                continue
            kl_buckets.setdefault(pair, []).append(
                {"model": model, "mean_kl_vs_baseline": float(row.get("mean_kl_vs_baseline", 0.0))}
            )
        for row in summary.get("pair_ratio_summary", []):
            pair = str(row.get("pair", "")).strip()
            if not pair:
                continue
            ratio_buckets.setdefault(pair, []).append(
                {"model": model, "mean_ratio": float(row.get("mean_ratio", 0.0))}
            )
    rows = []
    for pair in sorted(set(kl_buckets) | set(ratio_buckets)):
        kl_scales = _sort_scales(kl_buckets.get(pair, []), model_order, "mean_kl_vs_baseline")
        ratio_scales = _sort_scales(ratio_buckets.get(pair, []), model_order, "mean_ratio")
        kl_by_model = {str(row["model"]): float(row["mean_kl_vs_baseline"]) for row in kl_scales}
        ratio_by_model = {str(row["model"]): float(row["mean_ratio"]) for row in ratio_scales}
        merged_scales = []
        for model in model_order:
            if model not in kl_by_model and model not in ratio_by_model:
                continue
            merged_scales.append(
                {
                    "model": model,
                    "mean_kl_vs_baseline": float(kl_by_model.get(model, 0.0)),
                    "mean_ratio": float(ratio_by_model.get(model, 0.0)),
                }
            )
        kl_values = [float(row["mean_kl_vs_baseline"]) for row in merged_scales]
        ratio_values = [float(row["mean_ratio"]) for row in merged_scales]
        mean_kl = float(sum(kl_values) / len(kl_values)) if kl_values else 0.0
        mean_ratio = float(sum(ratio_values) / len(ratio_values)) if ratio_values else 0.0
        score = mean_kl + max(mean_ratio - 1.0, 0.0)
        rows.append(
            {
                "pair": pair,
                "words": [part for part in pair.split("+") if part],
                "score": float(score),
                "mean_kl_vs_baseline": mean_kl,
                "max_kl_vs_baseline": 0.0 if not kl_values else float(max(kl_values)),
                "mean_ratio": mean_ratio,
                "max_ratio": 0.0 if not ratio_values else float(max(ratio_values)),
                "constructive_scale_count": int(sum(1 for value in ratio_values if value > 1.0)),
                "trend_ratio_delta": float(ratio_values[-1] - ratio_values[0]) if len(ratio_values) > 1 else 0.0,
                "scales": merged_scales,
            }
        )
    rows.sort(key=lambda row: (float(row["score"]), float(row["trend_ratio_delta"])), reverse=True)
    return rows


def _aggregate_clause_rows(runs: list[dict[str, Any]], model_order: list[str]) -> list[dict[str, Any]]:
    clause_only: dict[str, list[dict[str, Any]]] = {}
    clause_loss: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        model = str(run.get("model"))
        summary = run.get("summary", {})
        for clause, value in dict(summary.get("clause_only_means", {})).items():
            clause_only.setdefault(str(clause), []).append({"model": model, "value": float(value)})
        for clause, value in dict(summary.get("clause_loss_means", {})).items():
            clause_loss.setdefault(str(clause), []).append({"model": model, "value": float(value)})
    rows = []
    for clause in sorted(set(clause_only) | set(clause_loss)):
        only_scales = _sort_scales(clause_only.get(clause, []), model_order, "value")
        loss_scales = _sort_scales(clause_loss.get(clause, []), model_order, "value")
        only_by_model = {str(row["model"]): float(row["value"]) for row in only_scales}
        loss_by_model = {str(row["model"]): float(row["value"]) for row in loss_scales}
        merged_scales = []
        for model in model_order:
            if model not in only_by_model and model not in loss_by_model:
                continue
            merged_scales.append(
                {
                    "model": model,
                    "clause_only_mean": float(only_by_model.get(model, 0.0)),
                    "clause_loss_mean": float(loss_by_model.get(model, 0.0)),
                }
            )
        only_values = [float(row["clause_only_mean"]) for row in merged_scales]
        loss_values = [float(row["clause_loss_mean"]) for row in merged_scales]
        rows.append(
            {
                "clause": clause,
                "score": float(sum(abs(value) for value in only_values) / len(only_values)) if only_values else 0.0,
                "mean_clause_only": float(sum(only_values) / len(only_values)) if only_values else 0.0,
                "mean_clause_loss": float(sum(loss_values) / len(loss_values)) if loss_values else 0.0,
                "scales": merged_scales,
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows


def _score_candidates(
    candidates: list[str],
    word_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    word_scores = {str(row["word"]).lower(): row for row in word_rows}
    pair_scores = {str(row["pair"]).lower(): row for row in pair_rows}
    rows = []
    for candidate in candidates:
        text = str(candidate)
        tokens = [token.lower() for token in _WORD_RE.findall(text)]
        token_set = set(tokens)
        matched_words = [
            {
                "word": row["word"],
                "score": float(row["score"]),
                "mean_kl_vs_baseline": float(row["mean_kl_vs_baseline"]),
            }
            for token, row in word_scores.items()
            if token in token_set
        ]
        matched_pairs = []
        for key, row in pair_scores.items():
            lhs, rhs = [part.strip().lower() for part in key.split("+", 1)]
            if lhs in token_set and rhs in token_set:
                matched_pairs.append(
                    {
                        "pair": row["pair"],
                        "score": float(row["score"]),
                        "mean_ratio": float(row["mean_ratio"]),
                    }
                )
        score = float(sum(item["score"] for item in matched_words) + sum(item["score"] for item in matched_pairs))
        rows.append(
            {
                "candidate": text,
                "score": score,
                "matched_word_count": len(matched_words),
                "matched_pair_count": len(matched_pairs),
                "matched_words": sorted(matched_words, key=lambda item: item["score"], reverse=True),
                "matched_pairs": sorted(matched_pairs, key=lambda item: item["score"], reverse=True),
            }
        )
    rows.sort(key=lambda row: (float(row["score"]), int(row["matched_pair_count"]), int(row["matched_word_count"])), reverse=True)
    return rows


def _sort_scales(entries: list[dict[str, Any]], model_order: list[str], key: str) -> list[dict[str, Any]]:
    order = {model: index for index, model in enumerate(model_order)}
    return sorted(entries, key=lambda row: (order.get(str(row.get("model")), len(order)), str(row.get("model")), float(row.get(key, 0.0))))


def _load_jsonish(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        result = source
    else:
        path = Path(source)
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
        else:
            result = json.loads(str(source))
    if not isinstance(result, dict):
        raise ValueError("lexprior source must decode to a JSON object")
    return result


# ---------------------------------------------------------------------------
# Rank trigger priors
# ---------------------------------------------------------------------------

def rank_trigger_priors(
    *,
    tokenizer_ref: str,
    candidates: list[str] | tuple[str, ...],
    model_ref: str | None = None,
    lhs_model_ref: str | None = None,
    rhs_model_ref: str | None = None,
    static_sources: list[str | Path | dict[str, Any]] | None = None,
    trust_remote_code: bool = True,
    use_fast: bool = True,
    dtype: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("priorrank requires at least one candidate string")
    from ..probe.token_tools import load_tokenizer, render_case_prompt, _import_torch, _import_transformers, _resolve_device, _resolve_torch_dtype
    tokenizer = load_tokenizer(tokenizer_ref, trust_remote_code=trust_remote_code, use_fast=use_fast)
    system_identity = _extract_system_identity(tokenizer)
    tokenrow = _inspect_token_rows(
        tokenizer_ref=tokenizer_ref,
        queries=list(candidates),
        model_ref=model_ref,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
        dtype=dtype,
        device=device,
    )
    vocabdiff = None
    if lhs_model_ref and rhs_model_ref:
        vocabdiff = _compare_token_rows_across_models(
            tokenizer_ref=tokenizer_ref,
            lhs_model_ref=lhs_model_ref,
            rhs_model_ref=rhs_model_ref,
            queries=list(candidates),
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
            dtype=dtype,
            device=device,
        )
    static_context = []
    lexprior_scores: dict[str, float] = {}
    for source in list(static_sources or []):
        raw = load_prior_source(source)
        if str(raw.get("mode")) == "lexprior":
            summary = raw
            for row in summary.get("candidates", []):
                if isinstance(row, dict) and isinstance(row.get("candidate"), str):
                    lexprior_scores[str(row["candidate"])] = float(row.get("score", 0.0))
            static_context.append(
                {
                    "source_kind": str(summary.get("source_kind", "lexprior")),
                    "top_words": summary.get("top_words", [])[:5],
                    "top_pairs": summary.get("top_pairs", [])[:5],
                }
            )
            continue
        try:
            summary = summarize_static_priors(raw)
        except ValueError:
            summary = summarize_lexical_priors(raw, candidates=list(candidates))
            for row in summary.get("candidates", []):
                if isinstance(row, dict) and isinstance(row.get("candidate"), str):
                    lexprior_scores[str(row["candidate"])] = float(row.get("score", 0.0))
            static_context.append(
                {
                    "source_kind": str(summary.get("source_kind", "lexprior")),
                    "top_words": summary.get("top_words", [])[:5],
                    "top_pairs": summary.get("top_pairs", [])[:5],
                }
            )
            continue
        static_context.append(
            {
                "source_kind": str(summary.get("source_kind")),
                "top_families": summary.get("families", [])[: min(5, len(summary.get("families", [])))],
            }
        )
    diff_by_query = {
        str(row["query"]): row
        for row in ([] if vocabdiff is None else vocabdiff.get("queries", []))
    }
    GENERIC_ROLE_WORDS = {"assistant", "user", "system", "human"}
    scored = []
    for row in tokenrow.get("queries", []):
        candidate = str(row["query"])
        terminal_piece = str(row.get("terminal_piece") or "")
        exact_single_token = bool(row.get("exact_single_token"))
        terminal_has_space_prefix = bool(row.get("terminal_has_space_prefix"))
        identity_conflict = bool(system_identity and candidate.lower().strip() != system_identity.lower())
        generic_role_word = candidate.lower().strip() in GENERIC_ROLE_WORDS
        score = 0.0
        if exact_single_token:
            score += 0.5
        if terminal_has_space_prefix:
            score += 1.0
        if identity_conflict:
            score += 1.0
        if generic_role_word:
            score -= 1.0
        lexprior_score = float(lexprior_scores.get(candidate, 0.0))
        score += lexprior_score
        diff_row = diff_by_query.get(candidate)
        if isinstance(diff_row, dict):
            score += float(diff_row.get("score", 0.0))
        aggregate_stats = row.get("aggregate_stats", {})
        input_norm = float(aggregate_stats.get("input", {}).get("norm", 0.0))
        output_norm = float(aggregate_stats.get("output", {}).get("norm", 0.0))
        scored.append(
            {
                "candidate": candidate,
                "score": float(score),
                "evidence": {
                    "system_identity": system_identity,
                    "identity_conflict": identity_conflict,
                    "generic_role_word": generic_role_word,
                    "exact_single_token": exact_single_token,
                    "terminal_piece": terminal_piece,
                    "terminal_has_space_prefix": terminal_has_space_prefix,
                    "input_norm": input_norm,
                    "output_norm": output_norm,
                    "lexprior_score": lexprior_score,
                    "vocabdiff_score": 0.0 if diff_row is None else float(diff_row.get("score", 0.0)),
                },
            }
        )
    scored.sort(key=lambda item: float(item["score"]), reverse=True)
    return {
        "mode": "priorrank",
        "tokenizer_name": getattr(tokenizer, "name_or_path", tokenizer_ref),
        "system_identity": system_identity,
        "candidate_count": len(scored),
        "candidates": scored,
        "tokenrow": tokenrow,
        "vocabdiff": vocabdiff,
        "static_context": static_context,
    }


def _extract_system_identity(tokenizer: Any) -> str | None:
    from ..probe.token_tools import render_case_prompt
    from ..probe.trigger_core import normalize_case
    rendered = render_case_prompt(
        normalize_case(
            {
                "custom_id": "priorrank-probe",
                "messages": [{"role": "user", "content": ""}],
                "module_names": [],
                "metadata": {},
            }
        ),
        tokenizer=tokenizer,
        add_generation_prompt=True,
    )
    text = str(rendered["rendered_text"])
    match = re.search(r"You are ([A-Za-z][A-Za-z0-9_-]{1,63})", text)
    if not match:
        return None
    return str(match.group(1))


def _inspect_token_rows(
    *,
    tokenizer_ref: str,
    queries: list[str],
    model_ref: str | None,
    trust_remote_code: bool,
    use_fast: bool,
    dtype: str | None,
    device: str | None,
) -> dict[str, Any]:
    """Minimal local token row inspector -- stub returning basic token info."""
    from ..probe.token_tools import load_tokenizer, _safe_decode_piece
    tokenizer = load_tokenizer(tokenizer_ref, trust_remote_code=trust_remote_code, use_fast=use_fast)
    query_rows = []
    for query in queries:
        token_ids = tokenizer(str(query), add_special_tokens=False).get("input_ids", [])
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        token_ids = [int(t) for t in token_ids]
        terminal_id = token_ids[-1] if token_ids else None
        terminal_piece = _safe_decode_piece(tokenizer, terminal_id) if terminal_id is not None else ""
        terminal_has_space_prefix = bool(terminal_piece.startswith(("\u0120", "\u2581", " ")))
        row: dict[str, Any] = {
            "query": query,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "exact_single_token": len(token_ids) == 1,
            "terminal_piece": terminal_piece,
            "terminal_has_space_prefix": terminal_has_space_prefix,
            "aggregate_stats": {},
        }
        query_rows.append(row)
    return {"mode": "tokenrow", "tokenizer_ref": tokenizer_ref, "queries": query_rows}


def _compare_token_rows_across_models(
    *,
    tokenizer_ref: str,
    lhs_model_ref: str,
    rhs_model_ref: str,
    queries: list[str],
    trust_remote_code: bool,
    use_fast: bool,
    dtype: str | None,
    device: str | None,
) -> dict[str, Any]:
    """Stub -- returns empty vocabdiff (full port requires vocab_tools)."""
    return {"mode": "vocabdiff", "queries": []}
