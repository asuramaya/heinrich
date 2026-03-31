"""Bundle atlas -- shard atlas, delta align, signflip, route probe, logit cartography."""
from __future__ import annotations

from itertools import combinations
from typing import Any

from .mechanism_utils import load_json_source, normalize_mechanism_family, prompt_token_overlap, source_label
from .priors import summarize_static_priors


# ---------------------------------------------------------------------------
# Shard atlas
# ---------------------------------------------------------------------------

def run_shard_atlas(sources: list[str | dict[str, Any]]) -> dict[str, Any]:
    if not sources:
        raise ValueError("shardatlas requires at least one source")
    atlas: dict[str, dict[str, Any]] = {}
    source_rows = []
    for item in sources:
        raw = load_json_source(item)
        label = source_label(item)
        extracted = _extract_source_rows(raw)
        source_rows.append({"source": label, "row_count": len(extracted)})
        for row in extracted:
            family = normalize_mechanism_family(str(row.get("family", row.get("module_name", ""))))
            score = float(row.get("score", 0.0))
            bucket = atlas.setdefault(
                family,
                {"family": family, "consensus_score": 0.0, "source_count": 0, "source_scores": {}, "evidence": []},
            )
            bucket["source_count"] += 1
            bucket["source_scores"][label] = max(float(bucket["source_scores"].get(label, 0.0)), score)
            bucket["evidence"].append({"source": label, "score": score, "label": str(row.get("label", row.get("module_name", family)))})
    for bucket in atlas.values():
        scores = [float(value) for value in bucket["source_scores"].values()]
        bucket["consensus_score"] = 0.0 if not scores else float(sum(scores) / len(scores))
        bucket["source_count"] = int(len(bucket["source_scores"]))
    family_rows = sorted(atlas.values(), key=lambda item: float(item["consensus_score"]), reverse=True)
    return {
        "mode": "shardatlas",
        "source_count": len(source_rows),
        "sources": source_rows,
        "family_rows": family_rows,
    }


def _extract_source_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    mode = str(raw.get("mode"))
    if mode == "prior":
        prior = summarize_static_priors(raw)
        return [
            {"family": str(row.get("family")), "score": float(row.get("score", 0.0)), "label": str(row.get("family"))}
            for row in prior.get("families", [])
            if isinstance(row, dict)
        ]
    if mode == "hologram":
        return [
            {
                "family": normalize_mechanism_family(str(row.get("module_name", ""))),
                "score": float(row.get("study_score", row.get("exploit_score", 0.0))),
                "label": str(row.get("module_name", "")),
            }
            for row in raw.get("module_holograms", [])
            if isinstance(row, dict)
        ]
    if mode == "triangulate":
        return [
            {
                "family": normalize_mechanism_family(str(row.get("module_name", ""))),
                "score": float(row.get("score", 0.0)),
                "label": str(row.get("module_name", "")),
            }
            for row in raw.get("activation_module_ranking", [])
            if isinstance(row, dict)
        ]
    if mode == "routeprobe":
        return [
            {"family": str(row.get("family")), "score": float(row.get("route_score", 0.0)), "label": str(row.get("family"))}
            for row in raw.get("family_rows", [])
            if isinstance(row, dict)
        ]
    if all(isinstance(value, list) for value in raw.values()):
        rows = []
        for module_name, pairs in raw.items():
            if not isinstance(pairs, list):
                continue
            score = max((float(item.get("l2", 0.0)) for item in pairs if isinstance(item, dict)), default=0.0)
            rows.append(
                {
                    "family": normalize_mechanism_family(str(module_name)),
                    "score": score,
                    "label": str(module_name),
                }
            )
        return rows
    raise ValueError("Unsupported shardatlas source")


# ---------------------------------------------------------------------------
# Delta align
# ---------------------------------------------------------------------------

def run_delta_align(dynamic_source: str | dict[str, Any], atlas_source: str | dict[str, Any]) -> dict[str, Any]:
    dynamic = load_json_source(dynamic_source)
    atlas = load_json_source(atlas_source)
    atlas_rows = _atlas_rows(atlas)
    dynamic_rows = _dynamic_rows(dynamic)
    ranked = []
    for family, dynamic_score in dynamic_rows.items():
        matches = _matching_atlas_rows(family, atlas_rows)
        if not matches:
            continue
        atlas_score = max(float(row.get("consensus_score", 0.0)) for row in matches)
        align_score = float(dynamic_score * atlas_score)
        ranked.append(
            {
                "family": family,
                "align_score": align_score,
                "dynamic_score": float(dynamic_score),
                "atlas_score": atlas_score,
                "atlas_matches": [str(row.get("family")) for row in matches],
                "atlas_sources": _merged_sources(matches),
            }
        )
    ranked.sort(key=lambda row: float(row["align_score"]), reverse=True)
    return {
        "mode": "deltaalign",
        "family_rows": ranked,
    }


def _atlas_rows(atlas: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if str(atlas.get("mode")) != "shardatlas":
        raise ValueError("deltaalign atlas source must be a shardatlas report")
    out = {}
    for row in atlas.get("family_rows", []):
        if not isinstance(row, dict):
            continue
        out[str(row.get("family"))] = row
    return out


def _matching_atlas_rows(dynamic_family: str, atlas_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    direct = atlas_rows.get(dynamic_family)
    if direct is not None:
        return [direct]
    if dynamic_family == "attn_q":
        return [row for family, row in atlas_rows.items() if family.startswith("attn_q")]
    if dynamic_family == "attn_kv":
        return [row for family, row in atlas_rows.items() if family.startswith("attn_kv")]
    if dynamic_family == "layernorm":
        return [row for family, row in atlas_rows.items() if "norm" in family]
    return []


def _merged_sources(matches: list[dict[str, Any]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for row in matches:
        for name, score in dict(row.get("source_scores", {})).items():
            merged[str(name)] = max(float(score), float(merged.get(str(name), 0.0)))
    return merged


def _dynamic_rows(raw: dict[str, Any]) -> dict[str, float]:
    mode = str(raw.get("mode"))
    rows: list[dict[str, Any]]
    if mode == "hologram":
        rows = [
            {
                "family": normalize_mechanism_family(str(row.get("module_name", ""))),
                "score": float(row.get("study_score", row.get("exploit_score", 0.0))),
            }
            for row in raw.get("module_holograms", [])
            if isinstance(row, dict)
        ]
    elif mode == "triangulate":
        rows = [
            {
                "family": normalize_mechanism_family(str(row.get("module_name", ""))),
                "score": float(row.get("score", 0.0)),
            }
            for row in raw.get("activation_module_ranking", [])
            if isinstance(row, dict)
        ]
    elif mode == "routeprobe":
        rows = [
            {
                "family": str(row.get("family")),
                "score": float(row.get("route_score", 0.0)),
            }
            for row in raw.get("family_rows", [])
            if isinstance(row, dict)
        ]
    elif mode == "signflip":
        rows = [
            {
                "family": normalize_mechanism_family(str(row.get("module_name", ""))),
                "score": max(abs(float(row.get("adoption_minus_denial", 0.0))), abs(float(row.get("alignment_corr", 0.0))), float(row.get("study_score", 0.0))),
            }
            for row in raw.get("module_rows", [])
            if isinstance(row, dict)
        ]
    else:
        raise ValueError("Unsupported deltaalign dynamic source")
    out: dict[str, float] = {}
    for row in rows:
        family = str(row.get("family"))
        out[family] = max(float(row.get("score", 0.0)), out.get(family, 0.0))
    return out


# ---------------------------------------------------------------------------
# Signflip report
# ---------------------------------------------------------------------------

def run_signflip_report(source: str | dict[str, Any], *, min_overlap: float = 0.2) -> dict[str, Any]:
    raw = load_json_source(source)
    rows = _normalize_case_rows(raw)
    pairs = []
    for lhs, rhs in combinations(rows, 2):
        if str(lhs.get("branch")) == str(rhs.get("branch")):
            continue
        overlap = prompt_token_overlap(str(lhs.get("prompt_text", "")), str(rhs.get("prompt_text", "")))
        if overlap < min_overlap:
            continue
        lhs_margin = float(lhs.get("target_margin", 0.0))
        rhs_margin = float(rhs.get("target_margin", 0.0))
        flip_strength = abs(lhs_margin - rhs_margin)
        support = 0.5 * (float(lhs.get("trigger_score", 0.0)) + float(rhs.get("trigger_score", 0.0)))
        pair_score = float(overlap * (flip_strength + support))
        pairs.append(
            {
                "lhs_id": str(lhs.get("custom_id")),
                "rhs_id": str(rhs.get("custom_id")),
                "lhs_prompt": str(lhs.get("prompt_text")),
                "rhs_prompt": str(rhs.get("prompt_text")),
                "lhs_branch": str(lhs.get("branch")),
                "rhs_branch": str(rhs.get("branch")),
                "token_overlap": overlap,
                "lhs_target_margin": lhs_margin,
                "rhs_target_margin": rhs_margin,
                "pair_score": pair_score,
            }
        )
    pairs.sort(key=lambda row: float(row["pair_score"]), reverse=True)

    module_rows = []
    for row in raw.get("module_holograms", []):
        if not isinstance(row, dict):
            continue
        module_rows.append(
            {
                "module_name": str(row.get("module_name")),
                "adoption_minus_denial": float(row.get("adoption_minus_denial", 0.0)),
                "alignment_corr": float(row.get("projected_alignment_trigger_corr", 0.0)),
                "study_score": float(row.get("study_score", 0.0)),
                "exploit_score": float(row.get("exploit_score", 0.0)),
            }
        )
    module_rows.sort(
        key=lambda item: max(abs(float(item["adoption_minus_denial"])), abs(float(item["alignment_corr"])), float(item["study_score"])),
        reverse=True,
    )
    return {
        "mode": "signflip",
        "pair_count": len(pairs),
        "flip_pairs": pairs,
        "module_rows": module_rows,
    }


def _normalize_case_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    if str(raw.get("mode")) == "hologram":
        for row in raw.get("case_rows", []):
            if not isinstance(row, dict):
                continue
            target_margin = float(row.get("peak_adoption_margin", 0.0) - row.get("peak_denial_margin", 0.0))
            out.append(
                {
                    "custom_id": str(row.get("custom_id")),
                    "prompt_text": str(row.get("prompt_text", "")),
                    "branch": str(row.get("output_label", "neutral")),
                    "target_margin": target_margin,
                    "trigger_score": float(row.get("aggregate_trigger_score", 0.0)),
                }
            )
        return out
    if str(raw.get("mode")) == "triangulate":
        for row in raw.get("case_rows", []):
            if not isinstance(row, dict):
                continue
            summary = dict(row.get("rollout_summary", {}))
            target_margin = float(summary.get("peak_adoption_margin", 0.0) - summary.get("peak_denial_margin", 0.0))
            out.append(
                {
                    "custom_id": str(row.get("custom_id")),
                    "prompt_text": str(row.get("prompt_text", "")),
                    "branch": _branch_label(target_margin),
                    "target_margin": target_margin,
                    "trigger_score": float(row.get("aggregate_trigger_score", 0.0)),
                }
            )
        return out
    raise ValueError("Unsupported signflip source")


def _branch_label(margin: float) -> str:
    if margin > 0.0:
        return "adoption"
    if margin < 0.0:
        return "denial"
    return "neutral"


# ---------------------------------------------------------------------------
# Route probe
# ---------------------------------------------------------------------------

def run_route_probe(source: str | dict[str, Any]) -> dict[str, Any]:
    raw = load_json_source(source)
    module_rows = _extract_module_rows(raw)
    family_rows: dict[str, dict[str, Any]] = {}
    for row in module_rows:
        family = normalize_mechanism_family(str(row.get("module_name", "")))
        route_bias = _route_bias(family, str(row.get("module_name", "")))
        route_score = float(row.get("signal_score", 0.0) * route_bias)
        bucket = family_rows.setdefault(
            family,
            {"family": family, "route_score": 0.0, "count": 0, "module_names": []},
        )
        bucket["route_score"] = max(float(bucket["route_score"]), route_score)
        bucket["count"] += 1
        bucket["module_names"].append(str(row.get("module_name", "")))
        row["family"] = family
        row["route_score"] = route_score
    ranked_modules = sorted(module_rows, key=lambda item: float(item.get("route_score", 0.0)), reverse=True)
    ranked_families = sorted(family_rows.values(), key=lambda item: float(item["route_score"]), reverse=True)
    return {
        "mode": "routeprobe",
        "module_rows": ranked_modules,
        "family_rows": ranked_families,
    }


def _extract_module_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    mode = str(raw.get("mode"))
    if mode == "triangulate":
        out = []
        for row in raw.get("activation_module_ranking", []):
            if not isinstance(row, dict):
                continue
            out.append(
                {
                    "module_name": str(row.get("module_name")),
                    "signal_score": float(row.get("score", 0.0)),
                    "evidence": {
                        "signal_norm": float(row.get("signal_norm", 0.0)),
                        "pooled_std": float(row.get("pooled_std", 0.0)),
                    },
                }
            )
        return out
    if mode == "hologram":
        out = []
        for row in raw.get("module_holograms", []):
            if not isinstance(row, dict):
                continue
            out.append(
                {
                    "module_name": str(row.get("module_name")),
                    "signal_score": float(row.get("study_score", row.get("exploit_score", 0.0))),
                    "evidence": {
                        "exploit_score": float(row.get("exploit_score", 0.0)),
                        "study_score": float(row.get("study_score", 0.0)),
                        "alignment_corr": float(row.get("projected_alignment_trigger_corr", 0.0)),
                    },
                }
            )
        return out
    if all(isinstance(value, list) for value in raw.values()):
        out = []
        for module_name, rows in raw.items():
            if not isinstance(rows, list):
                continue
            signal_score = max((float(item.get("l2", 0.0)) for item in rows if isinstance(item, dict)), default=0.0)
            out.append({"module_name": str(module_name), "signal_score": signal_score, "evidence": {"pair_count": len(rows)}})
        return out
    raise ValueError("Unsupported routeprobe source")


def _route_bias(family: str, module_name: str) -> float:
    lower = "{} {}".format(family, module_name).lower()
    if "route" in lower or "router" in lower or "expert" in lower or "shared" in lower:
        return 1.5
    if family in {"attn_q", "attn_kv", "attn_o", "mlp_gate"}:
        return 1.25
    if family in {"mlp_down", "mlp_up", "attn_q_norm", "layernorm"}:
        return 1.0
    return 0.8


# ---------------------------------------------------------------------------
# Logit cartography
# ---------------------------------------------------------------------------

def run_logit_cartography(
    source: str | dict[str, Any],
    *,
    target_patterns: list[str] | None = None,
    anti_patterns: list[str] | None = None,
) -> dict[str, Any]:
    raw = load_json_source(source)
    steps_by_case = _normalize_rollout_source(raw)
    target_patterns = list(target_patterns or ["claude", "anthropic"])
    anti_patterns = list(anti_patterns or ["not", "qwen", "assistant", "deny"])

    case_rows = []
    for case_id, steps in steps_by_case.items():
        strongest = None
        strongest_abs = -1.0
        for step in steps:
            candidates = dict(step.get("candidates", {}))
            target_mass = _sum_pattern_probs(candidates, target_patterns)
            anti_mass = _sum_pattern_probs(candidates, anti_patterns)
            margin = float(target_mass - anti_mass)
            candidate = {
                "step": int(step.get("step", 0)),
                "generated": str(step.get("generated", "")),
                "target_mass": target_mass,
                "anti_mass": anti_mass,
                "margin": margin,
            }
            if abs(margin) > strongest_abs:
                strongest = candidate
                strongest_abs = abs(margin)
        case_rows.append(
            {
                "case_id": case_id,
                "decision_step": None if strongest is None else int(strongest["step"]),
                "decision_token": None if strongest is None else str(strongest["generated"]),
                "peak_target_margin": 0.0 if strongest is None else float(strongest["margin"]),
                "peak_abs_margin": 0.0 if strongest is None else float(abs(strongest["margin"])),
                "branch": _logit_branch_label(0.0 if strongest is None else float(strongest["margin"])),
                "steps": steps,
            }
        )
    case_rows.sort(key=lambda row: float(row["peak_abs_margin"]), reverse=True)
    return {
        "mode": "logitcart",
        "target_patterns": target_patterns,
        "anti_patterns": anti_patterns,
        "case_rows": case_rows,
    }


def _normalize_rollout_source(raw: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if raw and all(isinstance(value, list) for value in raw.values()):
        return {str(key): [row for row in value if isinstance(row, dict)] for key, value in raw.items()}
    if str(raw.get("mode")) == "triangulate":
        out: dict[str, list[dict[str, Any]]] = {}
        for row in raw.get("case_rows", []):
            if not isinstance(row, dict):
                continue
            rollout = row.get("rollout")
            if isinstance(rollout, list):
                out[str(row.get("custom_id", "case"))] = [item for item in rollout if isinstance(item, dict)]
        return out
    raise ValueError("Unsupported logitcart source")


def _sum_pattern_probs(candidates: dict[str, Any], patterns: list[str]) -> float:
    total = 0.0
    lowered = [pattern.lower() for pattern in patterns]
    for key, value in candidates.items():
        text = str(key).lower()
        if any(pattern in text for pattern in lowered):
            total += float(value)
    return total


def _logit_branch_label(margin: float) -> str:
    if margin > 0.0:
        return "target"
    if margin < 0.0:
        return "anti"
    return "neutral"
