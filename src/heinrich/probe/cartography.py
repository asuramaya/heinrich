"""Cartography probes -- ported from conker-detect cartography.py and framecart.py."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .trigger_core import build_identity_suite, describe_provider, normalize_case
from .triangulate import load_case_suite, run_triangulation_report


DEFAULT_CARTOGRAPHY_IDENTITY_TEMPLATES = ("hello", "hi", "you_are")
DEFAULT_CARTOGRAPHY_CONTROL_QUERIES = (" Assistant",)
DEFAULT_FRAME_STATE_FAMILIES = ("SELF", "CONTEXT", "STOP", "AVOID")


def run_cartography_report(
    provider: Any,
    case: dict[str, Any],
    *,
    models: Sequence[str],
    tokenizer_ref: str | Path | None = None,
    model_ref: str | Path | None = None,
    module_names: Sequence[str] | None = None,
    meta_families: Sequence[str] | None = None,
    chain_lines: Sequence[str] | None = None,
    meta_probe_placement: str = "append",
    meta_cross_model: bool = False,
    meta_include_base: bool = True,
    repeats: int = 1,
    seed_pattern: str = r"^[A-Za-z][A-Za-z'\-]{2,15}$",
    seed_limit: int = 16,
    seed_verify_topk: int = 0,
    seed_batch_size: int = 16,
    static_sources: Sequence[str | Path | dict[str, Any]] | None = None,
    control_queries: Sequence[str] | None = None,
    identity_templates: Sequence[str] | None = None,
    rollout_steps: int = 6,
    rollout_topk: int = 8,
    dtype: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run a full cartography report: meta scan, seed scan, prior rank, triangulation, hologram.

    Requires a local provider with list_modules() support and a reachable model.
    """
    from .meta import run_meta_probe_scan
    from .seedscan import scan_seed_candidates
    from .hologram import run_hologram_report
    from ..bundle.priors import rank_trigger_priors

    normalized = normalize_case(case, default_id="case")
    unique_models = [str(model) for model in dict.fromkeys(models) if str(model)]
    if not unique_models:
        raise ValueError("cartography requires at least one model")
    resolved_tokenizer_ref = str(
        tokenizer_ref
        or getattr(provider, "_tokenizer_ref", "")
        or getattr(getattr(provider, "_tokenizer", None), "name_or_path", "")
    )
    resolved_model_ref = str(model_ref or getattr(provider, "_model_ref", "") or "")
    if not resolved_tokenizer_ref or not resolved_model_ref:
        raise ValueError("cartography requires tokenizer_ref/model_ref directly or via a local provider")
    resolved_modules = list(module_names or _suggest_modules(provider))

    metascan = run_meta_probe_scan(
        provider,
        normalized,
        models=unique_models,
        families=meta_families,
        chain_lines=chain_lines,
        probe_placement=meta_probe_placement,
        chat_repeats=repeats,
        cross_model=meta_cross_model,
        include_base=meta_include_base,
    )
    seedscan = scan_seed_candidates(
        tokenizer_ref=resolved_tokenizer_ref,
        model_ref=resolved_model_ref,
        pattern=seed_pattern,
        limit=seed_limit,
        verify_topk=seed_verify_topk,
        batch_size=seed_batch_size,
        dtype=dtype,
        device=device,
    )
    recovered_surfaces = [
        str(row["surface"])
        for row in seedscan.get("rows", [])[: min(5, len(seedscan.get("rows", [])))]
    ]
    priorrank = rank_trigger_priors(
        tokenizer_ref=resolved_tokenizer_ref,
        candidates=recovered_surfaces or ["Assistant"],
        model_ref=resolved_model_ref,
        static_sources=list(static_sources or []),
        dtype=dtype,
        device=device,
    )
    top_seed = recovered_surfaces[0] if recovered_surfaces else None
    system_identity = str(priorrank.get("system_identity") or "").strip()
    effective_controls = list(control_queries or DEFAULT_CARTOGRAPHY_CONTROL_QUERIES)
    if system_identity:
        system_query = f" {system_identity}"
        if system_query not in effective_controls:
            effective_controls.append(system_query)

    identity_suite = None
    triangulate = None
    hologram = None
    if top_seed:
        identity_suite = build_identity_suite(
            [top_seed],
            templates=list(identity_templates or DEFAULT_CARTOGRAPHY_IDENTITY_TEMPLATES),
            base_text="Hello Assistant",
            base_id="cartography-base",
        )
        triangulate = run_triangulation_report(
            provider,
            identity_suite,
            model=unique_models[0],
            tokenizer_ref=resolved_tokenizer_ref,
            model_ref=resolved_model_ref,
            module_names=resolved_modules,
            candidate_strings=[f" {top_seed}", " not", *effective_controls],
            rollout_steps=rollout_steps,
            repeats=repeats,
            rollout_topk=rollout_topk,
        )
        if resolved_modules:
            hologram = run_hologram_report(
                provider,
                identity_suite,
                model=unique_models[0],
                target_queries=[f" {top_seed}"],
                control_queries=effective_controls,
                module_names=resolved_modules,
                tokenizer_ref=resolved_tokenizer_ref,
                model_ref=resolved_model_ref,
                rollout_steps=rollout_steps,
                repeats=repeats,
                rollout_topk=rollout_topk,
                dtype=dtype,
                device=device,
            )

    synthesis = _build_cartography_synthesis(
        metascan=metascan,
        seedscan=seedscan,
        priorrank=priorrank,
        triangulate=triangulate,
        hologram=hologram,
        top_seed=top_seed,
    )
    return {
        "mode": "cartography",
        "provider": describe_provider(provider),
        "base_case": normalized,
        "models": unique_models,
        "tokenizer_ref": resolved_tokenizer_ref,
        "model_ref": resolved_model_ref,
        "module_names": resolved_modules,
        "meta_probe_placement": meta_probe_placement,
        "seed_pattern": seed_pattern,
        "seed_limit": int(seed_limit),
        "seedscan": seedscan,
        "priorrank": priorrank,
        "metascan": metascan,
        "identity_suite": identity_suite,
        "triangulate": triangulate,
        "hologram": hologram,
        "synthesis": synthesis,
    }


def run_frame_cartography(
    provider: Any,
    suite: dict[str, Any] | str | Path,
    *,
    model: str,
    tokenizer_ref: str | Path | None = None,
    model_ref: str | Path | None = None,
    module_names: Sequence[str] | None = None,
    repeats: int = 1,
    rollout_steps: int = 6,
    rollout_topk: int = 8,
    state_families: Sequence[str] | None = None,
    state_topk: int = 3,
    min_overlap: float = 0.0,
    rubric: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Run frame cartography: triangulation, sign-flip analysis, logit cartography, and state probes."""
    from .rubric import load_rubric
    from .identity import run_state_cartography
    from ..bundle.atlas import run_signflip_report, run_logit_cartography

    resolved_suite = load_case_suite(suite)
    resolved_rubric = load_rubric(rubric) if isinstance(rubric, str) else rubric
    triage = run_triangulation_report(
        provider,
        resolved_suite,
        model=model,
        tokenizer_ref=tokenizer_ref,
        model_ref=model_ref,
        module_names=list(module_names or []),
        repeats=repeats,
        rollout_steps=rollout_steps,
        rollout_topk=rollout_topk,
    )
    signflip = run_signflip_report(triage, min_overlap=min_overlap)
    logitcart = run_logit_cartography(triage)

    suite_cases = [resolved_suite["base_case"], *list(resolved_suite.get("variants", []))]
    case_lookup = {case["custom_id"]: case for case in suite_cases}

    selected_rows = [
        row
        for row in sorted(
            [row for row in triage.get("case_rows", []) if not bool(row.get("is_control"))],
            key=lambda item: float(item.get("aggregate_trigger_score", 0.0)),
            reverse=True,
        )[: max(int(state_topk), 0)]
    ]

    state_rows = []
    for row in selected_rows:
        case_id = str(row.get("custom_id"))
        case = case_lookup.get(case_id)
        if case is None:
            continue
        report = run_state_cartography(
            provider,
            case,
            model=model,
            families=state_families or DEFAULT_FRAME_STATE_FAMILIES,
            repeats=repeats,
            rubric=resolved_rubric,
        )
        state_rows.append({
            "custom_id": case_id,
            "prompt_text": str(row.get("prompt_text", "")),
            "aggregate_trigger_score": float(row.get("aggregate_trigger_score", 0.0)),
            "output_mode": dict(row.get("output_mode", {})),
            "rollout_summary": dict(row.get("rollout_summary", {})),
            "top_probes": list(report.get("top_probes", [])),
            "statecartography": report,
        })

    adoption_case = _top_case_by_label(triage.get("case_rows", []), "adoption")
    denial_case = _top_case_by_label(triage.get("case_rows", []), "denial")

    return {
        "mode": "framecart",
        "provider": describe_provider(provider),
        "model": model,
        "repeats": int(repeats),
        "triangulate": triage,
        "signflip": signflip,
        "logitcart": logitcart,
        "state_rows": state_rows,
        "summary": {
            "top_adoption_case": adoption_case,
            "top_denial_case": denial_case,
            "top_flip_pair": signflip.get("flip_pairs", [None])[0] if signflip.get("flip_pairs") else None,
            "top_state_case": state_rows[0] if state_rows else None,
            "recommended_confirmation_prompts": [
                row["prompt_text"]
                for row in [adoption_case, denial_case]
                if isinstance(row, dict) and str(row.get("prompt_text", ""))
            ],
        },
    }


def _suggest_modules(provider: Any) -> list[str]:
    if not hasattr(provider, "list_modules") or not callable(provider.list_modules):
        return []
    try:
        modules = [str(name) for name in provider.list_modules()]
    except Exception:
        return []
    preferred_suffixes = (
        ".self_attn.q_proj",
        ".self_attn.o_proj",
        ".mlp.down_proj",
        ".mlp.gate_proj",
        ".self_attn.q_a_layernorm",
        ".self_attn.q_a_proj",
        ".self_attn.q_b_proj",
    )
    selected: list[str] = []
    for suffix in preferred_suffixes:
        for name in modules:
            if name.endswith(suffix):
                selected.append(name)
                break
    if selected:
        return selected[:4]
    return modules[:4]


def _build_cartography_synthesis(
    *,
    metascan: dict[str, Any],
    seedscan: dict[str, Any],
    priorrank: dict[str, Any],
    triangulate: dict[str, Any] | None,
    hologram: dict[str, Any] | None,
    top_seed: str | None,
) -> dict[str, Any]:
    meta_rows = list(metascan.get("scored", {}).get("variants", []))
    seed_rows = list(seedscan.get("rows", []))
    prior_rows = list(priorrank.get("candidates", []))
    tri_rows = list([] if triangulate is None else triangulate.get("case_rows", []))
    holo_rows = list([] if hologram is None else hologram.get("module_holograms", []))
    return {
        "top_seed": top_seed,
        "top_seed_score": None if not seed_rows else float(seed_rows[0].get("aggregate_score", 0.0)),
        "top_seeds": [
            {
                "surface": str(row["surface"]),
                "aggregate_score": float(row.get("aggregate_score", 0.0)),
                "generic_identity_probability": float(row.get("generic_identity_probability", 0.0)),
            }
            for row in seed_rows[:5]
        ],
        "top_meta_variants": [
            {
                "variant_id": str(row.get("variant_id")),
                "description": str(row.get("description", "")),
                "max_combined_score": float(row.get("max_combined_score", row.get("max_pairwise_score", 0.0))),
                "family": str(row.get("metadata", {}).get("family", row.get("case", {}).get("metadata", {}).get("family", ""))),
                "probe_kind": str(row.get("metadata", {}).get("probe_kind", "")),
            }
            for row in meta_rows[:5]
        ],
        "top_prior_candidates": [
            {
                "candidate": str(row.get("candidate")),
                "score": float(row.get("score", 0.0)),
            }
            for row in prior_rows[:5]
        ],
        "top_identity_cases": [
            {
                "custom_id": str(row.get("custom_id")),
                "prompt_text": str(row.get("prompt_text", "")),
                "output_label": str(row.get("output_mode", {}).get("label", "")),
                "aggregate_trigger_score": float(row.get("aggregate_trigger_score", 0.0)),
            }
            for row in tri_rows[:5]
        ],
        "top_modules": [
            {
                "module_name": str(row.get("module_name")),
                "study_score": float(row.get("study_score", 0.0)),
                "exploit_score": float(row.get("exploit_score", 0.0)),
                "triangulation_activation_score": float(row.get("triangulation_activation_score", 0.0)),
            }
            for row in holo_rows[:5]
        ],
        "recommended_cases": _recommended_cases(seedscan, metascan),
    }


def _recommended_cases(seedscan: dict[str, Any], metascan: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in seedscan.get("top_cases", [])[:3]:
        out.append({"kind": "seedscan", **row})
    for row in metascan.get("scored", {}).get("variants", [])[:3]:
        out.append({
            "kind": "metascan",
            "variant_id": str(row.get("variant_id")),
            "description": str(row.get("description", "")),
            "case": row.get("case"),
        })
    return out


def _top_case_by_label(rows: Sequence[dict[str, Any]], label: str) -> dict[str, Any] | None:
    matches = [row for row in rows if str(row.get("output_mode", {}).get("label")) == label]
    if not matches:
        return None
    ranked = sorted(matches, key=lambda item: float(item.get("aggregate_trigger_score", 0.0)), reverse=True)
    top = dict(ranked[0])
    return {
        "custom_id": str(top.get("custom_id")),
        "prompt_text": str(top.get("prompt_text", "")),
        "aggregate_trigger_score": float(top.get("aggregate_trigger_score", 0.0)),
        "output_mode": dict(top.get("output_mode", {})),
        "rollout_summary": dict(top.get("rollout_summary", {})),
    }
