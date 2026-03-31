"""Compress a SignalStore into a context-window-optimized JSON document."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import heinrich
from ..signal import SignalStore


def compress_store(
    store: SignalStore,
    *,
    stages_run: list[str] | None = None,
    models: list[str] | None = None,
    base: str | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    summary = store.summary()
    top_signals = store.top(k=top_k)

    result = {
        "heinrich_version": heinrich.__version__,
        "models": models or _infer_models(store),
        "base": base,
        "stages_run": stages_run or [],
        "structural": _build_structural(store),
        "findings": _build_findings(store),
        "signals_summary": {
            "total": summary["total"],
            "by_kind": summary["by_kind"],
            "top_10": [
                {
                    "kind": s.kind,
                    "source": s.source,
                    "model": s.model,
                    "target": s.target,
                    "value": s.value,
                }
                for s in top_signals
            ],
        },
    }
    self_analysis = _build_self_analysis(store)
    if self_analysis is not None:
        result["self_analysis"] = self_analysis
    trajectory = _build_trajectory(store)
    if trajectory is not None:
        result["trajectory"] = trajectory
    return result


def _build_findings(store: SignalStore, top_k: int = 5) -> list[dict[str, Any]]:
    """Generate ranked findings from convergent signals."""
    target_signals: dict[str, list] = defaultdict(list)
    for s in store:
        if s.target and s.value > 0:
            target_signals[s.target].append(s)

    findings = []
    for target, signals in target_signals.items():
        kinds = {s.kind for s in signals}
        sources = {s.source for s in signals}
        if len(kinds) < 2:
            continue  # need convergence from multiple signal types
        mean_value = sum(s.value for s in signals) / len(signals)
        findings.append({
            "target": target,
            "converging_signals": len(signals),
            "signal_kinds": sorted(kinds),
            "sources": sorted(sources),
            "mean_value": mean_value,
            "confidence": min(1.0, len(kinds) / 5.0),  # rough heuristic
        })

    findings.sort(key=lambda f: (f["converging_signals"], f["mean_value"]), reverse=True)
    return findings[:top_k]


def _infer_models(store: SignalStore) -> list[str]:
    models: set[str] = set()
    for s in store:
        if s.model:
            models.add(s.model)
    return sorted(models)


def _build_self_analysis(store: SignalStore) -> dict[str, Any] | None:
    entropy = store.filter(kind="self_entropy")
    confidence = store.filter(kind="self_confidence")
    if not entropy and not confidence: return None
    layer_norms = store.filter(kind="self_layer_norm")
    norm_mean = store.filter(kind="self_norm_mean")
    attn_heads = store.filter(kind="self_attn_max_head")
    return {
        "entropy": entropy[-1].value if entropy else None,
        "confidence": confidence[-1].value if confidence else None,
        "layer_count": len(set(s.metadata.get("layer", -1) for s in layer_norms)),
        "norm_mean": norm_mean[-1].value if norm_mean else None,
        "dominant_heads": [int(s.value) for s in attn_heads[-3:]],
        "observation_count": len(entropy),
    }


def _build_trajectory(store: SignalStore) -> dict[str, Any] | None:
    scores = store.filter(kind="env_score")
    actions = store.filter(kind="action_taken")
    deltas = store.filter(kind="matrix_delta_norm")
    if not scores and not actions: return None
    import numpy as np
    return {
        "turns": len(actions),
        "scores": [s.value for s in scores],
        "score_trend": "improving" if len(scores) >= 2 and scores[-1].value > scores[0].value else ("declining" if len(scores) >= 2 and scores[-1].value < scores[0].value else "flat"),
        "total_grid_changes": len(deltas),
        "mean_delta": float(np.mean([s.value for s in deltas])) if deltas else 0.0,
    }


def _build_structural(store: SignalStore) -> dict[str, Any]:
    arch_signals = store.filter(kind="architecture_type")
    arch_type = arch_signals[0].metadata.get("model_type", "unknown") if arch_signals else "unknown"

    config_signals = store.filter(kind="config_field")
    config = {s.target: s.value for s in config_signals}

    return {
        "architecture_type": arch_type,
        "config": config,
    }
