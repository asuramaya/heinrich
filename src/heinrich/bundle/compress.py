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

    return {
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


def _build_structural(store: SignalStore) -> dict[str, Any]:
    arch_signals = store.filter(kind="architecture_type")
    arch_type = arch_signals[0].metadata.get("model_type", "unknown") if arch_signals else "unknown"

    config_signals = store.filter(kind="config_field")
    config = {s.target: s.value for s in config_signals}

    return {
        "architecture_type": arch_type,
        "config": config,
    }
