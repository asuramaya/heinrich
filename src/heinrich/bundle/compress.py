"""Compress a SignalStore into a context-window-optimized JSON document."""

from __future__ import annotations

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
