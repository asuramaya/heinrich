"""Signal scoring, ranking, and fusion."""
from __future__ import annotations
from typing import Any, Sequence
from ..signal import Signal, SignalStore


def rank_signals(store: SignalStore, *, top_k: int = 20) -> list[dict[str, Any]]:
    """Rank all signals by value, return top-k with context."""
    signals = sorted(store, key=lambda s: s.value, reverse=True)[:top_k]
    return [
        {
            "rank": i + 1,
            "kind": s.kind,
            "source": s.source,
            "model": s.model,
            "target": s.target,
            "value": s.value,
            "metadata": s.metadata,
        }
        for i, s in enumerate(signals)
    ]


def compute_convergence(store: SignalStore, target: str) -> dict[str, Any]:
    """How many different signal kinds point at the same target?"""
    matching = [s for s in store if s.target == target]
    kinds = {s.kind for s in matching}
    sources = {s.source for s in matching}
    return {
        "target": target,
        "signal_count": len(matching),
        "kind_count": len(kinds),
        "source_count": len(sources),
        "kinds": sorted(kinds),
        "mean_value": sum(s.value for s in matching) / len(matching) if matching else 0.0,
    }


def fuse_signals(stores: Sequence[SignalStore]) -> SignalStore:
    """Merge multiple signal stores into one."""
    merged = SignalStore()
    for store in stores:
        for signal in store:
            merged.add(signal)
    return merged
