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


def parse_markdown_table(text: str) -> list[dict[str, str]]:
    """Parse a markdown table into a list of row dicts."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip() and '|' in l]
    if len(lines) < 3:
        return []
    headers = [h.strip() for h in lines[0].split('|') if h.strip()]
    rows = []
    for line in lines[2:]:  # skip header + separator
        cells = [c.strip() for c in line.split('|') if c.strip()]
        if len(cells) >= len(headers):
            rows.append(dict(zip(headers, cells[:len(headers)])))
    return rows


def score_against_leaderboard(
    leaderboard: list[dict[str, str]],
    submission_score: float,
    *,
    score_column: str = "Score",
    label: str = "leaderboard",
) -> list[Signal]:
    """Compare a submission score against a parsed leaderboard."""
    scores = []
    for row in leaderboard:
        try:
            scores.append(float(row.get(score_column, "inf")))
        except ValueError:
            continue
    if not scores:
        return []
    best = min(scores)
    worst = max(scores)
    rank = sum(1 for s in scores if s < submission_score) + 1
    return [
        Signal("leaderboard_rank", "bundle", label, "rank", float(rank),
               {"total": len(scores), "best": best, "worst": worst}),
        Signal("leaderboard_percentile", "bundle", label, "percentile",
               float(rank / len(scores)), {}),
        Signal("leaderboard_gap", "bundle", label, "gap_to_best",
               submission_score - best, {"best": best, "submission": submission_score}),
    ]
