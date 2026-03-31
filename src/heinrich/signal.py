"""Signal schema and store — the spine of the heinrich pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class Signal:
    """A single typed measurement from a pipeline stage."""

    kind: str
    source: str
    model: str
    target: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SignalStore:
    """Accumulates signals across a pipeline run."""

    def __init__(self) -> None:
        self._signals: list[Signal] = []
        self._by_target: dict[str, list[Signal]] = {}  # index for O(1) target lookup

    def add(self, signal: Signal) -> None:
        self._signals.append(signal)
        self._by_target.setdefault(signal.target, []).append(signal)

    def extend(self, signals: Sequence[Signal]) -> None:
        for s in signals:
            self.add(s)

    def signals_for_target(self, target: str) -> list[Signal]:
        """Return all signals for a given target. O(1) index lookup."""
        return self._by_target.get(target, [])

    def __len__(self) -> int:
        return len(self._signals)

    def __iter__(self):
        return iter(self._signals)

    def filter(
        self,
        *,
        kind: str | None = None,
        source: str | None = None,
        model: str | None = None,
    ) -> list[Signal]:
        out = list(self._signals)
        if kind is not None:
            out = [s for s in out if s.kind == kind]
        if source is not None:
            out = [s for s in out if s.source == source]
        if model is not None:
            out = [s for s in out if s.model == model]
        return out

    def top(self, k: int = 10) -> list[Signal]:
        return sorted(self._signals, key=lambda s: s.value, reverse=True)[:k]

    def summary(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for s in self._signals:
            by_kind[s.kind] = by_kind.get(s.kind, 0) + 1
        return {"total": len(self._signals), "by_kind": by_kind}

    def to_json(self) -> str:
        return json.dumps([asdict(s) for s in self._signals], indent=2)

    @classmethod
    def from_json(cls, data: str) -> SignalStore:
        store = cls()
        for row in json.loads(data):
            store.add(Signal(**row))
        return store
