"""Pipeline runner — chains stages and manages the signal store."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from .signal import SignalStore


class Stage(Protocol):
    """Protocol for pipeline stages."""

    name: str

    def run(self, store: SignalStore, config: dict[str, Any]) -> None: ...


class Pipeline:
    """Runs a sequence of stages, accumulating signals in a shared store."""

    def __init__(self, stages: Sequence[Stage]) -> None:
        self._stages = list(stages)
        self.stages_run: list[str] = []

    def run(self, config: dict[str, Any]) -> SignalStore:
        store = SignalStore()
        self.stages_run = []
        for stage in self._stages:
            stage.run(store, config)
            self.stages_run.append(stage.name)
        return store
