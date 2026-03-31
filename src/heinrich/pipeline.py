"""Pipeline runner — chains stages and manages the signal store."""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence

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


class Loop:
    """Iterative pipeline — runs stages repeatedly, accumulating signals."""

    def __init__(
        self,
        stages: Sequence[Stage],
        *,
        act: Stage | None = None,
        terminate: Callable[[SignalStore], bool] | None = None,
        max_iterations: int = 50,
    ) -> None:
        self._stages = list(stages)
        self._act = act
        self._terminate = terminate or (lambda store: False)
        self._max_iterations = max_iterations
        self.iterations_run: int = 0
        self.stages_run: list[str] = []

    def run(self, config: dict[str, Any], store: SignalStore | None = None) -> SignalStore:
        if store is None:
            store = SignalStore()
        self.iterations_run = 0
        self.stages_run = []
        for i in range(self._max_iterations):
            config["_iteration"] = i
            for stage in self._stages:
                stage.run(store, config)
                if stage.name not in self.stages_run:
                    self.stages_run.append(stage.name)
            if self._act is not None:
                self._act.run(store, config)
                if self._act.name not in self.stages_run:
                    self.stages_run.append(self._act.name)
            self.iterations_run = i + 1
            if self._terminate(store):
                break
        return store


def has_convergent_finding(store: SignalStore, *, threshold: float = 0.9, min_kinds: int = 2) -> bool:
    """Check if the store has any target with convergent signals above threshold."""
    from collections import defaultdict
    targets: dict[str, set[str]] = defaultdict(set)
    for s in store:
        if s.value > 0 and s.target:
            targets[s.target].add(s.kind)
    return any(len(kinds) >= min_kinds for kinds in targets.values())
