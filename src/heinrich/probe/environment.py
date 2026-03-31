"""Environment protocol for step-based observe-act loops."""
from __future__ import annotations
from typing import Any, Protocol
import numpy as np
from ..signal import Signal, SignalStore
from ..inspect.matrix import analyze_matrix, diff_matrices


class Environment(Protocol):
    """Protocol for step-based environments."""
    def observe(self) -> dict[str, Any]: ...
    def act(self, action: Any) -> dict[str, Any]: ...
    def score(self) -> float | None: ...
    def done(self) -> bool: ...


class MockEnvironment:
    """Test environment with scripted grid states."""

    def __init__(self, states: list[np.ndarray], scores: list[float] | None = None) -> None:
        self._states = states
        self._scores = scores or [0.0] * len(states)
        self._step = 0

    def observe(self) -> dict[str, Any]:
        idx = min(self._step, len(self._states) - 1)
        return {"grid": self._states[idx], "step": self._step}

    def act(self, action: Any) -> dict[str, Any]:
        self._step += 1
        return self.observe()

    def score(self) -> float | None:
        idx = min(self._step, len(self._scores) - 1)
        return self._scores[idx]

    def done(self) -> bool:
        return self._step >= len(self._states)


class ObserveStage:
    """Pipeline stage that observes an environment and emits grid signals."""
    name = "observe"

    def run(self, store: SignalStore, config: dict[str, Any]) -> None:
        env = config.get("environment")
        if env is None:
            return
        label = config.get("model_label", "env")
        iteration = config.get("_iteration", 0)
        obs = env.observe()
        grid = obs.get("grid")
        if grid is not None and isinstance(grid, np.ndarray):
            signals = analyze_matrix(grid, label=label, name=f"frame_{iteration}")
            store.extend(signals)
            # Store grid for diffing with next frame
            prev_key = "_prev_grid"
            if prev_key in config and config[prev_key] is not None:
                delta_signals = diff_matrices(config[prev_key], grid, label=label, name=f"delta_{iteration}")
                store.extend(delta_signals)
            config[prev_key] = grid.copy()
        # Score signal
        s = env.score()
        if s is not None:
            store.add(Signal("env_score", "observe", label, f"step_{iteration}", float(s), {}))


class ActStage:
    """Pipeline stage that takes an action in the environment."""
    name = "act"

    def run(self, store: SignalStore, config: dict[str, Any]) -> None:
        env = config.get("environment")
        if env is None:
            return
        action = config.get("next_action", 1)
        env.act(action)
        store.add(Signal("action_taken", "act", config.get("model_label", "env"),
                         f"step_{config.get('_iteration', 0)}", float(action) if isinstance(action, (int, float)) else 0.0, {}))
