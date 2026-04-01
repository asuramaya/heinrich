"""Named high-level controls derived from the behavioral manifold."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .manifold import BehaviorCluster
from .surface import Knob


@dataclass
class Dial:
    """A named control that adjusts multiple knobs together."""
    name: str
    knob_ids: list[str]
    default_mode: str = "scale"
    value: float = 1.0    # 0.0 = fully suppress, 1.0 = normal, 2.0 = amplify


class ControlPanel:
    """Collection of named dials for controlling model behavior."""

    def __init__(self) -> None:
        self.dials: dict[str, Dial] = {}

    def add_dial(self, dial: Dial) -> None:
        self.dials[dial.name] = dial

    def set(self, name: str, value: float) -> None:
        if name in self.dials:
            self.dials[name] = Dial(
                name=self.dials[name].name,
                knob_ids=self.dials[name].knob_ids,
                default_mode=self.dials[name].default_mode,
                value=value,
            )

    def active_dials(self) -> list[Dial]:
        return [d for d in self.dials.values() if d.value != 1.0]

    @classmethod
    def from_clusters(cls, clusters: list[BehaviorCluster]) -> ControlPanel:
        panel = cls()
        for i, cluster in enumerate(clusters):
            name = cluster.name if not cluster.name.startswith("cluster_") else f"behavior_{i}"
            panel.add_dial(Dial(name=name, knob_ids=cluster.knob_ids))
        return panel

    def summary(self) -> dict[str, Any]:
        return {
            "dials": {name: {"knobs": len(d.knob_ids), "value": d.value}
                      for name, d in self.dials.items()},
            "active": [d.name for d in self.active_dials()],
        }
