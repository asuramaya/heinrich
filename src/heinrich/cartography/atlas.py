"""Persistent knob → effect map."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .perturb import PerturbResult
from .surface import Knob
from ..signal import Signal, SignalStore


class Atlas:
    """Maps knobs to their behavioral effects."""

    def __init__(self) -> None:
        self.results: dict[str, PerturbResult] = {}

    def add(self, result: PerturbResult) -> None:
        self.results[result.knob.id] = result

    def add_all(self, results: list[PerturbResult]) -> None:
        for r in results:
            self.add(r)

    def __len__(self) -> int:
        return len(self.results)

    def top_by_kl(self, k: int = 20) -> list[PerturbResult]:
        return sorted(self.results.values(), key=lambda r: r.kl_divergence, reverse=True)[:k]

    def top_by_entropy_delta(self, k: int = 20) -> list[PerturbResult]:
        return sorted(self.results.values(), key=lambda r: abs(r.entropy_delta), reverse=True)[:k]

    def top_token_changers(self) -> list[PerturbResult]:
        return [r for r in self.results.values() if r.top_token_changed]

    def for_layer(self, layer: int) -> list[PerturbResult]:
        return [r for r in self.results.values() if r.knob.layer == layer]

    def to_signals(self, store: SignalStore, label: str = "atlas") -> None:
        for r in self.results.values():
            store.add(Signal("atlas_entry", "cartography", label, r.knob.id, r.kl_divergence,
                             {"entropy_delta": r.entropy_delta, "top_changed": r.top_token_changed,
                              "layer": r.knob.layer, "kind": r.knob.kind}))

    def save(self, path: Path | str) -> None:
        data = []
        for r in self.results.values():
            data.append({
                "knob_id": r.knob.id, "knob_kind": r.knob.kind,
                "layer": r.knob.layer, "index": r.knob.index,
                "mode": r.mode,
                "baseline_entropy": r.baseline_entropy,
                "perturbed_entropy": r.perturbed_entropy,
                "entropy_delta": r.entropy_delta,
                "kl_divergence": r.kl_divergence,
                "top_token_changed": r.top_token_changed,
                "baseline_top": r.baseline_top,
                "perturbed_top": r.perturbed_top,
            })
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> Atlas:
        atlas = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for entry in data:
            knob = Knob(entry["knob_id"], entry["knob_kind"], entry["layer"], entry.get("index", 0))
            result = PerturbResult(
                knob=knob, mode=entry.get("mode", "zero"),
                baseline_entropy=entry["baseline_entropy"],
                perturbed_entropy=entry["perturbed_entropy"],
                entropy_delta=entry["entropy_delta"],
                kl_divergence=entry["kl_divergence"],
                top_token_changed=entry["top_token_changed"],
                baseline_top=entry.get("baseline_top", 0),
                perturbed_top=entry.get("perturbed_top", 0),
            )
            atlas.add(result)
        return atlas
