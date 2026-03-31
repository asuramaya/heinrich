"""Inspect stage — structural analysis of weight data."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from ..signal import Signal, SignalStore
from .family import classify_tensor_family
from .spectral import spectral_stats

__all__ = ["InspectStage"]

class InspectStage:
    name = "inspect"
    def run(self, store: SignalStore, config: dict[str, Any]) -> None:
        weights_path = config.get("weights_path")
        if weights_path is None:
            return
        label = config.get("model_label", "model")
        tensors = _load_tensors(Path(weights_path))
        for tensor_name, arr in tensors.items():
            family = classify_tensor_family(tensor_name)
            store.add(Signal("tensor_family", "inspect", label, tensor_name, 0.0, {"family": family}))
            if arr.ndim == 2:
                stats = spectral_stats(arr.astype(np.float64), topk=16)
                store.add(Signal("spectral_sigma1", "inspect", label, tensor_name, stats["sigma1"], stats))
                store.add(Signal("fro_norm", "inspect", label, tensor_name, stats["fro_norm"], {}))
                singular = np.linalg.svd(arr.astype(np.float64), compute_uv=False)
                energy = singular ** 2
                total = energy.sum()
                rank_95 = int(np.searchsorted(np.cumsum(energy) / total, 0.95)) + 1 if total > 0 else 0
                store.add(Signal("rank_at_95", "inspect", label, tensor_name, float(rank_95), {}))

def _load_tensors(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {name: np.array(data[name]) for name in data.files}
    return {}
