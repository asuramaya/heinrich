"""Diff stage — compare models and compute deltas."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from ..signal import Signal, SignalStore
from .weight import compare_tensors
from .circuit import score_vocabulary
from .embedding import project_delta_onto_embeddings

__all__ = ["DiffStage"]


class DiffStage:
    name = "diff"

    def run(self, store: SignalStore, config: dict[str, Any]) -> None:
        lhs_path = config.get("lhs_weights")
        rhs_path = config.get("rhs_weights")
        if lhs_path is None or rhs_path is None:
            return
        lhs_label = config.get("lhs_label", "lhs")
        rhs_label = config.get("rhs_label", "rhs")
        lhs = _load_npz(Path(lhs_path))
        rhs = _load_npz(Path(rhs_path))

        # Weight comparison
        signals = compare_tensors(lhs, rhs, lhs_label=lhs_label, rhs_label=rhs_label)
        store.extend(signals)

        # Embedding projection on deltas (if 2D tensors exist)
        for name in sorted(set(lhs) & set(rhs)):
            a = lhs[name].astype(np.float64)
            b = rhs[name].astype(np.float64)
            if a.shape != b.shape or a.ndim != 2:
                continue
            delta = a - b
            if float(np.abs(delta).max()) < 1e-8:
                continue
            # SVD for rank signal
            singular = np.linalg.svd(delta, compute_uv=False)
            energy = singular ** 2
            total = energy.sum()
            if total > 0:
                cumulative = np.cumsum(energy) / total
                rank_95 = int(np.searchsorted(cumulative, 0.95)) + 1
            else:
                rank_95 = 0
            store.add(Signal("delta_rank", "diff", f"{lhs_label}_vs_{rhs_label}", name,
                             float(rank_95), {"sigma1": float(singular[0])}))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.array(data[name]) for name in data.files}
