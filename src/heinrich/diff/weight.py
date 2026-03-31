"""Weight-level comparison between models."""
from __future__ import annotations
from typing import Any
import numpy as np
from ..signal import Signal, SignalStore


def compare_tensors(
    lhs: dict[str, np.ndarray],
    rhs: dict[str, np.ndarray],
    *,
    lhs_label: str = "lhs",
    rhs_label: str = "rhs",
) -> list[Signal]:
    """Compare matching tensors between two weight dicts. Return signals."""
    signals = []
    lhs_names = set(lhs.keys())
    rhs_names = set(rhs.keys())
    common = sorted(lhs_names & rhs_names)
    only_lhs = sorted(lhs_names - rhs_names)
    only_rhs = sorted(rhs_names - lhs_names)

    for name in only_lhs:
        signals.append(Signal("only_in_lhs", "diff", lhs_label, name, 0.0, {}))
    for name in only_rhs:
        signals.append(Signal("only_in_rhs", "diff", rhs_label, name, 0.0, {}))

    for name in common:
        a = lhs[name].astype(np.float64)
        b = rhs[name].astype(np.float64)
        if a.shape != b.shape:
            signals.append(Signal("shape_mismatch", "diff", f"{lhs_label}_vs_{rhs_label}", name, 0.0,
                                  {"lhs_shape": list(a.shape), "rhs_shape": list(b.shape)}))
            continue
        delta = a - b
        l2 = float(np.linalg.norm(delta))
        max_abs = float(np.abs(delta).max()) if delta.size > 0 else 0.0
        is_identical = l2 == 0.0

        if is_identical:
            signals.append(Signal("identical_tensor", "diff", f"{lhs_label}_vs_{rhs_label}", name, 0.0, {}))
        else:
            frac_changed = float(np.mean(np.abs(delta) > 1e-6)) if delta.size > 0 else 0.0
            signals.append(Signal("delta_norm", "diff", f"{lhs_label}_vs_{rhs_label}", name, l2,
                                  {"max_abs": max_abs, "frac_changed": frac_changed}))
    return signals
