"""Weight patching — apply corrections, merge models, export."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..signal import Signal

def patch_weights(
    base: dict[str, np.ndarray], delta: dict[str, np.ndarray],
    *, mode: str = "subtract", scale: float = 1.0, label: str = "patch",
) -> tuple[dict[str, np.ndarray], list[Signal]]:
    patched = dict(base)
    signals = []
    for name in sorted(set(base) & set(delta)):
        if base[name].shape != delta[name].shape:
            continue
        before = float(np.linalg.norm(base[name]))
        if mode == "subtract":
            patched[name] = base[name] - delta[name] * scale
        elif mode == "add":
            patched[name] = base[name] + delta[name] * scale
        elif mode == "zero":
            patched[name] = np.zeros_like(base[name])
        elif mode == "scale":
            patched[name] = base[name] * scale
        after = float(np.linalg.norm(patched[name]))
        signals.append(Signal("patch_applied", "diff", label, name, abs(after - before),
                              {"mode": mode, "scale": scale, "before_norm": before, "after_norm": after}))
    return patched, signals

def merge_weights(
    models: list[dict[str, np.ndarray]], *, weights: list[float] | None = None, label: str = "merge",
) -> tuple[dict[str, np.ndarray], list[Signal]]:
    if not models:
        return {}, []
    w = weights or [1.0 / len(models)] * len(models)
    all_names = sorted(set().union(*[set(m) for m in models]))
    merged = {}
    signals = []
    for name in all_names:
        arrays = [(m[name], wi) for m, wi in zip(models, w) if name in m]
        if not arrays:
            continue
        merged[name] = sum(a * wi for a, wi in arrays)
        signals.append(Signal("merge_applied", "diff", label, name, float(np.linalg.norm(merged[name])), {"sources": len(arrays)}))
    signals.append(Signal("merge_total", "diff", label, "total", float(len(merged)), {"model_count": len(models)}))
    return merged, signals

def export_npz(weights: dict[str, np.ndarray], path: Path | str) -> None:
    np.savez_compressed(str(path), **weights)

def export_safetensors(weights: dict[str, np.ndarray], path: Path | str) -> None:
    from safetensors.numpy import save_file
    save_file({k: v.astype(np.float32) for k, v in weights.items()}, str(path))
