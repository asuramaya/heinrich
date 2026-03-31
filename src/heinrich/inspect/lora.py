"""LoRA adapter loading and delta analysis."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..signal import Signal

def load_lora_deltas(path: Path | str, *, label: str = "lora") -> tuple[dict[str, np.ndarray], list[Signal]]:
    path = Path(path)
    signals = []
    from .safetensors import load_tensors
    try:
        tensors = load_tensors(path)
    except Exception:
        return {}, [Signal("lora_error", "inspect", label, str(path), 0.0, {"error": "failed to load"})]

    lora_a, lora_b = {}, {}
    for name, tensor in tensors.items():
        base = name.replace("base_model.model.", "").replace("base_model.", "")
        if "lora_A" in name:
            base = base.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
            lora_a[base] = tensor
        elif "lora_B" in name:
            base = base.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
            lora_b[base] = tensor

    deltas = {}
    for base_name in sorted(set(lora_a) & set(lora_b)):
        A = lora_a[base_name]
        B = lora_b[base_name]
        delta = B @ A
        deltas[base_name] = delta
        rank = min(A.shape[0], B.shape[0])
        signals.append(Signal("lora_rank", "inspect", label, base_name, float(rank),
                              {"shape_A": list(A.shape), "shape_B": list(B.shape)}))
        signals.append(Signal("lora_delta_norm", "inspect", label, base_name,
                              float(np.linalg.norm(delta)), {"rank": rank}))

    signals.append(Signal("lora_layer_count", "inspect", label, "total", float(len(deltas)), {}))
    return deltas, signals
