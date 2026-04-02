"""Model self-analysis — capture internal states during inference."""
from __future__ import annotations
import numpy as np
from ..signal import Signal


def analyze_logits(logits: np.ndarray, *, label: str = "self", step: int = 0) -> list[Signal]:
    if logits.ndim == 1:
        probs = _softmax(logits)
    elif logits.ndim >= 2:
        probs = _softmax(logits[-1] if logits.ndim == 2 else logits[0, -1])
    else:
        return []
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
    max_prob = float(np.max(probs))
    top_k = np.argsort(probs)[::-1][:5]
    return [
        Signal("self_entropy", "inspect", label, f"step_{step}", entropy, {"max_prob": max_prob}),
        Signal("self_confidence", "inspect", label, f"step_{step}", max_prob, {}),
        Signal("self_top1_id", "inspect", label, f"step_{step}", float(top_k[0]),
               {"top5_ids": top_k.tolist(), "top5_probs": probs[top_k].tolist()}),
    ]

def analyze_hidden_states(hidden_states: list[np.ndarray], *, label: str = "self", step: int = 0) -> list[Signal]:
    signals = []
    norms = []
    for i, h in enumerate(hidden_states):
        arr = np.asarray(h, dtype=np.float64)
        if arr.ndim >= 2: arr = arr[-1] if arr.ndim == 2 else arr[0, -1]
        norm = float(np.linalg.norm(arr))
        norms.append(norm)
        signals.append(Signal("self_layer_norm", "inspect", label, f"step_{step}_layer_{i}", norm, {"layer": i}))
    if norms:
        signals.append(Signal("self_norm_mean", "inspect", label, f"step_{step}", float(np.mean(norms)), {"num_layers": len(norms)}))
        signals.append(Signal("self_norm_std", "inspect", label, f"step_{step}", float(np.std(norms)), {}))
    return signals

def analyze_attention(attention_weights: list[np.ndarray], *, label: str = "self", step: int = 0) -> list[Signal]:
    signals = []
    for i, attn in enumerate(attention_weights):
        arr = np.asarray(attn, dtype=np.float64)
        if arr.ndim == 4: arr = arr[0]
        if arr.ndim != 3: continue
        num_heads = arr.shape[0]
        head_norms = np.linalg.norm(arr.reshape(num_heads, -1), axis=1)
        max_head = int(np.argmax(head_norms))
        signals.append(Signal("self_attn_max_head", "inspect", label, f"step_{step}_layer_{i}", float(max_head),
                              {"head_norms": head_norms.tolist(), "layer": i}))
        last_row = arr[max_head, -1]
        ent = -float(np.sum(last_row * np.log2(last_row + 1e-12)))
        signals.append(Signal("self_attn_entropy", "inspect", label, f"step_{step}_layer_{i}_head_{max_head}", ent,
                              {"layer": i, "head": max_head}))
    return signals

def compute_activation_novelty(current: np.ndarray, prior: list[np.ndarray]) -> float:
    if not prior: return 1.0
    curr = current.flatten().astype(np.float64)
    cn = np.linalg.norm(curr)
    if cn == 0: return 0.0
    sims = []
    for p in prior:
        pf = p.flatten().astype(np.float64)
        pn = np.linalg.norm(pf)
        if pn == 0: continue
        sims.append(abs(float(np.dot(curr, pf) / (cn * pn))))
    return 1.0 - max(sims) if sims else 1.0

def _softmax(x: np.ndarray) -> np.ndarray:
    """Softmax — prefer heinrich.cartography.metrics.softmax for new code."""
    from ..cartography.metrics import softmax
    return softmax(x)
