"""Per-head decomposition of multi-head attention deltas."""
from __future__ import annotations
import numpy as np
from ..signal import Signal


def decompose_heads(
    delta_qb: np.ndarray,
    num_heads: int,
    *,
    model_label: str = "model",
    layer: int = 0,
) -> list[Signal]:
    """Decompose q_b delta into per-head norms."""
    total_dim = delta_qb.shape[0]
    head_dim = total_dim // num_heads
    reshaped = delta_qb.reshape(num_heads, head_dim, -1)
    head_norms = np.linalg.norm(reshaped.reshape(num_heads, -1), axis=1)

    signals = []
    for head_idx in range(num_heads):
        signals.append(Signal(
            "per_head_norm", "diff", model_label, f"layer_{layer}_head_{head_idx}",
            float(head_norms[head_idx]),
            {"head_idx": head_idx, "layer": layer, "head_dim": head_dim},
        ))
    return signals


def head_trigger_tokens(
    delta_qb: np.ndarray,
    qa_base: np.ndarray,
    embeddings: np.ndarray,
    num_heads: int,
    head_idx: int,
    *,
    top_k: int = 10,
    model_label: str = "model",
    layer: int = 0,
) -> list[Signal]:
    """Find tokens that maximally activate a specific attention head's delta."""
    total_dim = delta_qb.shape[0]
    head_dim = total_dim // num_heads
    head_delta = delta_qb[head_idx * head_dim : (head_idx + 1) * head_dim]  # [head_dim, compressed]

    # Chain: head_delta @ qa_base @ embed.T -> [head_dim, vocab]
    sensitivity = head_delta @ qa_base @ embeddings.T  # [head_dim, vocab]
    per_token = np.linalg.norm(sensitivity, axis=0)  # [vocab]

    top_indices = np.argsort(per_token)[::-1][:top_k]
    mu, sigma = float(per_token.mean()), float(per_token.std())
    signals = []
    for rank, idx in enumerate(top_indices):
        z = (float(per_token[idx]) - mu) / sigma if sigma > 0 else 0.0
        signals.append(Signal(
            "head_trigger_token", "diff", model_label,
            f"layer_{layer}_head_{head_idx}_token_{int(idx)}",
            float(per_token[idx]),
            {"head_idx": head_idx, "token_id": int(idx), "z_score": z,
             "layer": layer, "rank": rank + 1},
        ))
    return signals
