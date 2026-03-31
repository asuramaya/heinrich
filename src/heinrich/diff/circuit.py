"""Full attention circuit simulation for trigger recovery."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from ..signal import Signal


def circuit_score_tokens(
    delta_qa: np.ndarray,
    delta_qb: np.ndarray,
    qa_base: np.ndarray,
    qb_base: np.ndarray,
    ln_weight: np.ndarray,
    embeddings: np.ndarray,
    token_ids: Sequence[int],
) -> float:
    """Score a set of token IDs through the full q_a -> ln -> q_b circuit."""
    x = sum(embeddings[tid] for tid in token_ids)
    if not isinstance(x, np.ndarray):
        # empty token_ids: sum of empty sequence is integer 0
        return 0.0
    term1 = qb_base @ (delta_qa @ x * ln_weight)
    term2 = delta_qb @ (qa_base @ x * ln_weight)
    return float(np.linalg.norm(term1 + term2))


def score_vocabulary(
    delta_qa: np.ndarray,
    delta_qb: np.ndarray,
    qa_base: np.ndarray,
    qb_base: np.ndarray,
    ln_weight: np.ndarray,
    embeddings: np.ndarray,
    *,
    model_label: str = "model",
    top_k: int = 50,
) -> list[Signal]:
    """Score every token in the vocabulary through the circuit. Return top-k signals."""
    vocab_size = embeddings.shape[0]
    scores = np.zeros(vocab_size)
    for tid in range(vocab_size):
        x = embeddings[tid]
        t1 = qb_base @ (delta_qa @ x * ln_weight)
        t2 = delta_qb @ (qa_base @ x * ln_weight)
        scores[tid] = float(np.linalg.norm(t1 + t2))

    top_indices = np.argsort(scores)[::-1][:top_k]
    signals = []
    for idx in top_indices:
        signals.append(Signal(
            "circuit_score", "diff", model_label, f"token_{int(idx)}",
            float(scores[idx]),
            {"token_id": int(idx), "rank": len(signals) + 1},
        ))
    return signals
