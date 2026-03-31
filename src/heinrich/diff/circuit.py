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


def score_vocabulary_vectorized(
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
    """Vectorized vocabulary scoring — handles 128K+ tokens efficiently."""
    # term1[i] = qb_base @ (delta_qa @ embed[i] * ln)
    # term2[i] = delta_qb @ (qa_base @ embed[i] * ln)
    # Batch: delta_qa @ embed.T -> [compressed, vocab], then ln scale, then qb_base @
    compressed_delta = delta_qa @ embeddings.T  # [compressed, vocab]
    compressed_delta *= ln_weight[:, np.newaxis]  # broadcast ln
    term1 = qb_base @ compressed_delta  # [heads_dim, vocab]

    compressed_base = qa_base @ embeddings.T  # [compressed, vocab]
    compressed_base *= ln_weight[:, np.newaxis]
    term2 = delta_qb @ compressed_base  # [heads_dim, vocab]

    combined = term1 + term2  # [heads_dim, vocab]
    scores = np.linalg.norm(combined, axis=0)  # [vocab]

    top_indices = np.argsort(scores)[::-1][:top_k]
    signals = []
    mu, sigma = float(scores.mean()), float(scores.std())
    for rank, idx in enumerate(top_indices):
        z = (float(scores[idx]) - mu) / sigma if sigma > 0 else 0.0
        signals.append(Signal(
            "circuit_score", "diff", model_label, f"token_{int(idx)}",
            float(scores[idx]),
            {"token_id": int(idx), "z_score": z, "rank": rank + 1},
        ))
    return signals


def aggregate_circuit_scores(
    layers: list[dict[str, np.ndarray]],
    embeddings: np.ndarray,
    ln_weights: list[np.ndarray],
    *,
    model_label: str = "model",
    top_k: int = 50,
) -> list[Signal]:
    """Aggregate circuit scores across multiple layers.

    Each entry in layers is a dict with keys: delta_qa, delta_qb, qa_base, qb_base.
    """
    vocab_size = embeddings.shape[0]
    total_scores = np.zeros(vocab_size)

    for layer_data, ln_w in zip(layers, ln_weights):
        compressed_delta = layer_data["delta_qa"] @ embeddings.T
        compressed_delta *= ln_w[:, np.newaxis]
        term1 = layer_data["qb_base"] @ compressed_delta

        compressed_base = layer_data["qa_base"] @ embeddings.T
        compressed_base *= ln_w[:, np.newaxis]
        term2 = layer_data["delta_qb"] @ compressed_base

        combined = term1 + term2
        total_scores += np.linalg.norm(combined, axis=0)

    mu, sigma = float(total_scores.mean()), float(total_scores.std())
    top_indices = np.argsort(total_scores)[::-1][:top_k]
    signals = []
    for rank, idx in enumerate(top_indices):
        z = (float(total_scores[idx]) - mu) / sigma if sigma > 0 else 0.0
        signals.append(Signal(
            "circuit_score_agg", "diff", model_label, f"token_{int(idx)}",
            float(total_scores[idx]),
            {"token_id": int(idx), "z_score": z, "rank": rank + 1, "num_layers": len(layers)},
        ))
    return signals
