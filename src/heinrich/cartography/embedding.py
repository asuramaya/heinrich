"""Embedding space operations — search, cluster, and measure in token space.

The embedding table is the model's vocabulary mapped to vectors. This module
provides operations that work directly on this space without forward passes:
- Find tokens near a direction (which tokens align with the safety direction?)
- Cluster tokens by embedding similarity
- Measure embedding-space distances between concepts
- Score all tokens against a direction (fast shart-like analysis)

These operations are O(vocab_size) per query, not O(forward_pass).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class EmbeddingNeighbor:
    token_id: int
    token: str
    score: float       # projection onto direction
    distance: float    # L2 distance from query


def get_embedding_matrix(backend: Any) -> np.ndarray:
    """Extract the full embedding matrix [vocab_size, hidden_size] as numpy."""
    if hasattr(backend, 'model'):
        import mlx.core as mx
        inner = getattr(backend.model, "model", backend.model)
        return np.array(inner.embed_tokens.weight.astype(mx.float32))

    if hasattr(backend, 'hf_model'):
        embed = backend.hf_model.model.embed_tokens
        return embed.weight.float().cpu().numpy()

    raise NotImplementedError("Requires MLX or HF backend with model access")


def get_unembedding_matrix(backend: Any) -> np.ndarray:
    """Extract the unembedding (lm_head) matrix [vocab_size, hidden_size] as numpy."""
    if hasattr(backend, 'model'):
        import mlx.core as mx
        return np.array(backend.model.lm_head.weight.astype(mx.float32))

    if hasattr(backend, 'hf_model'):
        return backend.hf_model.lm_head.weight.float().cpu().numpy()

    raise NotImplementedError("Requires MLX or HF backend with model access")


def tokens_along_direction(
    backend: Any,
    direction: np.ndarray,
    *,
    top_k: int = 50,
    space: str = "embedding",
) -> tuple[list[EmbeddingNeighbor], list[EmbeddingNeighbor]]:
    """Find tokens most aligned and most anti-aligned with a direction.

    space: "embedding" uses embed_tokens, "unembedding" uses lm_head.

    Returns (positive_aligned, negative_aligned) — tokens that score
    highest and lowest when projected onto the direction.
    """
    if space == "embedding":
        matrix = get_embedding_matrix(backend)
    else:
        matrix = get_unembedding_matrix(backend)

    # Normalize direction
    d = direction / (np.linalg.norm(direction) + 1e-12)

    # Project all tokens
    scores = matrix @ d  # [vocab_size]

    # Top positive
    pos_idx = np.argsort(scores)[::-1][:top_k]
    positive = [
        EmbeddingNeighbor(
            token_id=int(i), token=backend.decode([int(i)]),
            score=float(scores[i]), distance=0.0,
        )
        for i in pos_idx
    ]

    # Top negative
    neg_idx = np.argsort(scores)[:top_k]
    negative = [
        EmbeddingNeighbor(
            token_id=int(i), token=backend.decode([int(i)]),
            score=float(scores[i]), distance=0.0,
        )
        for i in neg_idx
    ]

    return positive, negative


def nearest_tokens(
    backend: Any,
    vector: np.ndarray,
    *,
    top_k: int = 20,
    space: str = "embedding",
) -> list[EmbeddingNeighbor]:
    """Find tokens nearest to a vector in embedding space (by cosine similarity)."""
    if space == "embedding":
        matrix = get_embedding_matrix(backend)
    else:
        matrix = get_unembedding_matrix(backend)

    # Cosine similarity
    v_norm = vector / (np.linalg.norm(vector) + 1e-12)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    similarities = (matrix / norms) @ v_norm

    top_idx = np.argsort(similarities)[::-1][:top_k]
    return [
        EmbeddingNeighbor(
            token_id=int(i), token=backend.decode([int(i)]),
            score=float(similarities[i]),
            distance=float(np.linalg.norm(matrix[i] - vector)),
        )
        for i in top_idx
    ]


def score_all_tokens(
    backend: Any,
    direction: np.ndarray,
    *,
    space: str = "unembedding",
) -> np.ndarray:
    """Score every token against a direction. Returns [vocab_size] array.

    Uses unembedding space by default — this scores tokens by how much
    the model would want to output them if the residual were pushed
    along this direction.
    """
    if space == "embedding":
        matrix = get_embedding_matrix(backend)
    else:
        matrix = get_unembedding_matrix(backend)

    d = direction / (np.linalg.norm(direction) + 1e-12)
    return matrix @ d


def embedding_clusters(
    backend: Any,
    tokens: list[str],
    *,
    n_clusters: int = 5,
) -> list[dict]:
    """Cluster a set of tokens by their embedding similarity.

    Returns list of clusters with member tokens and centroid.
    """
    matrix = get_embedding_matrix(backend)
    token_ids = []
    for t in tokens:
        ids = backend.tokenize(t)
        if ids:
            token_ids.append(ids[0])

    if len(token_ids) < n_clusters:
        n_clusters = max(1, len(token_ids))

    embeddings = np.array([matrix[tid] for tid in token_ids])

    # Simple k-means
    rng = np.random.default_rng(42)
    centroids = embeddings[rng.choice(len(embeddings), n_clusters, replace=False)].copy()

    for _ in range(20):
        dists = np.array([
            [np.linalg.norm(embeddings[i] - c) for c in centroids]
            for i in range(len(embeddings))
        ])
        labels = dists.argmin(axis=1)
        for c in range(n_clusters):
            members = embeddings[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)

    clusters = []
    for c in range(n_clusters):
        member_idx = np.where(labels == c)[0]
        member_tokens = [tokens[i] for i in member_idx]
        member_ids = [token_ids[i] for i in member_idx]
        clusters.append({
            "cluster_id": c,
            "n_members": len(member_idx),
            "tokens": member_tokens[:10],
            "token_ids": member_ids[:10],
        })

    return clusters


def direction_vocabulary_overlap(
    backend: Any,
    dir_a: np.ndarray,
    dir_b: np.ndarray,
    *,
    top_k: int = 100,
) -> dict:
    """Measure how much two directions affect the same tokens.

    If the safety direction and political direction move the same tokens,
    they're entangled. If they move different tokens, they're independent.
    """
    scores_a = score_all_tokens(backend, dir_a)
    scores_b = score_all_tokens(backend, dir_b)

    top_a = set(np.argsort(np.abs(scores_a))[::-1][:top_k])
    top_b = set(np.argsort(np.abs(scores_b))[::-1][:top_k])

    overlap = top_a & top_b
    jaccard = len(overlap) / len(top_a | top_b) if top_a | top_b else 0

    return {
        "jaccard": round(jaccard, 4),
        "overlap_count": len(overlap),
        "overlap_tokens": [backend.decode([int(tid)]) for tid in list(overlap)[:20]],
        "a_only_count": len(top_a - top_b),
        "b_only_count": len(top_b - top_a),
    }
