"""Tests for heinrich.cartography.embedding."""
import numpy as np
from unittest.mock import MagicMock
from heinrich.cartography.embedding import (
    tokens_along_direction, nearest_tokens,
    score_all_tokens, direction_vocabulary_overlap,
    EmbeddingNeighbor,
)


def _make_mock_backend(vocab_size=100, hidden_size=16):
    """Create a mock backend with a fake embedding matrix."""
    backend = MagicMock()
    # Create a simple embedding matrix where token i has value i in dim 0
    matrix = np.random.randn(vocab_size, hidden_size).astype(np.float32)
    # Make token 0 strongly positive in dim 0
    matrix[0] = np.zeros(hidden_size)
    matrix[0][0] = 10.0
    # Make token 1 strongly negative in dim 0
    matrix[1] = np.zeros(hidden_size)
    matrix[1][0] = -10.0

    import mlx.core as mx
    inner = MagicMock()
    weight = MagicMock()
    weight.astype.return_value = matrix
    inner.embed_tokens.weight = weight
    backend.model = MagicMock()
    backend.model.model = inner
    # Override for get_embedding_matrix to work
    backend.model.lm_head.weight = weight

    backend.decode = lambda ids: f"tok_{ids[0]}"
    backend.tokenize = lambda text: [0]
    return backend, matrix


class TestScoreAllTokens:
    def test_direction_scoring(self):
        backend, matrix = _make_mock_backend()
        direction = np.zeros(16)
        direction[0] = 1.0  # point along dim 0

        scores = score_all_tokens(backend, direction, space="embedding")
        assert scores.shape == (100,)
        # Token 0 should score highest (10.0 in dim 0)
        assert scores[0] > scores[1]


class TestDirectionVocabularyOverlap:
    def test_same_direction(self):
        backend, matrix = _make_mock_backend()
        d = np.random.randn(16)
        result = direction_vocabulary_overlap(backend, d, d, top_k=20)
        assert result["jaccard"] == 1.0
        assert result["overlap_count"] == 20

    def test_orthogonal_directions(self):
        backend, matrix = _make_mock_backend()
        d1 = np.zeros(16); d1[0] = 1.0
        d2 = np.zeros(16); d2[1] = 1.0
        result = direction_vocabulary_overlap(backend, d1, d2, top_k=10)
        # Orthogonal directions should have less overlap
        assert result["jaccard"] < 1.0


class TestEmbeddingNeighbor:
    def test_creation(self):
        n = EmbeddingNeighbor(token_id=5, token="hello", score=0.95, distance=0.1)
        assert n.score == 0.95
