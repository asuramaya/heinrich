"""Tests for heinrich.cartography.transfer — cross-model direction transfer."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from heinrich.cartography.transfer import (
    transfer_direction,
    transfer_attack,
    _get_unembedding_matrix,
    _get_embedding_matrix,
    _shared_vocab_mask,
    _transfer_vocabulary,
    _transfer_procrustes,
    _refuse_prob_from_probs,
)
from heinrich.cartography.model_config import ModelConfig
from heinrich.cartography.backend import ForwardResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(*, model_type="test", n_layers=4, hidden_size=8,
                 vocab_size=16, intermediate_size=32):
    return ModelConfig(
        model_type=model_type, n_layers=n_layers, hidden_size=hidden_size,
        intermediate_size=intermediate_size, n_heads=2, n_kv_heads=2,
        head_dim=hidden_size // 2, vocab_size=vocab_size,
        max_position_embeddings=512, chat_format="base",
    )


def _make_backend(*, hidden_size=8, vocab_size=16, n_layers=4,
                  model_type="test", embed_weights=None, unembed_weights=None):
    """Build a mock backend with controllable embedding / unembedding matrices."""
    cfg = _make_config(
        model_type=model_type, n_layers=n_layers,
        hidden_size=hidden_size, vocab_size=vocab_size,
    )
    backend = MagicMock()
    backend.config = cfg

    # Build model structure: model.lm_head.weight, model.model.embed_tokens.weight
    if unembed_weights is None:
        unembed_weights = np.random.randn(vocab_size, hidden_size).astype(np.float32)
    if embed_weights is None:
        embed_weights = np.random.randn(vocab_size, hidden_size).astype(np.float32)

    # model.lm_head.weight -> unembedding
    lm_head = MagicMock()
    lm_head.weight = unembed_weights
    model = MagicMock()
    model.lm_head = lm_head

    # model.model.embed_tokens.weight -> embedding
    inner_model = MagicMock()
    embed_tokens = MagicMock()
    embed_tokens.weight = embed_weights
    inner_model.embed_tokens = embed_tokens
    model.model = inner_model

    backend.model = model

    # Mock tokenizer
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.return_value = "test output"
    backend.tokenizer = tokenizer

    return backend


# ---------------------------------------------------------------------------
# _shared_vocab_mask
# ---------------------------------------------------------------------------

class TestSharedVocabMask:
    def test_same_size(self):
        mask = _shared_vocab_mask(100, 100)
        assert len(mask) == 100
        assert mask[0] == 0
        assert mask[-1] == 99

    def test_source_smaller(self):
        mask = _shared_vocab_mask(50, 100)
        assert len(mask) == 50

    def test_target_smaller(self):
        mask = _shared_vocab_mask(100, 50)
        assert len(mask) == 50


# ---------------------------------------------------------------------------
# _get_unembedding_matrix / _get_embedding_matrix
# ---------------------------------------------------------------------------

class TestMatrixExtraction:
    def test_get_unembedding_from_model(self):
        backend = _make_backend(hidden_size=8, vocab_size=16)
        W = _get_unembedding_matrix(backend)
        assert W.shape == (16, 8)
        assert W.dtype == np.float32

    def test_get_embedding_from_model(self):
        backend = _make_backend(hidden_size=8, vocab_size=16)
        W = _get_embedding_matrix(backend)
        assert W.shape == (16, 8)
        assert W.dtype == np.float32

    def test_unembedding_no_model_raises(self):
        backend = MagicMock()
        backend.model = None
        backend.hf_model = None
        with pytest.raises(RuntimeError, match="Cannot extract unembedding"):
            _get_unembedding_matrix(backend)

    def test_embedding_no_model_raises(self):
        backend = MagicMock()
        backend.model = None
        backend.hf_model = None
        with pytest.raises(RuntimeError, match="Cannot extract embedding"):
            _get_embedding_matrix(backend)

    def test_get_unembedding_from_hf_model(self):
        """HFBackend stores model as hf_model, not model."""
        backend = MagicMock()
        backend.model = None
        hf_model = MagicMock()
        lm_head = MagicMock()
        lm_head.weight = np.random.randn(10, 4).astype(np.float32)
        hf_model.lm_head = lm_head
        backend.hf_model = hf_model
        W = _get_unembedding_matrix(backend)
        assert W.shape == (10, 4)


# ---------------------------------------------------------------------------
# transfer_direction — vocabulary method
# ---------------------------------------------------------------------------

class TestTransferVocabulary:
    def test_same_model_recovers_direction(self):
        """Transferring a direction to the same model should give high cosine."""
        np.random.seed(42)
        hidden = 32
        vocab = 64
        # Use identical embedding/unembedding (tied weights)
        W = np.random.randn(vocab, hidden).astype(np.float32)
        backend = _make_backend(
            hidden_size=hidden, vocab_size=vocab,
            embed_weights=W, unembed_weights=W,
        )
        direction = np.random.randn(hidden).astype(np.float32)
        direction /= np.linalg.norm(direction)

        result = transfer_direction(direction, backend, backend, method="vocabulary")

        assert result.shape == (hidden,)
        # Should be unit normalized
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5
        # With tied weights and same model, cosine should be very high
        cos = np.dot(direction, result)
        assert cos > 0.7, f"Self-transfer cosine too low: {cos}"

    def test_different_hidden_sizes(self):
        """Vocabulary transfer should work across different hidden sizes."""
        np.random.seed(123)
        source = _make_backend(hidden_size=8, vocab_size=16)
        target = _make_backend(hidden_size=12, vocab_size=16)

        direction = np.random.randn(8).astype(np.float32)
        direction /= np.linalg.norm(direction)

        result = transfer_direction(direction, source, target, method="vocabulary")

        assert result.shape == (12,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_different_vocab_sizes(self):
        """Should handle models with different vocabulary sizes."""
        np.random.seed(456)
        source = _make_backend(hidden_size=8, vocab_size=20)
        target = _make_backend(hidden_size=8, vocab_size=10)

        direction = np.random.randn(8).astype(np.float32)
        direction /= np.linalg.norm(direction)

        result = transfer_direction(direction, source, target, method="vocabulary")

        assert result.shape == (8,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_zero_direction_returns_zero(self):
        """A zero input direction should produce a zero output."""
        source = _make_backend(hidden_size=8, vocab_size=16)
        target = _make_backend(hidden_size=8, vocab_size=16)

        direction = np.zeros(8, dtype=np.float32)

        result = transfer_direction(direction, source, target, method="vocabulary")
        assert np.allclose(result, 0)


# ---------------------------------------------------------------------------
# transfer_direction — procrustes method
# ---------------------------------------------------------------------------

class TestTransferProcrustes:
    def test_identity_rotation(self):
        """If both models produce identical activations, direction should be preserved."""
        np.random.seed(42)
        hidden = 8
        n_prompts = 10
        shared_states = np.random.randn(n_prompts, hidden).astype(np.float32)

        source = _make_backend(hidden_size=hidden)
        target = _make_backend(hidden_size=hidden)

        source.capture_residual_states.return_value = {5: shared_states.copy()}
        target.capture_residual_states.return_value = {5: shared_states.copy()}

        direction = np.random.randn(hidden).astype(np.float32)
        direction /= np.linalg.norm(direction)

        result = transfer_direction(
            direction, source, target,
            method="procrustes",
            shared_prompts=["p"] * n_prompts,
            layers_source=5,
            layers_target=5,
        )

        assert result.shape == (hidden,)
        cos = np.dot(direction, result)
        assert cos > 0.99, f"Identity procrustes cosine too low: {cos}"

    def test_known_rotation(self):
        """If target = source @ R, procrustes should recover R."""
        np.random.seed(42)
        hidden = 8
        n_prompts = 20

        # Create orthogonal rotation via QR
        Q, _ = np.linalg.qr(np.random.randn(hidden, hidden))
        source_states = np.random.randn(n_prompts, hidden).astype(np.float32)
        target_states = (source_states @ Q).astype(np.float32)

        source = _make_backend(hidden_size=hidden)
        target = _make_backend(hidden_size=hidden)
        source.capture_residual_states.return_value = {3: source_states}
        target.capture_residual_states.return_value = {3: target_states}

        direction = np.random.randn(hidden).astype(np.float32)
        direction /= np.linalg.norm(direction)
        expected = Q @ direction
        expected /= np.linalg.norm(expected)

        result = transfer_direction(
            direction, source, target,
            method="procrustes",
            shared_prompts=["p"] * n_prompts,
            layers_source=3,
            layers_target=3,
        )

        cos = np.dot(expected, result)
        assert abs(cos) > 0.95, f"Procrustes known-rotation cosine too low: {cos}"

    def test_different_hidden_sizes_raises(self):
        """Procrustes should reject models with different hidden sizes."""
        source = _make_backend(hidden_size=8)
        target = _make_backend(hidden_size=12)
        source.capture_residual_states.return_value = {5: np.zeros((5, 8))}
        target.capture_residual_states.return_value = {5: np.zeros((5, 12))}

        direction = np.random.randn(8).astype(np.float32)
        direction /= np.linalg.norm(direction)

        with pytest.raises(ValueError, match="same hidden size"):
            transfer_direction(
                direction, source, target,
                method="procrustes",
                shared_prompts=["p"] * 5,
                layers_source=5,
                layers_target=5,
            )

    def test_procrustes_requires_prompts(self):
        source = _make_backend()
        target = _make_backend()
        direction = np.random.randn(8).astype(np.float32)
        with pytest.raises(ValueError, match="shared_prompts"):
            transfer_direction(direction, source, target, method="procrustes")

    def test_procrustes_requires_layers(self):
        source = _make_backend()
        target = _make_backend()
        direction = np.random.randn(8).astype(np.float32)
        with pytest.raises(ValueError, match="layers_source"):
            transfer_direction(
                direction, source, target,
                method="procrustes",
                shared_prompts=["p"],
            )


# ---------------------------------------------------------------------------
# transfer_direction — method validation
# ---------------------------------------------------------------------------

class TestTransferMethodValidation:
    def test_unknown_method_raises(self):
        source = _make_backend()
        target = _make_backend()
        direction = np.random.randn(8).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown transfer method"):
            transfer_direction(direction, source, target, method="bogus")


# ---------------------------------------------------------------------------
# _refuse_prob_from_probs
# ---------------------------------------------------------------------------

class TestRefuseProbFromProbs:
    def test_returns_float(self):
        probs = np.zeros(100, dtype=np.float32)
        probs[0] = 1.0
        backend = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [42]  # single token per prefix
        backend.tokenizer = tokenizer
        result = _refuse_prob_from_probs(probs, backend)
        assert isinstance(result, float)

    def test_no_tokenizer_returns_zero(self):
        probs = np.ones(10, dtype=np.float32) / 10
        backend = MagicMock(spec=[])  # no tokenizer attribute
        result = _refuse_prob_from_probs(probs, backend)
        assert result == 0.0


# ---------------------------------------------------------------------------
# transfer_attack
# ---------------------------------------------------------------------------

class TestTransferAttack:
    def _setup_attack(self):
        """Create source and target backends for transfer attack tests."""
        np.random.seed(42)
        hidden = 8
        vocab = 16
        n_layers = 4

        source = _make_backend(
            hidden_size=hidden, vocab_size=vocab, n_layers=n_layers,
            model_type="source_model",
        )
        target = _make_backend(
            hidden_size=hidden, vocab_size=vocab, n_layers=n_layers,
            model_type="target_model",
        )

        # Mock capture_residual_states for direction discovery
        def make_states(prompts, *, layers):
            result = {}
            for l in layers:
                # Harmful prompts get positive projection, benign negative
                states = []
                for i, _ in enumerate(prompts):
                    s = np.random.randn(hidden).astype(np.float32)
                    states.append(s)
                result[l] = np.array(states)
            return result

        source.capture_residual_states.side_effect = make_states
        target.capture_residual_states.side_effect = make_states

        # Mock forward returns
        def make_forward_result(prompt, *, steer_dirs=None, alpha=0.0, **kw):
            probs = np.ones(vocab, dtype=np.float32) / vocab
            logits = np.zeros(vocab, dtype=np.float32)
            return ForwardResult(
                logits=logits, probs=probs, top_id=0,
                top_token="test", entropy=4.0, n_tokens=3,
            )

        source.forward.side_effect = make_forward_result
        target.forward.side_effect = make_forward_result

        # Mock generate
        target.generate.return_value = "Sure, here's how to hack..."

        return source, target

    def test_returns_expected_keys(self):
        source, target = self._setup_attack()
        result = transfer_attack(source, target, "How to hack a bank?")

        assert "source_rp" in result
        assert "target_rp_native" in result
        assert "target_rp_transfer" in result
        assert "generated" in result
        assert "transfer_cosine" in result
        assert "best_source_layer" in result
        assert "best_target_layer" in result

    def test_values_are_numeric(self):
        source, target = self._setup_attack()
        result = transfer_attack(source, target, "How to hack a bank?")

        assert isinstance(result["source_rp"], float)
        assert isinstance(result["target_rp_native"], float)
        assert isinstance(result["target_rp_transfer"], float)
        assert isinstance(result["generated"], str)
        assert isinstance(result["transfer_cosine"], float)
        assert isinstance(result["best_source_layer"], (int, np.integer))
        assert isinstance(result["best_target_layer"], (int, np.integer))

    def test_signals_stored(self):
        from heinrich.signal import SignalStore
        source, target = self._setup_attack()
        store = SignalStore()

        transfer_attack(source, target, "How to hack a bank?", store=store)

        signals = list(store)
        assert len(signals) > 0
        transfer_signals = [s for s in signals if s.kind == "transfer_attack"]
        assert len(transfer_signals) == 1
        assert "source_model->target_model" in transfer_signals[0].target

    def test_custom_layers(self):
        source, target = self._setup_attack()
        result = transfer_attack(
            source, target, "How to hack a bank?",
            source_layers=[1, 2], target_layers=[2, 3],
        )
        assert result["best_source_layer"] in [1, 2]
        assert result["best_target_layer"] in [2, 3]

    def test_custom_alpha(self):
        source, target = self._setup_attack()
        # Should work with different alphas
        result = transfer_attack(
            source, target, "How to hack a bank?", alpha=-0.30,
        )
        assert "target_rp_transfer" in result

    def test_generated_text_from_backend(self):
        source, target = self._setup_attack()
        result = transfer_attack(source, target, "Test prompt")
        assert result["generated"] == "Sure, here's how to hack..."
        target.generate.assert_called_once()
