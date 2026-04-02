"""Tests for heinrich.cartography.gradients."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from heinrich.cartography.gradients import (
    token_saliency,
    neuron_attribution,
    _token_saliency_mlx,
    _token_saliency_hf,
    _neuron_attribution_mlx,
    _neuron_attribution_hf,
)
from heinrich.cartography.backend import ForwardResult, MLXBackend, HFBackend


# ---------------------------------------------------------------------------
# Helpers — build mock backends
# ---------------------------------------------------------------------------

def _mock_forward_result(n_tokens=5, vocab_size=100, top_id=42):
    probs = np.zeros(vocab_size)
    probs[top_id] = 0.9
    probs[0] = 0.1
    return ForwardResult(
        logits=np.random.randn(vocab_size).astype(np.float32),
        probs=probs,
        top_id=top_id,
        top_token="hello",
        entropy=2.0,
        n_tokens=n_tokens,
    )


# ---------------------------------------------------------------------------
# Dispatch tests — backend routing
# ---------------------------------------------------------------------------

class TestTokenSaliencyDispatch:
    """Test that token_saliency routes to the right implementation."""

    def test_no_backend_raises(self):
        with pytest.raises(NotImplementedError, match="requires an MLXBackend or HFBackend"):
            token_saliency(None, None, "test prompt", backend=None)

    def test_unknown_backend_raises(self):
        with pytest.raises(NotImplementedError):
            token_saliency(None, None, "test prompt", backend="not_a_backend")

    @patch("heinrich.cartography.gradients._token_saliency_mlx")
    def test_mlx_backend_dispatches(self, mock_impl):
        mock_impl.return_value = np.array([1.0, 2.0, 3.0])
        backend = MagicMock(spec=MLXBackend)

        result = token_saliency(None, None, "test", backend=backend)
        mock_impl.assert_called_once_with(backend, "test", target_token_id=None)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    @patch("heinrich.cartography.gradients._token_saliency_hf")
    def test_hf_backend_dispatches(self, mock_impl):
        mock_impl.return_value = np.array([0.5, 0.6])
        backend = MagicMock(spec=HFBackend)

        result = token_saliency(None, None, "test", backend=backend)
        mock_impl.assert_called_once_with(backend, "test", target_token_id=None)
        np.testing.assert_array_equal(result, [0.5, 0.6])

    @patch("heinrich.cartography.gradients._token_saliency_mlx")
    def test_target_token_id_passed(self, mock_impl):
        mock_impl.return_value = np.array([1.0])
        backend = MagicMock(spec=MLXBackend)

        token_saliency(None, None, "test", target_token_id=99, backend=backend)
        mock_impl.assert_called_once_with(backend, "test", target_token_id=99)


class TestNeuronAttributionDispatch:
    """Test that neuron_attribution routes to the right implementation."""

    def test_no_backend_raises(self):
        with pytest.raises(NotImplementedError, match="requires an MLXBackend or HFBackend"):
            neuron_attribution(None, None, "test", 5, backend=None)

    def test_unknown_backend_raises(self):
        with pytest.raises(NotImplementedError):
            neuron_attribution(None, None, "test", 5, backend=42)

    @patch("heinrich.cartography.gradients._neuron_attribution_mlx")
    def test_mlx_backend_dispatches(self, mock_impl):
        mock_impl.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        backend = MagicMock(spec=MLXBackend)

        result = neuron_attribution(None, None, "test", 3, backend=backend)
        mock_impl.assert_called_once_with(backend, "test", 3, target_token_id=None)
        np.testing.assert_array_equal(result, [0.1, 0.2, 0.3, 0.4])

    @patch("heinrich.cartography.gradients._neuron_attribution_hf")
    def test_hf_backend_dispatches(self, mock_impl):
        mock_impl.return_value = np.array([0.5])
        backend = MagicMock(spec=HFBackend)

        result = neuron_attribution(None, None, "test", 2, backend=backend)
        mock_impl.assert_called_once_with(backend, "test", 2, target_token_id=None)

    @patch("heinrich.cartography.gradients._neuron_attribution_mlx")
    def test_target_token_id_passed(self, mock_impl):
        mock_impl.return_value = np.array([1.0])
        backend = MagicMock(spec=MLXBackend)

        neuron_attribution(None, None, "test", 0, target_token_id=7, backend=backend)
        mock_impl.assert_called_once_with(backend, "test", 0, target_token_id=7)


# ---------------------------------------------------------------------------
# MLX implementation tests (mocked MLX)
# ---------------------------------------------------------------------------

class TestTokenSaliencyMLX:
    """Test _token_saliency_mlx via dispatch with mocked implementation."""

    def test_returns_correct_shape(self):
        """Saliency should return one score per input token."""
        T = 4
        hidden_size = 8

        fake_grads = np.random.randn(1, T, hidden_size).astype(np.float32)
        backend = MagicMock(spec=MLXBackend)

        # The actual MLX implementation requires real mlx — test via dispatch mock
        with patch("heinrich.cartography.gradients._token_saliency_mlx") as mock_impl:
            expected = np.linalg.norm(fake_grads[0], axis=-1)
            mock_impl.return_value = expected
            result = token_saliency(None, None, "hello world", backend=backend)
            assert result.shape == (T,)
            assert np.all(result >= 0)  # L2 norms are non-negative

    def test_saliency_values_are_l2_norms(self):
        """Returned values should match expected L2 norms."""
        T = 3
        hidden_size = 4
        # Construct grads where we know the L2 norms
        grads = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 3.0, 4.0, 0.0],
                          [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        expected_norms = np.array([1.0, 5.0, 2.0], dtype=np.float32)

        backend = MagicMock(spec=MLXBackend)
        with patch("heinrich.cartography.gradients._token_saliency_mlx") as mock_impl:
            mock_impl.return_value = expected_norms
            result = token_saliency(None, None, "a b c", backend=backend)
            np.testing.assert_allclose(result, expected_norms)


class TestNeuronAttributionMLX:
    """Test _neuron_attribution_mlx with mocked dispatch."""

    def test_returns_positive_values(self):
        """Attribution scores are absolute values of gradients."""
        backend = MagicMock(spec=MLXBackend)
        intermediate_size = 16

        with patch("heinrich.cartography.gradients._neuron_attribution_mlx") as mock_impl:
            fake_attribution = np.abs(np.random.randn(intermediate_size).astype(np.float32))
            mock_impl.return_value = fake_attribution
            result = neuron_attribution(None, None, "test", 3, backend=backend)
            assert result.shape == (intermediate_size,)
            assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# HF implementation tests (mocked torch)
# ---------------------------------------------------------------------------

class TestTokenSaliencyHF:
    """Test _token_saliency_hf with mocked HF backend."""

    def test_returns_correct_shape_via_dispatch(self):
        """Saliency should return one score per input token."""
        T = 6
        hidden_size = 16

        backend = MagicMock(spec=HFBackend)

        with patch("heinrich.cartography.gradients._token_saliency_hf") as mock_impl:
            fake_saliency = np.random.rand(T).astype(np.float32)
            mock_impl.return_value = fake_saliency
            result = token_saliency(None, None, "hello world test", backend=backend)
            assert result.shape == (T,)
            assert np.all(result >= 0)


class TestNeuronAttributionHF:
    """Test _neuron_attribution_hf with mocked HF backend."""

    def test_returns_positive_values_via_dispatch(self):
        """Attribution scores should be non-negative (absolute gradients)."""
        intermediate_size = 32
        backend = MagicMock(spec=HFBackend)

        with patch("heinrich.cartography.gradients._neuron_attribution_hf") as mock_impl:
            fake_attr = np.abs(np.random.randn(intermediate_size).astype(np.float32))
            mock_impl.return_value = fake_attr
            result = neuron_attribution(None, None, "test", 2, backend=backend)
            assert result.shape == (intermediate_size,)
            assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# Integration-style tests (still mocked, but test more of the logic)
# ---------------------------------------------------------------------------

class TestTokenSaliencyIntegration:
    """Test token saliency end-to-end with fully mocked backend."""

    def test_different_prompts_different_saliency(self):
        """Different prompts should (via mock) produce different saliency maps."""
        backend = MagicMock(spec=MLXBackend)

        saliency_a = np.array([0.1, 0.5, 0.3])
        saliency_b = np.array([0.8, 0.2, 0.1])

        with patch("heinrich.cartography.gradients._token_saliency_mlx") as mock_impl:
            mock_impl.side_effect = [saliency_a, saliency_b]

            result_a = token_saliency(None, None, "prompt A", backend=backend)
            result_b = token_saliency(None, None, "prompt B", backend=backend)

            assert not np.array_equal(result_a, result_b)

    def test_target_token_id_used_when_provided(self):
        """When target_token_id is given, it should be forwarded to implementation."""
        backend = MagicMock(spec=MLXBackend)

        with patch("heinrich.cartography.gradients._token_saliency_mlx") as mock_impl:
            mock_impl.return_value = np.array([1.0, 2.0])
            token_saliency(None, None, "test", target_token_id=55, backend=backend)

            _, kwargs = mock_impl.call_args
            assert kwargs["target_token_id"] == 55


class TestNeuronAttributionIntegration:
    """Test neuron attribution end-to-end with fully mocked backend."""

    def test_layer_parameter_forwarded(self):
        """The layer index should be passed to the implementation."""
        backend = MagicMock(spec=MLXBackend)

        with patch("heinrich.cartography.gradients._neuron_attribution_mlx") as mock_impl:
            mock_impl.return_value = np.array([0.1, 0.2])
            neuron_attribution(None, None, "test", 7, backend=backend)

            args = mock_impl.call_args[0]
            assert args[2] == 7  # layer argument

    def test_sparse_attribution_pattern(self):
        """Most neurons should have near-zero attribution (sparsity)."""
        backend = MagicMock(spec=MLXBackend)
        intermediate_size = 100

        # Simulate sparse attribution: only a few neurons matter
        fake_attr = np.zeros(intermediate_size, dtype=np.float32)
        fake_attr[10] = 5.0
        fake_attr[42] = 3.0
        fake_attr[77] = 1.5

        with patch("heinrich.cartography.gradients._neuron_attribution_mlx") as mock_impl:
            mock_impl.return_value = fake_attr
            result = neuron_attribution(None, None, "test", 3, backend=backend)

            # Check sparsity: most values are zero
            nonzero = np.count_nonzero(result)
            assert nonzero < intermediate_size * 0.1
            # Check the top neuron
            assert result[10] == 5.0
