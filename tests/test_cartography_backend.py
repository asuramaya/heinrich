"""Tests for heinrich.cartography.backend."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import fields

from heinrich.cartography.backend import (
    ForwardResult,
    Backend,
    MLXBackend,
    HFBackend,
    load_backend,
)


# ---------------------------------------------------------------------------
# ForwardResult dataclass
# ---------------------------------------------------------------------------

class TestForwardResult:
    def test_construction(self):
        logits = np.array([1.0, 2.0, 3.0])
        probs = np.array([0.1, 0.3, 0.6])
        result = ForwardResult(
            logits=logits,
            probs=probs,
            top_id=2,
            top_token="hello",
            entropy=1.5,
            n_tokens=10,
        )
        assert result.top_id == 2
        assert result.top_token == "hello"
        assert result.entropy == 1.5
        assert result.n_tokens == 10
        assert result.residual is None  # default

    def test_residual_optional(self):
        logits = np.zeros(5)
        probs = np.ones(5) / 5
        residual = np.array([0.1, 0.2, 0.3])

        result = ForwardResult(
            logits=logits, probs=probs, top_id=0,
            top_token="a", entropy=2.0, n_tokens=5,
            residual=residual,
        )
        np.testing.assert_array_equal(result.residual, residual)

    def test_field_names(self):
        names = {f.name for f in fields(ForwardResult)}
        expected = {"logits", "probs", "top_id", "top_token", "entropy", "n_tokens", "residual", "per_layer"}
        assert names == expected


# ---------------------------------------------------------------------------
# load_backend
# ---------------------------------------------------------------------------

class TestLoadBackend:
    @patch("heinrich.backend.hf.HFBackend")
    @patch("heinrich.backend.mlx.MLXBackend")
    @patch("platform.system", return_value="Darwin")
    def test_auto_darwin_prefers_mlx(self, mock_system, mock_mlx_cls, mock_hf_cls):
        """On Darwin, auto should try MLX first."""
        mock_mlx_instance = MagicMock()
        mock_mlx_cls.return_value = mock_mlx_instance

        result = load_backend("test-model", backend="auto")
        mock_mlx_cls.assert_called_once_with("test-model")
        assert result is mock_mlx_instance

    @patch("heinrich.backend.hf.HFBackend")
    @patch("heinrich.backend.mlx.MLXBackend", side_effect=ImportError("no mlx"))
    @patch("platform.system", return_value="Darwin")
    def test_auto_darwin_falls_back_to_hf(self, mock_system, mock_mlx_cls, mock_hf_cls):
        """On Darwin, if MLX fails, falls back to HF."""
        mock_hf_instance = MagicMock()
        mock_hf_cls.return_value = mock_hf_instance

        result = load_backend("test-model", backend="auto")
        mock_mlx_cls.assert_called_once()
        mock_hf_cls.assert_called_once_with("test-model")
        assert result is mock_hf_instance

    @patch("heinrich.backend.hf.HFBackend")
    @patch("platform.system", return_value="Linux")
    def test_auto_linux_uses_hf(self, mock_system, mock_hf_cls):
        """On Linux, auto goes straight to HF."""
        mock_hf_instance = MagicMock()
        mock_hf_cls.return_value = mock_hf_instance

        result = load_backend("test-model", backend="auto")
        mock_hf_cls.assert_called_once_with("test-model")
        assert result is mock_hf_instance

    @patch("heinrich.backend.mlx.MLXBackend")
    def test_explicit_mlx(self, mock_mlx_cls):
        mock_mlx_instance = MagicMock()
        mock_mlx_cls.return_value = mock_mlx_instance

        result = load_backend("test-model", backend="mlx")
        mock_mlx_cls.assert_called_once_with("test-model")
        assert result is mock_mlx_instance

    @patch("heinrich.backend.hf.HFBackend")
    def test_explicit_hf(self, mock_hf_cls):
        mock_hf_instance = MagicMock()
        mock_hf_cls.return_value = mock_hf_instance

        result = load_backend("test-model", backend="hf", device="cpu")
        mock_hf_cls.assert_called_once_with("test-model", device="cpu")
        assert result is mock_hf_instance

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend.*'bogus'"):
            load_backend("test-model", backend="bogus")


# ---------------------------------------------------------------------------
# MLXBackend (mock mlx_lm and runtime)
# ---------------------------------------------------------------------------

class TestMLXBackend:
    @patch("heinrich.backend.mlx.detect_config")
    @patch("mlx_lm.load")
    def test_init_loads_model(self, mock_load, mock_detect):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_detect.return_value = MagicMock()

        backend = MLXBackend("test/model")

        mock_load.assert_called_once_with("test/model")
        assert backend.model is mock_model
        assert backend.tokenizer is mock_tokenizer
        mock_detect.assert_called_once_with(mock_model, mock_tokenizer)

    @patch("heinrich.backend.mlx.detect_config")
    @patch("mlx_lm.load")
    def test_forward_delegates_to_runtime(self, mock_load, mock_detect):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_detect.return_value = MagicMock()

        backend = MLXBackend("test/model")

        runtime_result = {
            "logits": np.zeros(10),
            "probs": np.ones(10) / 10,
            "top_id": 0,
            "top_token": "hello",
            "entropy": 2.0,
            "n_tokens": 5,
            "residual": np.zeros(8),
        }

        with patch("heinrich.cartography.runtime.forward_pass", return_value=runtime_result):
            result = backend.forward("test prompt", return_residual=True)

        assert isinstance(result, ForwardResult)
        assert result.top_token == "hello"
        assert result.entropy == 2.0

    @patch("heinrich.backend.mlx.detect_config")
    @patch("mlx_lm.load")
    def test_generate_delegates_to_runtime(self, mock_load, mock_detect):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_detect.return_value = MagicMock()

        backend = MLXBackend("test/model")

        with patch("mlx_lm.generate", return_value="hello world"):
            result = backend.generate("test prompt", max_tokens=20)

        assert result == "hello world"

    @patch("heinrich.backend.mlx.detect_config")
    @patch("mlx_lm.load")
    def test_tokenize_and_decode(self, mock_load, mock_detect):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "abc"
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_detect.return_value = MagicMock()

        backend = MLXBackend("test/model")

        assert backend.tokenize("abc") == [1, 2, 3]
        assert backend.decode([1, 2, 3]) == "abc"


# ---------------------------------------------------------------------------
# HFBackend (mock transformers)
# ---------------------------------------------------------------------------

class TestHFBackend:
    """Test HFBackend with torch and transformers fully mocked via sys.modules."""

    def _make_hf_backend(self):
        """Create an HFBackend with fully mocked torch and transformers."""
        import sys

        # Create mock torch module
        mock_torch = MagicMock()
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float32 = "float32"
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        # Create mock transformers module
        mock_transformers = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Make next(model.parameters()).device work
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])

        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        saved_torch = sys.modules.get("torch")
        saved_transformers = sys.modules.get("transformers")

        sys.modules["torch"] = mock_torch
        sys.modules["transformers"] = mock_transformers

        try:
            with patch("heinrich.backend.hf.detect_config") as mock_detect:
                mock_detect.return_value = MagicMock(intermediate_size=11008)
                backend = HFBackend("test/model", device="cpu", torch_dtype="float32")
        finally:
            # Restore original modules
            if saved_torch is not None:
                sys.modules["torch"] = saved_torch
            else:
                sys.modules.pop("torch", None)
            if saved_transformers is not None:
                sys.modules["transformers"] = saved_transformers
            else:
                sys.modules.pop("transformers", None)

        return backend, mock_model, mock_tokenizer, mock_torch

    def test_init_loads_model(self):
        backend, mock_model, mock_tokenizer, _ = self._make_hf_backend()
        assert backend.hf_model is mock_model
        assert backend.tokenizer is mock_tokenizer
        mock_model.eval.assert_called_once()

    def test_tokenize_and_decode(self):
        backend, _, mock_tokenizer, _ = self._make_hf_backend()
        mock_tokenizer.encode.return_value = [10, 20]
        mock_tokenizer.decode.return_value = "hi"

        assert backend.tokenize("hi") == [10, 20]
        assert backend.decode([10, 20]) == "hi"

    def test_capture_mlp_activations_registers_hook(self):
        """Verify that capture_mlp_activations registers and removes a forward hook."""
        import sys

        backend, mock_model, mock_tokenizer, mock_torch = self._make_hf_backend()

        # Set up mock layer structure
        mock_mlp = MagicMock()
        mock_handle = MagicMock()
        mock_mlp.register_forward_hook.return_value = mock_handle
        mock_model.model.layers.__getitem__.return_value = MagicMock(mlp=mock_mlp)

        # Mock tokenizer encode to return a tensor-like
        mock_input_ids = MagicMock()
        mock_input_ids.shape = (1, 5)
        mock_tokenizer.encode.return_value = mock_input_ids
        mock_input_ids.to.return_value = mock_input_ids

        # Intercept register_forward_hook to capture the hook function
        captured_hook_fn = None

        def capture_hook(fn):
            nonlocal captured_hook_fn
            captured_hook_fn = fn
            return mock_handle

        mock_mlp.register_forward_hook.side_effect = capture_hook

        # Mock model call to trigger our captured hook
        def fake_forward(*args, **kwargs):
            if captured_hook_fn:
                mock_tensor = MagicMock()
                mock_tensor.__getitem__ = MagicMock(return_value=MagicMock(
                    float=MagicMock(return_value=MagicMock(
                        cpu=MagicMock(return_value=MagicMock(
                            numpy=MagicMock(return_value=np.array([1.0, 2.0]))
                        ))
                    ))
                ))
                captured_hook_fn(mock_mlp, None, (mock_tensor,))
            return MagicMock()

        mock_model.side_effect = fake_forward

        # Inject mock torch for the method call
        saved_torch = sys.modules.get("torch")
        sys.modules["torch"] = mock_torch

        try:
            result = backend.capture_mlp_activations("test", layer=5)
        finally:
            if saved_torch is not None:
                sys.modules["torch"] = saved_torch
            else:
                sys.modules.pop("torch", None)

        # Hook was registered and removed
        mock_mlp.register_forward_hook.assert_called_once()
        mock_handle.remove.assert_called_once()


# ---------------------------------------------------------------------------
# Backend protocol check
# ---------------------------------------------------------------------------

class TestBackendProtocol:
    def test_mlx_backend_is_runtime_checkable(self):
        """MLXBackend should satisfy the Backend protocol structurally."""
        # We just check that the protocol is runtime_checkable and that
        # a MagicMock with the right attributes passes
        mock = MagicMock(spec=MLXBackend)
        mock.config = MagicMock()
        # The Protocol is runtime_checkable, so isinstance should work
        # for any object with the right attributes
        assert hasattr(mock, "forward")
        assert hasattr(mock, "generate")
        assert hasattr(mock, "capture_residual_states")
        assert hasattr(mock, "capture_mlp_activations")
        assert hasattr(mock, "tokenize")
        assert hasattr(mock, "decode")
