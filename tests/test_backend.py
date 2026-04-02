"""Tests for backend.py -- ablation mode dispatch and HF steering hooks.

Uses mocked models (no real model loading) to verify:
1. forward_pass() dispatches zero_attn and zero_mlp ablation modes correctly
2. HFBackend._install_steer_hooks() registers and removes hooks properly
3. HFBackend.forward() and .generate() apply steering when steer_dirs is provided
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Issue 1: Granular ablation modes in runtime.forward_pass
# ---------------------------------------------------------------------------

class FakeMXArray:
    """Stand-in for mx.array that supports the arithmetic forward_pass needs."""

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)
        self.shape = self.data.shape

    def astype(self, dtype):
        return self

    def __add__(self, other):
        if isinstance(other, FakeMXArray):
            return FakeMXArray(self.data + other.data)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, FakeMXArray):
            return FakeMXArray(other.data + self.data)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, FakeMXArray):
            return FakeMXArray(self.data - other.data)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return FakeMXArray(self.data * other)
        return NotImplemented

    def __getitem__(self, key):
        result = self.data[key]
        if isinstance(result, np.ndarray):
            return FakeMXArray(result)
        return result

    def __array__(self, dtype=None):
        return self.data if dtype is None else self.data.astype(dtype)


class FakeMX:
    """Minimal stand-in for the mx module."""
    float32 = "float32"
    float16 = "float16"

    @staticmethod
    def array(x):
        if isinstance(x, FakeMXArray):
            return x
        return FakeMXArray(np.asarray(x, dtype=np.float32))


def _build_ablation_env(n_layers=1):
    """Build fake model env for ablation tests.

    Returns (fake_inner, fake_model, calls) where calls collects
    the names of sub-component invocations.
    """
    hidden = np.ones((1, 3, 8), dtype=np.float32)
    calls = []

    class FakeLayer:
        def __init__(self, name):
            self._name = name

        def input_layernorm(self, h):
            calls.append(f"{self._name}.input_ln")
            return h

        def self_attn(self, h, mask=None, cache=None):
            calls.append(f"{self._name}.self_attn")
            return FakeMXArray(np.full_like(h.data, 0.1))

        def post_attention_layernorm(self, h):
            calls.append(f"{self._name}.post_attn_ln")
            return h

        def mlp(self, h):
            calls.append(f"{self._name}.mlp")
            return FakeMXArray(np.full_like(h.data, 0.2))

        def __call__(self, h, mask=None, cache=None):
            calls.append(f"{self._name}.full")
            return FakeMXArray(h.data + 0.3)

    layers = [FakeLayer(f"L{i}") for i in range(n_layers)]

    fake_inner = SimpleNamespace(
        embed_tokens=lambda ids: FakeMXArray(hidden),
        layers=layers,
        norm=lambda h: h,
    )
    fake_model = SimpleNamespace(
        model=fake_inner,
        lm_head=lambda h: FakeMXArray(
            np.random.default_rng(42).standard_normal((1, 3, 16))
        ),
    )
    return fake_inner, fake_model, calls


def _run_forward(fake_inner, fake_model, **kwargs):
    """Patch _setup_forward and call forward_pass."""
    with patch("heinrich.cartography.runtime._setup_forward") as mock_setup:
        mock_setup.return_value = (fake_inner, None, None, [1, 2, 3], FakeMX)
        from heinrich.cartography.runtime import forward_pass
        return forward_pass(fake_model, None, "test", **kwargs)


def test_forward_pass_zero_attn_calls_mlp_not_attn():
    fake_inner, fake_model, calls = _build_ablation_env(1)
    _run_forward(fake_inner, fake_model, ablate_layers={0}, ablate_mode="zero_attn")

    assert "L0.mlp" in calls
    assert "L0.post_attn_ln" in calls
    assert "L0.self_attn" not in calls
    assert "L0.full" not in calls


def test_forward_pass_zero_mlp_calls_attn_not_mlp():
    fake_inner, fake_model, calls = _build_ablation_env(1)
    _run_forward(fake_inner, fake_model, ablate_layers={0}, ablate_mode="zero_mlp")

    assert "L0.self_attn" in calls
    assert "L0.input_ln" in calls
    assert "L0.mlp" not in calls
    assert "L0.full" not in calls


def test_forward_pass_non_ablated_layer_runs_normally():
    fake_inner, fake_model, calls = _build_ablation_env(2)
    _run_forward(fake_inner, fake_model, ablate_layers={0}, ablate_mode="zero_attn")

    assert "L0.full" not in calls
    assert "L1.full" in calls


def test_forward_pass_zero_mode_still_works():
    fake_inner, fake_model, calls = _build_ablation_env(1)
    result = _run_forward(fake_inner, fake_model, ablate_layers={0}, ablate_mode="zero")

    assert "L0.full" in calls
    assert result["probs"] is not None


def test_forward_pass_scale_mode_still_works():
    fake_inner, fake_model, calls = _build_ablation_env(1)
    result = _run_forward(fake_inner, fake_model, ablate_layers={0}, ablate_mode="scale")

    assert "L0.full" in calls
    assert result["probs"] is not None


def test_forward_pass_invalid_ablate_mode_raises():
    fake_inner, fake_model, calls = _build_ablation_env(1)

    with pytest.raises(ValueError, match="Unknown ablate_mode"):
        _run_forward(fake_inner, fake_model, ablate_layers={0}, ablate_mode="bogus")


def test_forward_pass_no_ablation_runs_all_layers():
    fake_inner, fake_model, calls = _build_ablation_env(3)
    _run_forward(fake_inner, fake_model)

    assert calls == ["L0.full", "L1.full", "L2.full"]


# ---------------------------------------------------------------------------
# Issue 2: HFBackend steering via hooks
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")

from heinrich.cartography.backend import HFBackend, ForwardResult


def _make_fake_hf_backend(n_layers=4, hidden_size=16):
    """Create an HFBackend with real torch.nn layers (for hook support)."""
    layers = torch.nn.ModuleList(
        [torch.nn.Linear(hidden_size, hidden_size, bias=False)
         for _ in range(n_layers)]
    )

    fake_hf_model = MagicMock()
    fake_hf_model.model.layers = layers

    backend = object.__new__(HFBackend)
    backend.hf_model = fake_hf_model
    backend.tokenizer = MagicMock()
    backend.tokenizer.encode = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
    backend.tokenizer.decode = MagicMock(return_value="test")
    backend._device = "cpu"
    backend.config = MagicMock(intermediate_size=hidden_size * 4)

    return backend, list(layers)


def test_install_steer_hooks_registers_on_correct_layers():
    backend, layers = _make_fake_hf_backend(n_layers=4, hidden_size=8)
    direction = np.ones(8, dtype=np.float32)
    steer_dirs = {1: (direction, 2.0), 3: (direction, 1.5)}

    handles = backend._install_steer_hooks(steer_dirs, alpha=1.0)
    assert len(handles) == 2
    assert len(layers[1]._forward_hooks) == 1
    assert len(layers[3]._forward_hooks) == 1
    assert len(layers[0]._forward_hooks) == 0
    assert len(layers[2]._forward_hooks) == 0

    for h in handles:
        h.remove()


def test_install_steer_hooks_removes_cleanly():
    backend, layers = _make_fake_hf_backend(n_layers=4, hidden_size=8)
    direction = np.ones(8, dtype=np.float32)
    steer_dirs = {1: (direction, 2.0)}

    handles = backend._install_steer_hooks(steer_dirs, alpha=1.0)
    assert len(layers[1]._forward_hooks) == 1

    for h in handles:
        h.remove()
    assert len(layers[1]._forward_hooks) == 0


def test_install_steer_hooks_skips_out_of_range_layers():
    backend, layers = _make_fake_hf_backend(n_layers=4, hidden_size=8)
    direction = np.ones(8, dtype=np.float32)
    steer_dirs = {1: (direction, 2.0), 99: (direction, 1.0)}

    handles = backend._install_steer_hooks(steer_dirs, alpha=1.0)
    assert len(handles) == 1

    for h in handles:
        h.remove()


def test_steer_hook_modifies_hidden_state():
    backend, layers = _make_fake_hf_backend(n_layers=4, hidden_size=8)

    direction = np.ones(8, dtype=np.float32) * 0.5
    mean_gap = 2.0
    alpha = 3.0
    steer_dirs = {1: (direction, mean_gap)}

    handles = backend._install_steer_hooks(steer_dirs, alpha=alpha)

    hook_fn = list(layers[1]._forward_hooks.values())[0]
    hidden = torch.zeros(1, 5, 8)
    output = (hidden.clone(),)

    result = hook_fn(layers[1], None, output)

    expected = direction * mean_gap * alpha
    actual = result[0][0, -1, :].float().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5)

    for pos in range(4):
        np.testing.assert_allclose(
            result[0][0, pos, :].float().numpy(), np.zeros(8), atol=1e-7
        )

    for h in handles:
        h.remove()


def test_steer_hook_preserves_tuple_structure():
    backend, layers = _make_fake_hf_backend(n_layers=4, hidden_size=8)
    direction = np.ones(8, dtype=np.float32)
    steer_dirs = {0: (direction, 1.0)}
    handles = backend._install_steer_hooks(steer_dirs, alpha=1.0)

    hidden = torch.zeros(1, 3, 8)
    extra = torch.ones(1, 3, 8)
    output = (hidden.clone(), extra)

    hook_fn = list(layers[0]._forward_hooks.values())[0]
    result = hook_fn(layers[0], None, output)

    assert isinstance(result, tuple) and len(result) == 2
    assert torch.allclose(result[1], extra)

    for h in handles:
        h.remove()


def _mock_hf_output():
    """Build a fake HF model output with .logits."""
    logits = torch.randn(1, 3, 32)
    outputs = MagicMock()
    outputs.logits = logits
    outputs.hidden_states = None
    return outputs


def test_forward_no_steering_does_not_install_hooks():
    backend, layers = _make_fake_hf_backend(n_layers=4, hidden_size=8)
    backend.hf_model.return_value = _mock_hf_output()

    backend.forward("test prompt")

    for layer in layers:
        assert len(layer._forward_hooks) == 0


def test_forward_with_steering_installs_and_removes_hooks():
    backend, layers = _make_fake_hf_backend(n_layers=4, hidden_size=8)
    direction = np.ones(8, dtype=np.float32)
    steer_dirs = {1: (direction, 2.0)}
    backend.hf_model.return_value = _mock_hf_output()

    backend.forward("test prompt", steer_dirs=steer_dirs, alpha=1.0)

    assert len(layers[1]._forward_hooks) == 0


def test_generate_with_steering_installs_and_removes_hooks():
    backend, layers = _make_fake_hf_backend(n_layers=4, hidden_size=8)
    direction = np.ones(8, dtype=np.float32)
    steer_dirs = {2: (direction, 1.5)}
    backend.hf_model.generate = MagicMock(
        return_value=torch.tensor([[1, 2, 3, 4, 5]])
    )

    backend.generate("test prompt", steer_dirs=steer_dirs, alpha=2.0)
    assert len(layers[2]._forward_hooks) == 0


def test_forward_hooks_removed_even_on_error():
    backend, layers = _make_fake_hf_backend(n_layers=4, hidden_size=8)
    direction = np.ones(8, dtype=np.float32)
    steer_dirs = {1: (direction, 2.0)}
    backend.hf_model.side_effect = RuntimeError("simulated failure")

    with pytest.raises(RuntimeError, match="simulated failure"):
        backend.forward("test", steer_dirs=steer_dirs, alpha=1.0)

    assert len(layers[1]._forward_hooks) == 0


def test_generate_hooks_removed_even_on_error():
    backend, layers = _make_fake_hf_backend(n_layers=4, hidden_size=8)
    direction = np.ones(8, dtype=np.float32)
    steer_dirs = {2: (direction, 1.0)}
    backend.hf_model.generate = MagicMock(
        side_effect=RuntimeError("generation failed")
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        backend.generate("test", steer_dirs=steer_dirs, alpha=1.0)

    assert len(layers[2]._forward_hooks) == 0


def test_forward_zero_alpha_skips_hooks():
    backend, layers = _make_fake_hf_backend(n_layers=4, hidden_size=8)
    direction = np.ones(8, dtype=np.float32)
    steer_dirs = {1: (direction, 2.0)}
    backend.hf_model.return_value = _mock_hf_output()

    backend.forward("test", steer_dirs=steer_dirs, alpha=0.0)

    for layer in layers:
        assert len(layer._forward_hooks) == 0
