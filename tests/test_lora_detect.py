"""Tests for heinrich.cartography.lora_detect — LoRA safety-signature detection."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from heinrich.cartography.lora_detect import (
    analyze_lora_weights,
    compare_lora_to_safety,
    simulate_safety_lora,
    _check_late_layer_concentration,
    _extract_layer_index,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeConfig:
    """Minimal stand-in for ModelConfig."""

    def __init__(self, n_layers=32, hidden_size=64):
        self.n_layers = n_layers
        self.hidden_size = hidden_size


class FakeBackend:
    """Minimal stand-in for Backend with a safety direction."""

    def __init__(self, *, n_layers=32, hidden_size=64, safety_direction=None):
        self.config = FakeConfig(n_layers=n_layers, hidden_size=hidden_size)
        if safety_direction is not None:
            self.safety_direction = safety_direction
        else:
            rng = np.random.default_rng(0)
            d = rng.standard_normal(hidden_size).astype(np.float32)
            self.safety_direction = d / np.linalg.norm(d)


def _make_lora_npz(
    layers: list[int],
    hidden_size: int = 64,
    rank: int = 4,
    *,
    seed: int = 42,
) -> Path:
    """Create a mock LoRA .npz file targeting specific layers."""
    rng = np.random.default_rng(seed)
    d = Path(tempfile.mkdtemp())
    path = d / "adapter.npz"

    tensors = {}
    for layer in layers:
        prefix = f"base_model.model.layers.{layer}.self_attn.q_proj"
        tensors[f"{prefix}.lora_A.weight"] = (
            rng.standard_normal((rank, hidden_size)).astype(np.float32)
        )
        tensors[f"{prefix}.lora_B.weight"] = (
            rng.standard_normal((hidden_size, rank)).astype(np.float32)
        )

    np.savez_compressed(path, **tensors)
    return path


def _make_safety_aligned_lora_npz(
    safety_direction: np.ndarray,
    layers: list[int],
    hidden_size: int = 64,
    rank: int = 4,
) -> Path:
    """Create a LoRA that is intentionally aligned with the safety direction."""
    d = safety_direction / (np.linalg.norm(safety_direction) + 1e-12)
    rng = np.random.default_rng(99)
    tmpdir = Path(tempfile.mkdtemp())
    path = tmpdir / "safety_aligned_adapter.npz"

    tensors = {}
    for layer in layers:
        prefix = f"base_model.model.layers.{layer}.self_attn.q_proj"
        # A's first row is the safety direction, B's first column is -d
        A = np.zeros((rank, hidden_size), dtype=np.float32)
        B = np.zeros((hidden_size, rank), dtype=np.float32)
        A[0] = d
        B[:, 0] = -d * 5.0  # large magnitude to dominate

        # Pad with small noise
        if rank > 1:
            A[1:] = rng.standard_normal((rank - 1, hidden_size)).astype(np.float32) * 0.01
            B[:, 1:] = rng.standard_normal((hidden_size, rank - 1)).astype(np.float32) * 0.01

        tensors[f"{prefix}.lora_A.weight"] = A
        tensors[f"{prefix}.lora_B.weight"] = B

    np.savez_compressed(path, **tensors)
    return path


# ---------------------------------------------------------------------------
# Tests: _extract_layer_index
# ---------------------------------------------------------------------------


class TestExtractLayerIndex:
    def test_standard_key(self):
        assert _extract_layer_index("layers.24.self_attn.q_proj") == 24

    def test_with_prefix(self):
        assert _extract_layer_index("base_model.model.layers.0.mlp.gate_proj") == 0

    def test_no_layers_keyword(self):
        assert _extract_layer_index("embed_tokens.weight") is None

    def test_non_numeric_after_layers(self):
        assert _extract_layer_index("layers.foo.weight") is None


# ---------------------------------------------------------------------------
# Tests: _check_late_layer_concentration
# ---------------------------------------------------------------------------


class TestLateLayerConcentration:
    def test_all_late(self):
        frac, suspicious = _check_late_layer_concentration([24, 25, 26, 27], 28)
        assert frac == 1.0
        assert suspicious is True

    def test_all_early(self):
        frac, suspicious = _check_late_layer_concentration([0, 1, 2, 3], 28)
        assert frac == 0.0
        assert suspicious is False

    def test_mixed(self):
        frac, suspicious = _check_late_layer_concentration([2, 10, 22, 25], 28)
        # threshold = 28*3//4 = 21, so layers 22 and 25 are late => 2/4 = 0.5
        assert frac == 0.5
        assert suspicious is False  # 0.5 <= 0.6

    def test_empty(self):
        frac, suspicious = _check_late_layer_concentration([], 28)
        assert frac == 0.0
        assert suspicious is False


# ---------------------------------------------------------------------------
# Tests: compare_lora_to_safety
# ---------------------------------------------------------------------------


class TestCompareLoraToSafety:
    def test_aligned_delta(self):
        """A rank-1 delta built from the safety direction should score high."""
        rng = np.random.default_rng(7)
        d = rng.standard_normal(64).astype(np.float32)
        d /= np.linalg.norm(d)

        # delta = -outer(d, d)  -> projects d onto itself, flips sign
        delta = -5.0 * np.outer(d, d)

        result = compare_lora_to_safety(delta, d, threshold=0.3)
        assert result["is_safety_relevant"]
        assert abs(result["cosine_similarity"]) > 0.3

    def test_orthogonal_delta(self):
        """A delta orthogonal to the safety direction should score low."""
        rng = np.random.default_rng(8)
        d = np.zeros(64, dtype=np.float32)
        d[0] = 1.0

        # Build a delta that only touches dimensions orthogonal to d
        delta = np.zeros((64, 64), dtype=np.float32)
        delta[1:, 1:] = rng.standard_normal((63, 63)).astype(np.float32) * 0.01

        result = compare_lora_to_safety(delta, d, threshold=0.3)
        assert not result["is_safety_relevant"]
        assert abs(result["cosine_similarity"]) < 0.3

    def test_dimension_mismatch(self):
        """When dimensions don't match the safety direction at all."""
        d = np.ones(64, dtype=np.float32)
        delta = np.ones((32, 32), dtype=np.float32)

        result = compare_lora_to_safety(delta, d)
        assert result["cosine_similarity"] == 0.0
        assert not result["is_safety_relevant"]

    def test_1d_delta_returns_zero(self):
        d = np.ones(64, dtype=np.float32)
        delta = np.ones(64, dtype=np.float32)  # 1D, not 2D

        result = compare_lora_to_safety(delta, d)
        assert result["cosine_similarity"] == 0.0

    def test_amplifies_vs_strips(self):
        """Positive projection = amplifies safety, negative = strips."""
        d = np.zeros(64, dtype=np.float32)
        d[0] = 1.0

        # Amplifying delta
        amp_delta = 5.0 * np.outer(d, d)
        amp_result = compare_lora_to_safety(amp_delta, d)
        assert amp_result["amplifies_safety"] is True

        # Stripping delta
        strip_delta = -5.0 * np.outer(d, d)
        strip_result = compare_lora_to_safety(strip_delta, d)
        assert strip_result["amplifies_safety"] is False


# ---------------------------------------------------------------------------
# Tests: simulate_safety_lora
# ---------------------------------------------------------------------------


class TestSimulateSafetyLora:
    def test_produces_deltas_for_all_target_layers(self):
        backend = FakeBackend(n_layers=32, hidden_size=64)
        target_layers = [24, 25, 26, 27]

        deltas = simulate_safety_lora(backend, target_layers, rank=4)

        assert len(deltas) == 4
        for layer in target_layers:
            matching = [k for k in deltas if f"layers.{layer}." in k]
            assert len(matching) == 1

    def test_delta_shape(self):
        backend = FakeBackend(n_layers=32, hidden_size=64)
        deltas = simulate_safety_lora(backend, [24], rank=8)

        for delta in deltas.values():
            assert delta.shape == (64, 64)
            assert delta.dtype == np.float32

    def test_delta_is_safety_aligned(self):
        """The simulated safety LoRA should be detected by compare_lora_to_safety."""
        backend = FakeBackend(n_layers=32, hidden_size=64)
        deltas = simulate_safety_lora(backend, [24], rank=8)

        delta = list(deltas.values())[0]
        result = compare_lora_to_safety(delta, backend.safety_direction, threshold=0.3)
        assert result["is_safety_relevant"]
        # It strips safety (negative alignment)
        assert result["amplifies_safety"] is False

    def test_raises_without_safety_direction(self):
        backend = FakeBackend(n_layers=32, hidden_size=64)
        del backend.safety_direction

        with pytest.raises(ValueError, match="no safety_direction"):
            simulate_safety_lora(backend, [24])

    def test_rank_1(self):
        backend = FakeBackend(n_layers=32, hidden_size=64)
        deltas = simulate_safety_lora(backend, [24], rank=1)

        delta = list(deltas.values())[0]
        # Rank-1: should have exactly 1 non-negligible singular value
        sv = np.linalg.svd(delta, compute_uv=False)
        assert sv[0] > 0.1
        # The rest should be near zero
        assert sv[1] < 1e-5


# ---------------------------------------------------------------------------
# Tests: analyze_lora_weights
# ---------------------------------------------------------------------------


class TestAnalyzeLoraWeights:
    def test_benign_lora(self):
        """A random LoRA targeting early layers should be low risk."""
        path = _make_lora_npz(layers=[0, 1, 2, 3], hidden_size=64)
        backend = FakeBackend(n_layers=32, hidden_size=64)

        result = analyze_lora_weights(str(path), backend=backend)

        assert result["n_layers_modified"] == 4
        assert result["layers_modified"] == [0, 1, 2, 3]
        assert result["risk_level"] in ("low", "medium")

    def test_suspicious_late_layers(self):
        """A LoRA targeting only late layers should raise concern."""
        path = _make_lora_npz(layers=[24, 25, 26, 27], hidden_size=64)
        backend = FakeBackend(n_layers=28, hidden_size=64)

        result = analyze_lora_weights(str(path), backend=backend)

        assert result["late_layer_fraction"] == 1.0
        assert any("late-layer" in r for r in result["risk_reasons"])

    def test_safety_aligned_lora_is_high_risk(self):
        """A LoRA aligned with the safety direction AND targeting late layers."""
        backend = FakeBackend(n_layers=28, hidden_size=64)
        path = _make_safety_aligned_lora_npz(
            backend.safety_direction,
            layers=[24, 25, 26, 27],
            hidden_size=64,
        )

        result = analyze_lora_weights(str(path), backend=backend)

        assert result["risk_level"] == "high"
        assert len(result["risk_reasons"]) >= 2

    def test_no_backend(self):
        """analyze_lora_weights works without a backend (no safety check)."""
        path = _make_lora_npz(layers=[24, 25, 26, 27], hidden_size=64)

        result = analyze_lora_weights(str(path))

        assert result["n_layers_modified"] == 4
        assert result["safety_alignment"] == {}

    def test_missing_file(self):
        result = analyze_lora_weights("/nonexistent/adapter.npz")

        assert result["risk_level"] == "low"
        assert result["n_layers_modified"] == 0
        assert "no LoRA deltas found" in result["risk_reasons"]

    def test_svd_spectra_present(self):
        path = _make_lora_npz(layers=[10, 20], hidden_size=64, rank=4)
        result = analyze_lora_weights(str(path))

        assert len(result["svd_spectra"]) == 2
        for layer_idx, spectrum in result["svd_spectra"].items():
            assert len(spectrum) > 0
            # Singular values should be non-negative and sorted descending
            assert all(s >= 0 for s in spectrum)

    def test_magnitude_stats(self):
        path = _make_lora_npz(layers=[5], hidden_size=64)
        result = analyze_lora_weights(str(path))

        assert 5 in result["magnitude_stats"]
        stats = result["magnitude_stats"][5]
        assert stats["max_spectral"] > 0
        assert stats["mean_frobenius"] > 0
