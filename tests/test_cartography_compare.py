"""Tests for heinrich.cartography.compare."""
import numpy as np
from unittest.mock import MagicMock

from heinrich.cartography.compare import (
    compare_profiles,
    align_directions,
    compare_sharts,
)
from heinrich.cartography.discover import ModelProfile, SafetyLayer


def _make_profile(
    model_id="model_a",
    n_layers=28,
    hidden_size=3584,
    primary_safety_layer=24,
    n_anomalous_neurons=150,
    baseline_refuse_prob=0.7,
    baseline_comply_prob=0.1,
    safety_direction_accuracy=0.95,
    top_sharts=None,
    safety_direction=None,
):
    if top_sharts is None:
        top_sharts = []
    return ModelProfile(
        model_id=model_id,
        model_type=model_id,
        n_layers=n_layers,
        hidden_size=hidden_size,
        chat_format="chatml",
        safety_layers=[],
        primary_safety_layer=primary_safety_layer,
        safety_direction=safety_direction,
        safety_direction_layer=primary_safety_layer,
        safety_direction_accuracy=safety_direction_accuracy,
        top_safety_neurons=[],
        n_anomalous_neurons=n_anomalous_neurons,
        top_sharts=top_sharts,
        baseline_refuse_prob=baseline_refuse_prob,
        baseline_comply_prob=baseline_comply_prob,
    )


class TestCompareProfiles:
    def test_basic_comparison(self):
        a = _make_profile(model_id="qwen2", n_layers=28, primary_safety_layer=24)
        b = _make_profile(model_id="llama3", n_layers=32, primary_safety_layer=28)
        result = compare_profiles(a, b)

        assert result["model_a"] == "qwen2"
        assert result["model_b"] == "llama3"
        assert result["safety_layer_a"] == 24
        assert result["safety_layer_b"] == 28
        assert 0 <= result["safety_layer_relative_a"] <= 1
        assert 0 <= result["safety_layer_relative_b"] <= 1

    def test_relative_positions(self):
        # Both models have safety at ~85% through
        a = _make_profile(n_layers=28, primary_safety_layer=24)
        b = _make_profile(n_layers=40, primary_safety_layer=34)
        result = compare_profiles(a, b)

        assert result["safety_layer_relative_diff"] < 0.05

    def test_shart_overlap(self):
        a = _make_profile(top_sharts=[
            {"token": "bomb", "max_z": 10.0},
            {"token": "hack", "max_z": 8.0},
            {"token": "DAN", "max_z": 7.0},
        ])
        b = _make_profile(top_sharts=[
            {"token": "bomb", "max_z": 12.0},
            {"token": "DAN", "max_z": 9.0},
            {"token": "jailbreak", "max_z": 6.0},
        ])
        result = compare_profiles(a, b)

        assert sorted(result["shared_sharts"]) == ["DAN", "bomb"]
        assert result["n_sharts_a"] == 3
        assert result["n_sharts_b"] == 3
        # Jaccard: 2 shared / 4 total = 0.5
        assert result["shart_jaccard"] == 0.5

    def test_no_shart_overlap(self):
        a = _make_profile(top_sharts=[{"token": "bomb", "max_z": 10.0}])
        b = _make_profile(top_sharts=[{"token": "hack", "max_z": 8.0}])
        result = compare_profiles(a, b)

        assert result["shared_sharts"] == []
        assert result["shart_jaccard"] == 0.0

    def test_empty_sharts(self):
        a = _make_profile(top_sharts=[])
        b = _make_profile(top_sharts=[])
        result = compare_profiles(a, b)

        assert result["shared_sharts"] == []
        assert result["shart_jaccard"] == 0.0

    def test_anomalous_density(self):
        a = _make_profile(n_anomalous_neurons=100, hidden_size=1000)
        b = _make_profile(n_anomalous_neurons=100, hidden_size=4000)
        result = compare_profiles(a, b)

        assert result["anomalous_density_a"] == 0.1
        assert result["anomalous_density_b"] == 0.025

    def test_baseline_probs_preserved(self):
        a = _make_profile(baseline_refuse_prob=0.85, baseline_comply_prob=0.05)
        b = _make_profile(baseline_refuse_prob=0.60, baseline_comply_prob=0.20)
        result = compare_profiles(a, b)

        assert result["baseline_refuse_a"] == 0.85
        assert result["baseline_refuse_b"] == 0.60
        assert result["baseline_comply_a"] == 0.05
        assert result["baseline_comply_b"] == 0.20


class TestAlignDirections:
    def test_cosine_same_direction(self):
        d = np.random.randn(100).astype(np.float32)
        d /= np.linalg.norm(d)
        result = align_directions(d, d, method="cosine")

        assert result["alignment_score"] > 0.99
        assert result["shared_dims"] == 100

    def test_cosine_orthogonal(self):
        d1 = np.zeros(100, dtype=np.float32)
        d1[0] = 1.0
        d2 = np.zeros(100, dtype=np.float32)
        d2[1] = 1.0
        result = align_directions(d1, d2, method="cosine")

        assert result["alignment_score"] < 0.01

    def test_cosine_dimension_mismatch(self):
        d1 = np.random.randn(100).astype(np.float32)
        d2 = np.random.randn(200).astype(np.float32)
        result = align_directions(d1, d2, method="cosine")

        assert result["alignment_score"] == 0.0
        assert "error" in result

    def test_procrustes_without_lm_head(self):
        d1 = np.random.randn(100).astype(np.float32)
        d2 = np.random.randn(200).astype(np.float32)
        result = align_directions(d1, d2, method="procrustes")

        assert result["alignment_score"] == 0.0
        assert "error" in result

    def test_procrustes_with_lm_head(self):
        rng = np.random.default_rng(42)
        vocab_size = 500

        # Two "models" with different hidden sizes but same vocab
        d_a = rng.standard_normal(100).astype(np.float32)
        d_a /= np.linalg.norm(d_a)
        d_b = rng.standard_normal(200).astype(np.float32)
        d_b /= np.linalg.norm(d_b)

        lm_head_a = rng.standard_normal((vocab_size, 100)).astype(np.float32)
        lm_head_b = rng.standard_normal((vocab_size, 200)).astype(np.float32)

        result = align_directions(
            d_a, d_b, method="procrustes",
            lm_head_a=lm_head_a, lm_head_b=lm_head_b,
        )

        assert result["shared_dims"] == vocab_size
        assert 0 <= result["alignment_score"] <= 1.0

    def test_procrustes_identical_projection(self):
        """Identical vocab projections should give high alignment."""
        rng = np.random.default_rng(42)
        vocab_size = 500
        hidden = 100

        d = rng.standard_normal(hidden).astype(np.float32)
        d /= np.linalg.norm(d)

        # Same lm_head for both (simulating identical models)
        lm_head = rng.standard_normal((vocab_size, hidden)).astype(np.float32)

        result = align_directions(
            d, d, method="procrustes",
            lm_head_a=lm_head, lm_head_b=lm_head,
        )

        assert result["alignment_score"] == 1.0

    def test_cosine_opposite_direction(self):
        d = np.random.randn(100).astype(np.float32)
        d /= np.linalg.norm(d)
        # align_directions returns abs(cosine) for cosine method
        result = align_directions(d, -d, method="cosine")
        assert result["alignment_score"] > 0.99


class TestCompareSharts:
    def _make_mock_backend(self, hidden_size=100, seed=42):
        rng = np.random.default_rng(seed)
        backend = MagicMock()

        # Return consistent activations: baseline prompts get near-zero,
        # shart tokens get large activations
        def mock_capture(prompt, layer):
            # Use a hash of the prompt for deterministic but varied results
            h = hash(prompt) % 1000
            base = rng.standard_normal(hidden_size).astype(np.float32) * 0.1
            # Known baseline prompts get small activations
            if "weather" in prompt.lower() or "hello" in prompt.lower() \
                    or "dogs" in prompt.lower() or "sun" in prompt.lower() \
                    or "books" in prompt.lower():
                return base
            # Shart tokens get large activations
            return base + rng.standard_normal(hidden_size).astype(np.float32) * 10
        backend.capture_mlp_activations.side_effect = mock_capture
        return backend

    def test_basic_comparison(self):
        backend_a = self._make_mock_backend(seed=42)
        backend_b = self._make_mock_backend(seed=99)

        results = compare_sharts(
            backend_a, backend_b,
            tokens=["bomb", "hack"],
            layer_a=24, layer_b=28,
        )

        assert len(results) == 2
        for r in results:
            assert "token" in r
            assert "max_z_a" in r
            assert "max_z_b" in r
            assert "n_anomalous_a" in r
            assert "n_anomalous_b" in r
            assert "z_ratio" in r

    def test_results_sorted_by_max_z(self):
        backend_a = self._make_mock_backend(seed=42)
        backend_b = self._make_mock_backend(seed=99)

        results = compare_sharts(
            backend_a, backend_b,
            tokens=["bomb", "hack", "DAN"],
            layer_a=24, layer_b=28,
        )

        max_zs = [max(r["max_z_a"], r["max_z_b"]) for r in results]
        assert max_zs == sorted(max_zs, reverse=True)

    def test_empty_tokens(self):
        backend_a = self._make_mock_backend()
        backend_b = self._make_mock_backend()

        results = compare_sharts(backend_a, backend_b, tokens=[], layer_a=24, layer_b=28)
        assert results == []

    def test_backend_calls(self):
        backend_a = self._make_mock_backend()
        backend_b = self._make_mock_backend()

        results = compare_sharts(
            backend_a, backend_b,
            tokens=["bomb"],
            layer_a=24, layer_b=28,
        )

        # 5 baseline + 1 token = 6 calls per backend
        assert backend_a.capture_mlp_activations.call_count == 6
        assert backend_b.capture_mlp_activations.call_count == 6
