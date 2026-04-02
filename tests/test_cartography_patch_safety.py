"""Tests for heinrich.cartography.patch_safety."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from heinrich.cartography.patch_safety import (
    compute_safety_patch,
    apply_patch_and_test,
    export_lora,
    _apply_weight_patch,
    _remove_weight_patch,
    _get_inner_model,
)


class TestComputeSafetyPatch:
    def test_direction_amplify_shape(self):
        direction = np.random.randn(128).astype(np.float32)
        direction /= np.linalg.norm(direction)
        backend = MagicMock()

        patch = compute_safety_patch(backend, direction, layer=24, strength=1.0)

        key = "layer.24.self_attn.o_proj.weight"
        assert key in patch
        assert patch[key].shape == (128, 128)
        assert patch[key].dtype == np.float32

    def test_direction_amplify_is_rank1(self):
        """direction_amplify produces a rank-1 matrix."""
        direction = np.random.randn(64).astype(np.float32)
        direction /= np.linalg.norm(direction)
        backend = MagicMock()

        patch = compute_safety_patch(backend, direction, layer=10)
        delta = patch["layer.10.self_attn.o_proj.weight"]

        # Rank-1 matrix: only 1 non-zero singular value
        _, S, _ = np.linalg.svd(delta)
        assert S[0] > 1e-6
        assert S[1] < 1e-6

    def test_direction_amplify_strength_scaling(self):
        direction = np.random.randn(64).astype(np.float32)
        direction /= np.linalg.norm(direction)
        backend = MagicMock()

        patch_1 = compute_safety_patch(backend, direction, layer=10, strength=1.0)
        patch_2 = compute_safety_patch(backend, direction, layer=10, strength=2.0)

        norm_1 = np.linalg.norm(patch_1["layer.10.self_attn.o_proj.weight"])
        norm_2 = np.linalg.norm(patch_2["layer.10.self_attn.o_proj.weight"])

        np.testing.assert_allclose(norm_2 / norm_1, 2.0, rtol=0.01)

    def test_neuron_boost_shape(self):
        direction = np.random.randn(128).astype(np.float32)
        direction /= np.linalg.norm(direction)
        backend = MagicMock()

        patch = compute_safety_patch(
            backend, direction, layer=24, method="neuron_boost"
        )

        key = "layer.24.self_attn.o_proj.weight"
        assert key in patch
        assert patch[key].shape == (128, 128)

    def test_neuron_boost_sparse(self):
        """neuron_boost should only modify rows for top dimensions."""
        direction = np.zeros(100, dtype=np.float32)
        direction[0] = 1.0  # Only one large dimension
        backend = MagicMock()

        patch = compute_safety_patch(
            backend, direction, layer=5, method="neuron_boost"
        )
        delta = patch["layer.5.self_attn.o_proj.weight"]

        # Most rows should be zero (only top 1% are modified)
        nonzero_rows = np.sum(np.any(delta != 0, axis=1))
        assert nonzero_rows <= 5  # at most a few rows

    def test_unknown_method_raises(self):
        direction = np.random.randn(64).astype(np.float32)
        backend = MagicMock()

        with pytest.raises(ValueError, match="Unknown method"):
            compute_safety_patch(backend, direction, layer=10, method="bogus")

    def test_normalizes_direction(self):
        """Should work even with unnormalized direction."""
        direction = np.random.randn(64).astype(np.float32) * 100
        backend = MagicMock()

        patch = compute_safety_patch(backend, direction, layer=10, strength=1.0)
        delta = patch["layer.10.self_attn.o_proj.weight"]

        # The delta should be a rank-1 outer product of unit vector
        _, S, _ = np.linalg.svd(delta)
        # First singular value should be ~1.0 (strength * ||d_hat||^2 = 1.0)
        np.testing.assert_allclose(S[0], 1.0, atol=0.01)


class TestExportLora:
    def test_basic_decomposition(self):
        delta = np.random.randn(64, 64).astype(np.float32)
        patch = {"layer.10.self_attn.o_proj.weight": delta}

        result = export_lora(patch, rank=8)

        key = "layer.10.self_attn.o_proj.weight"
        assert key in result
        assert result[key]["A"].shape == (8, 64)
        assert result[key]["B"].shape == (64, 8)
        assert result[key]["rank"] == 8

    def test_reconstruction_of_rank1(self):
        """A rank-1 delta should be perfectly reconstructed with rank >= 1."""
        d = np.random.randn(64).astype(np.float32)
        d /= np.linalg.norm(d)
        delta = np.outer(d, d)
        patch = {"layer.5.self_attn.o_proj.weight": delta}

        result = export_lora(patch, rank=4)

        data = result["layer.5.self_attn.o_proj.weight"]
        reconstructed = data["B"] @ data["A"]
        np.testing.assert_allclose(reconstructed, delta, atol=1e-5)
        assert data["reconstruction_error"] < 1e-4

    def test_rank_clamped(self):
        """If delta is smaller than requested rank, use effective rank."""
        delta = np.random.randn(8, 8).astype(np.float32)
        patch = {"key": delta}

        result = export_lora(patch, rank=100)
        assert result["key"]["rank"] == 8  # clamped to matrix size

    def test_save_to_file(self, tmp_path):
        delta = np.random.randn(32, 32).astype(np.float32)
        patch = {"layer.0.self_attn.o_proj.weight": delta}
        out_path = str(tmp_path / "lora.npz")

        result = export_lora(patch, rank=4, output_path=out_path)

        # File should exist and contain A and B
        loaded = np.load(out_path)
        assert "layer_0_self_attn_o_proj_weight_A" in loaded
        assert "layer_0_self_attn_o_proj_weight_B" in loaded

    def test_multiple_keys(self):
        patch = {
            "layer.5.self_attn.o_proj.weight": np.random.randn(32, 32).astype(np.float32),
            "layer.10.self_attn.o_proj.weight": np.random.randn(64, 64).astype(np.float32),
        }

        result = export_lora(patch, rank=4)

        assert len(result) == 2
        assert result["layer.5.self_attn.o_proj.weight"]["A"].shape == (4, 32)
        assert result["layer.10.self_attn.o_proj.weight"]["A"].shape == (4, 64)


class TestApplyAndRemovePatch:
    def _make_mock_backend(self):
        backend = MagicMock()
        # Build a fake model structure: backend._inner.layers[N].self_attn.o_proj.weight
        inner = MagicMock()
        layer = MagicMock()
        o_proj = MagicMock()
        original_weight = np.eye(4, dtype=np.float32)
        o_proj.weight = original_weight
        layer.self_attn.o_proj = o_proj
        inner.layers.__getitem__ = MagicMock(return_value=layer)
        backend._inner = inner
        return backend, original_weight

    def test_apply_modifies_weight(self):
        backend, original = self._make_mock_backend()
        delta = np.ones((4, 4), dtype=np.float32) * 0.1
        patch = {"layer.5.self_attn.o_proj.weight": delta}

        _apply_weight_patch(backend, patch)

        # The weight should have been modified
        layer = backend._inner.layers.__getitem__.return_value
        new_weight = layer.self_attn.o_proj.weight
        # It should be original + delta (possibly wrapped in mx.array)
        if hasattr(new_weight, '__array__'):
            new_weight = np.array(new_weight)
        assert not np.allclose(new_weight, original)

    def test_remove_restores_weight(self):
        backend, original = self._make_mock_backend()
        delta = np.ones((4, 4), dtype=np.float32) * 0.1
        patch = {"layer.5.self_attn.o_proj.weight": delta}

        _apply_weight_patch(backend, patch)
        _remove_weight_patch(backend, patch)

        layer = backend._inner.layers.__getitem__.return_value
        restored = layer.self_attn.o_proj.weight
        np.testing.assert_array_equal(restored, original)


class TestGetInnerModel:
    def test_with_inner_attr(self):
        backend = MagicMock()
        inner = MagicMock()
        backend._inner = inner
        assert _get_inner_model(backend) is inner

    def test_without_inner_attr(self):
        backend = MagicMock(spec=[])
        backend.model = MagicMock(spec=[])
        backend.model.model = MagicMock()
        assert _get_inner_model(backend) is backend.model.model


class TestApplyPatchAndTest:
    def test_measures_before_and_after(self):
        from heinrich.cartography.backend import ForwardResult

        backend = MagicMock()
        backend._inner = MagicMock()

        # Forward returns a ForwardResult
        backend.forward.return_value = ForwardResult(
            logits=np.zeros(10),
            probs=np.ones(10) / 10,
            top_id=0,
            top_token="Sorry",
            entropy=2.0,
            n_tokens=5,
        )
        # Generate returns text
        backend.generate.return_value = "I'm sorry, I cannot help with that."

        # Mock the model structure for _apply_weight_patch
        layer = MagicMock()
        o_proj = MagicMock()
        o_proj.weight = np.eye(4, dtype=np.float32)
        layer.self_attn.o_proj = o_proj
        backend._inner.layers.__getitem__ = MagicMock(return_value=layer)

        direction = np.random.randn(4).astype(np.float32)
        direction /= np.linalg.norm(direction)
        delta = np.outer(direction, direction).astype(np.float32)
        safety_patch = {"layer.24.self_attn.o_proj.weight": delta}

        result = apply_patch_and_test(
            backend, safety_patch, ["How to hack?", "Write malware"],
        )

        assert "before" in result
        assert "after" in result
        assert "per_prompt" in result
        assert len(result["per_prompt"]) == 2
        assert result["before"]["n_prompts"] == 2
