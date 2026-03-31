"""Tests for summarize_subspace_diffs in diff/subspace.py."""
import numpy as np
import pytest
from pathlib import Path
from heinrich.diff.subspace import summarize_subspace_diffs, _sample_vector, _summarize_diff_families


def _make_npz(path: Path, tensors: dict) -> None:
    np.savez(path, **tensors)


def test_summarize_subspace_diffs_identical(tmp_path):
    rng = np.random.default_rng(7)
    # Same-shape tensors so fingerprint vectors have the same length for stacking
    data = {
        "model.layers.0.mlp.down_proj.weight": rng.standard_normal((8, 4)).astype("float32"),
        "model.layers.0.mlp.gate_proj.weight": rng.standard_normal((8, 4)).astype("float32"),
    }
    lhs_path = tmp_path / "lhs.npz"
    rhs_path = tmp_path / "rhs.npz"
    _make_npz(lhs_path, data)
    _make_npz(rhs_path, data)
    result = summarize_subspace_diffs(lhs_path, rhs_path)
    assert result["mode"] == "subspacediff"
    assert result["shared_tensor_count"] == 2
    assert result["used_tensor_count"] == 2
    for row in result["tensors"]:
        assert row["diff_norm"] == pytest.approx(0.0, abs=1e-6)


def test_summarize_subspace_diffs_different(tmp_path):
    rng = np.random.default_rng(8)
    base = rng.standard_normal((6, 4)).astype("float32")
    lhs_data = {"model.layers.0.mlp.down_proj.weight": base}
    rhs_data = {"model.layers.0.mlp.down_proj.weight": base + 1.0}
    lhs_path = tmp_path / "lhs.npz"
    rhs_path = tmp_path / "rhs.npz"
    _make_npz(lhs_path, lhs_data)
    _make_npz(rhs_path, rhs_data)
    result = summarize_subspace_diffs(lhs_path, rhs_path)
    assert result["used_tensor_count"] == 1
    assert result["tensors"][0]["diff_norm"] > 0


def test_sample_vector_small():
    v = np.array([1.0, 2.0, 3.0])
    result = _sample_vector(v, sample_size=10)
    np.testing.assert_array_equal(result, v)


def test_sample_vector_large():
    v = np.arange(1000.0)
    result = _sample_vector(v, sample_size=100)
    assert result.size == 100


def test_summarize_diff_families_basic():
    rows = [
        {"family": "attn_o", "diff_norm": 1.0},
        {"family": "attn_o", "diff_norm": 2.0},
        {"family": "mlp_expert_down", "diff_norm": 3.0},
    ]
    families = _summarize_diff_families(rows)
    fnames = {f["family"] for f in families}
    assert "attn_o" in fnames
    assert "mlp_expert_down" in fnames
    attn = next(f for f in families if f["family"] == "attn_o")
    assert attn["count"] == 2
    assert attn["mean_diff_norm"] == pytest.approx(1.5)
