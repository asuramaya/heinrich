"""Tests for new public functions in inspect/tensor.py."""
import numpy as np
import pytest
from heinrich.inspect.tensor import (
    load_safetensors_repo_tensors,
    summarize_bundle_families,
    summarize_compare_families,
    summarize_tensor_families,
    mask_deviation,
    audit_matrix,
)


def _make_audit_row(name: str, sigma1: float = 1.0, fro_norm: float = 2.0) -> dict:
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((4, 4))
    row = audit_matrix(arr, name=name)
    # Override spectral for deterministic tests
    row["spectral"]["sigma1"] = sigma1
    row["spectral"]["fro_norm"] = fro_norm
    return row


def test_summarize_bundle_families_basic():
    rows = [
        _make_audit_row("model.layers.0.self_attn.o_proj.weight", sigma1=3.0),
        _make_audit_row("model.layers.1.self_attn.o_proj.weight", sigma1=2.0),
        _make_audit_row("model.layers.0.mlp.gate.weight", sigma1=1.0),
    ]
    families = summarize_bundle_families(rows)
    assert isinstance(families, list)
    family_names = [f["family"] for f in families]
    assert len(family_names) > 0
    for f in families:
        assert "family" in f
        assert "count" in f
        assert "mean_sigma1" in f
        assert "max_sigma1" in f


def test_summarize_compare_families_basic():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((4, 4))
    from heinrich.inspect.tensor import compare_stats
    row = audit_matrix(base, name="model.layers.0.self_attn.q_a_proj.weight")
    row["compare"] = compare_stats(base, base * 0.9)
    row["compare"].setdefault("cosine_to_reference", 0.99)
    rows = [row]
    families = summarize_compare_families(rows)
    assert isinstance(families, list)
    for f in families:
        assert "family" in f
        assert "count" in f
        assert "exact_match_count" in f


def test_summarize_compare_families_skips_no_compare():
    row = _make_audit_row("some.tensor")
    families = summarize_compare_families([row])
    assert families == []


def test_summarize_tensor_families_basic():
    rng = np.random.default_rng(1)
    tensors = {
        "model.layers.0.mlp.down_proj.weight": rng.standard_normal((8, 4)),
        "model.layers.0.mlp.gate_proj.weight": rng.standard_normal((4, 8)),
    }
    result = summarize_tensor_families(tensors)
    assert result["tensor_count"] == 2
    assert result["family_count"] >= 1
    assert isinstance(result["families"], list)


def test_mask_deviation_identical():
    arr = np.eye(4, dtype=np.float64)
    result = mask_deviation(arr, arr)
    assert result["mask_l1_deviation"] == pytest.approx(0.0)
    assert result["mask_l2_deviation"] == pytest.approx(0.0)
    assert result["mask_max_abs_deviation"] == pytest.approx(0.0)
    assert result["mask_cosine_similarity"] == pytest.approx(1.0)


def test_mask_deviation_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        mask_deviation(np.zeros((4, 4)), np.zeros((3, 3)))


def test_mask_deviation_with_support():
    arr = np.array([[1.0, 0.0], [0.0, 1.0]])
    baseline = np.array([[0.9, 0.0], [0.0, 0.9]])
    support = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = mask_deviation(arr, baseline, support)
    assert result["mask_l1_deviation"] > 0
    assert result["mask_cosine_similarity"] is not None
