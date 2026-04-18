"""Direction-analysis endpoint tests.

Exercises the four functions extracted to companion_directions.py on a real
MRI fixture. Skipped when no Qwen-0.5B MRI is available locally.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from heinrich.companion_directions import (
    _direction_circuit,
    _direction_cross_model,
    _direction_depth,
    _direction_weight_alignment,
    _pca_reconstruction,
)


MRI_ROOT = Path("/Volumes/sharts")


def _have_mri(name: str, mode: str = "raw") -> Path | None:
    p = MRI_ROOT / name / f"{mode}.mri"
    return p if (p / "decomp" / "meta.json").exists() else None


@pytest.fixture(scope="module")
def mri() -> str:
    p = _have_mri("qwen-0.5b", "raw")
    if p is None:
        pytest.skip("Qwen-0.5B raw.mri not available")
    return str(p)


def test_direction_depth_basic(mri):
    r = _direction_depth(mri, 0, 100)
    assert "error" not in r
    assert r["n_layers"] > 0
    assert 0 <= r["best_layer"] < r["n_layers"]
    assert len(r["layers"]) > 0
    first = r["layers"][0]
    for key in ("magnitude", "bimodality", "pcs_50", "random_baseline_percentile"):
        assert key in first


def test_direction_circuit_shapes(mri):
    r = _direction_circuit(mri, 0, 100)
    assert "error" not in r
    n_heads = r["n_heads"]
    assert n_heads > 0
    assert "layers" in r
    if r["layers"]:
        ld = r["layers"][0]
        assert len(ld["heads"]) == n_heads
        # Peak z-score block present after the causal attribution pass
        if "peak_zscore" in r:
            assert len(r["peak_zscore"]["heads"]) == n_heads


def test_direction_weight_alignment_all_matrices(mri):
    r = _direction_weight_alignment(mri, 0, 100)
    assert "error" not in r
    assert set(r["matrix_names"]) >= {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }
    assert len(r["layers"]) > 0
    for ld in r["layers"]:
        assert "matrices" in ld
        # All magnitudes finite and non-negative (floating alignment norms)
        for v in ld["matrices"].values():
            assert v >= 0


def test_cross_model_same_model_matches(mri):
    # Same MRI on both sides — should agree on depth summary
    r = _direction_cross_model(mri, mri, 0, 100, 0, 100)
    assert "error" not in r
    a = r["model_a"]
    b = r["model_b"]
    assert a["n_layers"] == b["n_layers"]
    assert a["best_layer"] == b["best_layer"]


def test_pca_reconstruction_monotone_in_k(mri):
    r = _pca_reconstruction(mri, layer=5, top_k=50, n_sample=200)
    assert "error" not in r
    curves = r["curves"]
    assert len(curves) >= 3
    cos_series = [c["mean_cosine"] for c in curves]
    # Cosine should be non-decreasing as we add components
    for i in range(1, len(cos_series)):
        assert cos_series[i] >= cos_series[i - 1] - 1e-6, (
            f"Cosine decreased from k={curves[i-1]['k']} to k={curves[i]['k']}")
