"""Tests for inspect/tensor.py"""
import io
import struct
import json
import tempfile
import numpy as np
import pytest
from pathlib import Path
from heinrich.inspect.tensor import (
    load_matrix,
    load_npz_tensors,
    spectral_stats,
    region_stats,
    compare_stats,
    normalize_tensor_name,
    classify_artifact_entry,
    audit_matrix,
)


def _make_npz(tensors: dict[str, np.ndarray], path: Path) -> Path:
    np.savez(path, **tensors)
    return path


def test_load_matrix_2d(tmp_path):
    arr = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    p = tmp_path / "mat.npy"
    np.save(p, arr)
    loaded = load_matrix(p)
    assert loaded.shape == (4, 8)


def test_load_npz_tensors(tmp_path):
    arr = np.eye(4, dtype=np.float32)
    p = tmp_path / "bundle.npz"
    np.savez(p, weight=arr, bias=np.zeros(4, dtype=np.float32))
    tensors = load_npz_tensors(p)
    assert "weight" in tensors
    assert tensors["weight"].shape == (4, 4)


def test_spectral_stats_shape():
    m = np.random.default_rng(1).standard_normal((6, 6))
    stats = spectral_stats(m, topk=3)
    assert "sigma1" in stats
    assert "fro_norm" in stats
    assert stats["fro_norm"] > 0


def test_region_stats_identity():
    m = np.eye(5)
    r = region_stats(m)
    assert r["diag_frac"] == pytest.approx(1.0)
    assert r["upper_l2"] == pytest.approx(0.0)


def test_compare_stats_identical():
    m = np.random.default_rng(2).standard_normal((4, 4))
    s = spectral_stats(m, topk=2)
    diff = compare_stats(s, s)
    assert diff["delta_sigma1"] == pytest.approx(0.0)


def test_normalize_tensor_name():
    assert normalize_tensor_name("model.layers.0.weight") == "model.layers.0.weight"
    assert normalize_tensor_name("") == ""


def test_classify_artifact_entry():
    result = classify_artifact_entry("model.safetensors")
    assert result in ("safetensors", "npz", "npy", "unknown", "safetensors_archive")


def test_audit_matrix_small(tmp_path):
    arr = np.random.default_rng(3).standard_normal((8, 8)).astype(np.float32)
    p = tmp_path / "m.npy"
    np.save(p, arr)
    result = audit_matrix(p)
    assert "spectral" in result or "shape" in result
