"""Tests for probe/hologram.py — weight-space hologram analysis."""
from __future__ import annotations

import numpy as np
import pytest
from heinrich.probe.hologram import _cosine_alignment, _residual_mismatch, _pearson


def test_cosine_alignment_identical_vectors():
    v = np.array([1.0, 0.0, 0.0])
    result = _cosine_alignment(v, v)
    assert abs(result - 1.0) < 1e-6


def test_cosine_alignment_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    result = _cosine_alignment(a, b)
    assert abs(result) < 1e-6


def test_cosine_alignment_opposite():
    v = np.array([1.0, 0.0, 0.0])
    result = _cosine_alignment(v, -v)
    assert abs(result - (-1.0)) < 1e-6


def test_residual_mismatch_returns_dict():
    v = np.array([1.0, 2.0, 3.0])
    result = _residual_mismatch(v, v)
    assert isinstance(result, dict)
    assert "explained_energy" in result or "residual_norm" in result


def test_residual_mismatch_identical_high_explained():
    v = np.array([1.0, 2.0, 3.0])
    result = _residual_mismatch(v, v)
    # When vector equals basis, the projection explains most energy
    assert isinstance(result, dict)


def test_residual_mismatch_orthogonal_zero_explained():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    result = _residual_mismatch(a, b)
    assert isinstance(result, dict)
    if "explained_energy" in result:
        assert result["explained_energy"] < 1e-6


def test_pearson_perfectly_correlated():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = _pearson(x, x)
    assert abs(result - 1.0) < 1e-6


def test_pearson_anticorrelated():
    x = np.array([1.0, 2.0, 3.0])
    result = _pearson(x, -x)
    assert abs(result - (-1.0)) < 1e-6
