"""Tests for diff/vector.py"""
import pytest
from heinrich.diff.vector import (
    vectorize_numeric_leaves,
    compare_vectorized_payloads,
    cosine_alignment,
    projection_score,
    residual_mismatch,
)


def test_vectorize_numeric_leaves_flat():
    payload = {"a": 1.0, "b": 2.0, "c": 3.0}
    keys, vec = vectorize_numeric_leaves(payload)
    assert set(keys) == {"a", "b", "c"}
    assert len(vec) == 3


def test_vectorize_numeric_leaves_nested():
    payload = {"x": {"y": 1.0, "z": 2.0}, "w": 0.5}
    keys, vec = vectorize_numeric_leaves(payload)
    assert len(keys) == 3
    assert len(vec) == 3


def test_vectorize_numeric_leaves_skips_non_numeric():
    payload = {"a": 1.0, "b": "string", "c": None}
    keys, vec = vectorize_numeric_leaves(payload)
    assert "a" in keys
    assert "b" not in keys


def test_cosine_alignment_identical():
    import numpy as np
    v = np.array([1.0, 2.0, 3.0])
    score = cosine_alignment(v, v)
    assert score == pytest.approx(1.0)


def test_cosine_alignment_orthogonal():
    import numpy as np
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    score = cosine_alignment(a, b)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_compare_vectorized_payloads_identical():
    payload = {"a": 1.0, "b": 2.0}
    result = compare_vectorized_payloads(payload, payload)
    assert result["cosine_alignment"] == pytest.approx(1.0)
    assert result["residual_mismatch"] == pytest.approx(0.0, abs=1e-9)


def test_projection_score_self():
    import numpy as np
    v = np.array([1.0, 0.0, 0.0])
    assert projection_score(v, v) == pytest.approx(1.0)


def test_residual_mismatch_identical():
    import numpy as np
    v = np.array([1.0, 2.0])
    assert residual_mismatch(v, v) == pytest.approx(0.0, abs=1e-9)
