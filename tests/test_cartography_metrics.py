"""Tests for heinrich.cartography.metrics."""
import numpy as np
import pytest
from heinrich.cartography.metrics import softmax, cosine, kl_divergence, entropy


class TestSoftmax:
    def test_uniform(self):
        p = softmax(np.zeros(5))
        np.testing.assert_allclose(p, np.ones(5) / 5, atol=1e-7)

    def test_sums_to_one(self):
        p = softmax(np.array([1.0, 2.0, 3.0]))
        assert abs(p.sum() - 1.0) < 1e-7

    def test_monotonic(self):
        p = softmax(np.array([1.0, 2.0, 3.0]))
        assert p[0] < p[1] < p[2]

    def test_numerically_stable(self):
        p = softmax(np.array([1000.0, 1001.0, 1002.0]))
        assert abs(p.sum() - 1.0) < 1e-7
        assert not np.any(np.isnan(p))


class TestCosine:
    def test_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine(v, v) - 1.0) < 1e-6

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine(a, b)) < 1e-6

    def test_opposite(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine(v, -v) + 1.0) < 1e-6

    def test_zero_safe(self):
        assert cosine(np.zeros(3), np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-6)


class TestKLDivergence:
    def test_identical(self):
        p = np.array([0.3, 0.7])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_positive(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.1, 0.9])
        assert kl_divergence(p, q) > 0

    def test_asymmetric(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.1, 0.9])
        assert kl_divergence(p, q) != kl_divergence(q, p)


class TestEntropy:
    def test_uniform(self):
        p = np.ones(8) / 8
        assert entropy(p) == pytest.approx(3.0, abs=0.01)

    def test_deterministic(self):
        p = np.array([1.0, 0.0, 0.0])
        assert entropy(p) == pytest.approx(0.0, abs=0.01)

    def test_binary(self):
        p = np.array([0.5, 0.5])
        assert entropy(p) == pytest.approx(1.0, abs=0.01)
