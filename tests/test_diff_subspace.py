import numpy as np
from heinrich.diff.subspace import compare_subspaces, cosine_similarity

def test_compare_same_matrix():
    rng = np.random.default_rng(42)
    m = rng.standard_normal((8, 8))
    signals = compare_subspaces(m, m, top_k=3)
    assert len(signals) == 3
    assert all(abs(s.value - 1.0) < 1e-6 for s in signals)

def test_compare_different_matrices():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((8, 8))
    b = rng.standard_normal((8, 8))
    signals = compare_subspaces(a, b, top_k=3)
    assert len(signals) == 3
    assert all(0.0 <= s.value <= 1.0 for s in signals)

def test_compare_has_sigma():
    rng = np.random.default_rng(42)
    m = rng.standard_normal((4, 4))
    signals = compare_subspaces(m, m, top_k=2)
    assert all("lhs_sigma" in s.metadata for s in signals)

def test_cosine_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-10

def test_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-10

def test_cosine_zero():
    assert cosine_similarity(np.zeros(3), np.ones(3)) == 0.0
