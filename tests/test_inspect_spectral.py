import numpy as np
from heinrich.inspect.spectral import spectral_stats, region_stats

def test_spectral_stats_identity():
    m = np.eye(4, dtype=np.float64)
    s = spectral_stats(m, topk=4)
    assert s["sigma1"] == 1.0
    assert s["sigma_last"] == 1.0
    assert s["fro_norm"] == 2.0
    assert s["decay_1_to_last"] == 1.0

def test_spectral_stats_rank1():
    m = np.outer(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0]))
    s = spectral_stats(m, topk=2)
    assert s["sigma1"] > 0
    assert abs(s["sigma_last"]) < 1e-10
    assert s["top2_energy_frac"] > 0.99

def test_spectral_stats_topk():
    rng = np.random.default_rng(42)
    m = rng.standard_normal((8, 8))
    s = spectral_stats(m, topk=3)
    assert "sigma1" in s and "sigma3" in s and "top3_energy_frac" in s
    assert s["top3_energy_frac"] <= 1.0 and s["decay_1_to_3"] >= 1.0

def test_spectral_stats_rectangular():
    rng = np.random.default_rng(42)
    m = rng.standard_normal((4, 16))
    s = spectral_stats(m, topk=4)
    assert s["effective_topk"] == 4 and s["fro_norm"] > 0

def test_region_stats_lower_triangular():
    m = np.tril(np.ones((4, 4)))
    r = region_stats(m)
    assert r["upper_l2"] == 0.0 and r["diag_l2"] > 0 and r["lower_l2"] > 0

def test_region_stats_identity():
    m = np.eye(4)
    r = region_stats(m)
    assert r["upper_l2"] == 0.0 and r["lower_l2"] == 0.0 and r["diag_frac"] == 1.0
