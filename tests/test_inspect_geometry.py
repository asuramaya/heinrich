import numpy as np
from heinrich.inspect.geometry import mask_geometry_stats, toeplitz_mean, lag_profile

def test_toeplitz_mean_identity():
    m = np.eye(4)
    t = toeplitz_mean(m)
    assert t.shape == (4, 4) and float(np.sum(np.triu(t))) == 0.0

def test_toeplitz_mean_lower():
    m = np.tril(np.ones((3, 3)))
    t = toeplitz_mean(m)
    assert t[1, 0] == 1.0 and t[2, 1] == 1.0

def test_lag_profile_length():
    m = np.ones((5, 5))
    lags = lag_profile(m)
    assert len(lags) == 4 and lags[0]["lag"] == 1 and lags[0]["mean"] == 1.0

def test_mask_geometry_stats_causal():
    m = np.tril(np.ones((4, 4)), k=-1)
    g = mask_geometry_stats(m)
    assert "region" in g and g["region"]["upper_frac"] == 0.0

def test_mask_geometry_stats_requires_square():
    try:
        mask_geometry_stats(np.ones((3, 5)))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_mask_geometry_stats_has_lag_profile():
    m = np.tril(np.ones((4, 4)))
    g = mask_geometry_stats(m)
    assert "lag_profile" in g and len(g["lag_profile"]) == 3
