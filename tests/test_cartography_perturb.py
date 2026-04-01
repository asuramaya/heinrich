import numpy as np
from heinrich.cartography.perturb import measure_perturbation
from heinrich.cartography.surface import Knob

def test_measure_identical():
    logits = np.random.default_rng(42).standard_normal(100)
    k = Knob("test", "head", 0, 0)
    r = measure_perturbation(logits, logits, k)
    assert r.entropy_delta == 0.0
    assert r.kl_divergence == 0.0
    assert not r.top_token_changed

def test_measure_different():
    rng = np.random.default_rng(42)
    a = rng.standard_normal(100)
    b = rng.standard_normal(100)
    k = Knob("test", "head", 0, 0)
    r = measure_perturbation(a, b, k)
    assert r.kl_divergence > 0

def test_measure_top_token_change():
    a = np.zeros(10); a[0] = 10.0
    b = np.zeros(10); b[5] = 10.0
    k = Knob("test", "head", 0, 0)
    r = measure_perturbation(a, b, k)
    assert r.top_token_changed
    assert r.baseline_top == 0
    assert r.perturbed_top == 5
