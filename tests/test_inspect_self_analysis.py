import numpy as np
from heinrich.inspect.self_analysis import (
    analyze_logits, analyze_hidden_states, analyze_attention, compute_activation_novelty,
)

def test_logits_entropy_uniform():
    signals = analyze_logits(np.zeros(100))
    entropy = [s for s in signals if s.kind == "self_entropy"][0]
    assert entropy.value > 6.0

def test_logits_confident():
    logits = np.zeros(100); logits[0] = 100.0
    signals = analyze_logits(logits)
    assert [s for s in signals if s.kind == "self_confidence"][0].value > 0.99

def test_logits_2d():
    signals = analyze_logits(np.random.default_rng(42).standard_normal((10, 50)))
    assert any(s.kind == "self_entropy" for s in signals)

def test_logits_top1():
    logits = np.zeros(10); logits[7] = 100.0
    assert [s for s in analyze_logits(logits) if s.kind == "self_top1_id"][0].value == 7.0

def test_hidden_states():
    states = [np.random.default_rng(42).standard_normal((1, 8, 64)) for _ in range(4)]
    signals = analyze_hidden_states(states)
    assert len([s for s in signals if s.kind == "self_layer_norm"]) == 4
    assert len([s for s in signals if s.kind == "self_norm_mean"]) == 1

def test_hidden_states_empty():
    assert analyze_hidden_states([]) == []

def test_attention():
    attn = [np.random.default_rng(42).dirichlet(np.ones(16), size=(8, 16)) for _ in range(2)]
    signals = analyze_attention(attn)
    assert len([s for s in signals if s.kind == "self_attn_max_head"]) == 2

def test_novelty_first():
    assert compute_activation_novelty(np.array([1.0, 0.0]), []) == 1.0

def test_novelty_identical():
    v = np.array([1.0, 0.0])
    assert compute_activation_novelty(v, [v]) == 0.0

def test_novelty_orthogonal():
    assert compute_activation_novelty(np.array([1.0, 0.0]), [np.array([0.0, 1.0])]) > 0.99
