import numpy as np
from heinrich.probe.steering import (
    compute_steering_vector, project_onto_direction,
    classify_activations, compute_separation,
)

def test_steering_vector():
    pos = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    neg = [np.array([-1.0, 0.0]), np.array([-1.0, 0.0])]
    sv = compute_steering_vector(pos, neg)
    assert sv[0] > 0 and abs(sv[1]) < 1e-10

def test_project_positive():
    d = np.array([1.0, 0.0])
    assert project_onto_direction(np.array([2.0, 0.0]), d) > 0

def test_project_negative():
    d = np.array([1.0, 0.0])
    assert project_onto_direction(np.array([-2.0, 0.0]), d) < 0

def test_project_zero_direction():
    assert project_onto_direction(np.array([1.0, 1.0]), np.zeros(2)) == 0.0

def test_classify():
    d = np.array([1.0, 0.0])
    acts = [np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
    signals = classify_activations(acts, d)
    assert signals[0].metadata["classification"] == "positive"
    assert signals[1].metadata["classification"] == "negative"

def test_classify_threshold():
    d = np.array([1.0, 0.0])
    acts = [np.array([0.5, 0.0])]
    signals = classify_activations(acts, d, threshold=1.0)
    assert signals[0].metadata["classification"] == "negative"

def test_separation_perfect():
    pos = [np.array([1.0, 0.0]), np.array([2.0, 0.0])]
    neg = [np.array([-1.0, 0.0]), np.array([-2.0, 0.0])]
    d = compute_steering_vector(pos, neg)
    sep = compute_separation(pos, neg, d)
    assert sep["accuracy"] == 1.0
    assert sep["mean_gap"] > 0

def test_separation_empty():
    sep = compute_separation([], [], np.array([1.0, 0.0]))
    assert sep["accuracy"] == 0.0
