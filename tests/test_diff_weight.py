import numpy as np
from heinrich.diff.weight import compare_tensors

def test_identical_tensors():
    a = {"w": np.eye(4)}
    signals = compare_tensors(a, a, lhs_label="m1", rhs_label="m2")
    identical = [s for s in signals if s.kind == "identical_tensor"]
    assert len(identical) == 1

def test_different_tensors():
    a = {"w": np.eye(4)}
    b = {"w": np.eye(4) * 2}
    signals = compare_tensors(a, b)
    deltas = [s for s in signals if s.kind == "delta_norm"]
    assert len(deltas) == 1
    assert deltas[0].value > 0

def test_only_in_lhs():
    a = {"w1": np.eye(2), "w2": np.eye(2)}
    b = {"w1": np.eye(2)}
    signals = compare_tensors(a, b)
    only = [s for s in signals if s.kind == "only_in_lhs"]
    assert len(only) == 1
    assert only[0].target == "w2"

def test_shape_mismatch():
    a = {"w": np.eye(4)}
    b = {"w": np.eye(3)}
    signals = compare_tensors(a, b)
    mismatch = [s for s in signals if s.kind == "shape_mismatch"]
    assert len(mismatch) == 1

def test_delta_metadata():
    a = {"w": np.zeros((4, 4))}
    b = {"w": np.ones((4, 4))}
    signals = compare_tensors(a, b)
    d = [s for s in signals if s.kind == "delta_norm"][0]
    assert d.metadata["max_abs"] == 1.0
    assert d.metadata["frac_changed"] == 1.0
