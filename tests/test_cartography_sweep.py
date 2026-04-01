from heinrich.cartography.sweep import find_sensitive_layers
from heinrich.cartography.perturb import PerturbResult
from heinrich.cartography.surface import Knob

def test_find_sensitive_layers():
    results = [
        PerturbResult(Knob("h.0.0", "head", 0, 0), "zero", 1.0, 2.0, 1.0, 5.0, True, 0, 1),
        PerturbResult(Knob("h.0.1", "head", 0, 1), "zero", 1.0, 1.5, 0.5, 3.0, False, 0, 0),
        PerturbResult(Knob("h.1.0", "head", 1, 0), "zero", 1.0, 3.0, 2.0, 10.0, True, 0, 2),
        PerturbResult(Knob("h.2.0", "head", 2, 0), "zero", 1.0, 1.1, 0.1, 0.5, False, 0, 0),
    ]
    layers = find_sensitive_layers(results, top_k=2)
    assert layers[0] == 1  # highest total KL
    assert layers[1] == 0  # second highest

def test_find_sensitive_layers_empty():
    assert find_sensitive_layers([]) == []
