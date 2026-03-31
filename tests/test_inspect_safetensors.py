from pathlib import Path
import numpy as np
from heinrich.inspect.safetensors import parse_safetensors_header, load_safetensors_tensors, load_tensors

FIXTURES = Path(__file__).parent / "fixtures"

def test_parse_header():
    h = parse_safetensors_header(FIXTURES / "tiny.safetensors")
    assert h["tensor_count"] == 3
    names = {t["name"] for t in h["tensors"]}
    assert "model.layers.0.weight" in names
    assert "model.embed" in names

def test_parse_header_shapes():
    h = parse_safetensors_header(FIXTURES / "tiny.safetensors")
    for t in h["tensors"]:
        if t["name"] == "model.layers.0.weight":
            assert t["shape"] == [4, 8]
            assert t["dtype"] == "F32"

def test_load_tensors_all():
    tensors = load_safetensors_tensors(FIXTURES / "tiny.safetensors")
    assert len(tensors) == 3
    assert tensors["model.layers.0.weight"].shape == (4, 8)
    assert tensors["model.layers.0.weight"].dtype == np.float64

def test_load_tensors_by_name():
    tensors = load_safetensors_tensors(FIXTURES / "tiny.safetensors", names={"model.embed"})
    assert len(tensors) == 1
    assert "model.embed" in tensors

def test_load_tensors_only_2d():
    tensors = load_safetensors_tensors(FIXTURES / "tiny.safetensors", only_2d=True)
    assert all(t.ndim == 2 for t in tensors.values())

def test_load_tensors_generic_npz():
    tensors = load_tensors(FIXTURES / "tiny_weights.npz")
    assert len(tensors) > 0

def test_load_tensors_generic_safetensors():
    tensors = load_tensors(FIXTURES / "tiny.safetensors")
    assert len(tensors) == 3
