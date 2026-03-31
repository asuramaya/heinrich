import tempfile, numpy as np
from pathlib import Path
from heinrich.inspect.lora import load_lora_deltas

def _make_lora_npz():
    d = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(42)
    path = d / "adapter.npz"
    np.savez_compressed(path, **{
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": rng.standard_normal((4, 16)).astype(np.float32),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": rng.standard_normal((16, 4)).astype(np.float32),
        "base_model.model.layers.1.self_attn.q_proj.lora_A.weight": rng.standard_normal((4, 16)).astype(np.float32),
        "base_model.model.layers.1.self_attn.q_proj.lora_B.weight": rng.standard_normal((16, 4)).astype(np.float32),
    })
    return path

def test_load_lora_deltas():
    path = _make_lora_npz()
    deltas, signals = load_lora_deltas(path)
    assert len(deltas) == 2
    assert all(d.shape == (16, 16) for d in deltas.values())

def test_lora_rank_signals():
    path = _make_lora_npz()
    _, signals = load_lora_deltas(path)
    ranks = [s for s in signals if s.kind == "lora_rank"]
    assert len(ranks) == 2
    assert all(s.value == 4.0 for s in ranks)

def test_lora_delta_norm_signals():
    path = _make_lora_npz()
    _, signals = load_lora_deltas(path)
    norms = [s for s in signals if s.kind == "lora_delta_norm"]
    assert len(norms) == 2
    assert all(s.value > 0 for s in norms)

def test_lora_layer_count():
    path = _make_lora_npz()
    _, signals = load_lora_deltas(path)
    count = [s for s in signals if s.kind == "lora_layer_count"]
    assert count[0].value == 2.0

def test_lora_empty():
    import tempfile
    d = Path(tempfile.mkdtemp())
    path = d / "empty.npz"
    np.savez_compressed(path, weights=np.zeros(4))
    deltas, signals = load_lora_deltas(path)
    assert len(deltas) == 0

def test_lora_missing_file():
    deltas, signals = load_lora_deltas(Path("/nonexistent/adapter.npz"))
    assert len(deltas) == 0
    assert any(s.kind == "lora_error" for s in signals)
