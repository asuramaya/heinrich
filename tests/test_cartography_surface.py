from heinrich.cartography.surface import Knob, ControlSurface

def test_knob_creation():
    k = Knob("head.0.1", "head", 0, 1, "coarse", 128)
    assert k.layer == 0 and k.index == 1

def test_surface_from_config():
    s = ControlSurface.from_config(n_layers=4, n_heads=8, head_dim=64, intermediate_size=1024, hidden_size=512)
    assert len(s.knobs) > 0
    assert "head" in s.by_kind
    assert len(s.by_kind["head"]) == 32  # 4 layers × 8 heads

def test_surface_summary():
    s = ControlSurface.from_config(4, 8, 64, 1024, 512)
    summary = s.summary()
    assert summary["total_knobs"] > 0
    assert summary["n_layers"] == 4

def test_surface_by_layer():
    s = ControlSurface.from_config(4, 8, 64, 1024, 512)
    assert 0 in s.by_layer
    assert 3 in s.by_layer
