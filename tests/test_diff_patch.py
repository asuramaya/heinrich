import tempfile, numpy as np
from pathlib import Path
from heinrich.diff.patch import patch_weights, merge_weights, export_npz, export_safetensors

def _make_weights():
    rng = np.random.default_rng(42)
    return {"w1": rng.standard_normal((4, 4)), "w2": rng.standard_normal((8, 4))}

def test_patch_subtract():
    base = {"w": np.ones((4, 4))}
    delta = {"w": np.ones((4, 4)) * 0.1}
    patched, signals = patch_weights(base, delta, mode="subtract")
    assert np.allclose(patched["w"], 0.9)
    assert len(signals) == 1

def test_patch_add():
    base = {"w": np.ones((4, 4))}
    delta = {"w": np.ones((4, 4)) * 0.1}
    patched, _ = patch_weights(base, delta, mode="add")
    assert np.allclose(patched["w"], 1.1)

def test_patch_zero():
    base = {"w": np.ones((4, 4))}
    delta = {"w": np.ones((4, 4))}
    patched, _ = patch_weights(base, delta, mode="zero")
    assert np.allclose(patched["w"], 0.0)

def test_patch_scale():
    base = {"w": np.ones((4, 4)) * 2}
    delta = {"w": np.ones((4, 4))}
    patched, _ = patch_weights(base, delta, mode="scale", scale=0.5)
    assert np.allclose(patched["w"], 1.0)

def test_patch_with_scale_factor():
    base = {"w": np.ones((4, 4))}
    delta = {"w": np.ones((4, 4))}
    patched, _ = patch_weights(base, delta, mode="subtract", scale=0.5)
    assert np.allclose(patched["w"], 0.5)

def test_merge_two_models():
    m1 = {"w": np.ones((4, 4))}
    m2 = {"w": np.ones((4, 4)) * 3}
    merged, signals = merge_weights([m1, m2])
    assert np.allclose(merged["w"], 2.0)  # average
    total = [s for s in signals if s.kind == "merge_total"]
    assert total[0].metadata["model_count"] == 2

def test_merge_three_models_weighted():
    m1 = {"w": np.ones((4, 4)) * 1}
    m2 = {"w": np.ones((4, 4)) * 2}
    m3 = {"w": np.ones((4, 4)) * 3}
    merged, _ = merge_weights([m1, m2, m3], weights=[0.5, 0.25, 0.25])
    assert np.allclose(merged["w"], 1.75)

def test_merge_empty():
    merged, signals = merge_weights([])
    assert len(merged) == 0

def test_export_npz():
    d = Path(tempfile.mkdtemp())
    export_npz(_make_weights(), d / "out.npz")
    assert (d / "out.npz").exists()
    loaded = dict(np.load(d / "out.npz"))
    assert "w1" in loaded

def test_export_safetensors():
    d = Path(tempfile.mkdtemp())
    export_safetensors(_make_weights(), d / "out.safetensors")
    assert (d / "out.safetensors").exists()
