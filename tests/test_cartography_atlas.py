import tempfile, json
from pathlib import Path
from heinrich.cartography.atlas import Atlas
from heinrich.cartography.perturb import PerturbResult
from heinrich.cartography.surface import Knob
from heinrich.signal import SignalStore

def _make_result(layer, head, kl, changed=False):
    return PerturbResult(Knob(f"h.{layer}.{head}", "head", layer, head), "zero",
                         1.0, 1.0 + kl * 0.1, kl * 0.1, kl, changed, 0, 1 if changed else 0)

def test_atlas_add_and_len():
    a = Atlas()
    a.add(_make_result(0, 0, 5.0))
    assert len(a) == 1

def test_atlas_top_by_kl():
    a = Atlas()
    a.add_all([_make_result(0, 0, 1.0), _make_result(0, 1, 5.0), _make_result(1, 0, 3.0)])
    top = a.top_by_kl(k=2)
    assert top[0].kl_divergence == 5.0

def test_atlas_token_changers():
    a = Atlas()
    a.add_all([_make_result(0, 0, 1.0, False), _make_result(0, 1, 5.0, True)])
    changers = a.top_token_changers()
    assert len(changers) == 1

def test_atlas_save_load():
    a = Atlas()
    a.add_all([_make_result(0, 0, 2.0), _make_result(1, 1, 4.0)])
    p = Path(tempfile.mkdtemp()) / "atlas.json"
    a.save(p)
    loaded = Atlas.load(p)
    assert len(loaded) == 2
    assert loaded.top_by_kl(1)[0].kl_divergence == 4.0

def test_atlas_to_signals():
    a = Atlas()
    a.add(_make_result(0, 0, 3.0))
    store = SignalStore()
    a.to_signals(store)
    assert len(store) == 1
    assert store.filter(kind="atlas_entry")[0].value == 3.0
