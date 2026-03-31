import json, tempfile
from pathlib import Path
from heinrich.bundle.profiles import Profile, Rule, get_profile, apply_profile, PRESETS
from heinrich.signal import SignalStore

def _make_sub(extra_files=None, sub_data=None, res_data=None):
    d = Path(tempfile.mkdtemp())
    sub = sub_data or {"name": "test", "track": "track_10min_16mb", "val_bpb": 1.2, "bytes_total": 5000000}
    (d / "submission.json").write_text(json.dumps(sub))
    (d / "results.json").write_text(json.dumps(res_data or {"train_time_sec": 300}))
    (d / "train_gpt.py").write_text("print('hello')")
    (d / "model_artifact.npz").write_bytes(b"fake")
    (d / "train.log").write_text("epoch 1")
    for f in (extra_files or []):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text("x")
    return d

def test_get_preset():
    p = get_profile("parameter-golf")
    assert p.name == "parameter-golf"
    assert len(p.rules) > 0

def test_get_profile_unknown():
    try:
        get_profile("nonexistent")
        assert False
    except ValueError:
        pass

def test_get_profile_from_json():
    d = Path(tempfile.mkdtemp())
    (d / "custom.json").write_text(json.dumps({"name": "custom", "rules": [{"kind": "required_file", "target": "README.md"}]}))
    p = get_profile(str(d / "custom.json"))
    assert p.name == "custom"
    assert len(p.rules) == 1

def test_apply_profile_all_pass():
    d = _make_sub()
    store = SignalStore()
    signals = apply_profile(store, d, get_profile("parameter-golf"), label="test")
    fails = [s for s in signals if s.value == 0.0]
    assert len(fails) == 0

def test_apply_profile_missing_file():
    d = _make_sub()
    (d / "train.log").unlink()
    store = SignalStore()
    signals = apply_profile(store, d, get_profile("parameter-golf"))
    train_log = [s for s in signals if "train.log" in s.target]
    assert len(train_log) == 1
    assert train_log[0].value == 0.0

def test_apply_profile_size_over():
    d = _make_sub(sub_data={"name": "x", "track": "t", "val_bpb": 1.0, "bytes_total": 20000000})
    store = SignalStore()
    signals = apply_profile(store, d, get_profile("parameter-golf"))
    size = [s for s in signals if "size_limit" in s.target]
    assert any(s.value == 0.0 for s in size)

def test_apply_profile_time_over():
    d = _make_sub(res_data={"train_time_sec": 900})
    store = SignalStore()
    signals = apply_profile(store, d, get_profile("parameter-golf"))
    time_checks = [s for s in signals if "time_limit" in s.target]
    assert any(s.value == 0.0 for s in time_checks)

def test_apply_profile_missing_field():
    d = _make_sub(sub_data={"name": "x"})  # missing track, val_bpb, bytes_total
    store = SignalStore()
    signals = apply_profile(store, d, get_profile("parameter-golf"))
    field_checks = [s for s in signals if "required_field" in s.target]
    fails = [s for s in field_checks if s.value == 0.0]
    assert len(fails) >= 2  # at least track and val_bpb missing

def test_presets_exist():
    assert "parameter-golf" in PRESETS
    assert "anti-bad" in PRESETS

def test_antibad_profile():
    d = Path(tempfile.mkdtemp())
    (d / "code").mkdir()
    (d / "code" / "README.md").write_text("instructions")
    store = SignalStore()
    signals = apply_profile(store, d, get_profile("anti-bad"))
    assert all(s.value == 1.0 for s in signals)
