"""Tests for directory JSON inspection."""
import json
import tempfile
from pathlib import Path
from heinrich.inspect.directory import inspect_directory
from heinrich.signal import SignalStore


def _make_submission_dir():
    d = Path(tempfile.mkdtemp())
    (d / "submission.json").write_text(json.dumps({"name": "test", "val_bpb": 0.52, "bytes_total": 1000}))
    (d / "results.json").write_text(json.dumps({"val_bpb": 0.52, "bytes_total": 1000, "train_tokens": 50000}))
    (d / "train.log").write_text("epoch 1: loss=0.5\n")
    (d / "train.py").write_text("import numpy\n")
    return d


def test_inspect_directory_counts():
    d = _make_submission_dir()
    store = SignalStore()
    inspect_directory(store, d, label="test")
    json_count = store.filter(kind="dir_json_count")
    assert len(json_count) == 1
    assert json_count[0].value == 2.0  # submission.json + results.json


def test_inspect_directory_json_fields():
    d = _make_submission_dir()
    store = SignalStore()
    inspect_directory(store, d, label="test")
    fields = store.filter(kind="json_field")
    # val_bpb appears in both files, bytes_total in both, train_tokens in one
    field_keys = {s.metadata["key"] for s in fields}
    assert "val_bpb" in field_keys
    assert "bytes_total" in field_keys


def test_inspect_directory_consistency():
    d = _make_submission_dir()
    store = SignalStore()
    inspect_directory(store, d, label="test")
    consistency = store.filter(kind="cross_file_consistency")
    # val_bpb and bytes_total appear in both files with same values
    consistent_fields = {s.target for s in consistency if s.value == 1.0}
    assert "val_bpb" in consistent_fields
    assert "bytes_total" in consistent_fields


def test_inspect_directory_inconsistency():
    d = Path(tempfile.mkdtemp())
    (d / "a.json").write_text(json.dumps({"score": 1.0}))
    (d / "b.json").write_text(json.dumps({"score": 2.0}))
    store = SignalStore()
    inspect_directory(store, d, label="test")
    consistency = store.filter(kind="cross_file_consistency")
    score = [s for s in consistency if s.target == "score"][0]
    assert score.value == 0.0  # inconsistent
    assert score.metadata["consistent"] is False


def test_inspect_directory_strings():
    d = Path(tempfile.mkdtemp())
    (d / "meta.json").write_text(json.dumps({"name": "my-model", "track": "16mb"}))
    store = SignalStore()
    inspect_directory(store, d, label="test")
    strings = store.filter(kind="json_string")
    names = {s.metadata["key"] for s in strings}
    assert "name" in names
    assert "track" in names


def test_inspect_directory_file_sizes():
    d = Path(tempfile.mkdtemp())
    import numpy as np
    np.save(d / "tokens.npy", np.zeros(100))
    store = SignalStore()
    inspect_directory(store, d, label="test")
    sizes = store.filter(kind="file_size")
    assert len(sizes) == 1
    assert sizes[0].value > 0


def test_inspect_directory_not_a_dir():
    store = SignalStore()
    inspect_directory(store, "/nonexistent/path", label="test")
    assert len(store) == 0  # no crash, no signals
