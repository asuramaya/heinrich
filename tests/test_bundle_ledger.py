import json
import tempfile
from pathlib import Path
from heinrich.bundle.ledger import scan_directory, extract_lineage, _classify_record
from heinrich.signal import SignalStore

def _make_dir(files: dict[str, dict]) -> Path:
    d = Path(tempfile.mkdtemp())
    for name, data in files.items():
        (d / name).write_text(json.dumps(data))
    return d

def test_scan_empty():
    d = Path(tempfile.mkdtemp())
    result = scan_directory(d)
    assert result["summary"]["total"] == 0

def test_scan_classifies_bridge():
    d = _make_dir({"run1.json": {"bpb": 0.5, "test_bpb": 0.6}})
    result = scan_directory(d)
    assert result["summary"]["by_kind"]["bridge"] == 1

def test_scan_classifies_full_eval():
    d = _make_dir({"eval1.json": {"eval_bpb": 0.5, "eval_tokens": 1000}})
    result = scan_directory(d)
    assert result["summary"]["by_kind"]["full_eval"] == 1

def test_scan_classifies_study():
    d = _make_dir({"study1.json": {"variants": [1, 2, 3]}})
    result = scan_directory(d)
    assert result["summary"]["by_kind"]["study"] == 1

def test_scan_emits_signals():
    d = _make_dir({"r.json": {"bpb": 0.5}})
    store = SignalStore()
    scan_directory(d, store=store)
    assert len(store.filter(kind="scan_total")) == 1

def test_scan_skips_invalid():
    d = Path(tempfile.mkdtemp())
    (d / "bad.json").write_text("not json{{{")
    result = scan_directory(d)
    assert result["summary"]["skipped"] == 1

def test_lineage_extraction():
    records = [
        {"loaded_state_path": "/a/parent.npz", "saved_state_path": "/b/child.npz"},
        {"loaded_state_path": None, "saved_state_path": "/c/orphan.npz"},
    ]
    edges = extract_lineage(records)
    assert len(edges) == 1
    assert edges[0]["parent"] == "/a/parent.npz"

def test_classify_unknown():
    assert _classify_record({"random": "data"}) == "unknown"
