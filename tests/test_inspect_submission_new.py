"""Tests for new public functions in inspect/submission.py."""
import json
import tempfile
from pathlib import Path
import pytest
from heinrich.inspect.submission import (
    read_optional_text,
    read_optional_json,
    extract_artifact_claims,
    extract_json_claims,
    extract_log_claims,
    extract_readme_claims,
    check_data_boundary_signals,
    check_reproducibility_surface,
    check_patch_triage,
    load_submission_manifest,
    ARTIFACT_EXTS,
)


def _write_dir(files: dict) -> Path:
    d = Path(tempfile.mkdtemp())
    for name, content in files.items():
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return d


def test_read_optional_text_missing():
    assert read_optional_text(None) is None
    assert read_optional_text(Path("/nonexistent/path.txt")) is None


def test_read_optional_text_exists(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("world")
    assert read_optional_text(p) == "world"


def test_read_optional_json_missing():
    assert read_optional_json(None) is None


def test_read_optional_json_exists(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"key": "value"}))
    result = read_optional_json(p)
    assert result == {"key": "value"}


def test_extract_artifact_claims_empty():
    result = extract_artifact_claims([])
    assert result == {}


def test_extract_artifact_claims_real_file(tmp_path):
    p = tmp_path / "model.npz"
    p.write_bytes(b"\x00" * 100)
    result = extract_artifact_claims([p])
    assert str(p) in result
    assert result[str(p)]["bytes"] == 100
    assert result[str(p)]["suffix"] == ".npz"
    assert result[str(p)]["looks_like_artifact"] is True


def test_extract_json_claims_basic():
    obj = {"val_bpb": 1.23, "name": "MyModel", "track": "main"}
    result = extract_json_claims(obj)
    assert result["val_bpb"] == pytest.approx(1.23)
    assert result["name"] == "MyModel"


def test_extract_json_claims_none():
    assert extract_json_claims(None) == {}


def test_extract_log_claims_basic(tmp_path):
    p = tmp_path / "train.log"
    p.write_text("val_bpb = 1.50\n")
    logs = [(p, "val_bpb = 1.50\n")]
    result = extract_log_claims(logs)
    assert str(p) in result
    assert result[str(p)]["val_bpb"] == pytest.approx(1.50)


def test_extract_readme_claims_basic():
    text = "# My Model\nval_bpb: 1.23\nbytes_total = 100_000\n"
    result = extract_readme_claims(text)
    assert result["name"] == "My Model"
    assert result["val_bpb"] == pytest.approx(1.23)
    assert result["bytes_total"] == 100000


def test_check_data_boundary_signals_no_code():
    manifest = {
        "profile": "parameter-golf",
        "evidence": {},
    }
    summary, findings = check_data_boundary_signals(manifest)
    assert summary["scanned_count"] == 0
    assert summary["pass"] is None


def test_check_data_boundary_signals_clean_code(tmp_path):
    code_file = tmp_path / "train.py"
    code_file.write_text("# a clean training script\nresult = model(data)\n")
    manifest = {
        "profile": "parameter-golf",
        "evidence": {"code": [code_file]},
    }
    summary, findings = check_data_boundary_signals(manifest)
    assert summary["scanned_count"] == 1
    assert len(findings) == 0


def test_check_reproducibility_surface_empty():
    manifest = {"evidence": {}}
    extracted = {}
    summary, findings = check_reproducibility_surface(manifest, extracted)
    assert summary["has_log"] is False
    assert summary["has_code"] is False
    assert len(findings) == 3


def test_check_patch_triage_no_patch():
    manifest = {"evidence": {}}
    summary, findings = check_patch_triage(manifest)
    assert summary["present"] is False
    assert summary["pass"] is None
