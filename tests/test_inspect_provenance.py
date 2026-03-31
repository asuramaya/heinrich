"""Tests for inspect/provenance.py"""
import json
import tempfile
from pathlib import Path
import pytest
from heinrich.inspect.provenance import (
    audit_provenance,
    load_provenance_manifest,
    check_selection_disclosure,
    check_dataset_fingerprints,
)


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_provenance_manifest_minimal(tmp_path):
    p = _write(tmp_path / "prov.json", {"submitted_run_id": "model_seed1", "selection_mode": "single"})
    manifest = load_provenance_manifest(p)
    assert manifest["submitted_run_id"] == "model_seed1"


def test_check_selection_disclosure_present():
    manifest = {"submitted_run_id": "abc", "selection_mode": "single"}
    finding = check_selection_disclosure(manifest)
    assert finding["pass"] is True


def test_check_selection_disclosure_missing():
    manifest = {}
    finding = check_selection_disclosure(manifest)
    assert finding["pass"] is False


def test_check_dataset_fingerprints_no_data():
    manifest = {}
    finding = check_dataset_fingerprints(manifest)
    assert "pass" in finding


def test_audit_provenance_missing_file():
    result = audit_provenance(Path("/tmp/does_not_exist_prov_test.json"))
    assert "verdict" in result or "findings" in result or "error" in result
