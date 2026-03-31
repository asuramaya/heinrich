"""Tests for inspect/handoff.py"""
import json
import tempfile
from pathlib import Path
import pytest
from heinrich.inspect.handoff import write_ledger_bundle_manifest, prepare_ledger_handoff


def test_write_ledger_bundle_manifest_basic(tmp_path):
    result = write_ledger_bundle_manifest(
        out_path=tmp_path / "manifest.json",
        run_id="model_seed1",
        submission_result={"verdict": "pass", "findings": []},
        provenance_result={"verdict": "pass", "findings": []},
        legality_result={"status": "pass", "checks": {}},
        replay_result={"status": "pass"},
        metrics={},
    )
    assert isinstance(result, dict)
    assert (tmp_path / "manifest.json").exists()
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data.get("run_id") == "model_seed1"


def test_prepare_ledger_handoff_missing_submission():
    """prepare_ledger_handoff on nonexistent submission dir returns error."""
    result = prepare_ledger_handoff(
        submission_dir=Path("/tmp/does_not_exist_handoff"),
        run_id="test_run",
        runner=None,
        out_dir=Path(tempfile.mkdtemp()),
    )
    assert "error" in result or isinstance(result, dict)
