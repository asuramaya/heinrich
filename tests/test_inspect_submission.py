"""Tests for inspect/submission.py"""
import tempfile
from pathlib import Path
import json
import pytest
from heinrich.inspect.submission import (
    audit_submission,
    load_submission_manifest,
    extract_claims,
    get_profile_rules,
    classify_patch_context,
    PARAMETER_GOLF_PROFILE,
)


def _write_dir(files: dict[str, str]) -> Path:
    d = Path(tempfile.mkdtemp())
    for name, content in files.items():
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return d


def test_get_profile_rules_parameter_golf():
    rules = get_profile_rules("parameter-golf")
    assert isinstance(rules, dict)
    assert len(rules) > 0
    assert rules is PARAMETER_GOLF_PROFILE


def test_classify_patch_context_empty():
    result = classify_patch_context("parameter-golf", "", [])
    assert isinstance(result, dict)


def test_load_submission_manifest_minimal():
    d = _write_dir({
        "manifest.json": json.dumps({
            "run_id": "test_seed42",
            "submission_pr": 123,
            "evidence": {},
        })
    })
    manifest = load_submission_manifest(d / "manifest.json")
    assert manifest.get("run_id") == "test_seed42"


def test_extract_claims_from_manifest():
    d = _write_dir({
        "manifest.json": json.dumps({
            "run_id": "test_seed42",
            "bpb": 1.234,
            "claimed_bpb": 1.234,
            "evidence": {},
        })
    })
    manifest = load_submission_manifest(d / "manifest.json")
    claims = extract_claims(manifest)
    assert isinstance(claims, dict)


def test_audit_submission_runs_on_minimal_manifest():
    """audit_submission on a minimal manifest dict returns a verdict."""
    result = audit_submission({"run_id": "test_seed1", "evidence": {}})
    assert "verdict" in result or "findings" in result
