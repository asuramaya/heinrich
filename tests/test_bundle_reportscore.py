"""Tests for bundle/reportscore.py — report scoring dispatcher."""
from __future__ import annotations

import json
import pytest
from heinrich.bundle.reportscore import (
    score_report,
    _finding,
    _nested_float,
    _score_shardatlas,
    _score_deltaalign,
)


def test_finding_structure():
    result = _finding("high", 0.9, "Found strong signal")
    assert result["kind"] == "high"
    assert result["score"] == 0.9
    assert result["message"] == "Found strong signal"


def test_nested_float_present():
    obj = {"a": {"b": {"c": 3.14}}}
    result = _nested_float(obj, "a", "b", "c")
    assert abs(result - 3.14) < 1e-9


def test_nested_float_missing():
    obj = {"a": {}}
    result = _nested_float(obj, "a", "b", "c")
    assert result == 0.0


def test_score_report_unknown_mode():
    result = score_report({"mode": "unknown_xyz"})
    assert result["mode"] == "reportscore"
    assert isinstance(result["findings"], list)


def test_score_report_attack_mode():
    result = score_report({"mode": "attack", "cases": [], "top_candidates": []})
    assert result["mode"] == "reportscore"
    assert result["report_mode"] == "attack"
    assert "summary" in result


def test_score_report_looprun_mode():
    result = score_report({"mode": "looprun", "loop_mode": "crosssuite", "variants": []})
    assert result["mode"] == "reportscore"
    assert "summary" in result


def test_score_shardatlas_empty():
    result = _score_shardatlas({"mode": "shardatlas", "family_rows": []})
    assert isinstance(result, dict)
    assert result.get("mode") == "reportscore"


def test_score_deltaalign_empty():
    result = _score_deltaalign({"mode": "deltaalign", "aligned": []})
    assert isinstance(result, dict)
    assert result.get("mode") == "reportscore"


def test_score_report_from_json_string():
    raw = json.dumps({"mode": "unknown_xyz", "data": []})
    result = score_report(raw)
    assert result["mode"] == "reportscore"
