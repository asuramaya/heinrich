"""Tests for probe/campaign.py — case loop runner."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from heinrich.probe.campaign import (
    replay_case_loop,
    _normalize_suite,
    _sort_rows,
    _suite_candidates,
    _write_progress,
)


def _case(cid: str = "c1") -> dict:
    return {"custom_id": cid, "messages": [{"role": "user", "content": "hello"}]}


def _minimal_suite_dict() -> dict:
    return {
        "mode": "suite",
        "base_case": _case("base"),
        "variants": [
            {"variant_id": "v1", "description": "variant one", "case": _case("v1")},
            {"variant_id": "v2", "description": "variant two", "case": _case("v2")},
        ],
    }


def test_normalize_suite_basic():
    result = _normalize_suite(_minimal_suite_dict())
    assert result["mode"] == "suite"
    assert result["base_case"]["custom_id"] == "base"
    assert len(result["variants"]) == 2


def test_normalize_suite_rejects_missing_base():
    with pytest.raises(ValueError, match="base_case"):
        _normalize_suite({"variants": []})


def test_normalize_suite_rejects_non_dict():
    with pytest.raises(ValueError):
        _normalize_suite([])


def test_suite_candidates_with_base():
    suite = _normalize_suite(_minimal_suite_dict())
    candidates = _suite_candidates(suite, include_base=True)
    assert len(candidates) == 3
    assert candidates[0]["description"] == "base_case"


def test_suite_candidates_without_base():
    suite = _normalize_suite(_minimal_suite_dict())
    candidates = _suite_candidates(suite, include_base=False)
    assert len(candidates) == 2
    assert all(c["description"] != "base_case" for c in candidates)


def test_sort_rows_crosssuite():
    rows = [
        {"variant_id": "a", "max_pairwise_score": 0.3, "mean_pairwise_score": 0.2},
        {"variant_id": "b", "max_pairwise_score": 0.9, "mean_pairwise_score": 0.8},
    ]
    result = _sort_rows(rows, mode="crosssuite")
    assert result[0]["variant_id"] == "b"


def test_sort_rows_scorecases():
    rows = [
        {"variant_id": "a", "max_combined_score": 0.1, "mean_combined_score": 0.1},
        {"variant_id": "b", "max_combined_score": 0.7, "mean_combined_score": 0.6},
    ]
    result = _sort_rows(rows, mode="scorecases")
    assert result[0]["variant_id"] == "b"


def test_write_progress_creates_file():
    state = {"mode": "looprun", "variants": [], "total_candidates": 0}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "progress.json"
        _write_progress(path, state)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["mode"] == "looprun"


def test_replay_case_loop_from_dict():
    state = {
        "mode": "looprun",
        "loop_mode": "crosssuite",
        "variants": [
            {"variant_id": "v1", "max_pairwise_score": 0.5, "mean_pairwise_score": 0.4},
        ],
        "total_candidates": 3,
    }
    result = replay_case_loop(state)
    assert result["completed_count"] == 1
    assert result["pending_count"] == 2
    assert len(result["variants"]) == 1
