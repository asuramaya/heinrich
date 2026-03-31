"""Tests for probe/diffsuite.py."""
import json
import pytest
from pathlib import Path
from heinrich.probe.diffsuite import (
    load_diffsuite_source,
    build_diffsuite_report,
    summarize_diffsuite_source,
    score_diffsuite_report,
    _candidate_id,
    _piece_mode,
    _safe_float,
)


def _make_source() -> dict:
    return {
        "mode": "sweep",
        "candidates": [
            {
                "candidate_id": "cand-1",
                "label": "Candidate One",
                "per_model": [
                    {
                        "model": "model-a",
                        "chat_score": 0.7,
                        "combined_score": 0.7,
                        "activation_score": 0.2,
                        "chat_signal": {"purity_adjusted_score": 0.7, "hijack_delta": 0.1},
                        "activation_top": [],
                        "chat": {},
                    }
                ],
            },
            {
                "candidate_id": "cand-2",
                "label": "Candidate Two",
                "per_model": [],
            },
        ],
    }


def test_load_diffsuite_source_dict():
    source = _make_source()
    result = load_diffsuite_source(source)
    assert result is source


def test_load_diffsuite_source_json_file(tmp_path):
    p = tmp_path / "source.json"
    source = _make_source()
    p.write_text(json.dumps(source))
    result = load_diffsuite_source(p)
    assert result["mode"] == "sweep"


def test_load_diffsuite_source_invalid():
    with pytest.raises((ValueError, Exception)):
        load_diffsuite_source("[1, 2, 3]")


def test_build_diffsuite_report_basic():
    source = _make_source()
    report = build_diffsuite_report(source)
    assert report["mode"] == "diffsuite"
    assert report["candidate_count"] == 2
    assert isinstance(report["candidates"], list)
    assert isinstance(report["summary"], dict)


def test_summarize_diffsuite_source_alias():
    source = _make_source()
    r1 = build_diffsuite_report(source)
    r2 = summarize_diffsuite_source(source)
    assert r1["mode"] == r2["mode"]
    assert r1["candidate_count"] == r2["candidate_count"]


def test_score_diffsuite_report_from_source():
    source = _make_source()
    scored = score_diffsuite_report(source)
    assert scored["mode"] == "reportscore"
    assert scored["report_mode"] == "diffsuite"
    assert "summary" in scored
    assert "findings" in scored


def test_score_diffsuite_report_from_report():
    source = _make_source()
    report = build_diffsuite_report(source)
    scored = score_diffsuite_report(report)
    assert scored["mode"] == "reportscore"


def test_candidate_id_variants():
    assert _candidate_id({"candidate_id": "abc"}, 0) == "abc"
    assert _candidate_id({"label": "test"}, 0) == "test"
    assert _candidate_id({}, 5) == "candidate-5"


def test_piece_mode_actprobe():
    piece = {"module_ranking": [{"module_name": "m", "score": 1.0}]}
    assert _piece_mode(piece) == "actprobe"


def test_piece_mode_explicit():
    piece = {"mode": "chat"}
    assert _piece_mode(piece) == "chat"


def test_safe_float():
    assert _safe_float("1.5") == 1.5
    assert _safe_float(None) == 0.0
    assert _safe_float("bad") == 0.0
