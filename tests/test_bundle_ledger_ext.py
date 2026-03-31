"""Tests for extended functions in bundle/ledger.py"""
import json
import tempfile
from pathlib import Path
import pytest
from heinrich.bundle.ledger import (
    dumps_json,
    load_json,
    finite_or_none,
    infer_run_id_from_stem,
    parse_bridge_record,
    parse_full_eval_record,
    parse_study_record,
    classify_record,
    scan_results,
    sort_records,
    lineage_rows,
    render_table,
    write_csv,
    infer_claim_level,
    survival_rows,
)


def test_dumps_json_nan():
    result = dumps_json({"v": float("nan")})
    # Python's json module produces unquoted NaN by default
    assert "NaN" in result


def test_dumps_json_inf():
    result = dumps_json({"v": float("inf")})
    assert "Infinity" in result


def test_load_json(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('{"key": 42}', encoding="utf-8")
    data = load_json(p)
    assert data["key"] == 42


def test_finite_or_none_valid():
    assert finite_or_none(1.5) == 1.5
    assert finite_or_none(0) == 0.0


def test_finite_or_none_nan():
    assert finite_or_none(float("nan")) is None


def test_finite_or_none_inf():
    assert finite_or_none(float("inf")) is None


def test_finite_or_none_string():
    assert finite_or_none("1.5") is None


def test_infer_run_id_from_stem_strips_date():
    assert infer_run_id_from_stem("model_seed1_2026-03-15") == "model_seed1"


def test_infer_run_id_from_stem_strips_fullval():
    result = infer_run_id_from_stem("model_seed1_fullval_test_abc123")
    assert "fullval" not in result


def test_parse_bridge_record(tmp_path):
    data = {"model": {"test_bpb": 1.23, "saved_state_path": "/tmp/run_seed1.npz", "params": 1e6}}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data))
    record = parse_bridge_record(path, data)
    assert record["kind"] == "bridge"
    assert record["bpb"] == pytest.approx(1.23)


def test_parse_full_eval_record(tmp_path):
    data = {"eval_bpb": 1.10, "eval_tokens": 5000, "state_npz": "/tmp/run_seed2.npz"}
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(data))
    record = parse_full_eval_record(path, data)
    assert record["kind"] == "full_eval"
    assert record["bpb"] == pytest.approx(1.10)
    assert record["quant_label"] == "fp16"


def test_parse_study_record(tmp_path):
    data = {"variants": [1, 2, 3], "models": []}
    path = tmp_path / "study.json"
    path.write_text(json.dumps(data))
    record = parse_study_record(path, data)
    assert record["kind"] == "study"
    assert record["variant_count"] == 3


def test_classify_record_bridge(tmp_path):
    data = {"model": {"test_bpb": 1.5}}
    path = tmp_path / "r.json"
    record = classify_record(path, data)
    assert record is not None
    assert record["kind"] == "bridge"


def test_classify_record_none(tmp_path):
    record = classify_record(tmp_path / "x.json", {"random": "data"})
    assert record is None


def test_scan_results_bridge(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"model": {"test_bpb": 1.0}}))
    result = scan_results(tmp_path)
    assert result["record_count"] == 1
    assert result["by_kind"]["bridge"] == 1


def test_sort_records_ascending():
    records = [{"bpb": 2.0}, {"bpb": 0.5}, {"bpb": 1.0}]
    sorted_r = sort_records(records, "bpb", ascending=True)
    assert sorted_r[0]["bpb"] == 0.5


def test_sort_records_none_last():
    records = [{"bpb": None}, {"bpb": 1.0}]
    sorted_r = sort_records(records, "bpb", ascending=True)
    assert sorted_r[0]["bpb"] == 1.0


def test_lineage_rows_from_bridge():
    records = [
        {
            "kind": "bridge",
            "loaded_state_path": "/tmp/parent_seed1.npz",
            "saved_state_path": "/tmp/child_seed2.npz",
            "family_id": "model",
            "run_id": "child_seed2",
            "bpb": 1.1,
            "seed": 2,
        }
    ]
    rows = lineage_rows(records)
    assert len(rows) == 1
    assert rows[0]["parent_run_id"] == "parent_seed1"
    assert rows[0]["child_run_id"] == "child_seed2"


def test_render_table_basic():
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    table = render_table(rows, ["a", "b"])
    assert "a" in table
    assert "x" in table


def test_render_table_empty():
    result = render_table([], ["a", "b"])
    assert "(empty)" in result or "(no rows)" in result


def test_write_csv(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    p = tmp_path / "out.csv"
    write_csv(p, rows, ["a", "b"])
    content = p.read_text()
    assert "a,b" in content
    assert "1,x" in content


def test_infer_claim_level_zero():
    result = infer_claim_level(None, None, None)
    assert result["level"] == 0


def test_infer_claim_level_bridge():
    result = infer_claim_level({"bpb": 1.0}, {"bridge": {"bpb": 1.0}}, {})
    assert result["level"] >= 1


def test_survival_rows_bridge_only():
    records = [
        {
            "kind": "bridge",
            "run_id": "model_seed1",
            "family_id": "model",
            "bpb": 1.5,
            "int6_bpb": None,
            "seed": 1,
            "path": "/tmp/r.json",
        }
    ]
    rows = survival_rows(records)
    assert len(rows) == 1
    assert rows[0]["status"] == "bridge_only"
