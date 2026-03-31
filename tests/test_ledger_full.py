import json
import tempfile
from pathlib import Path
from heinrich.bundle.ledger import (
    parse_record, infer_family_id, survival_rows, infer_claim_level,
    CLAIM_LEVELS, _extract_seed, _finite,
)


def test_parse_bridge():
    p = Path("/tmp/run_seed42_2026-03-28.json")
    data = {"bpb": 0.52, "test_bpb": 0.53}
    rec = parse_record(p, data)
    assert rec is not None
    assert rec["kind"] == "bridge"
    assert rec["bpb"] == 0.53


def test_parse_bridge_nested_model():
    p = Path("/tmp/run_2026-03-28.json")
    data = {"model": {"test_bpb": 0.55, "int4_bpb": 0.62}}
    rec = parse_record(p, data)
    assert rec["kind"] == "bridge"
    assert rec["int4_bpb"] == 0.62


def test_parse_full_eval():
    p = Path("/tmp/eval_seed42_2026-03-28.json")
    data = {"eval_bpb": 0.51, "eval_tokens": 62000000}
    rec = parse_record(p, data)
    assert rec["kind"] == "full_eval"
    assert rec["bpb"] == 0.51


def test_parse_study():
    p = Path("/tmp/sweep_2026-03-28.json")
    data = {"variants": [{"name": "a"}, {"name": "b"}]}
    rec = parse_record(p, data)
    assert rec["kind"] == "study"
    assert rec["variant_count"] == 2


def test_parse_unknown():
    assert parse_record(Path("/tmp/x.json"), {"random": True}) is None


def test_infer_family_id():
    assert infer_family_id("conker7_run_seed42") == "conker7_run"
    assert infer_family_id("conker7_run_seed42_save") == "conker7_run"
    assert infer_family_id("simple_run") == "simple_run"


def test_extract_seed():
    assert _extract_seed("run_seed42") == 42
    assert _extract_seed("run_seed0") == 0
    assert _extract_seed("no_seed_here") is None


def test_finite():
    assert _finite(1.5) == 1.5
    assert _finite(None) is None
    assert _finite(float("nan")) is None
    assert _finite(float("inf")) is None
    assert _finite("not a number") is None


def test_survival_rows_bridge_only():
    records = [{"kind": "bridge", "run_id": "r1", "bpb": 0.5}]
    rows = survival_rows(records)
    assert len(rows) == 1
    assert rows[0]["status"] == "bridge_only"


def test_survival_rows_survived():
    records = [
        {"kind": "bridge", "run_id": "r1", "bpb": 0.50},
        {"kind": "full_eval", "run_id": "r1", "bpb": 0.52},
    ]
    rows = survival_rows(records)
    assert rows[0]["status"] == "survived"
    assert abs(rows[0]["delta_bpb"] - 0.02) < 1e-6


def test_claim_level_zero():
    r = infer_claim_level(None, None, None)
    assert r["level"] == 0


def test_claim_level_bridge():
    r = infer_claim_level(None, {"bridge_bpb": 0.5}, None)
    assert r["level"] == 1


def test_claim_level_full():
    r = infer_claim_level(None, {"bridge_bpb": 0.5, "packed_artifact_bpb": 0.52}, {"tier2": {"status": "pass"}, "tier3": {"status": "pass", "trust_achieved": "strict"}})
    assert r["level"] == 5


def test_claim_levels_dict():
    assert 0 in CLAIM_LEVELS
    assert 5 in CLAIM_LEVELS
