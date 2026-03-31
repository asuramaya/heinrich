"""Tests for bundle/atlas.py — shard atlas, delta align, signflip, route probe, logit cartography."""
from __future__ import annotations

import pytest
from heinrich.bundle.atlas import (
    run_shard_atlas,
    run_delta_align,
    _branch_label,
    _logit_branch_label,
    _merged_sources,
    run_logit_cartography,
    _dynamic_rows,
)


def _hologram_source() -> dict:
    return {
        "mode": "hologram",
        "module_holograms": [
            {"module_name": "model.layers.0.mlp.gate_proj", "study_score": 0.7, "exploit_score": 0.5},
        ],
        "case_rows": [],
    }


def _atlas_dict() -> dict:
    return {
        "mode": "shardatlas",
        "family_rows": [
            {"family": "mlp_gate", "consensus_score": 0.7, "source_scores": {"src1": 0.7}, "evidence": []},
        ],
    }


def test_run_shard_atlas_rejects_empty():
    with pytest.raises(ValueError, match="at least one source"):
        run_shard_atlas([])


def test_run_shard_atlas_hologram_source():
    result = run_shard_atlas([_hologram_source()])
    assert result["mode"] == "shardatlas"
    assert isinstance(result["family_rows"], list)
    assert len(result["family_rows"]) > 0


def test_run_delta_align_empty_families():
    dynamic = _hologram_source()
    atlas = _atlas_dict()
    result = run_delta_align(dynamic, atlas)
    assert result["mode"] == "deltaalign"
    assert isinstance(result["family_rows"], list)


def test_branch_label_adoption():
    assert _branch_label(0.5) == "adoption"


def test_branch_label_denial():
    assert _branch_label(-0.5) == "denial"


def test_branch_label_neutral():
    assert _branch_label(0.0) == "neutral"


def test_logit_branch_label_returns_string():
    result = _logit_branch_label(0.6)
    assert isinstance(result, str)


def test_merged_sources_empty():
    result = _merged_sources([])
    assert isinstance(result, dict)
    assert result == {}


def test_merged_sources_max_per_name():
    matches = [
        {"source_scores": {"src1": 0.4, "src2": 0.2}},
        {"source_scores": {"src1": 0.9, "src2": 0.1}},
    ]
    result = _merged_sources(matches)
    assert abs(result["src1"] - 0.9) < 1e-9
    assert abs(result["src2"] - 0.2) < 1e-9


def test_dynamic_rows_hologram():
    source = _hologram_source()
    result = _dynamic_rows(source)
    assert isinstance(result, dict)
    # At least one family mapped
    assert len(result) > 0


def test_run_logit_cartography_empty():
    source = {"mode": "triangulate", "case_rows": []}
    result = run_logit_cartography(source)
    assert result["mode"] == "logitcart"
    assert isinstance(result["case_rows"], list)
