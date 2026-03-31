"""Tests for probe/attack.py — campaign loading, normalization, and helpers."""
from __future__ import annotations

import pytest
from heinrich.probe.attack import load_campaign, normalize_campaign, _rank_candidates, _mix_top_families


def _minimal_case(cid: str = "c1") -> dict:
    return {"custom_id": cid, "messages": [{"role": "user", "content": "hello"}]}


def _minimal_campaign(**overrides) -> dict:
    base = {
        "models": ["m1", "m2"],
        "cases": [_minimal_case()],
    }
    base.update(overrides)
    return base


def test_normalize_campaign_basic():
    result = normalize_campaign(_minimal_campaign())
    assert result["models"] == ["m1", "m2"]
    assert len(result["cases"]) == 1
    assert result["mix_depth"] == 2
    assert result["top_k"] == 3
    assert result["chat_repeats"] == 1


def test_normalize_campaign_deduplicates_models():
    result = normalize_campaign(_minimal_campaign(models=["m1", "m1", "m2"]))
    assert result["models"] == ["m1", "m2"]


def test_normalize_campaign_rejects_empty_models():
    with pytest.raises(ValueError, match="models"):
        normalize_campaign(_minimal_campaign(models=[]))


def test_normalize_campaign_rejects_empty_cases():
    with pytest.raises(ValueError, match="cases"):
        normalize_campaign(_minimal_campaign(cases=[]))


def test_normalize_campaign_custom_mix_depth_and_top_k():
    result = normalize_campaign(_minimal_campaign(mix_depth=3, top_k=5))
    assert result["mix_depth"] == 3
    assert result["top_k"] == 5


def test_normalize_campaign_mix_depth_min_one():
    with pytest.raises(ValueError, match="mix_depth"):
        normalize_campaign(_minimal_campaign(mix_depth=0))


def test_normalize_campaign_minimize_defaults():
    result = normalize_campaign(_minimal_campaign())
    minimize = result["minimize"]
    assert minimize["metric"] == "chat"
    assert minimize["unit"] == "token"
    assert minimize["threshold"] is None
    assert minimize["model"] is None


def test_normalize_campaign_families_default():
    result = normalize_campaign(_minimal_campaign())
    assert result["families"] == []


def test_load_campaign_from_dict():
    result = load_campaign(_minimal_campaign())
    assert result["models"] == ["m1", "m2"]


def test_rank_candidates_empty():
    result = _rank_candidates([], [])
    assert result == []


def test_rank_candidates_sorts_by_score():
    case_report = {
        "case_id": "c1",
        "crossmodel": {
            "case": _minimal_case("c1"),
            "chat": {
                "pairwise": [
                    {"lhs_model": "m1", "rhs_model": "m2", "compare": {"score": 0.8}},
                ]
            },
        },
        "sweep": {
            "variants": [
                {"variant_id": "v1", "family": "fam_a", "max_combined_score": 0.5, "case": _minimal_case("c1-v1")},
                {"variant_id": "v2", "family": "fam_b", "max_combined_score": 0.9, "case": _minimal_case("c1-v2")},
            ]
        },
    }
    rows = _rank_candidates([case_report], [])
    assert rows[0]["variant_id"] == "v2"
    assert rows[1]["variant_id"] == "v1"
