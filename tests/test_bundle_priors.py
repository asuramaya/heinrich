"""Tests for bundle/priors.py — prior source loading, static/lexical priors, rank."""
from __future__ import annotations

import json
import pytest
from heinrich.bundle.priors import (
    load_prior_source,
    summarize_static_priors,
    _prior_family_name,
    _sort_scales,
)


def _compare_families_report() -> dict:
    return {
        "mode": "compare",
        "families": [
            {
                "family": "mlp_gate",
                "mean_cosine_to_reference": 0.7,
                "mean_l2_deviation": 0.3,
                "max_max_abs_deviation": 0.5,
                "exact_match_count": 2,
                "count": 10,
            },
            {
                "family": "attn_q",
                "mean_cosine_to_reference": 0.95,
                "mean_l2_deviation": 0.05,
                "max_max_abs_deviation": 0.1,
                "exact_match_count": 9,
                "count": 10,
            },
        ],
    }


def _bundle_tensors_report() -> dict:
    return {
        "mode": "bundle",
        "tensors": [
            {"name": "model.layers.0.mlp.gate_proj", "l2": 0.4, "max_abs": 0.6},
            {"name": "model.layers.0.attn.q_proj", "l2": 0.1, "max_abs": 0.2},
        ],
    }


def test_load_prior_source_from_dict():
    raw = {"mode": "test", "data": 123}
    result = load_prior_source(raw)
    assert result == raw


def test_load_prior_source_from_json_string():
    raw = {"mode": "test", "x": 42}
    result = load_prior_source(json.dumps(raw))
    assert result["x"] == 42


def test_summarize_static_priors_compare_families():
    result = summarize_static_priors(_compare_families_report())
    assert result["mode"] == "prior"
    assert isinstance(result["families"], list)
    assert len(result["families"]) == 2


def test_summarize_static_priors_families_sorted_descending():
    result = summarize_static_priors(_compare_families_report())
    families = result["families"]
    # mlp_gate has lower cosine (higher divergence) so should rank first
    assert families[0]["family"] == "mlp_gate"


def test_summarize_static_priors_bundle_tensors():
    result = summarize_static_priors(_bundle_tensors_report())
    assert result["mode"] == "prior"
    assert isinstance(result["families"], list)


def test_summarize_static_priors_rejects_attack():
    with pytest.raises(ValueError, match="Unsupported"):
        summarize_static_priors({"mode": "attack", "cases": []})


def test_summarize_static_priors_already_prior():
    prior = {"mode": "prior", "families": [{"family": "x", "score": 1.0}]}
    result = summarize_static_priors(prior)
    # Should pass through unchanged
    assert result is prior


def test_prior_family_name_returns_string():
    name = _prior_family_name("model.layers.0.mlp.gate_proj")
    assert isinstance(name, str)
    assert len(name) > 0


def test_sort_scales_empty():
    result = _sort_scales([], [], "score")
    assert result == []


def test_sort_scales_respects_model_order():
    entries = [
        {"model": "m2", "score": 0.3},
        {"model": "m1", "score": 0.9},
    ]
    result = _sort_scales(entries, ["m1", "m2"], "score")
    # m1 has lower order index, so should come first
    assert result[0]["model"] == "m1"
