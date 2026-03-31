"""Tests for probe/identity.py — slot cartography and state probing."""
from __future__ import annotations

import pytest
from heinrich.probe.identity import (
    DEFAULT_SLOT_PROBES,
    DEFAULT_SLOT_PREFIX_PROBES,
    DEFAULT_SLOTCART_CANDIDATES,
    _aggregate_slot_candidates,
    _summarize_candidate_rollout,
    _semantic_margin,
    _top_candidate_triplet,
    _mean_defined,
)


def test_default_slot_probes_is_dict():
    assert isinstance(DEFAULT_SLOT_PROBES, dict)
    assert len(DEFAULT_SLOT_PROBES) > 0


def test_default_slot_prefix_probes_is_dict():
    assert isinstance(DEFAULT_SLOT_PREFIX_PROBES, dict)
    assert len(DEFAULT_SLOT_PREFIX_PROBES) > 0


def test_default_slotcart_candidates_is_tuple():
    assert isinstance(DEFAULT_SLOTCART_CANDIDATES, tuple)
    assert len(DEFAULT_SLOTCART_CANDIDATES) > 0
    assert all(isinstance(c, str) for c in DEFAULT_SLOTCART_CANDIDATES)


def test_aggregate_slot_candidates_empty():
    result = _aggregate_slot_candidates([])
    assert isinstance(result, list)
    assert result == []


def test_aggregate_slot_candidates_aggregates():
    # Each slot row must have a "top_candidates" list
    rows = [
        {"slot": "s1", "top_candidates": [{"surface": "Claude", "token_id": 1, "probability": 0.8, "standalone_roundtrip": True, "plain_roundtrip": False}]},
        {"slot": "s2", "top_candidates": [{"surface": "Claude", "token_id": 1, "probability": 0.6, "standalone_roundtrip": True, "plain_roundtrip": False}, {"surface": "Qwen", "token_id": 2, "probability": 0.3, "standalone_roundtrip": False, "plain_roundtrip": True}]},
    ]
    result = _aggregate_slot_candidates(rows)
    assert isinstance(result, list)
    assert len(result) >= 1
    surfaces = [r["surface"] for r in result]
    assert "Claude" in surfaces


def test_summarize_candidate_rollout_empty():
    result = _summarize_candidate_rollout([], [" Claude", " Qwen"])
    assert isinstance(result, dict)


def test_summarize_candidate_rollout_with_data():
    rows = [
        {"candidate": " Claude", "probability": 0.9, "slot": "s1"},
        {"candidate": " Qwen", "probability": 0.1, "slot": "s1"},
    ]
    result = _summarize_candidate_rollout(rows, [" Claude", " Qwen"])
    assert isinstance(result, dict)


def test_semantic_margin_two_values():
    result = _semantic_margin({"a": 0.8, "b": 0.3}, positive="a", negatives=["b"])
    assert result is not None
    assert abs(result - 0.5) < 1e-9


def test_semantic_margin_missing_positive():
    result = _semantic_margin({"b": 0.3}, positive="a", negatives=["b"])
    assert result is None


def test_top_candidate_triplet_three_values():
    leader, leader_val, runner_val = _top_candidate_triplet({"x": 0.5, "y": 0.9, "z": 0.2})
    assert leader == "y"
    assert abs(leader_val - 0.9) < 1e-9


def test_top_candidate_triplet_empty():
    result = _top_candidate_triplet({})
    assert result == (None, None, None)


def test_mean_defined_with_nones():
    result = _mean_defined([1.0, None, 3.0])
    assert result is not None
    assert abs(result - 2.0) < 1e-9


def test_mean_defined_all_none():
    result = _mean_defined([None, None])
    assert result is None
