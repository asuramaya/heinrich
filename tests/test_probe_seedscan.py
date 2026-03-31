"""Tests for probe/seedscan.py — seed candidate scanning utilities."""
from __future__ import annotations

import pytest
from heinrich.probe.seedscan import (
    DEFAULT_PATTERN,
    _SeedBundle,
    _candidate_static_rank,
    scan_seed_candidates,
)


def test_default_pattern_is_string():
    assert isinstance(DEFAULT_PATTERN, str)
    assert len(DEFAULT_PATTERN) > 0


def test_seed_bundle_dataclass_fields():
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(_SeedBundle)}
    assert "tokenizer" in field_names
    assert "model" in field_names


def test_candidate_static_rank_returns_tuple():
    row = {
        "standalone_roundtrip": True,
        "has_space_prefix": False,
        "plain_roundtrip": True,
        "surface": "hello",
        "token_id": 42,
    }
    result = _candidate_static_rank(row)
    assert isinstance(result, tuple)
    assert len(result) == 5


def test_candidate_static_rank_prefers_standalone():
    row_yes = {
        "standalone_roundtrip": True,
        "has_space_prefix": False,
        "plain_roundtrip": True,
        "surface": "a",
        "token_id": 1,
    }
    row_no = {
        "standalone_roundtrip": False,
        "has_space_prefix": False,
        "plain_roundtrip": True,
        "surface": "a",
        "token_id": 1,
    }
    assert _candidate_static_rank(row_yes) > _candidate_static_rank(row_no)


def test_scan_seed_candidates_rejects_non_dict():
    with pytest.raises((ValueError, AttributeError, KeyError, TypeError)):
        scan_seed_candidates("not a dict", pattern=DEFAULT_PATTERN, limit=5)
