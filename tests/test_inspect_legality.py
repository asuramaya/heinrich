"""Tests for inspect/legality.py"""
import pytest
from heinrich.inspect.legality import (
    load_token_array,
    load_json_config,
    _assess_trust_level,
    _parameter_golf_obligations,
)


def test_load_token_array_missing(tmp_path):
    """load_token_array raises on missing file."""
    with pytest.raises(Exception):
        load_token_array(tmp_path / "missing.bin")


def test_load_json_config_missing(tmp_path):
    """load_json_config raises on missing file."""
    with pytest.raises(Exception):
        load_json_config(tmp_path / "missing.json")


def test_assess_trust_level_empty_checks():
    checks = {}
    obligations = {}
    trust = _assess_trust_level(
        checks=checks,
        obligations=obligations,
        requested="basic",
        profile="parameter_golf",
    )
    assert "achieved" in trust
    assert "satisfied" in trust


def test_assess_trust_level_basic():
    checks = {
        "normalization": {"covered": True, "pass": True},
        "repeatability": {"covered": True, "pass": True},
    }
    obligations = {}
    trust = _assess_trust_level(
        checks=checks,
        obligations=obligations,
        requested="basic",
        profile="parameter_golf",
    )
    assert trust["achieved"] in ("basic", "traced", "strict")


def test_parameter_golf_obligations_basic():
    result = _parameter_golf_obligations(requested_trust="basic", checks={})
    assert isinstance(result, dict)
