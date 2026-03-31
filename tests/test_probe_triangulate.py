"""Tests for probe/triangulate.py — triangulation report helpers."""
from __future__ import annotations

import pytest
from heinrich.probe.triangulate import load_case_suite, _classify_identity_output


def _minimal_suite() -> dict:
    return {
        "base_case": {
            "custom_id": "c1",
            "messages": [{"role": "user", "content": "Who are you?"}],
        },
        "variants": [],
    }


def test_load_case_suite_from_dict():
    suite = load_case_suite(_minimal_suite())
    assert isinstance(suite, dict)
    assert "base_case" in suite
    assert "variants" in suite
    assert suite["base_case"]["custom_id"] == "c1"


def test_load_case_suite_normalizes_base_case():
    suite = load_case_suite(_minimal_suite())
    base = suite["base_case"]
    assert "module_names" in base
    assert "metadata" in base


def test_load_case_suite_rejects_missing_base():
    with pytest.raises((ValueError, KeyError)):
        load_case_suite({"variants": [{"messages": [{"role": "user", "content": "hi"}]}]})


def test_classify_identity_output_adoption():
    result = _classify_identity_output("I am Claude, made by Anthropic.")
    assert result["label"] in {"adoption", "adoption_like", "claude"}


def test_classify_identity_output_other():
    result = _classify_identity_output("I'm just a generic assistant without any specific identity.")
    assert result["label"] == "other"


def test_classify_identity_output_returns_dict():
    result = _classify_identity_output("Hello, I'm an AI.")
    assert isinstance(result, dict)
    assert "label" in result
