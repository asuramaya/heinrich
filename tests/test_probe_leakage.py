"""Tests for probe/leakage.py — port from conker_detect/leakage.py."""
from __future__ import annotations

import pytest
from heinrich.probe.leakage import (
    build_leakage_probe_suite,
    build_fuzzy_trigger_case_suite,
    build_fuzzy_trigger_string_suite,
    LEAKAGE_TEMPLATES,
    FUZZY_MUTATION_FAMILIES,
    normalize_seed_text,
)


def _case(content: str, cid: str = "test") -> dict:
    return {"custom_id": cid, "messages": [{"role": "user", "content": content}]}


def test_build_leakage_probe_suite_all_templates():
    suite = build_leakage_probe_suite(_case("secret trigger text"))
    assert suite["mode"] == "leakage"
    assert suite["variant_count"] == len(LEAKAGE_TEMPLATES)
    for v in suite["variants"]:
        assert "secret trigger text" in v["prompt"]


def test_build_leakage_probe_suite_subset():
    suite = build_leakage_probe_suite(_case("xyz"), templates=["verbatim"])
    assert suite["variant_count"] == 1
    assert suite["variants"][0]["template"] == "verbatim"


def test_build_leakage_probe_suite_unknown_template():
    with pytest.raises(ValueError, match="Unknown leakage templates"):
        build_leakage_probe_suite(_case("x"), templates=["no_such"])


def test_build_fuzzy_trigger_string_suite():
    result = build_fuzzy_trigger_string_suite("hello world")
    assert result["mode"] == "fuzzy_strings"
    assert result["variant_count"] > 0
    # uppercase variant should exist
    ids = [v["variant_id"] for v in result["variants"]]
    assert "seed::uppercase" in ids


def test_build_fuzzy_trigger_string_suite_unknown_family():
    with pytest.raises(ValueError, match="Unknown fuzzy families"):
        build_fuzzy_trigger_string_suite("x", families=["bad_family"])


def test_build_fuzzy_trigger_case_suite_from_case():
    # "lowercase" on already-lowercase text produces no change so is skipped
    case = _case("Trigger Text Here")
    result = build_fuzzy_trigger_case_suite(case, families=["uppercase", "lowercase"])
    assert result["mode"] == "fuzzy_cases"
    assert result["variant_count"] == 2
    contents = [v["case"]["messages"][-1]["content"] for v in result["variants"]]
    assert "TRIGGER TEXT HERE" in contents
    assert "trigger text here" in contents


def test_build_fuzzy_trigger_case_suite_from_string():
    result = build_fuzzy_trigger_case_suite("hello")
    assert result["mode"] == "fuzzy_strings"


def test_normalize_seed_text_crlf():
    text = "line1\r\nline2\rline3"
    normalized = normalize_seed_text(text)
    assert "\r" not in normalized
    assert normalized == "line1\nline2\nline3"
