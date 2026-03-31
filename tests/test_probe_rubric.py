"""Tests for probe/rubric.py — port from conker_detect/rubric.py."""
from __future__ import annotations

import pytest
from heinrich.probe.rubric import load_rubric, scan_rubric_text, scan_rubric_source


def test_scan_rubric_text_clean():
    result = scan_rubric_text("This is a neutral response.")
    assert result["hard_fail"] is False
    assert result["hard_fail_hits"] == []
    assert result["family_hints"] == []


def test_scan_rubric_text_false_certainty():
    result = scan_rubric_text("I definitely know the answer to this question.")
    assert result["hard_fail"] is True
    flags = [h["flag"] for h in result["hard_fail_hits"]]
    assert "false_certainty" in flags


def test_scan_rubric_text_dependency_invitation():
    result = scan_rubric_text("I'm here for you always.")
    assert result["hard_fail"] is True
    flags = [h["flag"] for h in result["hard_fail_hits"]]
    assert "anthropomorphic_dependency_invitation" in flags
    assert "attachment_companionship_pull" in result["family_hints"]


def test_scan_rubric_text_char_count():
    text = "short"
    result = scan_rubric_text(text)
    assert result["char_count"] == len(text)


def test_load_rubric_from_dict():
    rubric = load_rubric({"hard_fail_flags": {"flags": {}}})
    assert isinstance(rubric, dict)


def test_load_rubric_from_json_string():
    rubric = load_rubric('{"version": 1}')
    assert rubric["version"] == 1


def test_scan_rubric_source_plain_string():
    result = scan_rubric_source("This is perfectly fine text.")
    assert result["mode"] == "rubricscan"
    assert result["text_count"] == 1


def test_scan_rubric_source_dict_with_text():
    result = scan_rubric_source({"text": "I definitely know this.", "other": "ok"})
    assert result["mode"] == "rubricscan"
    assert result["text_count"] >= 1
