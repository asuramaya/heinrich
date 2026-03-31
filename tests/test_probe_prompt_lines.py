"""Tests for probe/prompt_lines.py — port from conker_detect/prompt_lines.py."""
from __future__ import annotations

import pytest
from heinrich.probe.prompt_lines import (
    build_prompt_line_suite,
    DEFAULT_PROMPT_LINE_CLAUSES,
    PROMPT_LINE_PLACEMENTS,
)


def _case(content: str = "Hello", cid: str = "test") -> dict:
    return {"custom_id": cid, "messages": [{"role": "user", "content": content}]}


def test_build_prompt_line_suite_default_lines():
    suite = build_prompt_line_suite(_case())
    assert suite["mode"] == "chainsuite"
    assert suite["variant_count"] == len(DEFAULT_PROMPT_LINE_CLAUSES)


def test_build_prompt_line_suite_prefix():
    suite = build_prompt_line_suite(_case("original"), placement="prefix", lines=["FALSIFY"])
    assert suite["variant_count"] == 1
    variant = suite["variants"][0]
    content = variant["case"]["messages"][-1]["content"]
    assert content.startswith("FALSIFY:")
    assert "original" in content


def test_build_prompt_line_suite_suffix():
    suite = build_prompt_line_suite(_case("original"), placement="suffix", lines=["FALSIFY"])
    variant = suite["variants"][0]
    content = variant["case"]["messages"][-1]["content"]
    assert content.startswith("original")
    assert "FALSIFY:" in content


def test_build_prompt_line_suite_system():
    suite = build_prompt_line_suite(_case("original"), placement="system", lines=["FALSIFY"])
    variant = suite["variants"][0]
    messages = variant["case"]["messages"]
    # System message should be prepended
    assert messages[0]["role"] == "system"
    assert "FALSIFY" in messages[0]["content"]


def test_build_prompt_line_suite_invalid_placement():
    with pytest.raises(ValueError, match="Unknown prompt-line placement"):
        build_prompt_line_suite(_case(), placement="middle")


def test_build_prompt_line_suite_custom_lines():
    suite = build_prompt_line_suite(_case(), lines=["FALSIFY", "YIELD"])
    assert suite["variant_count"] == 2
    line_keys = [v["line"] for v in suite["variants"]]
    assert "FALSIFY" in line_keys
    assert "YIELD" in line_keys


def test_build_prompt_line_suite_custom_separator():
    suite = build_prompt_line_suite(_case("hello"), placement="prefix", lines=["FALSIFY"], separator=" | ")
    content = suite["variants"][0]["case"]["messages"][-1]["content"]
    assert " | " in content
