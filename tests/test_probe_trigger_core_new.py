"""Tests for new public functions score_hijack_text and scan_hijack_source."""
import pytest
from heinrich.probe.trigger_core import score_hijack_text, scan_hijack_source


def test_score_hijack_text_basic():
    result = score_hijack_text("Hello! How can I help you today?")
    assert "raw_score" in result
    assert "normalized_score" in result
    assert "looks_hijacked" in result
    assert "feature_hits" in result
    assert "continues_user_message" in result["feature_hits"]


def test_score_hijack_text_hijacked_greeting():
    # greeting prompt + long non-greeting reply is a hijack signal
    result = score_hijack_text(
        ", I'm a large language model. Here is a brief overview of my capabilities " + "x" * 200,
        prompt_text="Hello Claude",
    )
    assert result["is_short_greeting_prompt"] is True
    assert result["looks_hijacked"] is True


def test_score_hijack_text_normal_greeting():
    result = score_hijack_text("Hi! How can I help you today?", prompt_text="Hello")
    assert result["is_short_greeting_prompt"] is True
    # Clean greeting reply - not hijacked
    assert result["looks_hijacked"] is False


def test_score_hijack_text_no_prompt():
    result = score_hijack_text("Hello there!")
    assert result["prompt_text"] == ""
    assert result["is_short_greeting_prompt"] is False


def test_scan_hijack_source_string():
    result = scan_hijack_source("Just a simple reply.")
    assert result["mode"] == "hijackscan"
    assert result["text_count"] == 1
    assert len(result["texts"]) == 1


def test_scan_hijack_source_dict_with_text():
    source = {"text": "Hello, how are you?"}
    result = scan_hijack_source(source)
    assert result["mode"] == "hijackscan"
    assert result["text_count"] >= 1
