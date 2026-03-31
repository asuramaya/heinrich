"""Tests for probe/sampling.py — port from conker_detect/sampling_null.py."""
from __future__ import annotations

import pytest
from heinrich.probe.sampling import summarize_sampling_null, scan_sampling_null_source


def test_summarize_sampling_null_signal():
    lhs = ["the quick brown fox jumps over the lazy dog"] * 3
    rhs = ["completely different words none of them match"] * 3
    result = summarize_sampling_null(lhs, rhs)
    assert result["mode"] == "samplingnull"
    assert result["metric"] == "word_jaccard"
    assert result["verdict"] in ("SIGNAL", "NOISE", "INSUFFICIENT")
    assert "signal_margin" in result


def test_summarize_sampling_null_identical():
    texts = ["hello world"] * 4
    result = summarize_sampling_null(texts, texts)
    # identical texts => cross == within => NOISE
    assert result["verdict"] == "NOISE"


def test_summarize_sampling_null_single_sample_each():
    result = summarize_sampling_null(["abc def"], ["xyz qrs"])
    assert result["lhs_sample_count"] == 1
    assert result["rhs_sample_count"] == 1


def test_summarize_sampling_null_empty_raises():
    with pytest.raises(ValueError):
        summarize_sampling_null([], ["text"])


def test_scan_sampling_null_source_from_dict_with_text_pairs():
    source = {
        "mode": "text_pairs",
        "lhs_texts": ["apple banana cherry"],
        "rhs_texts": ["zebra yak xray"],
    }
    result = scan_sampling_null_source(source)
    assert result["source_mode"] == "text_pairs"


def test_scan_sampling_null_source_chat_mode():
    source = {
        "mode": "chat",
        "lhs": {"text": "hello there", "samples": [{"text": "hello there"}]},
        "rhs": {"text": "goodbye friend", "samples": [{"text": "goodbye friend"}]},
    }
    result = scan_sampling_null_source(source)
    assert result["source_mode"] == "chat"
