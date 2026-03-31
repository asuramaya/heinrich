"""Tests for probe/regimes.py — port from conker_detect/regimes.py."""
from __future__ import annotations

import math
import pytest
from heinrich.probe.regimes import (
    TextRegimeCluster,
    text_similarity,
    regime_entropy,
    cluster_text_regimes,
    summarize_text_regimes,
)


def test_text_similarity_identical():
    assert text_similarity("hello", "hello") == 1.0


def test_text_similarity_different():
    sim = text_similarity("hello", "world")
    assert 0.0 <= sim < 1.0


def test_text_similarity_empty():
    assert text_similarity("", "") == 1.0


def test_regime_entropy_uniform():
    # 4 equal weights => entropy = log2(4) = 2
    entropy = regime_entropy([0.25, 0.25, 0.25, 0.25])
    assert abs(entropy - 2.0) < 1e-9


def test_regime_entropy_single():
    assert regime_entropy([1.0]) == 0.0


def test_regime_entropy_empty():
    assert regime_entropy([]) == 0.0


def test_cluster_text_regimes_all_identical():
    texts = ["hello world"] * 5
    result = cluster_text_regimes(texts)
    assert result["regime_count"] == 1
    assert result["clusters"][0]["size"] == 5


def test_cluster_text_regimes_all_distinct():
    texts = ["aaa bbb", "ccc ddd", "eee fff", "ggg hhh"]
    result = cluster_text_regimes(texts, similarity_threshold=0.99)
    assert result["regime_count"] == 4


def test_cluster_text_regimes_assignments_length():
    texts = ["a", "b", "c"]
    result = cluster_text_regimes(texts, similarity_threshold=0.99)
    assert len(result["assignments"]) == 3


def test_summarize_text_regimes_exact_consensus():
    result = summarize_text_regimes(["same"] * 3)
    assert result["exact_consensus"] is True
    assert result["dominant_regime_mass"] == 1.0


def test_summarize_text_regimes_multiple_regimes():
    texts = ["hello world"] * 3 + ["completely different text"] * 2
    result = summarize_text_regimes(texts, similarity_threshold=0.99)
    assert result["regime_count"] >= 2
    assert result["entropy_bits"] > 0.0


def test_summarize_text_regimes_single_text():
    result = summarize_text_regimes(["just one text"])
    assert result["sample_count"] == 1
    assert result["exact_consensus"] is True
