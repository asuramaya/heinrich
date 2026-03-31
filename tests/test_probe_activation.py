"""Tests for probe/activation.py — port from conker_detect/activation_probes.py."""
from __future__ import annotations

import numpy as np
import pytest
from heinrich.probe.activation import (
    LinearProbe,
    flatten_activation_map,
    build_feature_matrix,
    fit_binary_linear_probe,
    score_examples,
    rank_modules_by_separability,
    summarize_probe,
)


def _map(a, b):
    return {"layer0": np.array(a, dtype=float), "layer1": np.array(b, dtype=float)}


def test_flatten_activation_map_basic():
    amap = {"mod_a": np.array([1.0, 2.0])}
    flat = flatten_activation_map(amap)
    assert "mod_a" in flat
    np.testing.assert_array_equal(flat["mod_a"], [1.0, 2.0])


def test_flatten_activation_map_missing_module():
    with pytest.raises(KeyError, match="Missing module"):
        flatten_activation_map({"a": np.array([1.0])}, module_names=["b"])


def test_build_feature_matrix_from_maps():
    examples = [_map([1.0, 0.0], [0.0, 1.0]), _map([0.5, 0.5], [0.5, 0.5])]
    matrix, names, slices = build_feature_matrix(examples, module_names=["layer0", "layer1"])
    assert matrix.shape == (2, 4)
    assert names == ("layer0", "layer1")


def test_fit_ridge_probe():
    pos = [_map([1.0, 1.0], [1.0, 1.0])]
    neg = [_map([-1.0, -1.0], [-1.0, -1.0])]
    examples = pos + neg
    labels = [1, 0]
    probe = fit_binary_linear_probe(examples, labels, module_names=["layer0", "layer1"], method="ridge")
    assert isinstance(probe, LinearProbe)
    assert probe.method == "ridge"
    scores = score_examples(probe, examples)
    assert scores[0] > scores[1]


def test_fit_mean_difference_probe():
    pos = [_map([2.0], [2.0]), _map([3.0], [3.0])]
    neg = [_map([-2.0], [-2.0]), _map([-3.0], [-3.0])]
    probe = fit_binary_linear_probe(pos + neg, [1, 1, 0, 0], module_names=["layer0", "layer1"], method="mean_difference")
    assert probe.method == "mean_difference"
    scores = score_examples(probe, pos + neg)
    assert scores[0] > 0 and scores[1] > 0
    assert scores[2] < 0 and scores[3] < 0


def test_rank_modules_by_separability():
    examples = [_map([2.0], [0.0]), _map([3.0], [0.0]), _map([-2.0], [0.0]), _map([-3.0], [0.0])]
    labels = [1, 1, 0, 0]
    rows = rank_modules_by_separability(examples, labels, module_names=["layer0", "layer1"])
    # layer0 should rank higher since it discriminates, layer1 is constant
    assert rows[0]["module_name"] == "layer0"
    assert rows[0]["score"] > rows[1]["score"]


def test_summarize_probe_basic():
    probe = LinearProbe(
        weights=np.array([1.0, -1.0]),
        bias=0.0,
        method="mean_difference",
        module_names=("a", "b"),
        feature_slices=(("a", slice(0, 1)), ("b", slice(1, 2))),
    )
    summary = summarize_probe(probe)
    assert summary["method"] == "mean_difference"
    assert summary["feature_count"] == 2
    assert summary["bias"] == 0.0
