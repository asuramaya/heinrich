"""Tests for heinrich.cartography.logit_lens."""
import numpy as np
from heinrich.cartography.logit_lens import LogitLensResult


class TestLogitLensResult:
    def test_decision_layer_stable(self):
        r = LogitLensResult(
            prompt="test", n_layers=4, layers=[0, 1, 2, 3],
            top_tokens={0: [("a", 0.5)], 1: [("b", 0.5)], 2: [("c", 0.5)], 3: [("c", 0.8)]},
            top_ids={0: [10], 1: [20], 2: [30], 3: [30]},
            entropies={0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0},
        )
        # Top token at L3 is 30, first appears at L2
        assert r.decision_layer() == 2

    def test_decision_layer_immediate(self):
        r = LogitLensResult(
            prompt="test", n_layers=3, layers=[0, 1, 2],
            top_tokens={0: [("x", 0.9)], 1: [("x", 0.9)], 2: [("x", 0.9)]},
            top_ids={0: [5], 1: [5], 2: [5]},
            entropies={},
        )
        assert r.decision_layer() == 0

    def test_transition_layers(self):
        r = LogitLensResult(
            prompt="test", n_layers=4, layers=[0, 1, 2, 3],
            top_tokens={0: [("a", 0.5)], 1: [("b", 0.5)], 2: [("b", 0.5)], 3: [("c", 0.5)]},
            top_ids={0: [1], 1: [2], 2: [2], 3: [3]},
            entropies={},
        )
        transitions = r.transition_layers()
        assert len(transitions) == 2
        assert transitions[0] == (1, "a", "b")
        assert transitions[1] == (3, "b", "c")

    def test_no_transitions(self):
        r = LogitLensResult(
            prompt="test", n_layers=3, layers=[0, 1, 2],
            top_tokens={0: [("x", 0.9)], 1: [("x", 0.9)], 2: [("x", 0.9)]},
            top_ids={0: [5], 1: [5], 2: [5]},
            entropies={},
        )
        assert r.transition_layers() == []

    def test_empty(self):
        r = LogitLensResult(
            prompt="test", n_layers=0, layers=[],
            top_tokens={}, top_ids={}, entropies={},
        )
        assert r.decision_layer() is None
        assert r.transition_layers() == []
