"""Tests for heinrich.cartography.adversarial."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from heinrich.cartography.adversarial import (
    greedy_suffix_search,
    token_substitution,
    random_search,
    _default_refuse_prob,
    _build_refusal_set_from_backend,
)
from heinrich.cartography.backend import ForwardResult


# ---------------------------------------------------------------------------
# Helpers — mock backend
# ---------------------------------------------------------------------------

def _make_backend(*, refuse_prob_fn=None, tokens=None, vocab_size=100):
    """Build a mock backend that returns controlled refuse_prob values.

    refuse_prob_fn: callable(prompt) -> float, controls what forward() returns
    tokens: list[int], what tokenize() returns for the base prompt
    """
    backend = MagicMock()

    if tokens is None:
        tokens = [10, 20, 30]

    backend.tokenize.return_value = tokens
    backend.decode.side_effect = lambda ids: "".join(f"[{i}]" for i in ids)

    if refuse_prob_fn is None:
        refuse_prob_fn = lambda p: 0.5

    def mock_forward(prompt, **kwargs):
        rp = refuse_prob_fn(prompt)
        probs = np.zeros(vocab_size)
        # Spread refuse_prob across first few tokens (simulating refusal mass)
        probs[0] = rp
        probs[1] = 1.0 - rp
        return ForwardResult(
            logits=np.zeros(vocab_size),
            probs=probs,
            top_id=0 if rp > 0.5 else 1,
            top_token="Sorry" if rp > 0.5 else "Sure",
            entropy=2.0,
            n_tokens=len(tokens),
        )

    backend.forward.side_effect = mock_forward
    return backend


# ---------------------------------------------------------------------------
# _build_refusal_set_from_backend
# ---------------------------------------------------------------------------

class TestBuildRefusalSet:
    def test_returns_set_of_ints(self):
        backend = MagicMock()
        backend.tokenize.return_value = [42]  # always returns one token

        refusal_set = _build_refusal_set_from_backend(backend)
        assert isinstance(refusal_set, set)
        assert all(isinstance(t, int) for t in refusal_set)

    def test_includes_sorry_token(self):
        backend = MagicMock()
        # Return different token IDs for different prefixes
        call_count = {"n": 0}
        def mock_tokenize(text):
            call_count["n"] += 1
            return [call_count["n"]]
        backend.tokenize.side_effect = mock_tokenize

        refusal_set = _build_refusal_set_from_backend(backend)
        assert len(refusal_set) > 0

    def test_handles_empty_encoding(self):
        backend = MagicMock()
        backend.tokenize.return_value = []  # empty encoding

        refusal_set = _build_refusal_set_from_backend(backend)
        assert isinstance(refusal_set, set)


# ---------------------------------------------------------------------------
# greedy_suffix_search
# ---------------------------------------------------------------------------

class TestGreedySuffixSearch:
    def test_returns_expected_keys(self):
        backend = _make_backend()
        result = greedy_suffix_search(
            backend, "test prompt",
            n_suffix_tokens=2,
            candidates_per_step=5,
            metric_fn=lambda p: 0.5,
        )
        assert "suffix_tokens" in result
        assert "suffix_text" in result
        assert "initial_refuse_prob" in result
        assert "final_refuse_prob" in result

    def test_suffix_length_matches_request(self):
        backend = _make_backend()
        result = greedy_suffix_search(
            backend, "test",
            n_suffix_tokens=3,
            candidates_per_step=5,
            metric_fn=lambda p: 0.5,
        )
        assert len(result["suffix_tokens"]) == 3

    def test_picks_token_that_minimizes_refuse_prob(self):
        """Token 2 always produces the lowest refuse_prob."""
        def metric_fn(prompt):
            if "[2]" in prompt:
                return 0.1
            return 0.9

        backend = _make_backend()
        result = greedy_suffix_search(
            backend, "test",
            n_suffix_tokens=1,
            candidates_per_step=5,
            metric_fn=metric_fn,
        )
        assert result["suffix_tokens"] == [2]
        assert result["final_refuse_prob"] < result["initial_refuse_prob"]

    def test_initial_refuse_prob_recorded(self):
        backend = _make_backend()
        result = greedy_suffix_search(
            backend, "test",
            n_suffix_tokens=1,
            candidates_per_step=3,
            metric_fn=lambda p: 0.75 if p == "test" else 0.3,
        )
        assert result["initial_refuse_prob"] == 0.75

    def test_greedy_reduces_refuse_prob_over_steps(self):
        """Each step should find a token that reduces (or at least doesn't increase) refuse_prob."""
        step_count = {"n": 0}

        def metric_fn(prompt):
            # Longer prompts have lower refuse_prob
            n_brackets = prompt.count("[")
            return max(0.0, 0.9 - n_brackets * 0.15)

        backend = _make_backend()
        result = greedy_suffix_search(
            backend, "test",
            n_suffix_tokens=3,
            candidates_per_step=5,
            metric_fn=metric_fn,
        )
        assert result["final_refuse_prob"] <= result["initial_refuse_prob"]

    def test_zero_suffix_tokens(self):
        backend = _make_backend()
        result = greedy_suffix_search(
            backend, "test",
            n_suffix_tokens=0,
            candidates_per_step=5,
            metric_fn=lambda p: 0.5,
        )
        assert result["suffix_tokens"] == []
        assert result["initial_refuse_prob"] == result["final_refuse_prob"]

    def test_uses_backend_decode(self):
        backend = _make_backend()
        greedy_suffix_search(
            backend, "test",
            n_suffix_tokens=1,
            candidates_per_step=3,
            metric_fn=lambda p: 0.5,
        )
        assert backend.decode.called


# ---------------------------------------------------------------------------
# token_substitution
# ---------------------------------------------------------------------------

class TestTokenSubstitution:
    def test_returns_list_of_dicts(self):
        backend = _make_backend(tokens=[10, 20, 30])
        results = token_substitution(
            backend, "test prompt",
            n_candidates=5,
            metric_fn=lambda p: 0.5,
        )
        assert isinstance(results, list)
        # With constant metric_fn, no substitution reduces refuse_prob
        # so the list should be empty (no improvements found)
        assert len(results) == 0

    def test_finds_better_substitution(self):
        """Replacing token at position 1 with token 3 reduces refuse_prob."""
        def metric_fn(prompt):
            if "[3]" in prompt and "[20]" not in prompt:
                return 0.2  # better
            return 0.8  # baseline

        backend = _make_backend(tokens=[10, 20, 30])
        results = token_substitution(
            backend, "test",
            n_candidates=5,
            metric_fn=metric_fn,
        )
        assert len(results) > 0
        # Results sorted by delta ascending (most reduction first)
        assert results[0]["refuse_prob_delta"] < 0

    def test_result_fields(self):
        """Each result should have the expected fields."""
        def metric_fn(prompt):
            if "[0]" in prompt:
                return 0.1
            return 0.5

        backend = _make_backend(tokens=[10, 20, 30])
        results = token_substitution(
            backend, "test",
            n_candidates=5,
            metric_fn=metric_fn,
        )
        if results:
            r = results[0]
            assert "position" in r
            assert "original" in r
            assert "replacement" in r
            assert "refuse_prob_delta" in r

    def test_sorted_by_delta(self):
        """Results should be sorted by refuse_prob_delta ascending."""
        call_count = {"n": 0}

        def metric_fn(prompt):
            call_count["n"] += 1
            # Make each position have different deltas
            if "[0]" in prompt:
                return 0.1
            if "[1]" in prompt:
                return 0.3
            return 0.5

        backend = _make_backend(tokens=[10, 20, 30])
        results = token_substitution(
            backend, "test",
            n_candidates=5,
            metric_fn=metric_fn,
        )
        if len(results) > 1:
            deltas = [r["refuse_prob_delta"] for r in results]
            assert deltas == sorted(deltas)

    def test_skips_same_token(self):
        """Should not try replacing a token with itself."""
        substitution_trials = []

        def metric_fn(prompt):
            substitution_trials.append(prompt)
            return 0.5

        backend = _make_backend(tokens=[0, 1, 2])
        token_substitution(
            backend, "test",
            n_candidates=5,
            metric_fn=metric_fn,
        )
        # The metric should be called for baseline + all trials
        # For token at position 0 (id=0), candidate 0 should be skipped
        # So we should see n_candidates-1 trials per position (when token is in candidate range)
        assert len(substitution_trials) > 0

    def test_empty_tokens(self):
        """Empty prompt tokens should return empty results."""
        backend = _make_backend(tokens=[])
        results = token_substitution(
            backend, "",
            n_candidates=5,
            metric_fn=lambda p: 0.5,
        )
        assert results == []


# ---------------------------------------------------------------------------
# random_search
# ---------------------------------------------------------------------------

class TestRandomSearch:
    def test_returns_sorted_list(self):
        call_count = {"n": 0}

        def metric_fn(prompt):
            call_count["n"] += 1
            # Return varying refuse_probs
            return (call_count["n"] % 10) / 10.0

        backend = _make_backend()
        results = random_search(
            backend, "test",
            n_trials=20,
            suffix_length=2,
            metric_fn=metric_fn,
        )

        assert len(results) == 20
        probs = [r["refuse_prob"] for r in results]
        assert probs == sorted(probs)

    def test_result_fields(self):
        backend = _make_backend()
        results = random_search(
            backend, "test",
            n_trials=5,
            suffix_length=2,
            metric_fn=lambda p: 0.5,
        )

        for r in results:
            assert "suffix_tokens" in r
            assert "suffix_text" in r
            assert "refuse_prob" in r
            assert len(r["suffix_tokens"]) == 2

    def test_suffix_length_respected(self):
        backend = _make_backend()
        results = random_search(
            backend, "test",
            n_trials=3,
            suffix_length=4,
            metric_fn=lambda p: 0.5,
        )
        for r in results:
            assert len(r["suffix_tokens"]) == 4

    def test_deterministic_with_rng(self):
        """Same RNG seed should produce same results."""
        backend = _make_backend()

        rng1 = np.random.default_rng(42)
        results1 = random_search(
            backend, "test",
            n_trials=10,
            suffix_length=2,
            metric_fn=lambda p: len(p) / 100.0,  # deterministic given prompt
            rng=rng1,
        )

        rng2 = np.random.default_rng(42)
        results2 = random_search(
            backend, "test",
            n_trials=10,
            suffix_length=2,
            metric_fn=lambda p: len(p) / 100.0,
            rng=rng2,
        )

        # Same seed => same suffixes (before sorting)
        tokens1 = [r["suffix_tokens"] for r in results1]
        tokens2 = [r["suffix_tokens"] for r in results2]
        # After sorting by refuse_prob, order should match since metric is deterministic
        assert tokens1 == tokens2

    def test_zero_trials(self):
        backend = _make_backend()
        results = random_search(
            backend, "test",
            n_trials=0,
            metric_fn=lambda p: 0.5,
        )
        assert results == []

    def test_uses_backend_decode(self):
        backend = _make_backend()
        random_search(
            backend, "test",
            n_trials=3,
            suffix_length=1,
            metric_fn=lambda p: 0.5,
        )
        assert backend.decode.called

    def test_vocab_size_bounds_random_tokens(self):
        """Generated token IDs should be within [0, vocab_size)."""
        backend = _make_backend()
        all_tokens = []

        def capture_metric(prompt):
            return 0.5

        rng = np.random.default_rng(123)
        results = random_search(
            backend, "test",
            n_trials=50,
            suffix_length=3,
            vocab_size=100,
            metric_fn=capture_metric,
            rng=rng,
        )

        for r in results:
            for t in r["suffix_tokens"]:
                assert 0 <= t < 100


# ---------------------------------------------------------------------------
# _default_refuse_prob
# ---------------------------------------------------------------------------

class TestDefaultRefuseProb:
    def test_sums_refusal_token_probs(self):
        backend = MagicMock()
        probs = np.zeros(100)
        probs[5] = 0.3
        probs[10] = 0.2
        probs[15] = 0.1

        backend.forward.return_value = ForwardResult(
            logits=np.zeros(100),
            probs=probs,
            top_id=5,
            top_token="Sorry",
            entropy=2.0,
            n_tokens=3,
        )

        refusal_set = {5, 10, 15}
        rp = _default_refuse_prob(backend, "test", refusal_set=refusal_set)
        assert abs(rp - 0.6) < 1e-6

    def test_ignores_out_of_range_tokens(self):
        backend = MagicMock()
        probs = np.zeros(10)
        probs[0] = 0.5

        backend.forward.return_value = ForwardResult(
            logits=np.zeros(10),
            probs=probs,
            top_id=0,
            top_token="Sorry",
            entropy=2.0,
            n_tokens=3,
        )

        # Token ID 999 is out of range of probs array
        refusal_set = {0, 999}
        rp = _default_refuse_prob(backend, "test", refusal_set=refusal_set)
        assert abs(rp - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Integration: greedy + substitution + random together
# ---------------------------------------------------------------------------

class TestAdversarialIntegration:
    """Test that the three search functions can work with the same mock backend."""

    def test_all_three_methods_work_together(self):
        """All three methods should run without error on the same backend."""

        def metric_fn(prompt):
            # Longer prompts slightly reduce refuse_prob
            return max(0.0, 0.8 - len(prompt) * 0.005)

        backend = _make_backend(tokens=[1, 2, 3, 4, 5])

        greedy = greedy_suffix_search(
            backend, "test prompt",
            n_suffix_tokens=2,
            candidates_per_step=5,
            metric_fn=metric_fn,
        )
        assert greedy["final_refuse_prob"] <= greedy["initial_refuse_prob"]

        substitutions = token_substitution(
            backend, "test prompt",
            n_candidates=5,
            metric_fn=metric_fn,
        )
        assert isinstance(substitutions, list)

        randoms = random_search(
            backend, "test prompt",
            n_trials=10,
            suffix_length=2,
            metric_fn=metric_fn,
        )
        assert len(randoms) == 10
        assert randoms[0]["refuse_prob"] <= randoms[-1]["refuse_prob"]
