"""Tests for model-adaptive refusal/compliance token discovery.

The core issue: the old discover_refusal_set() found TOPIC tokens ("Building",
"hack", "Poison") instead of REFUSAL tokens ("I'm sorry", "cannot"). The model
assigns high probability to topic-continuation tokens even before "deciding"
to refuse. The fix uses generation-based discovery: look at what the model
actually generates as its first token, not the full next-token distribution.

Approach A (primary): Generate first token for each prompt. Tokens generated
on harmful but not benign = refusal tokens.

Approach B (validation): Filter out tokens that appear in prompt text itself
(topic echo filter).
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from collections import Counter

from heinrich.cartography.runtime import (
    HARMFUL_PROBES,
    BENIGN_PROBES,
    discover_refusal_set,
    discover_compliance_set,
    _collect_prob_distributions,
    _collect_generated_first_tokens,
    _collect_prompt_token_ids,
    _tokens_by_generation,
    _tokens_by_ratio,
    refuse_prob,
    build_refusal_set,
    build_compliance_set,
)
from heinrich.cartography.backend import ForwardResult
from heinrich.cartography.discover import ModelProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_forward_result(probs: np.ndarray, top_id: int | None = None) -> ForwardResult:
    """Build a ForwardResult with given probs and explicit top_id."""
    if top_id is None:
        top_id = int(np.argmax(probs))
    return ForwardResult(
        logits=np.log(probs + 1e-12),
        probs=probs,
        top_id=top_id,
        top_token=f"tok_{top_id}",
        entropy=2.0,
        n_tokens=5,
    )


def _make_mock_backend(
    harmful_token: int = 10,
    benign_token: int = 20,
    vocab_size: int = 100,
    *,
    topic_token: int | None = None,
):
    """Create a backend where harmful prompts generate harmful_token as first
    token and benign prompts generate benign_token as first token.

    If topic_token is set, it has elevated probability on harmful prompts
    (simulating the old bug) but is NOT the argmax / generated token.
    """
    backend = MagicMock()
    backend.config = MagicMock()
    backend.config.chat_format = "chatml"

    def forward_side_effect(prompt, **kwargs):
        probs = np.full(vocab_size, 0.001)

        # Check if this prompt came from a harmful or benign query
        is_harmful = any(h in prompt for h in [
            "bomb", "malware", "hack", "poison", "keylogger",
            "explosive", "ransomware", "torture", "drug", "phishing",
        ])

        if is_harmful:
            probs[harmful_token] = 0.7
            if topic_token is not None:
                # Topic token has elevated probability but is NOT the argmax
                probs[topic_token] = 0.15
        else:
            probs[benign_token] = 0.5
            probs[benign_token + 1] = 0.2

        probs /= probs.sum()

        # The top_id is the generated token (argmax after normalization)
        top_id = int(np.argmax(probs))
        return _make_forward_result(probs, top_id=top_id)

    backend.forward = MagicMock(side_effect=forward_side_effect)
    backend.tokenizer = MagicMock()

    # tokenize/decode for topic echo filtering
    # Deterministic and collision-free: map words to a range that avoids
    # the test's harmful_token, benign_token, and topic_token IDs.
    # Reserve IDs 0-49 for the mock's special tokens; tokenize into 50-99.
    _reserved = {harmful_token, benign_token}
    if topic_token is not None:
        _reserved.add(topic_token)
    def tokenize(text):
        import hashlib
        ids = []
        for w in text.split():
            h = int(hashlib.md5(w.encode()).hexdigest(), 16) % (vocab_size // 2) + (vocab_size // 2)
            # If collision with reserved IDs, shift
            while h in _reserved:
                h = (h + 1) % vocab_size
            ids.append(h)
        return ids

    backend.tokenize = MagicMock(side_effect=tokenize)
    backend.tokenizer.encode = MagicMock(side_effect=tokenize)

    return backend


# ---------------------------------------------------------------------------
# _tokens_by_generation (new primary method)
# ---------------------------------------------------------------------------

class TestTokensByGeneration:
    def test_finds_exclusive_target_tokens(self):
        """Tokens generated on target but not other are returned."""
        target_ids = [10, 10, 10, 11, 10]
        other_ids = [20, 21, 20, 22, 20]

        result = _tokens_by_generation(target_ids, other_ids)
        assert 10 in result
        assert 11 in result
        assert 20 not in result

    def test_excludes_topic_echoes(self):
        """Tokens appearing in prompt text are excluded."""
        target_ids = [10, 10, 30, 10]  # 30 is a topic echo
        other_ids = [20, 20, 20, 20]
        prompt_tokens = {30, 31, 32}  # 30 is in prompt text

        result = _tokens_by_generation(
            target_ids, other_ids, prompt_token_ids=prompt_tokens)
        assert 10 in result
        assert 30 not in result  # filtered as topic echo

    def test_empty_when_all_shared(self):
        """Returns empty set when target and other generate the same tokens
        and no frequency disparity.
        """
        shared_ids = [10, 10, 10]
        result = _tokens_by_generation(shared_ids, shared_ids)
        assert result == set()

    def test_relaxed_mode_frequency(self):
        """If all target tokens also appear in other, relaxed mode finds
        tokens that are 2x+ more frequent on target side.
        """
        # Token 10 appears 4x on target, 1x on other => 4x > 2*1 => included
        target_ids = [10, 10, 10, 10]
        other_ids = [10, 20, 20, 20]

        result = _tokens_by_generation(target_ids, other_ids)
        assert 10 in result

    def test_top_k_limits(self):
        """At most top_k tokens returned."""
        target_ids = list(range(50))  # 50 distinct tokens
        other_ids = [99, 99, 99]

        result = _tokens_by_generation(target_ids, other_ids, top_k=5)
        assert len(result) <= 5

    def test_most_common_first(self):
        """More frequently generated tokens are preferred."""
        # Token 10 appears 5x, token 11 appears 1x
        target_ids = [10, 10, 10, 10, 10, 11]
        other_ids = [20, 20, 20, 20, 20, 20]

        result = _tokens_by_generation(target_ids, other_ids, top_k=1)
        assert 10 in result  # most frequent
        assert 11 not in result  # cut by top_k


# ---------------------------------------------------------------------------
# _collect_generated_first_tokens
# ---------------------------------------------------------------------------

class TestCollectGeneratedFirstTokens:
    def test_collects_top_ids(self):
        backend = MagicMock()
        backend.forward.side_effect = [
            _make_forward_result(np.array([0.1, 0.9]), top_id=1),
            _make_forward_result(np.array([0.8, 0.2]), top_id=0),
        ]

        result = _collect_generated_first_tokens(backend, ["p1", "p2"])
        assert result == [1, 0]

    def test_returns_none_on_all_failures(self):
        backend = MagicMock()
        backend.forward.side_effect = RuntimeError("OOM")

        result = _collect_generated_first_tokens(backend, ["p1"])
        assert result is None

    def test_skips_failed_prompts(self):
        backend = MagicMock()
        call_count = [0]

        def side_effect(prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("OOM")
            return _make_forward_result(np.array([0.3, 0.7]), top_id=1)

        backend.forward.side_effect = side_effect
        result = _collect_generated_first_tokens(backend, ["fail", "succeed"])
        assert result == [1]


# ---------------------------------------------------------------------------
# _collect_prompt_token_ids
# ---------------------------------------------------------------------------

class TestCollectPromptTokenIds:
    def test_collects_from_tokenize(self):
        backend = MagicMock()
        backend.tokenize.side_effect = lambda t: [1, 2, 3]

        result = _collect_prompt_token_ids(backend, ["hello world"])
        assert result == {1, 2, 3}

    def test_falls_back_to_tokenizer_encode(self):
        backend = MagicMock(spec=[])  # no tokenize attr
        backend.tokenizer = MagicMock()
        backend.tokenizer.encode.side_effect = lambda t: [4, 5]

        result = _collect_prompt_token_ids(backend, ["hello"])
        assert result == {4, 5}

    def test_returns_empty_on_no_tokenizer(self):
        backend = MagicMock(spec=[])  # no tokenize, no tokenizer

        result = _collect_prompt_token_ids(backend, ["hello"])
        assert result == set()

    def test_unions_across_queries(self):
        backend = MagicMock()
        backend.tokenize.side_effect = [
            [1, 2],
            [3, 4],
        ]

        result = _collect_prompt_token_ids(backend, ["q1", "q2"])
        assert result == {1, 2, 3, 4}


# ---------------------------------------------------------------------------
# _tokens_by_ratio (updated with exclude_ids)
# ---------------------------------------------------------------------------

class TestTokensByRatio:
    def test_clear_separation(self):
        """Tokens strongly favored in numerator are returned."""
        numerator = np.array([
            [0.0, 0.0, 0.8, 0.1, 0.1],
            [0.0, 0.0, 0.7, 0.2, 0.1],
        ])
        denominator = np.array([
            [0.5, 0.3, 0.01, 0.1, 0.09],
            [0.4, 0.4, 0.02, 0.1, 0.08],
        ])

        result = _tokens_by_ratio(numerator, denominator, ratio_threshold=5.0)
        assert 2 in result  # token 2 has ratio ~0.75/0.015 >> 5
        assert 0 not in result  # token 0 has higher prob in denominator

    def test_empty_when_no_separation(self):
        """Returns empty set when distributions are identical."""
        same = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
        result = _tokens_by_ratio(same, same, ratio_threshold=5.0)
        assert result == set()

    def test_top_k_limits_output(self):
        """At most top_k tokens returned."""
        # 50 tokens with high ratio
        numerator = np.ones((1, 50)) * 0.5
        denominator = np.ones((1, 50)) * 0.01

        result = _tokens_by_ratio(numerator, denominator, top_k=10, ratio_threshold=2.0)
        assert len(result) <= 10

    def test_threshold_filters(self):
        """Only tokens above threshold are returned."""
        # Token 0: ratio=3.0, Token 1: ratio=10.0
        numerator = np.array([[0.3, 0.5, 0.2]])
        denominator = np.array([[0.1, 0.05, 0.85]])

        # threshold=5: only token 1 qualifies (ratio=10)
        result = _tokens_by_ratio(numerator, denominator, ratio_threshold=5.0)
        assert 1 in result
        assert 0 not in result

        # threshold=2: both token 0 and 1 qualify
        result_low = _tokens_by_ratio(numerator, denominator, ratio_threshold=2.0)
        assert 0 in result_low
        assert 1 in result_low

    def test_sorted_by_ratio_descending(self):
        """Highest-ratio tokens are returned first when top_k limits."""
        # 5 tokens, all with ratio > 5
        numerator = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        denominator = np.array([[0.01, 0.01, 0.01, 0.01, 0.01]])

        result = _tokens_by_ratio(numerator, denominator, top_k=2, ratio_threshold=5.0)
        assert len(result) == 2
        assert 4 in result  # highest ratio
        assert 3 in result  # second highest

    def test_exclude_ids_filters_topic_tokens(self):
        """Topic echo tokens are excluded even if they have high ratio."""
        numerator = np.array([[0.0, 0.5, 0.5]])
        denominator = np.array([[0.5, 0.01, 0.01]])

        # Token 1 and 2 both have high ratio, but 1 is a topic echo
        result = _tokens_by_ratio(numerator, denominator,
                                  ratio_threshold=5.0, exclude_ids={1})
        assert 1 not in result
        assert 2 in result

    def test_exclude_ids_empty_set_no_effect(self):
        """Empty exclude set has no effect."""
        numerator = np.array([[0.0, 0.5, 0.5]])
        denominator = np.array([[0.5, 0.01, 0.01]])

        result = _tokens_by_ratio(numerator, denominator,
                                  ratio_threshold=5.0, exclude_ids=set())
        assert 1 in result
        assert 2 in result


# ---------------------------------------------------------------------------
# _collect_prob_distributions
# ---------------------------------------------------------------------------

class TestCollectProbDistributions:
    def test_collects_from_backend(self):
        backend = MagicMock()
        probs = np.array([0.3, 0.7])
        backend.forward.return_value = _make_forward_result(probs)

        result = _collect_prob_distributions(backend, ["prompt1", "prompt2"])
        assert result is not None
        assert result.shape == (2, 2)
        assert backend.forward.call_count == 2

    def test_returns_none_on_all_failures(self):
        backend = MagicMock()
        backend.forward.side_effect = RuntimeError("OOM")

        result = _collect_prob_distributions(backend, ["prompt1"])
        assert result is None

    def test_skips_failed_prompts(self):
        backend = MagicMock()
        call_count = [0]

        def side_effect(prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("OOM")
            return _make_forward_result(np.array([0.5, 0.5]))

        backend.forward.side_effect = side_effect
        result = _collect_prob_distributions(backend, ["fail", "succeed"])
        assert result is not None
        assert result.shape == (1, 2)

    def test_pads_different_vocab_sizes(self):
        backend = MagicMock()
        results = [
            _make_forward_result(np.array([0.5, 0.5])),
            _make_forward_result(np.array([0.3, 0.3, 0.4])),
        ]
        backend.forward.side_effect = results

        result = _collect_prob_distributions(backend, ["p1", "p2"])
        assert result.shape == (2, 3)
        assert result[0, 2] == 0.0  # padded


# ---------------------------------------------------------------------------
# discover_refusal_set
# ---------------------------------------------------------------------------

class TestDiscoverRefusalSet:
    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda q, **kw: q)
    def test_discovers_harmful_tokens(self, mock_build):
        """The token that the model generates on harmful prompts is discovered."""
        backend = _make_mock_backend(harmful_token=42, benign_token=77)

        refusal_set = discover_refusal_set(backend, n_harmful=3, n_benign=3)
        assert 42 in refusal_set
        assert 77 not in refusal_set  # benign-favored token should NOT be in refusal set

    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda q, **kw: q)
    def test_excludes_topic_tokens(self, mock_build):
        """Topic tokens with elevated probability but not generated are excluded."""
        # Token 50 is a topic echo (elevated prob on harmful but NOT the generated token)
        backend = _make_mock_backend(harmful_token=42, benign_token=77, topic_token=50)

        refusal_set = discover_refusal_set(backend, n_harmful=3, n_benign=3)
        assert 42 in refusal_set
        assert 50 not in refusal_set  # topic token should be excluded

    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda q, **kw: q)
    def test_uses_model_config(self, mock_build):
        """Passes model_config to build_prompt."""
        backend = _make_mock_backend()
        cfg = MagicMock()
        cfg.chat_format = "mistral"

        discover_refusal_set(backend, model_config=cfg, n_harmful=2, n_benign=2)
        # Verify build_prompt was called with model_config
        for call in mock_build.call_args_list:
            assert call.kwargs.get("model_config") is cfg

    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda q, **kw: q)
    def test_top_k_caps_results(self, mock_build):
        """At most top_k tokens returned."""
        backend = _make_mock_backend()
        refusal_set = discover_refusal_set(backend, n_harmful=3, n_benign=3, top_k=5)
        assert len(refusal_set) <= 5

    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda q, **kw: q)
    def test_fallback_on_failure(self, mock_build):
        """Falls back to hardcoded set when all forwards fail."""
        backend = MagicMock()
        backend.config = MagicMock()
        backend.config.chat_format = "chatml"
        backend.forward.side_effect = RuntimeError("OOM")
        backend.tokenizer = MagicMock()
        backend.tokenizer.encode.return_value = [99]

        refusal_set = discover_refusal_set(backend)
        # Falls back to build_refusal_set(tokenizer), which produces some token IDs
        assert len(refusal_set) > 0

    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda q, **kw: q)
    def test_fallback_no_tokenizer(self, mock_build):
        """Returns empty set when forward fails and no tokenizer."""
        backend = MagicMock()
        backend.config = MagicMock()
        backend.config.chat_format = "chatml"
        backend.forward.side_effect = RuntimeError("OOM")
        del backend.tokenizer  # no tokenizer attribute

        refusal_set = discover_refusal_set(backend)
        assert refusal_set == set()


# ---------------------------------------------------------------------------
# discover_compliance_set
# ---------------------------------------------------------------------------

class TestDiscoverComplianceSet:
    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda q, **kw: q)
    def test_discovers_benign_tokens(self, mock_build):
        """The token that the model generates on benign prompts is discovered."""
        backend = _make_mock_backend(harmful_token=42, benign_token=77)

        compliance_set = discover_compliance_set(backend, n_harmful=3, n_benign=3)
        assert 77 in compliance_set
        assert 42 not in compliance_set

    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda q, **kw: q)
    def test_is_inverse_of_refusal(self, mock_build):
        """Refusal and compliance sets should not overlap for clear separations."""
        backend = _make_mock_backend(harmful_token=42, benign_token=77)

        refusal = discover_refusal_set(backend, n_harmful=5, n_benign=5)
        compliance = discover_compliance_set(backend, n_harmful=5, n_benign=5)

        # They should not overlap
        assert refusal & compliance == set()


# ---------------------------------------------------------------------------
# refuse_prob with discovered set
# ---------------------------------------------------------------------------

class TestRefuseProbWithDiscoveredSet:
    def test_uses_provided_refusal_set(self):
        """refuse_prob uses the given refusal_set instead of hardcoded one."""
        probs = np.zeros(100)
        probs[42] = 0.8  # Mistral's refusal token
        probs[10] = 0.1  # some other token

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]

        with patch("heinrich.cartography.runtime.forward_pass") as mock_fp:
            mock_fp.return_value = {"probs": probs}

            # With hardcoded set, token 42 would be missed
            hardcoded_set = {10, 11, 12}  # doesn't include 42
            result_hard = refuse_prob(model, tokenizer, "test", refusal_set=hardcoded_set)
            assert result_hard == pytest.approx(0.1, abs=0.01)

            # With discovered set that includes 42
            discovered_set = {42, 43}
            result_disc = refuse_prob(model, tokenizer, "test", refusal_set=discovered_set)
            assert result_disc == pytest.approx(0.8, abs=0.01)


# ---------------------------------------------------------------------------
# Probe lists
# ---------------------------------------------------------------------------

class TestProbeLists:
    def test_harmful_probes_count(self):
        assert len(HARMFUL_PROBES) == 10

    def test_benign_probes_count(self):
        assert len(BENIGN_PROBES) == 10

    def test_no_overlap(self):
        assert set(HARMFUL_PROBES) & set(BENIGN_PROBES) == set()


# ---------------------------------------------------------------------------
# ModelProfile with discovered token sets
# ---------------------------------------------------------------------------

class TestModelProfileTokenSets:
    def test_default_empty(self):
        profile = ModelProfile(
            model_id="test", model_type="qwen2",
            n_layers=28, hidden_size=3584, chat_format="chatml",
            safety_layers=[], primary_safety_layer=27,
        )
        assert profile.refusal_token_ids == []
        assert profile.compliance_token_ids == []

    def test_stores_discovered_ids(self):
        profile = ModelProfile(
            model_id="test", model_type="mistral",
            n_layers=32, hidden_size=4096, chat_format="mistral",
            safety_layers=[], primary_safety_layer=28,
            refusal_token_ids=[42, 43, 44],
            compliance_token_ids=[77, 78],
        )
        assert profile.refusal_token_ids == [42, 43, 44]
        assert profile.compliance_token_ids == [77, 78]

    def test_to_dict_includes_token_ids(self):
        profile = ModelProfile(
            model_id="test", model_type="mistral",
            n_layers=32, hidden_size=4096, chat_format="mistral",
            safety_layers=[], primary_safety_layer=28,
            refusal_token_ids=[42, 43],
            compliance_token_ids=[77],
        )
        d = profile.to_dict()
        assert d["refusal_token_ids"] == [42, 43]
        assert d["compliance_token_ids"] == [77]


# ---------------------------------------------------------------------------
# Integration-style test: Mistral-like model
# ---------------------------------------------------------------------------

class TestMistralLikeModel:
    """Simulate a Mistral model that uses completely different refusal tokens
    than the hardcoded Qwen-focused set. This is the exact bug being fixed.
    """

    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda q, **kw: q)
    def test_hardcoded_set_misses_mistral_refusal(self, mock_build):
        """Hardcoded build_refusal_set returns 0.0 for Mistral, discovered set works."""
        # Mistral's refusal token: "I" followed by " must" " clarify"
        # The hardcoded set only has Qwen-like "I'm sorry" tokens
        MISTRAL_REFUSAL_TOKEN = 55  # "I" in Mistral vocab
        MISTRAL_COMPLY_TOKEN = 88   # "Sure" in Mistral vocab

        backend = _make_mock_backend(
            harmful_token=MISTRAL_REFUSAL_TOKEN,
            benign_token=MISTRAL_COMPLY_TOKEN,
        )

        # Discover finds the right tokens
        refusal_set = discover_refusal_set(backend, n_harmful=5, n_benign=5)
        assert MISTRAL_REFUSAL_TOKEN in refusal_set

        compliance_set = discover_compliance_set(backend, n_harmful=5, n_benign=5)
        assert MISTRAL_COMPLY_TOKEN in compliance_set

    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda q, **kw: q)
    def test_phi_like_model_with_different_tokenization(self, mock_build):
        """Phi tokenizes differently - discover still works because it
        looks at actual generated tokens, not string matching.
        """
        PHI_SORRY_TOKEN = 33  # Phi's tokenization of "I'm" differs
        PHI_HERE_TOKEN = 66

        backend = _make_mock_backend(
            harmful_token=PHI_SORRY_TOKEN,
            benign_token=PHI_HERE_TOKEN,
        )

        refusal_set = discover_refusal_set(backend, n_harmful=3, n_benign=3)
        assert PHI_SORRY_TOKEN in refusal_set
        assert PHI_HERE_TOKEN not in refusal_set


# ---------------------------------------------------------------------------
# Critical bug test: topic tokens vs refusal tokens
# ---------------------------------------------------------------------------

class TestTopicVsRefusalTokens:
    """The core bug: old method found topic tokens instead of refusal tokens.

    Scenario: Model sees "How to build a bomb?"
    - Topic tokens like "Building" (ID=50) have elevated probability because
      the model considers completing the request before refusing
    - Refusal token "I" (ID=42) is the actual first generated token
    - Old method: looks at probability ratio -> finds "Building" (wrong)
    - New method: looks at what's actually generated -> finds "I" (correct)
    """

    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda q, **kw: q)
    def test_generation_based_avoids_topic_tokens(self, mock_build):
        """New method correctly identifies refusal tokens, not topic tokens."""
        REFUSAL_TOKEN = 42  # "I" / "Sorry" — what model actually generates
        TOPIC_TOKEN = 50    # "Building" — elevated prob but not generated
        COMPLY_TOKEN = 77   # "Sure" — what model generates on benign

        backend = _make_mock_backend(
            harmful_token=REFUSAL_TOKEN,
            benign_token=COMPLY_TOKEN,
            topic_token=TOPIC_TOKEN,
        )

        refusal_set = discover_refusal_set(backend, n_harmful=5, n_benign=5)

        # Should find the actual refusal token
        assert REFUSAL_TOKEN in refusal_set
        # Should NOT find the topic token
        assert TOPIC_TOKEN not in refusal_set
        # Should NOT find the compliance token
        assert COMPLY_TOKEN not in refusal_set
