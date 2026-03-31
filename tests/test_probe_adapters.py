"""Tests for provider-agnostic probe adapters."""
import pytest
import numpy as np

try:
    import mlx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class FakeProvider:
    def forward_with_internals(self, text, *, model=""):
        rng = np.random.default_rng(hash(text) % 2**31)
        return {"logits": rng.standard_normal(100)}


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(text[0]) % 100] if text else []


def test_vocab_scan():
    from heinrich.probe.vocab_scan import scan_vocabulary_via_provider
    signals = scan_vocabulary_via_provider(
        FakeProvider(), "Hello", ["Claude", "Qwen", "GPT"],
        label="test",
    )
    assert len(signals) == 3
    assert all(s.kind == "vocab_kl" for s in signals)


def test_vocab_scan_sorted_descending():
    from heinrich.probe.vocab_scan import scan_vocabulary_via_provider
    signals = scan_vocabulary_via_provider(
        FakeProvider(), "Hello", ["Claude", "Qwen", "GPT"],
        label="test",
    )
    values = [s.value for s in signals]
    assert values == sorted(values, reverse=True)


def test_logit_probe():
    from heinrich.probe.logit_probe import probe_next_tokens
    signals = probe_next_tokens(
        FakeProvider(), "Hello", ["Hi", "Hey", "Yo"],
        FakeTokenizer(), label="test",
    )
    assert len(signals) == 3
    assert all(s.kind == "nexttoken_prob" for s in signals)


def test_logit_probe_sorted_descending():
    from heinrich.probe.logit_probe import probe_next_tokens
    signals = probe_next_tokens(
        FakeProvider(), "Hello", ["Hi", "Hey", "Yo"],
        FakeTokenizer(), label="test",
    )
    values = [s.value for s in signals]
    assert values == sorted(values, reverse=True)


def test_vocab_scan_no_logits():
    from heinrich.probe.vocab_scan import scan_vocabulary_via_provider

    class NoLogitsProvider:
        def forward_with_internals(self, text, *, model=""):
            return {}

    signals = scan_vocabulary_via_provider(
        NoLogitsProvider(), "Hello", ["Claude"],
        label="test",
    )
    assert signals == []


def test_logit_probe_empty_candidates():
    from heinrich.probe.logit_probe import probe_next_tokens
    signals = probe_next_tokens(
        FakeProvider(), "Hello", [],
        FakeTokenizer(), label="test",
    )
    assert signals == []


@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
def test_vocab_scan_real_model():
    from heinrich.probe.mlx_provider import MLXProvider
    from heinrich.probe.vocab_scan import scan_vocabulary_via_provider
    provider = MLXProvider({"model": "Qwen/Qwen2.5-7B-Instruct"})
    signals = scan_vocabulary_via_provider(
        provider, "Hello", ["Claude", "Qwen"], model="q",
    )
    assert len(signals) == 2
