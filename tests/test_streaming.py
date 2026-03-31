"""Tests for streaming self-analysis stub."""
from heinrich.probe.streaming import generate_with_signals
from heinrich.probe.provider import MockProvider


def test_generate_with_signals():
    provider = MockProvider({"stream": "Hello there!"})
    text, store = generate_with_signals(provider, "Hi", model="test")
    assert len(store) > 0
    assert any(s.kind == "generated_text" for s in store)


def test_generate_with_callback():
    provider = MockProvider({"stream": "Response"})
    captured = []

    def on_token(step, signal):
        captured.append((step, signal.kind))

    text, store = generate_with_signals(provider, "Hi", model="test", on_token=on_token)
    # MockProvider has no forward_with_internals, so no logit signals
    # but generated_text should be there
    assert any(s.kind == "generated_text" for s in store)


def test_generate_returns_text():
    provider = MockProvider({"stream": "Hello world"})
    text, store = generate_with_signals(provider, "Hi", model="test")
    assert text == "Hello world"


def test_generate_text_stored_in_metadata():
    provider = MockProvider({"stream": "Short response"})
    text, store = generate_with_signals(provider, "Hi", model="test")
    gen_sigs = [s for s in store if s.kind == "generated_text"]
    assert len(gen_sigs) == 1
    assert gen_sigs[0].metadata["text"] == "Short response"


def test_generate_with_logits_provider():
    import numpy as np

    class LogitsProvider:
        def forward_with_internals(self, text, *, model=""):
            rng = np.random.default_rng(42)
            return {"logits": rng.standard_normal(100)}

        def chat_completions(self, cases, *, model):
            return [{"custom_id": c.get("custom_id", ""), "text": "ok"} for c in cases]

    provider = LogitsProvider()
    text, store = generate_with_signals(provider, "Hello", model="test")
    kinds = {s.kind for s in store}
    assert "generated_text" in kinds
    assert "self_entropy" in kinds  # logit signals present
