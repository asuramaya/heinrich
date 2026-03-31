from heinrich.probe.provider import MockProvider
from heinrich.probe.trigger import build_case, score_trigger_cases, detect_identity

def test_build_case():
    c = build_case("test-1", "Hello Claude")
    assert c["custom_id"] == "test-1"
    assert c["messages"][0]["content"] == "Hello Claude"

def test_score_with_control():
    provider = MockProvider({
        "control": "I am Qwen, an AI assistant.",
        "trigger": "I am Claude, made by Anthropic.",
    })
    control = build_case("control", "Hello")
    cases = [build_case("trigger", "Hello Claude")]
    signals = score_trigger_cases(provider, cases, model="test", control_case=control)
    assert len(signals) == 1
    assert signals[0].kind == "trigger_score"
    assert signals[0].value > 0  # divergence from control

def test_score_without_control():
    provider = MockProvider({"q": "Hello there"})
    cases = [build_case("q", "Hi")]
    signals = score_trigger_cases(provider, cases, model="test")
    assert len(signals) == 1
    assert signals[0].value == 0.0  # no control = no divergence

def test_score_has_text_preview():
    provider = MockProvider({"q": "A long response about many things"})
    signals = score_trigger_cases(provider, [build_case("q", "Hi")], model="test")
    assert "text_preview" in signals[0].metadata

def test_detect_identity_claude():
    assert detect_identity("I am Claude, an AI assistant")["label"] == "claude"

def test_detect_identity_qwen():
    assert detect_identity("I'm Qwen, created by Alibaba")["label"] == "qwen"

def test_detect_identity_other():
    assert detect_identity("Hello! How can I help?")["label"] == "other"
