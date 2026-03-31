from heinrich.probe.provider import MockProvider, HuggingFaceLocalProvider

def test_mock_provider_chat():
    p = MockProvider({"q1": "Hello"})
    results = p.chat_completions([{"custom_id": "q1", "messages": []}], model="m")
    assert results[0]["text"] == "Hello"

def test_mock_provider_activations():
    p = MockProvider()
    results = p.activations([{"custom_id": "q1"}], model="m")
    assert "activations" in results[0]

def test_hf_provider_init():
    p = HuggingFaceLocalProvider({"model": "test-model"})
    assert p.describe()["provider_type"] == "hf-local"

def test_hf_provider_no_model_raises():
    p = HuggingFaceLocalProvider({})
    try:
        p._ensure_loaded()
        assert False
    except (ValueError, ImportError):
        pass  # either no model specified or torch not available

def test_hf_provider_describe():
    p = HuggingFaceLocalProvider({"model": "my-model", "device": "cpu"})
    d = p.describe()
    assert d["model"] == "my-model"
