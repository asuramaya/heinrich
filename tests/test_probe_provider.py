from heinrich.probe.provider import MockProvider

def test_mock_provider_describe():
    p = MockProvider()
    assert p.describe()["provider_type"] == "mock"

def test_mock_provider_chat():
    p = MockProvider({"q1": "Hello from mock"})
    cases = [{"custom_id": "q1", "messages": [{"role": "user", "content": "Hi"}]}]
    results = p.chat_completions(cases, model="test")
    assert len(results) == 1
    assert results[0]["text"] == "Hello from mock"

def test_mock_provider_default_response():
    p = MockProvider()
    results = p.chat_completions([{"custom_id": "x"}], model="m")
    assert "Mock response" in results[0]["text"]

def test_mock_provider_activations():
    p = MockProvider()
    results = p.activations([{"custom_id": "x"}], model="m")
    assert len(results) == 1
    assert "activations" in results[0]
