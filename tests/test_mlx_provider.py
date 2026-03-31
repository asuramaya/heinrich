import pytest
import sys

try:
    import mlx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
def test_mlx_provider_describe():
    from heinrich.probe.mlx_provider import MLXProvider
    p = MLXProvider({"model": "test-model"})
    assert p.describe()["provider_type"] == "mlx"

@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
def test_mlx_provider_no_model_raises():
    from heinrich.probe.mlx_provider import MLXProvider
    p = MLXProvider({})
    try:
        p._ensure_loaded()
        assert False
    except ValueError:
        pass

@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
def test_mlx_provider_self_analyze():
    """Full integration: load real model, self-analyze, get signals."""
    from heinrich.probe.mlx_provider import MLXProvider
    from heinrich.probe.self_analyze import SelfAnalyzeStage
    from heinrich.signal import SignalStore
    
    provider = MLXProvider({"model": "Qwen/Qwen2.5-7B-Instruct"})
    store = SignalStore()
    stage = SelfAnalyzeStage()
    stage.run(store, {"provider": provider, "text": "Hello", "_iteration": 0})
    
    entropy = store.filter(kind="self_entropy")
    assert len(entropy) == 1
    assert entropy[0].value > 0
    
    confidence = store.filter(kind="self_confidence")
    assert len(confidence) == 1
    assert 0 < confidence[0].value <= 1.0

@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
def test_mlx_provider_chat():
    from heinrich.probe.mlx_provider import MLXProvider
    provider = MLXProvider({"model": "Qwen/Qwen2.5-7B-Instruct", "max_new_tokens": 20})
    results = provider.chat_completions([
        {"custom_id": "t1", "messages": [{"role": "user", "content": "Say hi"}]}
    ], model="qwen")
    assert len(results) == 1
    assert len(results[0]["text"]) > 0

@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
def test_mcp_configure_mlx():
    from heinrich.mcp import ToolServer
    server = ToolServer()
    r = server.call_tool("heinrich_configure", {"provider": "mlx", "model": "Qwen/Qwen2.5-7B-Instruct"})
    assert r["configured"]
    assert r["provider"]["provider_type"] == "mlx"
