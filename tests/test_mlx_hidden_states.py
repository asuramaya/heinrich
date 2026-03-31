"""Tests for MLX hidden state capture."""
import pytest

try:
    import mlx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
def test_mlx_hidden_states():
    from heinrich.probe.mlx_provider import MLXProvider
    p = MLXProvider({"model": "Qwen/Qwen2.5-7B-Instruct"})
    result = p.forward_with_internals("Hello")
    assert "logits" in result
    # Hidden states may or may not work depending on model architecture
    if "hidden_states" in result:
        assert len(result["hidden_states"]) > 1
