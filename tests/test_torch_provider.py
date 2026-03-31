"""Tests for HuggingFace local (torch) provider integration."""
import pytest


def test_torch_provider_configure():
    from heinrich.mcp import ToolServer
    server = ToolServer()
    try:
        r = server.call_tool("heinrich_configure", {"provider": "hf-local", "model": "test-model", "device": "cpu"})
        assert r["configured"]
    except ImportError:
        pytest.skip("torch not installed")


def test_torch_provider_describe():
    from heinrich.probe.provider import HuggingFaceLocalProvider
    p = HuggingFaceLocalProvider({"model": "test", "device": "cpu"})
    assert p.describe()["provider_type"] == "hf-local"


def test_torch_provider_describe_model():
    from heinrich.probe.provider import HuggingFaceLocalProvider
    p = HuggingFaceLocalProvider({"model": "some/model", "device": "cuda"})
    info = p.describe()
    assert info["model"] == "some/model"


def test_torch_provider_configure_returns_provider_info():
    from heinrich.mcp import ToolServer
    server = ToolServer()
    try:
        r = server.call_tool("heinrich_configure", {"provider": "hf-local", "model": "test-model"})
        assert r["provider"]["provider_type"] == "hf-local"
    except ImportError:
        pytest.skip("torch not installed")
