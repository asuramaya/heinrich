"""Tests for the heinrich_configure MCP tool."""
from heinrich.mcp import ToolServer


def test_configure_mock():
    s = ToolServer()
    r = s.call_tool("heinrich_configure", {"provider": "mock"})
    assert r["configured"]
    assert r["provider"]["provider_type"] == "mock"


def test_configure_hf_local_init():
    s = ToolServer()
    r = s.call_tool("heinrich_configure", {"provider": "hf-local", "model": "test-model"})
    assert r["configured"]
    assert r["provider"]["provider_type"] == "hf-local"


def test_configure_changes_probe_provider():
    s = ToolServer()
    s.call_tool("heinrich_configure", {"provider": "mock"})
    r = s.call_tool("heinrich_probe", {"prompts": ["Hello"], "model": "m"})
    assert r["signals_summary"]["total"] > 0


def test_configure_tool_exists():
    names = {t["name"] for t in ToolServer().list_tools()}
    assert "heinrich_configure" in names
