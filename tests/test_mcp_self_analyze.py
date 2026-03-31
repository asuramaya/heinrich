"""Tests for the heinrich_self_analyze MCP tool."""
from heinrich.mcp import ToolServer


def test_self_analyze_tool_exists():
    names = {t["name"] for t in ToolServer().list_tools()}
    assert "heinrich_self_analyze" in names


def test_self_analyze_no_provider():
    # Default MockProvider doesn't have forward_with_internals, should handle gracefully
    s = ToolServer()
    r = s.call_tool("heinrich_self_analyze", {"text": "Hello"})
    # Should not crash — stage checks for forward_with_internals
    assert "stages_run" in r
