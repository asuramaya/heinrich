from pathlib import Path
from heinrich.mcp import ToolServer

FIXTURES = Path(__file__).parent / "fixtures"


def test_list_tools():
    server = ToolServer()
    tools = server.list_tools()
    names = {t["name"] for t in tools}
    assert "heinrich_fetch" in names
    assert "heinrich_inspect" in names
    assert "heinrich_diff" in names
    assert "heinrich_probe" in names
    assert "heinrich_bundle" in names
    assert "heinrich_signals" in names
    assert "heinrich_status" in names
    assert "heinrich_pipeline" in names
    assert len(tools) == 13


def test_fetch_tool():
    server = ToolServer()
    result = server.call_tool("heinrich_fetch", {"source": str(FIXTURES)})
    assert "heinrich_version" in result
    assert result["signals_summary"]["total"] > 0
    assert "fetch" in result["stages_run"]


def test_inspect_tool():
    server = ToolServer()
    result = server.call_tool("heinrich_inspect", {"source": str(FIXTURES / "tiny_weights.npz")})
    assert result["signals_summary"]["total"] > 0


def test_diff_tool():
    server = ToolServer()
    result = server.call_tool("heinrich_diff", {
        "lhs": str(FIXTURES / "tiny_weights_base.npz"),
        "rhs": str(FIXTURES / "tiny_weights_modified.npz"),
    })
    assert result["signals_summary"]["total"] > 0


def test_probe_tool():
    server = ToolServer()
    result = server.call_tool("heinrich_probe", {"prompts": ["Hello Claude", "Hello"]})
    assert result["signals_summary"]["total"] > 0


def test_signals_filter():
    server = ToolServer()
    server.call_tool("heinrich_fetch", {"source": str(FIXTURES)})
    result = server.call_tool("heinrich_signals", {"kind": "config_field"})
    assert result["count"] > 0
    assert all(s["kind"] == "config_field" for s in result["signals"])


def test_status():
    server = ToolServer()
    result = server.call_tool("heinrich_status", {})
    assert result["signal_count"] == 0
    server.call_tool("heinrich_fetch", {"source": str(FIXTURES)})
    result = server.call_tool("heinrich_status", {})
    assert result["signal_count"] > 0
    assert "fetch" in result["stages_run"]


def test_stateful_accumulation():
    server = ToolServer()
    server.call_tool("heinrich_fetch", {"source": str(FIXTURES)})
    count1 = server.call_tool("heinrich_status", {})["signal_count"]
    server.call_tool("heinrich_inspect", {"source": str(FIXTURES / "tiny_weights.npz")})
    count2 = server.call_tool("heinrich_status", {})["signal_count"]
    assert count2 > count1


def test_unknown_tool():
    server = ToolServer()
    result = server.call_tool("nonexistent", {})
    assert "error" in result


def test_bundle_tool():
    server = ToolServer()
    server.call_tool("heinrich_fetch", {"source": str(FIXTURES)})
    result = server.call_tool("heinrich_bundle", {"top_k": 5})
    assert len(result["signals_summary"]["top_10"]) <= 5
