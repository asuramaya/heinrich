from heinrich.mcp import ToolServer

def test_observe_grid():
    s = ToolServer()
    r = s.call_tool("heinrich_observe", {"grid": [[0,1],[1,0]], "label": "t"})
    assert r["signals_summary"]["total"] > 0

def test_observe_logits():
    s = ToolServer()
    r = s.call_tool("heinrich_observe", {"logits": [0.1, 0.2, 0.7], "label": "t"})
    assert r["signals_summary"]["total"] > 0

def test_loop_states():
    s = ToolServer()
    r = s.call_tool("heinrich_loop", {"states": [[[0,0],[0,0]], [[1,1],[1,1]]], "max_turns": 2})
    assert r["signals_summary"]["total"] > 0

def test_tools_list():
    names = {t["name"] for t in ToolServer().list_tools()}
    assert "heinrich_observe" in names
    assert "heinrich_loop" in names
