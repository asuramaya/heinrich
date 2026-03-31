import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def _call_server(messages):
    """Send JSON-RPC messages to the server and collect responses."""
    input_text = "\n".join(json.dumps(m) for m in messages) + "\n"
    result = subprocess.run(
        [sys.executable, "-m", "heinrich.mcp_transport"],
        input=input_text, capture_output=True, text=True, timeout=10,
    )
    responses = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            responses.append(json.loads(line))
    return responses


def test_initialize():
    responses = _call_server([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    ])
    assert len(responses) >= 1
    assert responses[0]["result"]["serverInfo"]["name"] == "heinrich"


def test_tools_list():
    responses = _call_server([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ])
    tools_response = [r for r in responses if r.get("id") == 2][0]
    tools = tools_response["result"]["tools"]
    names = {t["name"] for t in tools}
    assert "heinrich_fetch" in names
    assert "heinrich_diff" in names
    assert len(tools) == 9


def test_tool_call_fetch():
    responses = _call_server([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "heinrich_fetch",
            "arguments": {"source": str(FIXTURES)},
        }},
    ])
    call_response = [r for r in responses if r.get("id") == 2][0]
    content = call_response["result"]["content"][0]["text"]
    data = json.loads(content)
    assert data["signals_summary"]["total"] > 0


def test_tool_call_status():
    responses = _call_server([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "heinrich_status",
            "arguments": {},
        }},
    ])
    call_response = [r for r in responses if r.get("id") == 2][0]
    content = json.loads(call_response["result"]["content"][0]["text"])
    assert "signal_count" in content


def test_unknown_method():
    responses = _call_server([
        {"jsonrpc": "2.0", "id": 1, "method": "bogus", "params": {}},
    ])
    assert "error" in responses[0]
