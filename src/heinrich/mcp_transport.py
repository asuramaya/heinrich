"""MCP stdio transport — JSON-RPC over stdin/stdout for the heinrich ToolServer."""
from __future__ import annotations

import json
import sys
from typing import Any

from .mcp import ToolServer, TOOLS


def run_stdio_server() -> None:
    """Run the MCP tool server over stdin/stdout using JSON-RPC."""
    server = ToolServer()

    # Build tool list once at startup
    tools_list = []
    for name, defn in TOOLS.items():
        tool_schema = {
            "name": name,
            "description": defn["description"],
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
        params = defn.get("parameters", {})
        for param_name, param_info in params.items():
            tool_schema["inputSchema"]["properties"][param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
            if param_info.get("required"):
                tool_schema["inputSchema"]["required"].append(param_name)
        tools_list.append(tool_schema)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _send_error(None, -32700, "Parse error")
            continue

        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            _send_result(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "heinrich", "version": "0.2.0"},
            })
        elif method == "tools/list":
            _send_result(req_id, {"tools": tools_list})
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            try:
                result = server.call_tool(tool_name, arguments)
                content = json.dumps(result, indent=2, default=str)
                _send_result(req_id, {
                    "content": [{"type": "text", "text": content}],
                })
            except Exception as e:
                _send_result(req_id, {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "isError": True,
                })
        elif method == "notifications/initialized":
            pass  # client acknowledgment, no response needed
        else:
            _send_error(req_id, -32601, f"Method not found: {method}")


def _send_result(req_id: Any, result: dict) -> None:
    response = {"jsonrpc": "2.0", "id": req_id, "result": result}
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def _send_error(req_id: Any, code: int, message: str) -> None:
    response = {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
