"""Tests for heinrich_validate MCP tool."""
import json
import tempfile
from pathlib import Path
from heinrich.mcp import ToolServer


def _make_submission():
    d = Path(tempfile.mkdtemp())
    (d / "submission.json").write_text(json.dumps({"name": "test", "val_bpb": 0.5, "bytes_total": 1000}))
    (d / "results.json").write_text(json.dumps({"val_bpb": 0.5, "bytes_total": 1000}))
    bundle = d / "audits_bundle"
    bundle.mkdir()
    (bundle / "claim.json").write_text(json.dumps({"candidate_id": "test"}))
    (bundle / "metrics.json").write_text(json.dumps({"bridge_bpb": 0.5}))
    (bundle / "audits.json").write_text(json.dumps({"tier1": {"status": "pass"}}))
    return d


def test_validate_submission_directory():
    d = _make_submission()
    server = ToolServer()
    result = server.call_tool("heinrich_validate", {"source": str(d), "label": "test-sub"})
    assert result["signals_summary"]["total"] > 0
    assert "validate" in result["stages_run"]


def test_validate_emits_consistency():
    d = _make_submission()
    server = ToolServer()
    server.call_tool("heinrich_validate", {"source": str(d), "label": "test"})
    consistency = server.call_tool("heinrich_signals", {"kind": "cross_file_consistency"})
    # val_bpb appears in both files with same value
    assert consistency["count"] > 0


def test_validate_emits_claim_level():
    d = _make_submission()
    server = ToolServer()
    server.call_tool("heinrich_validate", {"source": str(d), "label": "test"})
    claims = server.call_tool("heinrich_signals", {"kind": "claim_level"})
    assert claims["count"] == 1
    assert claims["signals"][0]["value"] == 1.0  # bridge_bpb present = level 1


def test_validate_list_tools_includes_validate():
    server = ToolServer()
    tools = server.list_tools()
    names = {t["name"] for t in tools}
    assert "heinrich_validate" in names
