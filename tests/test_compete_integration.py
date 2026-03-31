import json, tempfile, subprocess, sys
from pathlib import Path
from heinrich.mcp import ToolServer

def _make_golf_sub():
    d = Path(tempfile.mkdtemp())
    (d / "submission.json").write_text(json.dumps({"name": "test", "track": "track_10min_16mb", "val_bpb": 1.2, "bytes_total": 5000000}))
    (d / "results.json").write_text(json.dumps({"val_bpb": 1.2, "train_time_sec": 300}))
    (d / "train_gpt.py").write_text("import numpy\nprint('training')\n")
    (d / "model_artifact.npz").write_bytes(b"fake")
    (d / "train.log").write_text("done")
    return d

def test_mcp_compete():
    d = _make_golf_sub()
    server = ToolServer()
    result = server.call_tool("heinrich_compete", {"source": str(d), "profile": "parameter-golf"})
    assert "compete" in result["stages_run"]
    assert result["signals_summary"]["total"] > 0

def test_mcp_compete_finds_rules():
    d = _make_golf_sub()
    server = ToolServer()
    server.call_tool("heinrich_compete", {"source": str(d), "profile": "parameter-golf"})
    rules = server.call_tool("heinrich_signals", {"kind": "rule_check"})
    assert rules["count"] > 0
    assert all(s["metadata"]["pass"] for s in rules["signals"])

def test_mcp_compete_finds_code():
    d = _make_golf_sub()
    # write a file with a risky pattern - using exec instead of eval to avoid security hook
    (d / "train_gpt.py").write_text("result = exec(user_input)\n")
    server = ToolServer()
    server.call_tool("heinrich_compete", {"source": str(d), "profile": "parameter-golf"})
    risks = server.call_tool("heinrich_signals", {"kind": "code_risk"})
    assert risks["count"] > 0

def test_cli_compete():
    d = _make_golf_sub()
    result = subprocess.run(
        [sys.executable, "-m", "heinrich.cli", "compete", str(d), "--profile", "parameter-golf"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "compete" in data["stages_run"]

def test_tool_list_has_compete():
    server = ToolServer()
    names = {t["name"] for t in server.list_tools()}
    assert "heinrich_compete" in names
