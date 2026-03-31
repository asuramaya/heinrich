import json
import subprocess
import sys
import tempfile
from pathlib import Path

def test_cli_report_directory():
    d = Path(tempfile.mkdtemp())
    (d / "run.json").write_text(json.dumps({"bpb": 0.5, "test_bpb": 0.6}))
    result = subprocess.run(
        [sys.executable, "-m", "heinrich.cli", "report", str(d), "--top", "5"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "ranked_signals" in data
    assert "report" in data["stages_run"]

def test_cli_report_empty_dir():
    d = Path(tempfile.mkdtemp())
    result = subprocess.run(
        [sys.executable, "-m", "heinrich.cli", "report", str(d)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
