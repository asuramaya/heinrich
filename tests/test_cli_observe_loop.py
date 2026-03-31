"""Tests for the CLI observe and loop commands."""
import json
import tempfile
import subprocess
import sys
from pathlib import Path


def test_cli_observe():
    d = Path(tempfile.mkdtemp())
    (d / "input.json").write_text(json.dumps({"grid": [[0, 1], [1, 0]]}))
    r = subprocess.run(
        [sys.executable, "-m", "heinrich.cli", "observe", str(d / "input.json")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert json.loads(r.stdout)["signals_summary"]["total"] > 0


def test_cli_loop():
    d = Path(tempfile.mkdtemp())
    (d / "states.json").write_text(json.dumps({"states": [[[0, 0], [0, 0]], [[1, 1], [1, 1]]]}))
    r = subprocess.run(
        [sys.executable, "-m", "heinrich.cli", "loop", str(d / "states.json"), "--max-turns", "2"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert json.loads(r.stdout)["signals_summary"]["total"] > 0
