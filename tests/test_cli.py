import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
PYTHON = str(Path(__file__).parent.parent / ".venv" / "bin" / "python")


def test_cli_fetch_local():
    result = subprocess.run(
        [PYTHON, "-m", "heinrich.cli", "fetch", str(FIXTURES)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["stages_run"] == ["fetch"]
    assert output["signals_summary"]["total"] > 0


def test_cli_fetch_json_flag():
    result = subprocess.run(
        [PYTHON, "-m", "heinrich.cli", "fetch", str(FIXTURES), "--json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "heinrich_version" in data


def test_cli_status_no_session():
    result = subprocess.run(
        [PYTHON, "-m", "heinrich.cli", "status"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
