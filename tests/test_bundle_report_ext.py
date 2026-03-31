import json
import subprocess
import sys
import tempfile
from pathlib import Path
from heinrich.bundle.report import render_table, write_csv


def test_render_table():
    rows = [{"name": "a", "score": 1.0}, {"name": "bb", "score": 2.5}]
    text = render_table(rows, ["name", "score"])
    assert "name" in text
    assert "bb" in text


def test_render_table_empty():
    assert "(empty)" in render_table([], ["col"])


def test_render_table_top():
    rows = [{"x": i} for i in range(10)]
    text = render_table(rows, ["x"], top=3)
    lines = [l for l in text.strip().split("\n") if l.strip() and not l.startswith("-")]
    assert len(lines) == 4  # header + 3 data rows


def test_write_csv():
    path = Path(tempfile.mkdtemp()) / "test.csv"
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    write_csv(path, rows, ["a", "b"])
    content = path.read_text()
    assert "a,b" in content
    assert "1,x" in content


def test_cli_bundle():
    manifest_dir = Path(tempfile.mkdtemp())
    manifest = {"bundle_id": "cli-test", "metrics": {"bpb": 0.5}}
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    out_dir = manifest_dir / "out"
    result = subprocess.run(
        [sys.executable, "-m", "heinrich.cli", "bundle", str(manifest_path), str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["bundle_id"] == "cli-test"
