import tempfile, json, zipfile
from pathlib import Path
from heinrich.bundle.formats import package_zip, copy_record, generate_triage_report

def test_package_zip():
    d = Path(tempfile.mkdtemp())
    (d / "a.txt").write_text("hello")
    (d / "b.txt").write_text("world")
    out = d / "out.zip"
    package_zip({"a.txt": d / "a.txt", "b.txt": d / "b.txt"}, out)
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        assert set(zf.namelist()) == {"a.txt", "b.txt"}

def test_package_zip_directory():
    d = Path(tempfile.mkdtemp())
    sub = d / "code"
    sub.mkdir()
    (sub / "main.py").write_text("pass")
    out = d / "out.zip"
    package_zip({"code": sub}, out)
    with zipfile.ZipFile(out) as zf:
        assert any("main.py" in n for n in zf.namelist())

def test_copy_record():
    src = Path(tempfile.mkdtemp())
    (src / "file.txt").write_text("data")
    dst = Path(tempfile.mkdtemp()) / "copy"
    copy_record(src, dst)
    assert (dst / "file.txt").read_text() == "data"

def test_triage_all_pass():
    report = generate_triage_report([])
    assert "All checks passed" in report

def test_triage_with_issues():
    signals = [
        {"kind": "rule_check", "target": "required_file:results.json", "metadata": {"pass": False, "rule": "required_file"}},
        {"kind": "code_risk", "target": "dangerous_func", "value": 4.0, "metadata": {"severity": "critical", "description": "dangerous call", "first_line": 10}},
    ]
    report = generate_triage_report(signals)
    assert "Rule Violations" in report
    assert "Code Risks" in report
    assert "2 problem(s)" in report

def test_triage_consistency():
    signals = [{"kind": "cross_file_consistency", "target": "val_bpb", "metadata": {"consistent": False, "values": [1.0, 2.0], "files": ["a.json", "b.json"]}}]
    report = generate_triage_report(signals)
    assert "Consistency Issues" in report
