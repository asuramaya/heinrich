import tempfile
from pathlib import Path
from heinrich.inspect.codescan import scan_file, scan_text, scan_directory

def test_scan_clean():
    signals = scan_text("import numpy as np\nx = np.array([1,2,3])\n", label="test")
    risks = [s for s in signals if s.kind == "code_risk"]
    assert len(risks) == 0

def test_scan_eval():
    signals = scan_text("result = eval(user_input)\n")
    risks = [s for s in signals if s.kind == "code_risk"]
    assert len(risks) == 1
    assert risks[0].metadata["severity"] == "critical"

def test_scan_subprocess():
    signals = scan_text("subprocess.run(['ls'])\n")
    risks = [s for s in signals if s.kind == "code_risk"]
    assert len(risks) == 1
    assert risks[0].metadata["severity"] == "high"

def test_scan_multiple_risks():
    code = "eval(x)\nexec(y)\nos.system('rm -rf /')\n"
    signals = scan_text(code)
    risks = [s for s in signals if s.kind == "code_risk"]
    assert len(risks) == 3

def test_scan_ignores_comments():
    signals = scan_text("# eval(x)\nprint('safe')\n")
    risks = [s for s in signals if s.kind == "code_risk"]
    assert len(risks) == 0

def test_scan_line_count():
    signals = scan_text("a\nb\nc\n")
    lines = [s for s in signals if s.kind == "code_lines"]
    assert lines[0].value == 3.0

def test_scan_file():
    d = Path(tempfile.mkdtemp())
    (d / "script.py").write_text("eval('bad')\n")
    signals = scan_file(d / "script.py")
    assert any(s.kind == "code_risk" for s in signals)

def test_scan_directory():
    d = Path(tempfile.mkdtemp())
    (d / "a.py").write_text("eval('x')\n")
    (d / "b.py").write_text("print('safe')\n")
    (d / "c.txt").write_text("eval('ignored')\n")  # not .py
    signals = scan_directory(d)
    risks = [s for s in signals if s.kind == "code_risk"]
    assert len(risks) == 1  # only a.py
