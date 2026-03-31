"""Source code risk scanning."""
from __future__ import annotations
from pathlib import Path
from ..signal import Signal

RISK_PATTERNS: dict[str, tuple[str, str]] = {
    "eval(": ("critical", "Arbitrary code execution via eval()"),
    "exec(": ("critical", "Arbitrary code execution via exec()"),
    "os.system(": ("critical", "Shell command execution"),
    "subprocess.run": ("high", "Subprocess execution"),
    "subprocess.Popen": ("high", "Subprocess execution"),
    "subprocess.call": ("high", "Subprocess execution"),
    "pickle.load": ("high", "Pickle deserialization"),
    "__import__": ("high", "Dynamic import"),
    "importlib": ("medium", "Dynamic module loading"),
    "requests.get": ("medium", "Network GET request"),
    "requests.post": ("medium", "Network POST request"),
    "urllib.request": ("medium", "Network access"),
    "socket.socket": ("high", "Raw socket access"),
    "ctypes.": ("high", "C foreign function interface"),
    "compile(": ("medium", "Dynamic code compilation"),
}

SEVERITY_SCORES = {"critical": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0}

def scan_file(path: Path | str, *, label: str = "code") -> list[Signal]:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return scan_text(text, label=label, filename=path.name)

def scan_text(text: str, *, label: str = "code", filename: str = "source") -> list[Signal]:
    lines = text.splitlines()
    signals = []
    for pattern, (severity, description) in RISK_PATTERNS.items():
        hits = [(i + 1, line.strip()) for i, line in enumerate(lines) if pattern in line and not line.strip().startswith("#")]
        if hits:
            signals.append(Signal(
                "code_risk", "inspect", label, pattern,
                SEVERITY_SCORES.get(severity, 1.0),
                {"severity": severity, "description": description,
                 "hit_count": len(hits), "first_line": hits[0][0],
                 "preview": hits[0][1][:100], "filename": filename},
            ))
    signals.append(Signal("code_lines", "inspect", label, filename, float(len(lines)), {}))
    return signals

def scan_directory(path: Path | str, *, label: str = "code", extensions: tuple[str, ...] = (".py",)) -> list[Signal]:
    root = Path(path)
    if not root.is_dir():
        return []
    signals = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix in extensions:
            signals.extend(scan_file(f, label=label))
    return signals
