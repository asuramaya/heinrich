"""Submission format adapters and triage report generation."""
from __future__ import annotations
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any
from ..signal import Signal


def package_zip(contents: dict[str, Path | str], out_path: Path | str) -> Path:
    """Package files into a zip. contents = {archive_name: local_path}."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
        for arc_name, local_path in contents.items():
            local = Path(local_path)
            if local.is_file():
                zf.write(local, arc_name)
            elif local.is_dir():
                for f in sorted(local.rglob("*")):
                    if f.is_file():
                        zf.write(f, f"{arc_name}/{f.relative_to(local)}")
    return out


def copy_record(source: Path | str, dest: Path | str) -> Path:
    """Copy a record directory to a destination."""
    source, dest = Path(source), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, dirs_exist_ok=True)
    return dest


def generate_triage_report(signals: list[dict[str, Any]], *, title: str = "Heinrich Triage") -> str:
    """Generate a markdown triage report from signal dicts."""
    issues = [s for s in signals if s.get("kind") == "rule_check" and not s.get("metadata", {}).get("pass", True)]
    risks = [s for s in signals if s.get("kind") == "code_risk"]
    consistency = [s for s in signals if s.get("kind") == "cross_file_consistency" and not s.get("metadata", {}).get("consistent", True)]

    lines = [f"# {title}", ""]

    total_problems = len(issues) + len(risks) + len(consistency)
    if total_problems == 0:
        lines.append("All checks passed.")
        return "\n".join(lines) + "\n"

    lines.append(f"**{total_problems} problem(s) found**")
    lines.append("")

    if issues:
        lines.append("## Rule Violations")
        for s in issues:
            lines.append(f"- FAIL: `{s.get('target', '')}` ({s.get('metadata', {}).get('rule', '')})")
        lines.append("")

    if risks:
        lines.append("## Code Risks")
        for s in sorted(risks, key=lambda s: s.get("value", 0), reverse=True):
            m = s.get("metadata", {})
            lines.append(f"- [{m.get('severity', '?')}] `{s.get('target', '')}`: {m.get('description', '')} (line {m.get('first_line', '?')})")
        lines.append("")

    if consistency:
        lines.append("## Consistency Issues")
        for s in consistency:
            m = s.get("metadata", {})
            lines.append(f"- `{s.get('target', '')}`: values {m.get('values', [])} across {m.get('files', [])}")
        lines.append("")

    return "\n".join(lines) + "\n"
