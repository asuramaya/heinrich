"""Report generation — tables, CSV, and text summaries."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Any


def render_table(rows: list[dict[str, Any]], columns: list[str], *, top: int | None = None) -> str:
    """Format rows as an aligned ASCII table."""
    display = rows[:top] if top else rows
    if not display or not columns:
        return "(empty)\n"
    widths = {col: max(len(col), max((len(str(row.get(col, ""))) for row in display), default=0)) for col in columns}
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)
    lines = [header, separator]
    for row in display:
        lines.append(" | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))
    return "\n".join(lines) + "\n"


def write_csv(path: Path | str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write rows as a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
