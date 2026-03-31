"""Ledger functions — scan, classify, and track experiment records."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
from ..signal import Signal, SignalStore

DATE_SUFFIX_RE = re.compile(r"_\d{4}-\d{2}-\d{2}$")
SEED_RE = re.compile(r"_seed(\d+)")


def scan_directory(
    root: Path | str,
    *,
    store: SignalStore | None = None,
    model_label: str = "ledger",
) -> dict[str, Any]:
    """Walk a directory of JSON files, classify and count records."""
    root = Path(root)
    records: list[dict[str, Any]] = []
    by_kind: dict[str, int] = {"bridge": 0, "full_eval": 0, "study": 0, "unknown": 0}
    skipped = 0

    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue
        if not isinstance(data, dict):
            skipped += 1
            continue
        kind = _classify_record(data)
        by_kind[kind] = by_kind.get(kind, 0) + 1
        run_id = _infer_run_id(path)
        records.append({"path": str(path), "kind": kind, "run_id": run_id})

    summary = {"total": len(records), "by_kind": by_kind, "skipped": skipped}

    if store is not None:
        store.add(Signal("scan_total", "bundle", model_label, "scan", float(len(records)), summary))
        for kind, count in by_kind.items():
            store.add(Signal("scan_kind_count", "bundle", model_label, kind, float(count), {}))

    return {"records": records, "summary": summary}


def extract_lineage(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract parent-child lineage from loaded_state_path/saved_state_path fields."""
    edges = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        parent = rec.get("loaded_state_path") or rec.get("raw", {}).get("loaded_state_path")
        child = rec.get("saved_state_path") or rec.get("raw", {}).get("saved_state_path")
        if parent and child:
            edges.append({"parent": str(parent), "child": str(child)})
    return edges


def _classify_record(data: dict[str, Any]) -> str:
    if "eval_bpb" in data or "eval_tokens" in data:
        return "full_eval"
    if "model" in data and isinstance(data["model"], dict):
        model = data["model"]
        if "test_bpb" in model or "bpb" in model:
            return "bridge"
    if "bpb" in data or "test_bpb" in data:
        return "bridge"
    if "variants" in data or "models" in data:
        return "study"
    return "unknown"


def _infer_run_id(path: Path) -> str:
    stem = path.stem
    stem = DATE_SUFFIX_RE.sub("", stem)
    return stem
