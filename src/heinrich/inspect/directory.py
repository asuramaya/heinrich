"""Inspect a directory of JSON files — emit signals from their structure and contents."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from ..signal import Signal, SignalStore


def inspect_directory(
    store: SignalStore,
    path: Path | str,
    *,
    label: str = "dir",
) -> None:
    """Walk a directory, read JSON files, emit structural and content signals."""
    root = Path(path)
    if not root.is_dir():
        return

    json_files = sorted(root.rglob("*.json"))
    npy_files = sorted(root.rglob("*.npy"))
    npz_files = sorted(root.rglob("*.npz"))
    py_files = sorted(root.rglob("*.py"))
    log_files = sorted(root.rglob("*.log"))

    # File inventory signals
    store.add(Signal("dir_json_count", "inspect", label, "json_files", float(len(json_files)), {}))
    store.add(Signal("dir_npy_count", "inspect", label, "npy_files", float(len(npy_files)), {}))
    store.add(Signal("dir_npz_count", "inspect", label, "npz_files", float(len(npz_files)), {}))
    store.add(Signal("dir_py_count", "inspect", label, "py_files", float(len(py_files)), {}))
    store.add(Signal("dir_log_count", "inspect", label, "log_files", float(len(log_files)), {}))

    # File size signals for non-JSON files
    for f in npy_files + npz_files:
        size = f.stat().st_size
        store.add(Signal("file_size", "inspect", label, str(f.relative_to(root)), float(size), {}))

    # Read and emit signals from each JSON file
    for json_path in json_files:
        rel = str(json_path.relative_to(root))
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            store.add(Signal("json_parse_error", "inspect", label, rel, 0.0, {}))
            continue

        if not isinstance(data, dict):
            continue

        # Emit numeric fields as signals
        for key, value in data.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                store.add(Signal("json_field", "inspect", label, f"{rel}:{key}", float(value), {"file": rel, "key": key}))
            elif isinstance(value, str) and len(value) < 200:
                store.add(Signal("json_string", "inspect", label, f"{rel}:{key}", 0.0, {"file": rel, "key": key, "value": value}))
            elif isinstance(value, dict):
                # One level deep for nested dicts
                for subkey, subval in value.items():
                    if isinstance(subval, (int, float)) and not isinstance(subval, bool):
                        store.add(Signal("json_field", "inspect", label, f"{rel}:{key}.{subkey}", float(subval),
                                         {"file": rel, "key": f"{key}.{subkey}"}))

    # Cross-file consistency: find common fields across JSON files
    field_values: dict[str, list[tuple[str, float]]] = {}
    for json_path in json_files:
        rel = str(json_path.relative_to(root))
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    field_values.setdefault(key, []).append((rel, float(value)))

    # Emit consistency signals for fields that appear in multiple files
    for key, occurrences in field_values.items():
        if len(occurrences) < 2:
            continue
        values = [v for _, v in occurrences]
        all_same = all(v == values[0] for v in values)
        store.add(Signal(
            "cross_file_consistency", "inspect", label, key,
            1.0 if all_same else 0.0,
            {"files": [f for f, _ in occurrences], "values": values, "consistent": all_same},
        ))
