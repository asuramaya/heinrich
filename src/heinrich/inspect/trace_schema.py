"""Trace schema parsing for sample-level audit traces."""
from __future__ import annotations

from typing import Any

import numpy as np


TRACE_FIELDS = (
    "gold_logprobs",
    "loss_nats",
    "weights",
    "counted",
    "path_ids",
    "state_hash_before",
    "state_hash_after",
)

NUMERIC_TRACE_FIELDS = {"gold_logprobs", "loss_nats", "weights"}
BOOL_TRACE_FIELDS = {"counted"}
OBJECT_TRACE_FIELDS = {"path_ids", "state_hash_before", "state_hash_after"}


def parse_sample_trace(outputs: dict[str, Any], positions: list[int]) -> dict[str, Any]:
    trace: dict[int, dict[str, Any]] = {int(pos): {} for pos in positions}
    present_fields: set[str] = set()

    if "sample_gold_logprobs" in outputs:
        gold = _coerce_trace_array(outputs["sample_gold_logprobs"], len(positions), "sample_gold_logprobs", kind="numeric")
        present_fields.add("gold_logprobs")
        for idx, pos in enumerate(positions):
            trace[int(pos)]["gold_logprobs"] = float(gold[idx])

    raw_trace = outputs.get("sample_trace")
    if raw_trace is not None:
        if not isinstance(raw_trace, dict):
            raise ValueError("sample_trace must be a dict when provided")
        for field in TRACE_FIELDS:
            if field not in raw_trace:
                continue
            kind = _field_kind(field)
            values = _coerce_trace_array(raw_trace[field], len(positions), f"sample_trace.{field}", kind=kind)
            present_fields.add(field)
            for idx, pos in enumerate(positions):
                trace[int(pos)][field] = _coerce_scalar(values[idx], kind=kind)

    return {"by_position": trace, "present_fields": sorted(present_fields)}


def _field_kind(field: str) -> str:
    if field in NUMERIC_TRACE_FIELDS:
        return "numeric"
    if field in BOOL_TRACE_FIELDS:
        return "bool"
    if field in OBJECT_TRACE_FIELDS:
        return "object"
    raise ValueError(f"Unknown trace field: {field}")


def _coerce_trace_array(value: Any, length: int, label: str, *, kind: str) -> np.ndarray:
    if kind == "numeric":
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    elif kind == "bool":
        arr = np.asarray(value, dtype=bool).reshape(-1)
    elif kind == "object":
        arr = np.asarray(value, dtype=object).reshape(-1)
    else:
        raise ValueError(f"Unknown trace array kind: {kind}")
    if arr.shape[0] != length:
        raise ValueError(f"{label} length must match the number of requested sample_positions")
    return arr


def _coerce_scalar(value: Any, *, kind: str) -> Any:
    if kind == "numeric":
        return float(value)
    if kind == "bool":
        return bool(value)
    return value.item() if hasattr(value, "item") else value
