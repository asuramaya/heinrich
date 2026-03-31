"""Ledger functions — scan, classify, and track experiment records."""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Any
from ..signal import Signal, SignalStore

DATE_SUFFIX_RE = re.compile(r"_\d{4}-\d{2}-\d{2}$")
SEED_RE = re.compile(r"_seed(\d+)")
FULL_EVAL_SUFFIX_RE = re.compile(r"_fullval_(?:train|test)_[a-z0-9]+$")

CLAIM_LEVELS = {
    0: "No justified claim yet",
    1: "Bridge metric only",
    2: "Fresh-process held-out replay confirmed",
    3: "Packed-artifact replay confirmed",
    4: "Structural audit passed",
    5: "Behavioral legality audit passed",
}


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


def parse_record(path: Path, data: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a JSON record into a typed record dict."""
    if not isinstance(data, dict):
        return None
    if "eval_bpb" in data or "eval_tokens" in data:
        return _parse_full_eval(path, data)
    if isinstance(data.get("model"), dict):
        model = data["model"]
        if "test_bpb" in model or "bpb" in model:
            return _parse_bridge(path, data)
    if "bpb" in data or "test_bpb" in data:
        return _parse_bridge(path, data)
    if "variants" in data or "models" in data:
        return _parse_study(path, data)
    return None


def _parse_bridge(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    model = data.get("model", {}) if isinstance(data.get("model"), dict) else {}
    run_id = _infer_run_id(path)
    family_id = infer_family_id(run_id)
    seed = _extract_seed(run_id)
    bpb = _finite(model.get("test_bpb") or data.get("test_bpb") or data.get("bpb"))
    return {
        "kind": "bridge", "path": str(path), "run_id": run_id, "family_id": family_id,
        "seed": seed, "bpb": bpb,
        "int4_bpb": _finite(model.get("int4_bpb") or data.get("int4_bpb")),
        "int6_bpb": _finite(model.get("int6_bpb") or data.get("int6_bpb")),
        "params": model.get("params") or data.get("params"),
        "saved_state_path": str(model.get("saved_state_path") or data.get("saved_state_path") or ""),
        "loaded_state_path": str(model.get("loaded_state_path") or data.get("loaded_state_path") or ""),
    }


def _parse_full_eval(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    run_id = _infer_run_id(path)
    family_id = infer_family_id(run_id)
    seed = _extract_seed(run_id)
    quant_bits = int(data.get("quant_bits", 0))
    quant_label = f"int{quant_bits}" if quant_bits > 0 else "fp16"
    return {
        "kind": "full_eval", "path": str(path), "run_id": run_id, "family_id": family_id,
        "seed": seed, "quant_label": quant_label, "quant_bits": quant_bits,
        "bpb": _finite(data.get("eval_bpb") or data.get("bpb")),
        "eval_tokens": data.get("eval_tokens"),
        "artifact_bytes": data.get("artifact_bytes"),
    }


def _parse_study(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    run_id = _infer_run_id(path)
    variants = data.get("variants") or data.get("models") or []
    variant_count = len(variants) if isinstance(variants, list) else 0
    return {
        "kind": "study", "path": str(path), "run_id": run_id,
        "family_id": infer_family_id(run_id),
        "variant_count": variant_count,
    }


def infer_family_id(run_id: str) -> str:
    """Remove seed and _save suffixes to get a family identifier."""
    result = run_id
    result = SEED_RE.sub("", result)
    if result.endswith("_save"):
        result = result[:-5]
    return result


def survival_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join bridge and full_eval records by run_id to compute survival."""
    bridge_by_id: dict[str, dict[str, Any]] = {}
    full_by_id: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        rid = rec.get("run_id", "")
        if rec.get("kind") == "bridge":
            bridge_by_id[rid] = rec
        elif rec.get("kind") == "full_eval":
            full_by_id.setdefault(rid, []).append(rec)

    rows = []
    all_ids = sorted(set(bridge_by_id) | set(full_by_id))
    for rid in all_ids:
        bridge = bridge_by_id.get(rid)
        fulls = full_by_id.get(rid, [])
        bridge_bpb = bridge["bpb"] if bridge and bridge.get("bpb") is not None else None
        full_bpb = min((f["bpb"] for f in fulls if f.get("bpb") is not None), default=None)
        delta = None
        if bridge_bpb is not None and full_bpb is not None:
            delta = full_bpb - bridge_bpb
        if fulls and full_bpb is not None:
            status = "survived"
        elif bridge and not fulls:
            status = "bridge_only"
        else:
            status = "unknown"
        rows.append({
            "run_id": rid,
            "family_id": infer_family_id(rid),
            "bridge_bpb": bridge_bpb,
            "full_bpb": full_bpb,
            "delta_bpb": delta,
            "status": status,
        })
    return rows


def infer_claim_level(claim: Any, metrics: Any, audits: Any) -> dict[str, Any]:
    """Determine claim level (0-5) from claim, metrics, and audits."""
    level = 0
    notes: list[str] = []
    if isinstance(metrics, dict):
        if metrics.get("bridge_bpb") is not None or metrics.get("bpb") is not None:
            level = max(level, 1)
            notes.append("Bridge metric present")
        if metrics.get("fresh_process_bpb") is not None:
            level = max(level, 2)
            notes.append("Fresh process replay confirmed")
        if metrics.get("packed_artifact_bpb") is not None:
            level = max(level, 3)
            notes.append("Packed artifact replay confirmed")
    if isinstance(audits, dict):
        tier2 = audits.get("tier2", {})
        if isinstance(tier2, dict) and tier2.get("status") == "pass":
            level = max(level, 4)
            notes.append("Structural audit passed")
        tier3 = audits.get("tier3", {})
        if isinstance(tier3, dict) and tier3.get("status") == "pass":
            trust = str(tier3.get("trust_achieved", ""))
            if trust in {"traced", "strict"}:
                level = max(level, 5)
                notes.append(f"Behavioral legality passed (trust={trust})")
    return {"level": level, "label": CLAIM_LEVELS.get(level, "Unknown"), "notes": notes}


def _extract_seed(run_id: str) -> int | None:
    match = SEED_RE.search(run_id)
    return int(match.group(1)) if match else None


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None
