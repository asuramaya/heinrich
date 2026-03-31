"""Ledger functions — scan, classify, and track experiment records."""
from __future__ import annotations
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
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
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"bridge": None, "full": {}})
    for record in records:
        rid = record.get("run_id", "")
        if record.get("kind") == "bridge":
            grouped[rid]["bridge"] = record
        elif record.get("kind") == "full_eval":
            grouped[rid]["full"][record.get("quant_label") or "unknown"] = record

    rows: list[dict[str, Any]] = []
    for run_id, group in sorted(grouped.items()):
        bridge = group["bridge"]
        full = group["full"]
        if bridge is None:
            continue
        bridge_bpb = bridge.get("bpb")
        bridge_int6 = bridge.get("int6_bpb")
        full_fp16 = full.get("fp16", {}).get("bpb") if "fp16" in full else None
        full_int6 = full.get("int6", {}).get("bpb") if "int6" in full else None
        # Generic full_bpb: prefer fp16, then first available quant bpb
        full_bpb = full_fp16
        if full_bpb is None and full:
            for _v in full.values():
                _b = _v.get("bpb")
                if _b is not None:
                    full_bpb = _b
                    break
        status = "bridge_only"
        if full:
            if any(v.get("bpb") is None for v in full.values()):
                status = "full_eval_failed"
            else:
                status = "survived"
        rows.append(
            {
                "run_id": run_id,
                "family_id": infer_family_id(run_id),
                "seed": bridge.get("seed"),
                "bridge_fp16": bridge_bpb,
                "bridge_int6": bridge_int6,
                "full_fp16": full_fp16,
                "full_int6": full_int6,
                "delta_fp16": None if bridge_bpb is None or full_fp16 is None else full_fp16 - bridge_bpb,
                "delta_int6": None if bridge_int6 is None or full_int6 is None else full_int6 - bridge_int6,
                "delta_bpb": None if bridge_bpb is None or full_bpb is None else full_bpb - bridge_bpb,
                "status": status,
                "bridge_path": bridge.get("path"),
                "full_paths": {k: v.get("path") for k, v in full.items()},
            }
        )
    return rows


def infer_claim_level(claim: Any, metrics: Any, audits: Any) -> dict[str, Any]:
    """Determine claim level (0-5) from claim, metrics, and audits."""
    level = 0
    if claim not in (None, "", {}, []) or metrics not in (None, "", {}, []):
        level = 1
    if _dict_has_payload(metrics, ("fresh_process_full", "fresh_process_replay", "held_out_replay")):
        level = max(level, 2)
    if _dict_has_payload(metrics, ("packed_artifact_full", "packed_artifact_replay", "packed_replay")):
        level = max(level, 3)
    if _audit_status(audits, "tier2") == "pass":
        level = max(level, 4)
    tier3_credit = _tier3_claim_credit(audits)
    if tier3_credit["credited"]:
        level = max(level, 5)
    label = CLAIM_LEVELS.get(level, "Unknown")
    if level == 5 and tier3_credit.get("trust_achieved") in TIER3_PROMOTION_TRUST_LEVELS:
        label = f"{label} ({tier3_credit['trust_achieved']})"
    return {
        "level": level,
        "label": label,
        "tier3_credit": tier3_credit,
        "notes": tier3_credit.get("notes", []),
    }


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


# ---------------------------------------------------------------------------
# Extended ledger functions (ported from conker-ledger)
# ---------------------------------------------------------------------------

TIER3_PROMOTION_TRUST_LEVELS = {"traced", "strict"}
LIMITED_TIER3_SCOPES = {"prefix-only", "one_shot_runtime_handoff"}


def _json_default(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps_json(value: Any) -> str:
    """Serialize value to indented JSON, converting NaN/Inf to strings."""
    return json.dumps(value, indent=2, default=_json_default)


def load_json(path: Path) -> Any:
    """Load a JSON file from path."""
    return json.loads(path.read_text(encoding="utf-8"))


def finite_or_none(value: Any) -> float | None:
    """Return float if finite, else None."""
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    return None


def infer_run_id_from_stem(stem: str) -> str:
    """Strip full-eval and date suffixes from a file stem."""
    stem = FULL_EVAL_SUFFIX_RE.sub("", stem)
    stem = DATE_SUFFIX_RE.sub("", stem)
    return stem


def parse_bridge_record(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Parse a bridge (training-run) JSON record into a normalized dict."""
    model = data.get("model", {}) if isinstance(data.get("model"), dict) else {}
    quant_rows = data.get("quantization", [])
    saved_state_path = model.get("saved_state_path")
    loaded_state_path = model.get("loaded_state_path")
    run_id = (
        infer_run_id_from_stem(Path(saved_state_path).stem)
        if saved_state_path
        else infer_run_id_from_stem(path.stem)
    )
    seed_match = SEED_RE.search(run_id)
    quant_by_bits: dict[str, float | None] = {}
    for row in quant_rows if isinstance(quant_rows, list) else []:
        bits = row.get("bits")
        key = f"int{int(bits)}" if isinstance(bits, (int, float)) else None
        if key:
            quant_by_bits[key] = finite_or_none(row.get("test_bpb"))
    return {
        "kind": "bridge",
        "path": str(path),
        "title": data.get("title"),
        "run_id": run_id,
        "family_id": infer_family_id(run_id),
        "seed": int(seed_match.group(1)) if seed_match else model.get("seed"),
        "bpb": finite_or_none(model.get("test_bpb")),
        "bits_per_token": finite_or_none(model.get("test_bits_per_token")),
        "loss": finite_or_none(model.get("test_eval_loss")),
        "train_time_sec": finite_or_none(model.get("train_time_sec")),
        "params": model.get("params"),
        "saved_state_path": saved_state_path,
        "loaded_state_path": loaded_state_path,
        "int4_bpb": quant_by_bits.get("int4"),
        "int6_bpb": quant_by_bits.get("int6"),
        "raw": {
            "preset": model.get("preset"),
            "variant": model.get("variant"),
            "scale": model.get("scale"),
            "learning_rate": model.get("learning_rate"),
        },
    }


def parse_full_eval_record(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Parse a full-evaluation JSON record into a normalized dict."""
    state_npz = data.get("state_npz")
    run_id = (
        infer_run_id_from_stem(Path(state_npz).stem)
        if isinstance(state_npz, str)
        else infer_run_id_from_stem(path.stem)
    )
    seed_match = SEED_RE.search(run_id)
    quant_bits = int(data.get("quant_bits", 0) or 0)
    quant_label = "fp16" if quant_bits == 0 else f"int{quant_bits}"
    artifact_bytes = data.get("artifact_bytes_zlib")
    return {
        "kind": "full_eval",
        "path": str(path),
        "title": data.get("title"),
        "run_id": run_id,
        "family_id": infer_family_id(run_id),
        "seed": int(seed_match.group(1)) if seed_match else None,
        "quant_label": quant_label,
        "quant_bits": quant_bits,
        "bpb": finite_or_none(data.get("eval_bpb")),
        "bits_per_token": finite_or_none(data.get("eval_bits_per_token")),
        "loss": finite_or_none(data.get("eval_loss")),
        "eval_tokens": data.get("eval_tokens"),
        "artifact_bytes": (
            int(artifact_bytes)
            if isinstance(artifact_bytes, (int, float)) and math.isfinite(float(artifact_bytes))
            else None
        ),
        "state_npz": state_npz,
        "summary_json": data.get("summary_json"),
    }


def parse_study_record(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Parse a hyperparameter-study JSON record into a normalized dict."""
    variants = data.get("variants", [])
    models = data.get("models", [])
    best_label = None
    best_metric = None
    metric_name = None
    if isinstance(models, list):
        ranked: list[tuple[float, str]] = []
        for model in models:
            if not isinstance(model, dict):
                continue
            label = model.get("label")
            test_mean = finite_or_none(model.get("test_mean"))
            if label and test_mean is not None:
                ranked.append((test_mean, str(label)))
        if ranked:
            ranked.sort()
            best_metric, best_label = ranked[0]
            metric_name = "test_mean"
    return {
        "kind": "study",
        "path": str(path),
        "title": data.get("title"),
        "run_id": infer_run_id_from_stem(path.stem),
        "family_id": infer_family_id(infer_run_id_from_stem(path.stem)),
        "variant_count": (
            len(variants) if isinstance(variants, list) else len(models) if isinstance(models, list) else 0
        ),
        "best_label": best_label,
        "best_metric": best_metric,
        "metric_name": metric_name,
    }


def classify_record(path: Path, data: Any) -> dict[str, Any] | None:
    """Classify and parse a JSON record; returns None for unrecognized records."""
    if not isinstance(data, dict):
        return None
    if "eval_bpb" in data:
        return parse_full_eval_record(path, data)
    model = data.get("model")
    if isinstance(model, dict) and "test_bpb" in model:
        return parse_bridge_record(path, data)
    if "variants" in data or "models" in data:
        return parse_study_record(path, data)
    return None


def scan_results(root: Path) -> dict[str, Any]:
    """Scan a directory for JSON records, classify them and return a summary."""
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = load_json(path)
            record = classify_record(path, data)
        except Exception as exc:  # pragma: no cover - defensive scan path
            skipped.append(f"{path.name}: {exc}")
            continue
        if record is None:
            skipped.append(path.name)
            continue
        records.append(record)

    by_kind = Counter(record["kind"] for record in records)
    by_family = Counter(record["family_id"] for record in records)
    return {
        "root": str(root),
        "record_count": len(records),
        "by_kind": dict(by_kind),
        "family_count": len(by_family),
        "top_families": by_family.most_common(20),
        "records": records,
        "skipped": skipped,
    }


def sort_records(
    records: list[dict[str, Any]], metric: str, *, ascending: bool = True
) -> list[dict[str, Any]]:
    """Sort records by a numeric metric, placing None values last."""

    def key_fn(record: dict[str, Any]) -> tuple[int, float]:
        value = record.get(metric)
        if value is None:
            return (1, float("inf"))
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, float("inf"))

    return sorted(records, key=key_fn, reverse=not ascending)


def lineage_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract parent-child lineage edges from bridge records."""
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["kind"] != "bridge":
            continue
        loaded = record.get("loaded_state_path")
        saved = record.get("saved_state_path")
        if not loaded or not saved:
            continue
        parent_id = infer_run_id_from_stem(Path(loaded).stem)
        child_id = infer_run_id_from_stem(Path(saved).stem)
        rows.append(
            {
                "parent_run_id": parent_id,
                "child_run_id": child_id,
                "family_id": record["family_id"],
                "seed": record.get("seed"),
                "child_bpb": record.get("bpb"),
                "child_path": record.get("path"),
            }
        )
    return rows


def render_table(rows: list[dict[str, Any]], columns: list[str], top: int | None = None) -> str:
    """Render a list of dicts as a fixed-width ASCII table."""
    if top is not None:
        rows = rows[:top]
    if not rows:
        return "(no rows)"
    widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in columns}
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    sep = "  ".join("-" * widths[col] for col in columns)
    body = [
        "  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write records to a CSV file with the specified column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


# ---------------------------------------------------------------------------
# Validity bundle helpers
# ---------------------------------------------------------------------------

def _resolve_input_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_output_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Bundle attachment destination must stay inside the bundle: {value}")
    return path


def _load_manifest_value(value: Any, base_dir: Path) -> Any:
    if value is None:
        return {}
    if isinstance(value, str):
        return load_json(_resolve_input_path(base_dir, value))
    return value


def _dict_has_payload(data: Any, keys: tuple[str, ...]) -> bool:
    if not isinstance(data, dict):
        return False
    for key in keys:
        if key in data and data[key] not in (None, "", {}, []):
            return True
    return False


def _audit_status(audits: Any, tier: str) -> str | None:
    if not isinstance(audits, dict):
        return None
    tier_data = audits.get(tier)
    if not isinstance(tier_data, dict):
        return None
    status = tier_data.get("status")
    return str(status) if status is not None else None


def _tier3_claim_credit(audits: Any) -> dict[str, Any]:
    if not isinstance(audits, dict):
        return {"considered": False, "credited": False, "notes": []}
    tier3 = audits.get("tier3")
    if not isinstance(tier3, dict):
        return {"considered": False, "credited": False, "notes": []}
    status = str(tier3.get("status")) if tier3.get("status") is not None else None
    scope = tier3.get("scope")
    trust_achieved = tier3.get("trust_level_achieved")
    trust_satisfied = tier3.get("trust_satisfied")
    notes: list[str] = []
    if status != "pass":
        return {
            "considered": True,
            "credited": False,
            "status": status,
            "scope": scope,
            "trust_achieved": trust_achieved,
            "trust_satisfied": trust_satisfied,
            "notes": notes,
        }
    credited = True
    if scope in LIMITED_TIER3_SCOPES:
        credited = False
        notes.append(f"Tier 3 was not promoted because its scope was limited: `{scope}`.")
    if trust_achieved is not None and str(trust_achieved) not in TIER3_PROMOTION_TRUST_LEVELS:
        credited = False
        notes.append(
            "Tier 3 was not promoted because the achieved legality trust level was "
            f"`{trust_achieved}`, below the promotion floor of `traced`."
        )
    return {
        "considered": True,
        "credited": credited,
        "status": status,
        "scope": scope,
        "trust_achieved": trust_achieved,
        "trust_satisfied": trust_satisfied,
        "notes": notes,
    }


def _tier3_detail_lines(audits: Any) -> list[str]:
    if not isinstance(audits, dict):
        return []
    tier3 = audits.get("tier3")
    if not isinstance(tier3, dict):
        return []
    lines: list[str] = []
    scope = tier3.get("scope")
    if scope not in (None, "", [], {}):
        lines.append(f"- tier3 scope: `{scope}`")
    trust_requested = tier3.get("trust_level_requested")
    trust_achieved = tier3.get("trust_level_achieved")
    trust_satisfied = tier3.get("trust_satisfied")
    if trust_requested not in (None, "", [], {}):
        lines.append(
            "- tier3 trust: "
            f"requested=`{trust_requested}`, "
            f"achieved=`{trust_achieved}`, "
            f"satisfied=`{trust_satisfied}`"
        )
    return lines


def _copy_attachment(base_dir: Path, out_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    source_value = spec.get("source") or spec.get("path")
    if not isinstance(source_value, str):
        raise ValueError("Each attachment needs a string source/path")
    source = _resolve_input_path(base_dir, source_value)
    if not source.exists():
        raise FileNotFoundError(source)
    dest_rel = _resolve_output_path(spec.get("dest") or f"artifacts/{source.name}")
    dest = out_dir / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    resolved_dest = dest.resolve()
    resolved_out = out_dir.resolve()
    if not resolved_dest.is_relative_to(resolved_out):
        raise ValueError(f"Resolved attachment destination escapes bundle: {dest_rel} -> {resolved_dest}")
    if source.is_dir():
        shutil.copytree(source, dest, dirs_exist_ok=True)
        kind = "directory"
    else:
        shutil.copy2(source, dest)
        kind = "file"
    return {"source": str(source), "dest": str(dest_rel), "kind": kind}


def _flatten_legality_checks(checks: dict[str, Any]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for key, value in checks.items():
        if isinstance(value, dict):
            covered = value.get("covered")
            passed = value.get("pass")
            if covered is False:
                rows[key] = "uncovered"
            elif passed is True:
                rows[key] = "pass"
            elif passed is False:
                rows[key] = "fail"
            else:
                rows[key] = "unknown"
        else:
            rows[key] = str(value)
    return rows


def _flatten_legality_obligations(obligations: dict[str, Any]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for key, value in obligations.items():
        if isinstance(value, dict):
            rows[key] = str(value.get("status", "unknown"))
        else:
            rows[key] = str(value)
    return rows


def _flatten_generic_checks(checks: dict[str, Any]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for key, value in checks.items():
        if isinstance(value, dict):
            passed = value.get("pass")
            if passed is True:
                rows[key] = "pass"
            elif passed is False:
                rows[key] = "fail"
            else:
                rows[key] = "unknown"
        else:
            rows[key] = str(value)
    return rows


def _collect_detector_attachment_summaries(
    out_dir: Path,
    attachments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in attachments:
        dest = spec.get("dest")
        if not isinstance(dest, str) or not dest.endswith(".json"):
            continue
        path = out_dir / dest
        if not path.is_file():
            continue
        try:
            data = load_json(path)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        checks = data.get("checks")
        obligations = data.get("obligations")
        profile = data.get("profile")
        if isinstance(checks, dict) and isinstance(obligations, dict) and profile is not None:
            trust = data.get("trust", {}) if isinstance(data.get("trust"), dict) else {}
            rows.append(
                {
                    "kind": "legality",
                    "dest": dest,
                    "profile": str(profile),
                    "trust_requested": trust.get("requested"),
                    "trust_achieved": trust.get("achieved"),
                    "trust_satisfied": trust.get("satisfied"),
                    "checks": _flatten_legality_checks(checks),
                    "obligations": _flatten_legality_obligations(obligations),
                }
            )
            continue
        if isinstance(checks, dict) and "submission" in data and "verdict" in data:
            rows.append(
                {
                    "kind": "submission",
                    "dest": dest,
                    "verdict": str(data.get("verdict")),
                    "checks": _flatten_generic_checks(checks),
                }
            )
            continue
        if isinstance(checks, dict) and "provenance" in data and "verdict" in data:
            provenance_data = data.get("provenance", {})
            rows.append(
                {
                    "kind": "provenance",
                    "dest": dest,
                    "verdict": str(data.get("verdict")),
                    "submitted_run_id": provenance_data.get("submitted_run_id"),
                    "selection_mode": provenance_data.get("selection_mode"),
                    "checks": _flatten_generic_checks(checks),
                }
            )
            continue
        aggregate = data.get("aggregate")
        repeatability = data.get("repeatability")
        if profile is not None and isinstance(aggregate, dict) and isinstance(repeatability, dict):
            rows.append(
                {
                    "kind": "replay",
                    "dest": dest,
                    "profile": str(profile),
                    "mean_bpb": aggregate.get("mean_bpb"),
                    "repeatability": (
                        "pass"
                        if repeatability.get("pass") is True
                        else "fail"
                        if repeatability.get("pass") is False
                        else "unknown"
                    ),
                }
            )
    return rows


def render_validity_bundle_readme(
    *,
    bundle_id: str,
    claim: Any,
    metrics: Any,
    provenance: Any,
    audits: Any,
    claim_level: dict[str, Any],
    attachments: list[dict[str, Any]],
    detector_summaries: list[dict[str, Any]],
) -> str:
    """Render a human-readable README for a validity bundle."""
    requested_label = None
    if isinstance(claim, dict):
        requested_label = claim.get("requested_label") or claim.get("requested_claim")
    bridge_bpb = (
        metrics.get("bridge", {}).get("bpb")
        if isinstance(metrics, dict) and isinstance(metrics.get("bridge"), dict)
        else None
    )
    fresh_bpb = (
        metrics.get("fresh_process_full", {}).get("bpb")
        if isinstance(metrics, dict) and isinstance(metrics.get("fresh_process_full"), dict)
        else None
    )
    packed_bpb = (
        metrics.get("packed_artifact_full", {}).get("bpb")
        if isinstance(metrics, dict) and isinstance(metrics.get("packed_artifact_full"), dict)
        else None
    )
    provenance_rows: list[str] = []
    if isinstance(provenance, dict):
        for key in ("run_id", "family_id", "submission_pr", "source_repo", "source_root", "report_dir", "source_commit"):
            value = provenance.get(key)
            if value not in (None, "", [], {}):
                provenance_rows.append(f"- {key}: `{value}`")
    tier_lines = []
    for tier in ("tier1", "tier2", "tier3"):
        status = _audit_status(audits, tier) or "missing"
        tier_lines.append(f"- {tier}: `{status}`")
    tier3_details = _tier3_detail_lines(audits)
    attachment_lines = [f"- `{row['dest']}` <= `{row['source']}`" for row in attachments] or ["- none"]
    detector_lines: list[str] = []
    for row in detector_summaries:
        kind = row.get("kind", "detector")
        detector_lines.append(f"- `{row['dest']}` kind=`{kind}`")
        if kind == "legality":
            checks = ", ".join(f"{key}={value}" for key, value in row["checks"].items()) or "none"
            obligations = ", ".join(f"{key}={value}" for key, value in row["obligations"].items()) or "none"
            detector_lines.append(f"  profile: `{row['profile']}`")
            if row.get("trust_requested") is not None:
                detector_lines.append(
                    "  trust: "
                    f"requested=`{row.get('trust_requested')}`, "
                    f"achieved=`{row.get('trust_achieved')}`, "
                    f"satisfied=`{row.get('trust_satisfied')}`"
                )
            detector_lines.append(f"  checks: {checks}")
            detector_lines.append(f"  obligations: {obligations}")
        elif kind == "submission":
            checks = ", ".join(f"{key}={value}" for key, value in row["checks"].items()) or "none"
            detector_lines.append(f"  verdict: `{row['verdict']}`")
            detector_lines.append(f"  checks: {checks}")
        elif kind == "provenance":
            checks = ", ".join(f"{key}={value}" for key, value in row["checks"].items()) or "none"
            detector_lines.append(f"  verdict: `{row['verdict']}`")
            detector_lines.append(
                f"  selection: submitted_run_id=`{row.get('submitted_run_id')}`, selection_mode=`{row.get('selection_mode')}`"
            )
            detector_lines.append(f"  checks: {checks}")
        elif kind == "replay":
            repeatability = row.get("repeatability", "unknown")
            mean_bpb = row.get("mean_bpb")
            detector_lines.append(f"  profile: `{row['profile']}`")
            detector_lines.append(f"  mean_bpb: `{mean_bpb}`")
            detector_lines.append(f"  repeatability: `{repeatability}`")
    metric_lines = []
    if bridge_bpb is not None:
        metric_lines.append(f"- bridge bpb: `{bridge_bpb}`")
    if fresh_bpb is not None:
        metric_lines.append(f"- fresh-process full bpb: `{fresh_bpb}`")
    if packed_bpb is not None:
        metric_lines.append(f"- packed-artifact full bpb: `{packed_bpb}`")
    if not metric_lines:
        if isinstance(metrics, dict) and metrics:
            for key, value in list(metrics.items())[:8]:
                if isinstance(value, dict):
                    fields = []
                    for subkey, subvalue in value.items():
                        if subvalue in (None, "", [], {}):
                            continue
                        fields.append(f"{subkey}={subvalue}")
                        if len(fields) == 3:
                            break
                    metric_lines.append(f"- {key}: " + (", ".join(fields) if fields else "object"))
                else:
                    metric_lines.append(f"- {key}: `{value}`")
        else:
            metric_lines.append("- no structured metric summary provided")
    lines = [
        "# Validity Bundle",
        "",
        f"- bundle id: `{bundle_id}`",
        f"- strongest justified claim: `Tier {claim_level['level']}: {claim_level['label']}`",
    ]
    for note in claim_level.get("notes", []):
        lines.append(f"- claim note: {note}")
    if requested_label:
        lines.append(f"- requested label: `{requested_label}`")
    lines.extend(
        [
            "",
            "## Audit Coverage",
            "",
            *tier_lines,
            *tier3_details,
            "",
            "## Metrics",
            "",
            *metric_lines,
            "",
            "## Provenance",
            "",
            *(provenance_rows or ["- no provenance summary provided"]),
            "",
            "## Attachments",
            "",
            *attachment_lines,
            "",
            "## Detector Summaries",
            "",
            *(detector_lines or ["- no detector JSON attachments summarized"]),
            "",
            "## Files",
            "",
            "- `claim.json`",
            "- `evidence/metrics.json`",
            "- `evidence/provenance.json`",
            "- `evidence/audits.json`",
            "- `bundle_manifest.json`",
            "- `report/README.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_validity_bundle(manifest_path: Path, out_dir: Path) -> dict[str, Any]:
    """Assemble a validity bundle from a manifest file into out_dir."""
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Bundle manifest must be a JSON object")
    base_dir = manifest_path.parent.resolve()

    claim = _load_manifest_value(manifest.get("claim"), base_dir)
    metrics = _load_manifest_value(manifest.get("metrics"), base_dir)
    provenance = _load_manifest_value(manifest.get("provenance"), base_dir)
    audits = _load_manifest_value(manifest.get("audits"), base_dir)

    bundle_id = (
        manifest.get("bundle_id")
        or (claim.get("candidate_id") if isinstance(claim, dict) else None)
        or manifest_path.stem
    )
    claim_level = infer_claim_level(claim, metrics, audits)

    out_dir.mkdir(parents=True, exist_ok=True)
    attachments = [
        _copy_attachment(base_dir, out_dir, spec)
        for spec in manifest.get("attachments", [])
    ]
    detector_summaries = _collect_detector_attachment_summaries(out_dir, attachments)

    normalized_manifest = {
        "bundle_id": bundle_id,
        "claim": claim,
        "metrics": metrics,
        "provenance": provenance,
        "audits": audits,
        "attachments": attachments,
        "source_manifest": str(manifest_path.resolve()),
        "claim_level": claim_level,
    }

    (out_dir / "claim.json").write_text(dumps_json(claim) + "\n", encoding="utf-8")
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "metrics.json").write_text(dumps_json(metrics) + "\n", encoding="utf-8")
    (evidence_dir / "provenance.json").write_text(dumps_json(provenance) + "\n", encoding="utf-8")
    (evidence_dir / "audits.json").write_text(dumps_json(audits) + "\n", encoding="utf-8")
    (out_dir / "bundle_manifest.json").write_text(dumps_json(normalized_manifest) + "\n", encoding="utf-8")
    (out_dir / "report").mkdir(parents=True, exist_ok=True)
    (out_dir / "report" / "README.md").write_text(
        render_validity_bundle_readme(
            bundle_id=str(bundle_id),
            claim=claim,
            metrics=metrics,
            provenance=provenance,
            audits=audits,
            claim_level=claim_level,
            attachments=attachments,
            detector_summaries=detector_summaries,
        ),
        encoding="utf-8",
    )

    return {
        "bundle_id": str(bundle_id),
        "claim_level": claim_level,
        "attachment_count": len(attachments),
        "detector_attachment_count": len(detector_summaries),
        "legality_attachment_count": sum(1 for row in detector_summaries if row.get("kind") == "legality"),
        "out_dir": str(out_dir),
    }


def write_report_bundle(root: Path, out_dir: Path, *, top: int = 20) -> dict[str, Any]:
    """Scan root directory and write a full report bundle to out_dir."""
    from .viz import (
        write_bar_svg,
        write_grouped_bar_svg,
        write_histogram_svg,
        write_pie_svg,
        write_scatter_svg,
        render_lineage_mermaid,
        render_survival_mermaid,
    )

    scanned = scan_results(root)
    records = scanned["records"]
    top_full_eval = sort_records([r for r in records if r["kind"] == "full_eval"], "bpb")[:top]
    top_bridge = sort_records([r for r in records if r["kind"] == "bridge"], "bpb")[:top]
    top_study = sort_records([r for r in records if r["kind"] == "study"], "best_metric")[:top]
    surv_rows = survival_rows(records)
    survival_non_bridge = [row for row in surv_rows if row["status"] != "bridge_only"]
    failed = [row for row in surv_rows if row["status"] == "full_eval_failed"]
    lin_rows = lineage_rows(records)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scan_summary.json").write_text(
        dumps_json(
            {
                "root": scanned["root"],
                "record_count": scanned["record_count"],
                "by_kind": scanned["by_kind"],
                "family_count": scanned["family_count"],
                "top_families": scanned["top_families"],
                "skipped": scanned["skipped"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "top_full_eval.json").write_text(dumps_json(top_full_eval) + "\n", encoding="utf-8")
    (out_dir / "top_bridge.json").write_text(dumps_json(top_bridge) + "\n", encoding="utf-8")
    (out_dir / "top_study.json").write_text(dumps_json(top_study) + "\n", encoding="utf-8")
    (out_dir / "survival.json").write_text(dumps_json(survival_non_bridge) + "\n", encoding="utf-8")
    (out_dir / "failed_full_eval.json").write_text(dumps_json(failed) + "\n", encoding="utf-8")
    (out_dir / "lineage.json").write_text(dumps_json(lin_rows) + "\n", encoding="utf-8")

    write_csv(
        out_dir / "top_full_eval.csv",
        top_full_eval,
        ["family_id", "run_id", "seed", "quant_label", "bpb", "artifact_bytes", "path"],
    )
    write_csv(
        out_dir / "top_study.csv",
        top_study,
        ["family_id", "run_id", "best_label", "best_metric", "metric_name", "variant_count", "path"],
    )
    write_csv(
        out_dir / "survival.csv",
        survival_non_bridge,
        ["family_id", "run_id", "seed", "bridge_fp16", "full_fp16", "bridge_int6", "full_int6", "delta_fp16", "delta_int6", "status"],
    )
    write_csv(
        out_dir / "failed_full_eval.csv",
        failed,
        ["family_id", "run_id", "seed", "bridge_fp16", "bridge_int6", "status", "bridge_path"],
    )

    # SVG charts
    full_eval_with_bpb = [row for row in top_full_eval[:12] if row.get("bpb") is not None]
    write_bar_svg(
        out_dir / "top_full_eval.svg",
        "Top Full-Eval Rows",
        [f"{row['family_id']}:{row.get('quant_label')}" for row in full_eval_with_bpb],
        [row["bpb"] for row in full_eval_with_bpb],
    )
    study_rows_top = [row for row in top_study if row.get("best_metric") is not None][:12]
    write_bar_svg(
        out_dir / "top_study.svg",
        "Top Study Rows",
        [f"{row['family_id']}:{row.get('best_label') or 'study'}" for row in study_rows_top],
        [float(row["best_metric"]) for row in study_rows_top],
    )
    gap_rows = [
        row for row in survival_non_bridge
        if row.get("bridge_fp16") is not None and row.get("full_fp16") is not None
    ][:20]
    write_scatter_svg(
        out_dir / "bridge_vs_full_fp16.svg",
        "Bridge FP16 vs Full FP16",
        gap_rows,
        x_key="bridge_fp16",
        y_key="full_fp16",
        label_key="family_id",
        reference_line=True,
    )
    conker7_rows = [row for row in surv_rows if str(row["family_id"]).startswith("conker7_")]
    conker7_with_bpb = [row for row in conker7_rows if row.get("bridge_fp16") is not None]
    write_bar_svg(
        out_dir / "conker7_bridge_fp16.svg",
        "Conker-7 Bridge FP16 Rows",
        [row["family_id"] for row in conker7_with_bpb],
        [row["bridge_fp16"] for row in conker7_with_bpb],
    )
    survived_count = sum(1 for r in surv_rows if r["status"] == "survived_full_eval")
    failed_count = len(failed)
    bridge_only_count = sum(1 for r in surv_rows if r["status"] == "bridge_only")
    pie_labels: list[str] = []
    pie_values: list[float] = []
    pie_colors: list[str] = []
    if survived_count:
        pie_labels.append("survived_full_eval")
        pie_values.append(survived_count)
        pie_colors.append("#2ca02c")
    if failed_count:
        pie_labels.append("full_eval_failed")
        pie_values.append(failed_count)
        pie_colors.append("#c23b22")
    if bridge_only_count:
        pie_labels.append("bridge_only")
        pie_values.append(bridge_only_count)
        pie_colors.append("#7f7f7f")
    write_pie_svg(out_dir / "survival_status.svg", "Survival Status", pie_labels, pie_values, pie_colors)
    deltas = [row["delta_fp16"] for row in survival_non_bridge if row.get("delta_fp16") is not None]
    write_histogram_svg(out_dir / "delta_fp16_histogram.svg", "Bridge-to-Full Delta (FP16)", deltas)
    family_best: dict[str, dict[str, Any]] = {}
    for row in survival_non_bridge:
        fid = row["family_id"]
        if row.get("bridge_fp16") is not None and row.get("full_fp16") is not None:
            if fid not in family_best or (row["full_fp16"] < family_best[fid]["full_fp16"]):
                family_best[fid] = row
    grouped_rows = sort_records(list(family_best.values()), "full_fp16")[:12]
    write_grouped_bar_svg(
        out_dir / "bridge_vs_full_grouped.svg",
        "Bridge vs Full-Eval by Family",
        grouped_rows,
        key_a="bridge_fp16",
        key_b="full_fp16",
        label_key="family_id",
    )

    # Mermaid diagrams
    lineage_mermaid = render_lineage_mermaid(lin_rows)
    survival_mermaid = render_survival_mermaid(surv_rows)

    # README
    summary_lines = [
        "# Public Backlog Report",
        "",
        f"- root: `{root}`",
        f"- normalized records: `{scanned['record_count']}`",
        f"- bridge rows: `{scanned['by_kind'].get('bridge', 0)}`",
        f"- full eval rows: `{scanned['by_kind'].get('full_eval', 0)}`",
        f"- study rows: `{scanned['by_kind'].get('study', 0)}`",
        f"- experiment families: `{scanned['family_count']}`",
        "",
        "## Headline",
        "",
    ]
    if top_full_eval and top_full_eval[0].get("bpb") is not None:
        best = top_full_eval[0]
        summary_lines.append(
            f"- best normalized full eval in this backlog: `{best['family_id']}` `{best['quant_label']}` at `{best['bpb']:.6f} bpb`"
        )
    elif top_study and top_study[0].get("best_metric") is not None:
        best = top_study[0]
        summary_lines.append(
            f"- best study quick-check in this backlog: `{best['family_id']}` `{best.get('best_label')}` at `{best['best_metric']:.6f}` `{best.get('metric_name') or 'metric'}`"
        )
    if failed:
        summary_lines.append(f"- full-eval failures detected after optimistic bridge results: `{len(failed)}`")
    summary_lines.extend(
        [
            "",
            "## Survival Pipeline",
            "",
            "```mermaid",
            survival_mermaid,
            "```",
            "",
            "## Lineage",
            "",
            "```mermaid",
            lineage_mermaid,
            "```",
            "",
            "## Files",
            "",
            "- `scan_summary.json`",
            "- `top_full_eval.json` / `top_full_eval.csv` / `top_full_eval.svg`",
            "- `top_bridge.json`",
            "- `top_study.json` / `top_study.csv` / `top_study.svg`",
            "- `survival.json` / `survival.csv` / `survival_status.svg`",
            "- `failed_full_eval.json` / `failed_full_eval.csv`",
            "- `lineage.json`",
            "- `bridge_vs_full_fp16.svg` / `bridge_vs_full_grouped.svg`",
            "- `delta_fp16_histogram.svg`",
            "- `conker7_bridge_fp16.svg`",
            "",
            "## Visuals",
            "",
            "### Top Study Rows",
            "",
            "![Top study rows](./top_study.svg)",
            "",
            "### Survival Status",
            "",
            "![Survival status](./survival_status.svg)",
            "",
            "### Top Full-Eval Rows",
            "",
            "![Top full eval rows](./top_full_eval.svg)",
            "",
            "### Bridge vs Full-Eval FP16",
            "",
            "![Bridge vs full fp16](./bridge_vs_full_fp16.svg)",
            "",
            "### Bridge vs Full-Eval by Family",
            "",
            "![Bridge vs full grouped](./bridge_vs_full_grouped.svg)",
            "",
            "### Delta Distribution (FP16)",
            "",
            "![Delta histogram](./delta_fp16_histogram.svg)",
            "",
            "### Conker-7 Bridge Rows",
            "",
            "![Conker-7 bridge rows](./conker7_bridge_fp16.svg)",
        ]
    )
    if failed:
        summary_lines.extend(["", "## Failed Full-Eval Rows", ""])
        for row in failed[:20]:
            summary_lines.append(
                f"- `{row['family_id']}` seed `{row.get('seed')}` bridge fp16 `{row.get('bridge_fp16')}` bridge int6 `{row.get('bridge_int6')}`"
            )
    (out_dir / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "scan_summary": {
            "record_count": scanned["record_count"],
            "by_kind": scanned["by_kind"],
            "family_count": scanned["family_count"],
        },
        "best_full_eval": top_full_eval[0] if top_full_eval else None,
        "failed_full_eval_count": len(failed),
        "report_dir": str(out_dir),
    }
