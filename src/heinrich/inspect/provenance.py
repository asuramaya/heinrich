"""Provenance audit — selection disclosure, dataset fingerprints, held-out identity."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def audit_provenance(source: Path | str | dict[str, Any]) -> dict[str, Any]:
    try:
        manifest = load_provenance_manifest(source)
    except (FileNotFoundError, OSError) as exc:
        return {"error": str(exc), "verdict": "error", "findings": []}
    except Exception as exc:
        return {"error": str(exc), "verdict": "error", "findings": []}

    checks_and_rows: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {
        "selection_disclosure": _check_selection_disclosure_internal(manifest),
        "dataset_fingerprints": _check_dataset_fingerprints_internal(manifest),
        "held_out_identity": _check_held_out_identity_internal(manifest),
    }

    checks: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    for name, (summary, rows) in checks_and_rows.items():
        checks[name] = summary
        findings.extend(rows)

    alerts = [row["message"] for row in findings if row["severity"] in {"warn", "fail"}]
    selection = manifest.get("selection", {})
    datasets = manifest.get("datasets", {})
    return {
        "profile": manifest["profile"],
        "verdict": _verdict(findings),
        "provenance": {
            "submitted_run_id": selection.get("submitted_run_id"),
            "selection_mode": selection.get("selection_mode"),
            "candidate_run_count": selection.get("candidate_run_count"),
            "dataset_keys": sorted(datasets),
        },
        "checks": checks,
        "findings": findings,
        "alerts": alerts,
    }


def load_provenance_manifest(source: Path | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        manifest = dict(source)
    else:
        manifest = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Provenance manifest must decode to a JSON object")
    return {
        **manifest,
        "profile": str(manifest.get("profile", "parameter-golf")),
        "selection": dict(manifest.get("selection", {})),
        "datasets": dict(manifest.get("datasets", {})),
    }


def _normalize_flat_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Normalize a flat manifest (with top-level keys) into the nested selection/datasets format."""
    # If already nested, return as-is
    if "selection" in manifest or "datasets" in manifest:
        return manifest
    # Detect flat format: submitted_run_id / selection_mode at top level
    flat_selection_keys = {"submitted_run_id", "selection_mode", "candidate_run_count"}
    if flat_selection_keys & set(manifest):
        selection = {k: manifest[k] for k in flat_selection_keys if k in manifest}
        return {**manifest, "selection": selection}
    return manifest


def check_selection_disclosure(manifest: dict[str, Any]) -> dict[str, Any]:
    """Public API: returns summary dict."""
    summary, _ = _check_selection_disclosure_internal(_normalize_flat_manifest(manifest))
    return summary


def _check_selection_disclosure_internal(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection = manifest.get("selection", {})
    findings: list[dict[str, Any]] = []
    if not selection:
        findings.append(finding("warn", "missing_selection_manifest", "No selection manifest was provided"))
        return {"pass": False, "finding_count": len(findings)}, findings

    submitted_run_id = selection.get("submitted_run_id")
    selection_mode = selection.get("selection_mode")
    candidate_run_count = selection.get("candidate_run_count")

    if not submitted_run_id:
        findings.append(finding("warn", "missing_submitted_run_id", "Selection manifest is missing submitted_run_id"))
    if not selection_mode:
        findings.append(finding("warn", "missing_selection_mode", "Selection manifest is missing selection_mode"))

    if selection_mode == "best_of_k":
        if candidate_run_count is None:
            findings.append(finding("warn", "missing_candidate_run_count", "best_of_k selection was declared without candidate_run_count"))
        elif int(candidate_run_count) <= 1:
            findings.append(finding("warn", "inconsistent_selection_population", "best_of_k selection was declared with candidate_run_count <= 1", details={"candidate_run_count": int(candidate_run_count)}))
        else:
            findings.append(finding("warn", "selection_bias_risk", "best_of_k selection was disclosed; reported metrics may be optimistic without an unseen test set", details={"candidate_run_count": int(candidate_run_count)}))
    elif candidate_run_count is not None and int(candidate_run_count) > 1:
        findings.append(finding("warn", "selection_population_without_mode", "candidate_run_count > 1 was declared without best_of_k selection_mode", details={"candidate_run_count": int(candidate_run_count), "selection_mode": selection_mode}))

    return {"pass": not any(row["severity"] == "fail" for row in findings), "selection_mode": selection_mode, "candidate_run_count": candidate_run_count, "finding_count": len(findings)}, findings


def check_dataset_fingerprints(manifest: dict[str, Any]) -> dict[str, Any]:
    """Public API: returns summary dict."""
    summary, _ = _check_dataset_fingerprints_internal(manifest)
    return summary


def _check_dataset_fingerprints_internal(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    datasets = manifest.get("datasets", {})
    findings: list[dict[str, Any]] = []
    if not datasets:
        findings.append(finding("warn", "missing_dataset_manifest", "No dataset provenance manifest was provided"))
        return {"pass": False, "dataset_count": 0, "finding_count": len(findings)}, findings

    normalized = {name: _normalize_dataset_entry(entry) for name, entry in datasets.items()}
    for name, entry in normalized.items():
        if entry["fingerprint"] is None:
            findings.append(finding("warn", "missing_dataset_fingerprint", f"Dataset {name} is missing a fingerprint"))

    for lhs, rhs in (("train", "validation"), ("train", "held_out_test"), ("validation", "held_out_test")):
        if lhs not in normalized or rhs not in normalized:
            continue
        lhs_fp = normalized[lhs]["fingerprint"]
        rhs_fp = normalized[rhs]["fingerprint"]
        if lhs_fp is not None and rhs_fp is not None and lhs_fp == rhs_fp:
            findings.append(finding("fail", "dataset_fingerprint_overlap", f"Datasets {lhs} and {rhs} have the same fingerprint", details={"lhs": lhs, "rhs": rhs, "fingerprint": lhs_fp}))

    return {"pass": not any(row["severity"] == "fail" for row in findings), "dataset_count": len(normalized), "finding_count": len(findings)}, findings


def check_held_out_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Public API: returns summary dict."""
    summary, _ = _check_held_out_identity_internal(manifest)
    return summary


def _check_held_out_identity_internal(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    datasets = manifest.get("datasets", {})
    findings: list[dict[str, Any]] = []
    held_out = _normalize_dataset_entry(datasets.get("held_out_test"))
    if held_out["name"] is None:
        findings.append(finding("warn", "missing_held_out_identity", "Held-out test identity was not declared in the provenance manifest"))
    if held_out["name"] is not None and held_out["fingerprint"] is None:
        findings.append(finding("warn", "missing_held_out_fingerprint", "Held-out test dataset was named but is missing a fingerprint", details={"name": held_out["name"]}))
    return {"pass": not any(row["severity"] == "fail" for row in findings), "held_out_name": held_out["name"], "finding_count": len(findings)}, findings


def _normalize_dataset_entry(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"name": None, "fingerprint": None}
    return {"name": raw.get("name"), "fingerprint": raw.get("fingerprint")}


def finding(severity: str, kind: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"severity": severity, "kind": kind, "message": message}
    if details:
        row["details"] = details
    return row


def _verdict(findings: list[dict[str, Any]]) -> str:
    severities = {row["severity"] for row in findings}
    if "fail" in severities:
        return "fail"
    if "warn" in severities:
        return "warn"
    return "pass"
