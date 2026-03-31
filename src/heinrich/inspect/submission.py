"""Submission audit — manifest loading, claim extraction, and presence/consistency checks."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTIFACT_EXTS = (".pt", ".pth", ".bin", ".npz", ".safetensors", ".ckpt", ".ptz")
PATCH_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)

NUMBER_RE = r"([0-9][0-9_,]*(?:\.[0-9]+)?)"
README_PATTERNS = {
    "val_bpb": re.compile(rf"\bval_bpb\b\s*[:=]\s*{NUMBER_RE}", re.IGNORECASE),
    "pre_quant_val_bpb": re.compile(rf"\bpre_quant_val_bpb\b\s*[:=]\s*{NUMBER_RE}", re.IGNORECASE),
    "bytes_total": re.compile(rf"\bbytes_total\b\s*[:=]\s*{NUMBER_RE}", re.IGNORECASE),
    "bytes_model_int6_zlib": re.compile(
        rf"\b(?:bytes_model_int6_zlib|artifact(?:\s+bytes)?)\b\s*[:=]\s*{NUMBER_RE}",
        re.IGNORECASE,
    ),
}
LOG_PATTERNS = README_PATTERNS

PARAMETER_GOLF_PROFILE: dict[str, Any] = {
    "required_evidence": ("readme", "submission_json"),
    "claim_fields": (
        "name",
        "track",
        "val_bpb",
        "pre_quant_val_bpb",
        "bytes_total",
        "bytes_model_int6_zlib",
    ),
    "protocol_risk_markers": (
        ("optimizer.step()", "optimizer_step"),
        (".backward(", "backward_call"),
        ("model.train()", "model_train_call"),
        ("set_grad_enabled(true)", "grad_enabled_true"),
    ),
    "protocol_reassurance_markers": (
        ("inference_mode", "inference_mode"),
        ("no_grad", "no_grad"),
        ("score-first", "score_first"),
    ),
    "data_boundary_risk_markers": (
        ("fineweb_val", "validation_path_reference"),
        ("urllib", "network_call"),
        ("requests.", "network_call"),
        ("http://", "network_call"),
        ("https://", "network_call"),
        ("wget ", "network_call"),
        ("curl ", "network_call"),
    ),
}


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def get_profile_rules(profile: str) -> dict[str, Any]:
    if profile != "parameter-golf":
        raise ValueError(f"Unknown submission profile: {profile}")
    return PARAMETER_GOLF_PROFILE


def classify_patch_context(profile: str, patch_text: str, files: list[str]) -> dict[str, Any]:
    if profile != "parameter-golf":
        raise ValueError(f"Unknown submission profile: {profile}")
    lower_patch = patch_text.lower()
    record_dirs = sorted(
        {
            path.split("/", 3)[2]
            for path in files
            if path.startswith("records/") and path.count("/") >= 3
        }
    )
    adds_submission_json = any(path.endswith("/submission.json") for path in files)
    adds_train_gpt = any(path.endswith("/train_gpt.py") or path.endswith("/train_gpt_mlx.py") for path in files)
    touches_core = any(path in {"train_gpt.py", "train_gpt_mlx.py"} for path in files)
    docs_only = bool(files) and all(
        path.endswith((".md", ".txt", ".json", ".log")) or path.startswith(".github/")
        for path in files
    )
    has_artifact = any(path.endswith(ARTIFACT_EXTS) for path in files)
    category = "other"
    if adds_submission_json and adds_train_gpt:
        category = "record_submission"
    elif touches_core:
        category = "core_code"
    elif docs_only:
        category = "docs_or_logs"
    elif record_dirs:
        category = "record_misc"
    return {
        "category": category,
        "record_dirs": record_dirs,
        "file_count": len(files),
        "adds_submission_json": adds_submission_json,
        "adds_train_gpt": adds_train_gpt,
        "touches_core": touches_core,
        "has_artifact_blob": has_artifact,
        "ttt_signal": "ttt" in lower_patch or "test-time" in lower_patch,
        "score_first_signal": "score-first" in lower_patch,
        "inference_mode_signal": "inference_mode" in lower_patch,
        "optimizer_in_eval_signal": "optimizer.step()" in patch_text
        or (".step()" in patch_text and "eval" in lower_patch and "optimizer" in lower_patch),
        "triage_ready": category in {"record_submission", "core_code", "record_misc"},
        "runtime_replay_ready": category == "record_submission" and has_artifact,
        "sample_files": files[:12],
    }


def patch_files(patch_text: str) -> list[str]:
    files: list[str] = []
    for match in PATCH_FILE_RE.finditer(patch_text):
        path = match.group(1)
        if path != "/dev/null":
            files.append(path)
    return files


# ---------------------------------------------------------------------------
# Manifest loading and claim extraction
# ---------------------------------------------------------------------------

def load_submission_manifest(source: Path | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        manifest = dict(source)
        manifest_path = None
    else:
        manifest_path = Path(source)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Submission manifest must decode to a JSON object")
    profile = str(manifest.get("profile", "parameter-golf"))
    repo_root = manifest.get("repo_root")
    if repo_root is None:
        repo_root_path = manifest_path.parent if manifest_path is not None else Path.cwd()
    else:
        repo_root_path = Path(repo_root)
        if not repo_root_path.is_absolute() and manifest_path is not None:
            repo_root_path = (manifest_path.parent / repo_root_path).resolve()
    submission_root = manifest.get("submission_root")
    if submission_root is None:
        submission_root_path = repo_root_path
    else:
        submission_root_path = Path(submission_root)
        if not submission_root_path.is_absolute():
            submission_root_path = (repo_root_path / submission_root_path).resolve()
    evidence = dict(manifest.get("evidence", {}))
    return {
        **manifest,
        "profile": profile,
        "repo_root": repo_root_path,
        "submission_root": submission_root_path,
        "evidence": _resolve_evidence_paths(evidence, submission_root_path),
    }


def _resolve_evidence_paths(evidence: dict[str, Any], submission_root: Path) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in evidence.items():
        if isinstance(value, list):
            resolved[key] = [_resolve_path(item, submission_root) for item in value]
        elif value is None:
            resolved[key] = None
        else:
            resolved[key] = _resolve_path(value, submission_root)
    return resolved


def _resolve_path(raw: str | Path, submission_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (submission_root / path).resolve()


def extract_claims(manifest: dict[str, Any]) -> dict[str, Any]:
    evidence = manifest["evidence"]
    readme_path = evidence.get("readme")
    submission_json_path = evidence.get("submission_json")
    results_json_path = evidence.get("results_json")
    log_paths = evidence.get("logs", [])
    artifact_paths = evidence.get("artifacts", [])
    code_paths = evidence.get("code", [])
    patch_path = evidence.get("patch")
    readme_text = _read_optional_text(readme_path)
    patch_text = _read_optional_text(patch_path)
    logs = [(path, _read_optional_text(path)) for path in log_paths]
    return {
        "submission_json": _extract_json_claims(_read_optional_json(submission_json_path)),
        "results_json": _extract_json_claims(_read_optional_json(results_json_path)),
        "readme": _extract_readme_claims(readme_text),
        "logs": _extract_log_claims(logs),
        "artifact_files": _extract_artifact_claims(artifact_paths),
        "code_files": [str(path) for path in code_paths if path.exists()],
        "patch": {"path": str(patch_path), "text": patch_text} if patch_text is not None and patch_path is not None else None,
    }


def _read_optional_text(path: Path | None) -> str | None:
    if path is None or not Path(path).exists():
        return None
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).exists():
        return None
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else None


def _extract_json_claims(obj: dict[str, Any] | None) -> dict[str, Any]:
    if not obj:
        return {}
    out: dict[str, Any] = {}
    for key in ("name", "track", "val_bpb", "pre_quant_val_bpb", "bytes_total", "bytes_model_int6_zlib"):
        if key in obj:
            out[key] = _coerce_scalar(obj[key])
    return out


def _extract_readme_claims(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    out: dict[str, Any] = {}
    heading = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if heading:
        out["name"] = heading.group(1).strip()
    for key, pattern in README_PATTERNS.items():
        match = pattern.search(text)
        if match:
            out[key] = _coerce_number(match.group(1))
    return out


def _extract_log_claims(logs: list[tuple[Path, str | None]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path, text in logs:
        if not text:
            continue
        row: dict[str, Any] = {"path": str(path)}
        for key, pattern in LOG_PATTERNS.items():
            match = pattern.search(text)
            if match:
                row[key] = _coerce_number(match.group(1))
        out[str(path)] = row
    return out


def _extract_artifact_claims(paths: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in paths:
        if Path(path).exists():
            out[str(path)] = {
                "bytes": int(Path(path).stat().st_size),
                "suffix": Path(path).suffix,
                "looks_like_artifact": Path(path).suffix in ARTIFACT_EXTS,
            }
    return out


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return _coerce_number(value)
        except ValueError:
            return value.strip()
    return value


def _coerce_number(raw: str) -> int | float:
    text = raw.replace(",", "").replace("_", "").strip()
    if any(ch in text for ch in ".eE"):
        return float(text)
    return int(text)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def finding(
    severity: str,
    kind: str,
    message: str,
    *,
    path: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"severity": severity, "kind": kind, "message": message}
    if path is not None:
        row["path"] = path
    if details:
        row["details"] = details
    return row


def check_presence(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rules = get_profile_rules(manifest["profile"])
    evidence = manifest["evidence"]
    findings: list[dict[str, Any]] = []
    checked = 0
    for key in rules["required_evidence"]:
        checked += 1
        path = evidence.get(key)
        if path is None or not Path(path).exists():
            findings.append(finding("fail", "missing_evidence", f"Required evidence {key} is missing", path=str(path) if path else None))
    return {"pass": not findings, "checked_count": checked, "finding_count": len(findings)}, findings


def check_claim_consistency(manifest: dict[str, Any], extracted: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rules = get_profile_rules(manifest["profile"])
    findings: list[dict[str, Any]] = []
    sources = {
        "submission_json": extracted.get("submission_json", {}),
        "results_json": extracted.get("results_json", {}),
        "readme": extracted.get("readme", {}),
    }
    for log_path, row in extracted.get("logs", {}).items():
        sources[f"log:{log_path}"] = row
    for key in rules["claim_fields"]:
        values = {name: claims[key] for name, claims in sources.items() if key in claims}
        if len(values) <= 1:
            continue
        canonical_source = "submission_json" if "submission_json" in values else next(iter(values))
        canonical_value = values[canonical_source]
        for source_name, value in values.items():
            if source_name == canonical_source:
                continue
            if not _values_match(canonical_value, value):
                severity = "fail" if source_name == "results_json" else "warn"
                findings.append(
                    finding(
                        severity,
                        "claim_mismatch",
                        f"{key} differs between {canonical_source} and {source_name}",
                        details={
                            "field": key,
                            "lhs_source": canonical_source,
                            "lhs_value": canonical_value,
                            "rhs_source": source_name,
                            "rhs_value": value,
                        },
                    )
                )
    return {"pass": not any(row["severity"] == "fail" for row in findings), "finding_count": len(findings)}, findings


def check_artifact_bytes(manifest: dict[str, Any], extracted: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    actual_artifacts = extracted.get("artifact_files", {})
    actual_sizes = {path: row["bytes"] for path, row in actual_artifacts.items()}
    claimed_sources = (
        ("submission_json", extracted.get("submission_json", {}).get("bytes_model_int6_zlib")),
        ("results_json", extracted.get("results_json", {}).get("bytes_model_int6_zlib")),
        ("readme", extracted.get("readme", {}).get("bytes_model_int6_zlib")),
    )
    if len(actual_sizes) == 1:
        actual_path, actual_size = next(iter(actual_sizes.items()))
        for source_name, claimed in claimed_sources:
            if claimed is None:
                continue
            if int(claimed) != int(actual_size):
                findings.append(
                    finding(
                        "fail",
                        "artifact_size_mismatch",
                        f"Artifact bytes claimed by {source_name} do not match the actual artifact size",
                        path=actual_path,
                        details={"source": source_name, "claimed": int(claimed), "actual": int(actual_size)},
                    )
                )
    return {"pass": not findings if actual_sizes else None, "artifact_count": len(actual_sizes), "finding_count": len(findings)}, findings


def check_protocol_shape(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rules = get_profile_rules(manifest["profile"])
    findings: list[dict[str, Any]] = []
    scanned = 0
    for path in _iter_scan_paths(manifest, include_patch=True):
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = text.lower()
        scanned += 1
        for marker, kind in rules["protocol_risk_markers"]:
            if marker.lower() in lowered:
                findings.append(finding("warn", kind, f"Protocol-risk marker found: {marker}", path=str(path)))
    pass_value: bool | None = True if scanned else None
    if findings:
        pass_value = False
    return {"pass": pass_value, "scanned_count": scanned, "finding_count": len(findings)}, findings


def _iter_scan_paths(manifest: dict[str, Any], *, include_patch: bool) -> list[Path]:
    evidence = manifest["evidence"]
    paths = [Path(path) for path in evidence.get("code", []) if Path(path).exists()]
    if include_patch and evidence.get("patch") is not None:
        patch_path = Path(evidence["patch"])
        if patch_path.exists():
            paths.append(patch_path)
    return paths


def _values_match(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        return abs(float(lhs) - float(rhs)) <= 1e-9
    return lhs == rhs


# ---------------------------------------------------------------------------
# Top-level audit
# ---------------------------------------------------------------------------

def audit_submission(source: Path | str | dict[str, Any]) -> dict[str, Any]:
    manifest = load_submission_manifest(source)
    extracted = extract_claims(manifest)

    check_rows = {
        "presence": check_presence(manifest),
        "claim_consistency": check_claim_consistency(manifest, extracted),
        "artifact_bytes": check_artifact_bytes(manifest, extracted),
        "protocol_shape": check_protocol_shape(manifest),
    }

    checks: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    for name, (summary, rows) in check_rows.items():
        checks[name] = summary
        findings.extend(rows)

    alerts = [row["message"] for row in findings if row["severity"] in {"warn", "fail"}]
    return {
        "profile": manifest["profile"],
        "verdict": _verdict(findings),
        "submission": {
            "repo_root": str(manifest["repo_root"]),
            "submission_root": str(manifest["submission_root"]),
            "track": manifest.get("claim_overrides", {}).get("track"),
            "name": extracted.get("submission_json", {}).get("name")
            or extracted.get("readme", {}).get("name"),
        },
        "extracted_claims": extracted,
        "checks": checks,
        "findings": findings,
        "alerts": alerts,
    }


def _verdict(findings: list[dict[str, Any]]) -> str:
    severities = {row["severity"] for row in findings}
    if "fail" in severities:
        return "fail"
    if "warn" in severities:
        return "warn"
    return "pass"


# ---------------------------------------------------------------------------
# Public claim-extraction helpers (ported from conker-detect submission_extract.py)
# ---------------------------------------------------------------------------

def read_optional_text(path: "Path | None") -> "str | None":
    """Read text from path if it exists, returning None otherwise."""
    return _read_optional_text(path)


def read_optional_json(path: "Path | None") -> "dict[str, Any] | None":
    """Load a JSON object from path if it exists, returning None otherwise."""
    return _read_optional_json(path)


def extract_artifact_claims(paths: "list[Path]") -> "dict[str, Any]":
    """Return a dict mapping artifact path strings to size/suffix metadata."""
    return _extract_artifact_claims(paths)


def extract_json_claims(obj: "dict[str, Any] | None") -> "dict[str, Any]":
    """Extract structured claims from a submission/results JSON object."""
    return _extract_json_claims(obj)


def extract_log_claims(logs: "list[tuple[Path, str | None]]") -> "dict[str, Any]":
    """Extract metric claims from a list of (path, text) log pairs."""
    return _extract_log_claims(logs)


def extract_readme_claims(text: "str | None") -> "dict[str, Any]":
    """Extract metric claims from README text using regex patterns."""
    return _extract_readme_claims(text)


# ---------------------------------------------------------------------------
# Additional submission checks (ported from conker-detect submission_checks.py)
# ---------------------------------------------------------------------------

def check_data_boundary_signals(manifest: dict[str, Any]) -> "tuple[dict[str, Any], list[dict[str, Any]]]":
    """Scan submission code/patch for data-boundary risk markers."""
    rules = get_profile_rules(manifest["profile"])
    findings: list[dict[str, Any]] = []
    scanned = 0
    for path in _iter_scan_paths(manifest, include_patch=True):
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = text.lower()
        scanned += 1
        for marker, kind in rules["data_boundary_risk_markers"]:
            if marker.lower() in lowered:
                findings.append(finding("warn", kind, f"Data-boundary risk marker found: {marker}", path=str(path)))
    pass_value: bool | None = True if scanned else None
    if findings:
        pass_value = False
    return {"pass": pass_value, "scanned_count": scanned, "finding_count": len(findings)}, findings


def check_reproducibility_surface(
    manifest: dict[str, Any],
    extracted: dict[str, Any],
) -> "tuple[dict[str, Any], list[dict[str, Any]]]":
    """Check that the submission includes logs, code, and artifact for reproducibility."""
    evidence = manifest["evidence"]
    findings: list[dict[str, Any]] = []
    has_log = any(Path(path).exists() for path in evidence.get("logs", []))
    has_code = any(Path(path).exists() for path in evidence.get("code", []))
    has_artifact = bool(extracted.get("artifact_files"))
    if not has_log:
        findings.append(finding("warn", "missing_logs", "No log file was provided for reproducibility"))
    if not has_code:
        findings.append(finding("warn", "missing_code", "No code file was provided for reproducibility"))
    if not has_artifact:
        findings.append(finding("warn", "missing_artifact", "No artifact file was provided"))
    return {
        "pass": not any(row["severity"] == "fail" for row in findings),
        "has_log": has_log,
        "has_code": has_code,
        "has_artifact": has_artifact,
        "finding_count": len(findings),
    }, findings


def check_patch_triage(manifest: dict[str, Any]) -> "tuple[dict[str, Any], list[dict[str, Any]]]":
    """Triage the submission patch for category and risk signals."""
    patch_path = manifest["evidence"].get("patch")
    if patch_path is None or not Path(patch_path).exists():
        return {"pass": None, "present": False}, []
    patch_text = Path(patch_path).read_text(encoding="utf-8", errors="replace")
    files = patch_files(patch_text)
    triage = classify_patch_context(manifest["profile"], patch_text, files)
    findings: list[dict[str, Any]] = []
    if triage["optimizer_in_eval_signal"]:
        findings.append(finding("warn", "optimizer_in_eval_signal", "Patch contains an optimizer-in-eval signal", path=str(patch_path)))
    return {"pass": not findings, "present": True, **triage}, findings
