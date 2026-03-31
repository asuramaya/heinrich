"""Ledger handoff — prepare and write handoff bundles from run directories."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .legality import audit_legality, load_adapter, load_json_config, load_token_array
from .provenance import audit_provenance
from .replay import replay_runtime
from .submission import audit_submission

ARTIFACT_EXTS = (".pt", ".pth", ".bin", ".npz", ".safetensors", ".ckpt", ".ptz")


def prepare_ledger_handoff(
    run_dir: Path | None = None,
    out_dir: Path | None = None,
    *,
    submission_dir: Path | None = None,
    run_id: str | None = None,
    runner: Any | None = None,
    bundle_id: str | None = None,
    profile: str = "parameter-golf",
    repo_root: Path | None = None,
    patch: Path | None = None,
    provenance_source: Path | None = None,
    adapter_ref: str | None = None,
    adapter_config_raw: str | None = None,
    tokens_path: Path | None = None,
    tokens_key: str | None = None,
    trust_level: str = "basic",
    chunk_size: int = 32_768,
    max_chunks: int | None = None,
    sample_chunks: int = 4,
    future_probes_per_chunk: int = 2,
    answer_probes_per_chunk: int = 2,
    positions_per_future_probe: int = 4,
    position_batch_size: int = 256,
    seed: int = 0,
    vocab_size: int | None = None,
    atol: float = 1e-7,
    rtol: float = 1e-7,
) -> dict[str, Any]:
    # Support new-style API: submission_dir, run_id, runner
    if submission_dir is not None:
        if run_dir is None:
            run_dir = submission_dir
        if not submission_dir.exists():
            return {"error": f"Submission directory not found: {submission_dir}", "run_id": run_id}
    if run_dir is None:
        return {"error": "run_dir or submission_dir must be provided"}
    if out_dir is None:
        import tempfile
        out_dir = Path(tempfile.mkdtemp())
    run_dir = run_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_id = bundle_id or run_id or run_dir.name
    repo_root = repo_root.resolve() if repo_root is not None else _infer_repo_root(run_dir)

    submission_manifest = _build_submission_manifest(run_dir, repo_root=repo_root, profile=profile, patch=patch)
    submission_report = audit_submission(submission_manifest)
    report_paths: dict[str, str] = {}
    report_paths["submission"] = _write_json(out_dir / "reports" / "submission.json", submission_report)

    provenance_report: dict[str, Any] | None = None
    if provenance_source is not None:
        provenance_report = audit_provenance(provenance_source)
        report_paths["provenance"] = _write_json(out_dir / "reports" / "provenance.json", provenance_report)

    legality_report: dict[str, Any] | None = None
    replay_report: dict[str, Any] | None = None
    if adapter_ref and tokens_path is not None:
        adapter_config = load_json_config(adapter_config_raw)
        adapter = load_adapter(adapter_ref, adapter_config)
        tokens = load_token_array(tokens_path, key=tokens_key)
        legality_report = audit_legality(
            adapter, tokens, profile=profile, trust_level=trust_level,
            chunk_size=chunk_size, max_chunks=max_chunks, sample_chunks=sample_chunks,
            future_probes_per_chunk=future_probes_per_chunk, answer_probes_per_chunk=answer_probes_per_chunk,
            positions_per_future_probe=positions_per_future_probe, seed=seed,
            vocab_size=vocab_size, atol=atol, rtol=rtol,
        )
        report_paths["legality"] = _write_json(out_dir / "reports" / "legality.json", legality_report)

        replay_adapter = load_adapter(adapter_ref, adapter_config)
        replay_report = replay_runtime(
            replay_adapter, tokens, profile=profile,
            chunk_size=chunk_size, max_chunks=max_chunks, sample_chunks=sample_chunks,
            position_batch_size=position_batch_size, seed=seed, atol=atol, rtol=rtol,
        )
        report_paths["replay"] = _write_json(out_dir / "reports" / "replay.json", replay_report)

    claim = _synthesize_claim(bundle_id, submission_report)
    metrics = _synthesize_metrics(submission_report, replay_report)
    provenance = _synthesize_provenance(run_dir, repo_root, provenance_report)
    audits = _synthesize_audits(submission_report, provenance_report, legality_report, replay_report)

    claim_path = _write_json(out_dir / "claim.json", claim)
    metrics_path = _write_json(out_dir / "metrics.json", metrics)
    provenance_path = _write_json(out_dir / "provenance.json", provenance)
    audits_path = _write_json(out_dir / "audits.json", audits)
    manifest = write_ledger_bundle_manifest(
        out_dir / "ledger_manifest.json",
        bundle_id=bundle_id,
        claim=Path(claim_path).name,
        metrics=Path(metrics_path).name,
        provenance=Path(provenance_path).name,
        audits=Path(audits_path).name,
        submission_report=_rel_if_present(report_paths.get("submission")),
        provenance_report=_rel_if_present(report_paths.get("provenance")),
        legality_report=_rel_if_present(report_paths.get("legality")),
        replay_report=_rel_if_present(report_paths.get("replay")),
    )

    return {
        "bundle_id": bundle_id,
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "reports": report_paths,
        "claim": claim_path,
        "metrics": metrics_path,
        "provenance": provenance_path,
        "audits": audits_path,
        "ledger_manifest": str((out_dir / "ledger_manifest.json").resolve()),
        "manifest": manifest,
    }


def write_ledger_bundle_manifest(
    out_path: Path,
    *,
    bundle_id: str | None = None,
    run_id: str | None = None,
    claim: str | None = None,
    metrics: Any | None = None,
    provenance: str | None = None,
    audits: str | None = None,
    submission_report: Any | None = None,
    submission_result: Any | None = None,
    provenance_report: Any | None = None,
    provenance_result: Any | None = None,
    legality_report: Any | None = None,
    legality_result: Any | None = None,
    replay_report: Any | None = None,
    replay_result: Any | None = None,
) -> dict[str, Any]:
    # Normalize aliases
    effective_id = bundle_id or run_id or ""
    effective_submission = submission_report or submission_result
    effective_provenance = provenance_report or provenance_result
    effective_legality = legality_report or legality_result
    effective_replay = replay_report or replay_result

    attachments: list[dict[str, Any]] = []
    if effective_submission and isinstance(effective_submission, str):
        attachments.append({"source": effective_submission, "dest": "audits/tier1/submission.json"})
    if effective_provenance and isinstance(effective_provenance, str):
        attachments.append({"source": effective_provenance, "dest": "audits/tier1/provenance.json"})
    if effective_legality and isinstance(effective_legality, str):
        attachments.append({"source": effective_legality, "dest": "audits/tier3/legality.json"})
    if effective_replay and isinstance(effective_replay, str):
        attachments.append({"source": effective_replay, "dest": "audits/tier3/replay.json"})

    manifest: dict[str, Any] = {"bundle_id": effective_id, "run_id": effective_id, "attachments": attachments}
    if claim:
        manifest["claim"] = claim
    if metrics is not None:
        manifest["metrics"] = metrics
    if provenance:
        manifest["provenance"] = provenance
    if audits:
        manifest["audits"] = audits
    if effective_submission and not isinstance(effective_submission, str):
        manifest["submission_result"] = effective_submission
    if effective_provenance and not isinstance(effective_provenance, str):
        manifest["provenance_result"] = effective_provenance
    if effective_legality and not isinstance(effective_legality, str):
        manifest["legality_result"] = effective_legality
    if effective_replay and not isinstance(effective_replay, str):
        manifest["replay_result"] = effective_replay

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _build_submission_manifest(run_dir: Path, *, repo_root: Path, profile: str, patch: Path | None) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    if (run_dir / "README.md").exists():
        evidence["readme"] = "README.md"
    if (run_dir / "submission.json").exists():
        evidence["submission_json"] = "submission.json"
    if (run_dir / "results.json").exists():
        evidence["results_json"] = "results.json"
    logs = sorted(path.name for path in run_dir.glob("*.log"))
    if logs:
        evidence["logs"] = logs
    artifacts = sorted(path.name for path in run_dir.iterdir() if path.is_file() and path.suffix in ARTIFACT_EXTS)
    if artifacts:
        evidence["artifacts"] = artifacts
    code = [name for name in ("train_gpt.py", "train_gpt_mlx.py") if (run_dir / name).exists()]
    if code:
        evidence["code"] = code
    if patch is not None:
        evidence["patch"] = str(patch.resolve())
    return {
        "profile": profile,
        "repo_root": str(repo_root),
        "submission_root": _submission_root_value(run_dir, repo_root),
        "evidence": evidence,
    }


def _synthesize_claim(bundle_id: str, submission_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": bundle_id,
        "requested_label": "Tier-1 reviewed",
        "submission_name": submission_report.get("submission", {}).get("name"),
        "track": submission_report.get("submission", {}).get("track"),
    }


def _synthesize_metrics(submission_report: dict[str, Any], replay_report: dict[str, Any] | None) -> dict[str, Any]:
    extracted = submission_report.get("extracted_claims", {})
    metrics: dict[str, Any] = {}
    pre_quant = _pick_claim_value(extracted, "pre_quant_val_bpb")
    val_bpb = _pick_claim_value(extracted, "val_bpb")
    artifact_bytes = _pick_claim_value(extracted, "bytes_model_int6_zlib")
    if pre_quant is not None:
        metrics["fresh_process_full"] = {"bpb": pre_quant}
    if val_bpb is not None or artifact_bytes is not None:
        packed: dict[str, Any] = {}
        if val_bpb is not None:
            packed["bpb"] = val_bpb
        if artifact_bytes is not None:
            packed["artifact_bytes"] = artifact_bytes
        metrics["packed_artifact_full"] = packed
    if replay_report is not None:
        metrics["replay"] = replay_report.get("aggregate", {})
    return metrics


def _synthesize_provenance(run_dir: Path, repo_root: Path, provenance_report: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {"run_id": run_dir.name, "source_root": str(run_dir), "source_repo": str(repo_root)}
    if provenance_report is not None:
        summary = provenance_report.get("provenance", {})
        row.update({key: summary.get(key) for key in ("submitted_run_id", "selection_mode", "candidate_run_count")})
    return row


def _synthesize_audits(
    submission_report: dict[str, Any],
    provenance_report: dict[str, Any] | None,
    legality_report: dict[str, Any] | None,
    replay_report: dict[str, Any] | None,
) -> dict[str, Any]:
    tier1 = _combine_status([submission_report.get("verdict"), provenance_report.get("verdict") if provenance_report is not None else None])
    out: dict[str, Any] = {"tier1": {"status": tier1, "submission": submission_report.get("verdict"), "provenance": provenance_report.get("verdict") if provenance_report is not None else None}}
    if legality_report is not None:
        legality_status = "pass" if not legality_report.get("alerts") else "warn"
        tier3: dict[str, Any] = {"status": "warn", "scope": "one_shot_runtime_handoff", "legality": legality_status}
        trust = legality_report.get("trust", {})
        if trust:
            tier3["trust_level_requested"] = trust.get("requested")
            tier3["trust_level_achieved"] = trust.get("achieved")
            tier3["trust_satisfied"] = trust.get("satisfied")
        if replay_report is not None:
            replay_pass = replay_report.get("repeatability", {}).get("pass")
            tier3["replay"] = "pass" if replay_pass is True else "warn" if replay_pass is False else "unknown"
        out["tier3"] = tier3
    return out


def _combine_status(values: list[str | None]) -> str:
    values = [value for value in values if value]
    if not values:
        return "missing"
    if "fail" in values:
        return "fail"
    if "warn" in values:
        return "warn"
    return "pass"


def _pick_claim_value(extracted: dict[str, Any], key: str) -> Any:
    for source_name in ("submission_json", "results_json", "readme"):
        source = extracted.get(source_name, {})
        if key in source:
            return source[key]
    logs = extracted.get("logs", {})
    if isinstance(logs, dict):
        for row in logs.values():
            if isinstance(row, dict) and key in row:
                return row[key]
    return None


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return str(path.resolve())


def _rel_if_present(path: str | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).relative_to(Path(path).parents[1]))


def _infer_repo_root(run_dir: Path) -> Path:
    for parent in (run_dir, *run_dir.parents):
        records_dir = parent / "records"
        if records_dir.exists() and _is_relative_to(run_dir, records_dir):
            return parent.resolve()
    for parent in (run_dir, *run_dir.parents):
        if (parent / ".git").exists():
            return parent.resolve()
    return run_dir.parent.resolve()


def _submission_root_value(run_dir: Path, repo_root: Path) -> str:
    try:
        return str(run_dir.relative_to(repo_root))
    except ValueError:
        return str(run_dir)


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False
