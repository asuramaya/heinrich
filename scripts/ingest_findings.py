#!/usr/bin/env python3
"""Ingest experimental findings from JSON data files into the SignalDB.

Reads:
  - data/open_questions_results.json  -> kind=finding
  - data/cracking_matrix.json         -> kind=crack
  - data/complete_signal_analysis.json -> kind=signal_matrix

Stores everything in ~/.heinrich/signals.db with run tracking and
derivation links where source signals are available.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heinrich.db import SignalDB
from heinrich.signal import Signal

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = Path("~/.heinrich/signals.db").expanduser()


def ingest_open_questions(db: SignalDB) -> tuple[int, int]:
    """Ingest open_questions_results.json.

    Each answered question becomes a Signal with:
      kind=finding, source=experiment, model=<model_tested>,
      target=<question_number>, value=1.0 (answered) or 0.0 (blocked),
      metadata contains the finding text.

    Returns (n_signals, n_runs).
    """
    path = DATA_DIR / "open_questions_results.json"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return 0, 0

    data = json.loads(path.read_text())
    model = data.get("model", "unknown")
    results = data.get("results", {})

    n_signals = 0
    with db.run("ingest_open_questions", model=model,
                script="ingest_findings.py",
                metadata={"source_file": str(path.name),
                          "experiment_date": data.get("experiment_date", "")}) as run_id:
        for q_key, q_data in results.items():
            status = q_data.get("status", "unknown")
            finding = q_data.get("finding", "")

            # Determine value: 1.0 = answered, 0.0 = blocked/unknown
            answered_statuses = {"answered", "answered_differently", "overturned",
                                 "confirmed_and_extended", "partially_answered"}
            value = 1.0 if status in answered_statuses else 0.0

            meta = {
                "finding": finding,
                "status": status,
                "question_key": q_key,
            }
            # Include extra fields if present
            if "overturns" in q_data:
                meta["overturns"] = q_data["overturns"]
            if "models_tested" in q_data:
                meta["models_tested"] = q_data["models_tested"]
            if "data" in q_data:
                meta["data"] = q_data["data"]
            if "missing" in q_data:
                meta["missing"] = q_data["missing"]
            if "note" in q_data:
                meta["note"] = q_data["note"]

            # Use model from per-question data if available, else top-level
            models_tested = q_data.get("models_tested", q_data.get("models", [model]))
            if isinstance(models_tested, list) and models_tested:
                signal_model = models_tested[0]
            else:
                signal_model = model

            signal = Signal(
                kind="finding",
                source="experiment",
                model=signal_model,
                target=q_key,
                value=value,
                metadata=meta,
            )
            db.add(signal, run_id=run_id)
            n_signals += 1

    return n_signals, 1


def ingest_cracking_matrix(db: SignalDB) -> tuple[int, int]:
    """Ingest cracking_matrix.json.

    Each model crack result becomes a Signal with:
      kind=crack, source=attack, model=<model_name>, target=<query_type>,
      value=alpha_at_crack, metadata has window width, method,
      content classification.

    Returns (n_signals, n_runs).
    """
    path = DATA_DIR / "cracking_matrix.json"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return 0, 0

    data = json.loads(path.read_text())
    results = data.get("results", {})
    universal = data.get("universal_findings", {})

    n_signals = 0
    with db.run("ingest_cracking_matrix", model="multi",
                script="ingest_findings.py",
                metadata={"source_file": str(path.name),
                          "experiment_date": data.get("date", ""),
                          "methodology": data.get("methodology", "")}) as run_id:

        # Per-model crack results
        for model_key, mdata in results.items():
            crack_method = mdata.get("crack_method", "unknown")
            window = mdata.get("window", "unknown")
            difficulty = mdata.get("difficulty", "unknown")
            notes = mdata.get("notes", "")

            # Parse alpha from crack_method string (e.g. "32 layers, α=-0.15")
            alpha = 0.0
            if "α=" in crack_method or "alpha=" in crack_method.lower():
                import re
                m = re.search(r'[αa](?:lpha)?=(-?[\d.]+)', crack_method)
                if m:
                    alpha = float(m.group(1))

            # Primary crack signal
            meta = {
                "crack_method": crack_method,
                "window": window,
                "difficulty": difficulty,
                "notes": notes,
            }
            # Include specific test results
            for key in ("meth_holdout", "pipe_bomb_at_015", "keylogger_at_015",
                        "suicide_at_015", "paradox", "forensic_combo",
                        "early_layer_attack", "immunity_debunked",
                        "subset_resistance", "content_at_crack", "content_at_015"):
                if key in mdata:
                    meta[key] = mdata[key]

            signal = Signal(
                kind="crack",
                source="attack",
                model=model_key,
                target="distributed_safety",
                value=alpha,
                metadata=meta,
            )
            db.add(signal, run_id=run_id)
            n_signals += 1

        # Universal findings as a summary signal
        if universal:
            signal = Signal(
                kind="crack",
                source="attack",
                model="universal",
                target="summary",
                value=float(data.get("models_cracked", 0)),
                metadata=universal,
            )
            db.add(signal, run_id=run_id)
            n_signals += 1

    return n_signals, 1


def ingest_signal_matrix(db: SignalDB) -> tuple[int, int]:
    """Ingest complete_signal_analysis.json.

    One signal per cell in the matrix:
      kind=signal_matrix, source=cross_signal, model=<model>,
      target=<prompt_name>_<framing>,
      value=refuse_prob, metadata has all measurements.

    Returns (n_signals, n_runs).
    """
    path = DATA_DIR / "complete_signal_analysis.json"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return 0, 0

    data = json.loads(path.read_text())
    model = data.get("model", "unknown")
    matrix = data.get("signal_matrix", [])

    n_signals = 0
    source_ids = []  # Track source signal IDs for derivation

    with db.run("ingest_signal_matrix", model=model,
                script="ingest_findings.py",
                metadata={"source_file": str(path.name),
                          "timestamp": data.get("timestamp", ""),
                          "n_measurements": data.get("n_measurements", 0)}) as run_id:

        for entry in matrix:
            prompt = entry.get("prompt", "unknown")
            framing = entry.get("framing", "direct")
            target = f"{prompt}_{framing}"

            meta = {
                "query": entry.get("query", ""),
                "category": entry.get("category", ""),
                "framing": framing,
                "comply_prob": entry.get("comply_prob", 0.0),
                "entropy": entry.get("entropy", 0.0),
                "top_token": entry.get("top_token", ""),
                "residual_proj": entry.get("residual_proj", 0.0),
                "logit_label": entry.get("logit_label", ""),
                "residual_label": entry.get("residual_label", ""),
                "consensus": entry.get("consensus", ""),
                "confidence": entry.get("confidence", 0.0),
                "surface_compliance": entry.get("surface_compliance", False),
            }

            signal = Signal(
                kind="signal_matrix",
                source="cross_signal",
                model=model,
                target=target,
                value=entry.get("refuse_prob", 0.0),
                metadata=meta,
            )
            sid = db.add(signal, run_id=run_id)
            source_ids.append(sid)
            n_signals += 1

        # Create derived summary signals linking to source matrix entries
        # Per-prompt summary: average refuse_prob across framings
        prompts = {}
        prompt_source_ids = {}
        for i, entry in enumerate(matrix):
            prompt = entry.get("prompt", "unknown")
            rp = entry.get("refuse_prob", 0.0)
            prompts.setdefault(prompt, []).append(rp)
            prompt_source_ids.setdefault(prompt, []).append(source_ids[i])

        for prompt, rps in prompts.items():
            avg_rp = sum(rps) / len(rps)
            derived_signal = Signal(
                kind="signal_matrix",
                source="cross_signal_summary",
                model=model,
                target=f"{prompt}_avg",
                value=avg_rp,
                metadata={"n_framings": len(rps), "prompt": prompt},
            )
            db.add_derived(
                derived_signal,
                source_ids=prompt_source_ids[prompt],
                relationship="mean_across_framings",
                run_id=run_id,
            )
            n_signals += 1

    return n_signals, 1


def main():
    print(f"SignalDB: {DB_PATH}")
    db = SignalDB(DB_PATH)

    total_signals = 0
    total_runs = 0

    # 1. Open questions
    print("\n--- Ingesting open_questions_results.json ---")
    n, r = ingest_open_questions(db)
    print(f"  {n} signals, {r} runs")
    total_signals += n
    total_runs += r

    # 2. Cracking matrix
    print("\n--- Ingesting cracking_matrix.json ---")
    n, r = ingest_cracking_matrix(db)
    print(f"  {n} signals, {r} runs")
    total_signals += n
    total_runs += r

    # 3. Signal matrix
    print("\n--- Ingesting complete_signal_analysis.json ---")
    n, r = ingest_signal_matrix(db)
    print(f"  {n} signals, {r} runs")
    total_signals += n
    total_runs += r

    # Summary
    print("\n" + "=" * 50)
    print(f"TOTAL: {total_signals} signals ingested, {total_runs} runs created")
    summary = db.summary()
    print(f"DB now: {summary['n_signals']} signals, {summary['n_runs']} runs, "
          f"{summary['n_blobs']} blobs ({summary['db_size_mb']} MB)")
    print(f"Kinds: {summary['kinds']}")

    db.close()


if __name__ == "__main__":
    main()
