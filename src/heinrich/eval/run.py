"""Run the full eval pipeline as subprocess chain.

Each model-loading step runs in a subprocess that exits after writing to DB.
Pattern-based scorers run in-process (no model needed).

Usage:
    python -m heinrich.eval.run \
        --model mlx-community/Qwen2.5-7B-Instruct-4bit \
        --prompts harmbench,simplesafety \
        --scorers word_match,regex_harm,qwen3guard \
        --output report.json \
        --conditions clean
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _json_safe(obj):
    """Recursively convert *obj* to JSON-serializable types.

    Raises TypeError for unknown types instead of silently converting
    via ``default=str`` (which hides real serialization bugs).
    """
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    # Explicitly convert known numeric types (numpy, etc.)
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}: {obj!r}")


def _resolve_conditions(db, model: str, conditions: list[str]) -> list[str]:
    """Resolve conditions list, handling 'auto' by reading from DB.

    When conditions contains 'auto', query the conditions table for this model.
    If conditions are found, use them. Otherwise fall back to ['clean'].
    """
    if "auto" not in conditions:
        return conditions

    mid_row = db._conn.execute("SELECT id FROM models WHERE name = ?", (model,)).fetchone()
    if mid_row is None:
        return ["clean"]

    db_conditions = db.get_conditions(mid_row["id"])
    if db_conditions:
        names = [c["name"] for c in db_conditions]
        print(f"[auto] Resolved conditions from DB: {names}")
        return names
    return ["clean"]


def run_pipeline(
    model: str,
    prompts: list[str],
    scorers: list[str],
    conditions: list[str] | None = None,
    output: str | None = None,
    db_path: str | None = None,
    max_prompts: int | None = None,
    timeout: int = 1800,
) -> dict:
    """Run the full eval pipeline. Returns the report dict."""
    if conditions is None:
        conditions = ["clean"]

    from heinrich.core.db import SignalDB
    from heinrich.eval.prompts import insert_prompts_to_db
    from heinrich.eval.score import load_scorer, score_all
    from heinrich.eval.calibrate import calibrate_all
    from heinrich.eval.report import build_report

    db = SignalDB(db_path) if db_path else SignalDB()
    db_file = str(db.path)

    # Resolve 'auto' conditions from DB
    conditions = _resolve_conditions(db, model, conditions)

    # Step 1: Load prompts into DB (no model needed, in-process)
    t0 = time.time()
    for prompt_set in prompts:
        insert_prompts_to_db(db, prompt_set, max_prompts=max_prompts)
    n_prompts = len(db.get_prompts())
    print(f"[{time.time()-t0:.1f}s] Loaded {n_prompts} prompts from {prompts}")

    # Scale timeout based on prompt count: at least 10 min, ~5s per prompt
    effective_timeout = max(timeout, max(600, n_prompts * 5))

    # Step 2: Generate (SUBPROCESS — loads model, generates, exits)
    # Check if generations already exist for this model+conditions
    mid_row = db._conn.execute("SELECT id FROM models WHERE name = ?", (model,)).fetchone()
    existing_gens = 0
    if mid_row:
        existing_gens = db._conn.execute(
            "SELECT COUNT(*) FROM generations WHERE model_id = ?", (mid_row["id"],)
        ).fetchone()[0]

    if existing_gens >= n_prompts * len(conditions):
        print(f"[{time.time()-t0:.1f}s] Skipping generation: {existing_gens} already exist")
    else:
        print(f"[{time.time()-t0:.1f}s] Generating outputs (subprocess)...")
        db.close()  # Close DB before subprocess
        _run_subprocess([
            sys.executable, "-m", "heinrich.eval.generate",
            "--model", model,
            "--db", db_file,
            "--conditions", ",".join(conditions),
        ], timeout=effective_timeout)
        db = SignalDB(db_file)  # Reopen

        # Issue 3: Verify generation subprocess actually wrote data
        mid_row = db._conn.execute("SELECT id FROM models WHERE name = ?", (model,)).fetchone()
        if mid_row:
            n_gens = db._conn.execute(
                "SELECT COUNT(*) FROM generations WHERE model_id = ?",
                (mid_row["id"],),
            ).fetchone()[0]
        else:
            n_gens = 0
        if n_gens == 0:
            raise RuntimeError("Generation subprocess produced no outputs")
        print(f"[{time.time()-t0:.1f}s] Generated {n_gens} outputs")

    # Step 3: Score (subprocess for model-based, in-process for pattern-based)
    # Issue 4: Continue on error — wrap each scorer in try/except
    for scorer_name in scorers:
        try:
            scorer_instance = load_scorer(scorer_name)

            # Check how many unscored generations exist
            unscored = db.get_unscored_generations(scorer_name)
            if not unscored:
                print(f"[{time.time()-t0:.1f}s] Skipping {scorer_name}: all scored")
                continue

            if scorer_instance.requires_model:
                # SUBPROCESS — loads scorer model, scores, exits
                print(f"[{time.time()-t0:.1f}s] Scoring with {scorer_name} (subprocess, {len(unscored)} to score)...")
                db.close()
                _run_subprocess([
                    sys.executable, "-m", "heinrich.eval.score",
                    "--scorer", scorer_name,
                    "--db", db_file,
                ], timeout=effective_timeout)
                db = SignalDB(db_file)

                # Issue 3: Verify scorer subprocess wrote scores
                n_new = db._conn.execute(
                    "SELECT COUNT(*) FROM scores WHERE scorer = ?",
                    (scorer_name,),
                ).fetchone()[0]
                if n_new == 0:
                    print(f"WARNING: {scorer_name} subprocess produced no scores")
            else:
                # IN-PROCESS — no model needed, instant
                print(f"[{time.time()-t0:.1f}s] Scoring with {scorer_name} (in-process, {len(unscored)} to score)...")
                n = score_all(db, scorer_name)
                print(f"  Scored {n}")
        except Exception as e:
            print(f"WARNING: Scorer {scorer_name} failed: {e}")
            # Reopen DB if it was closed before the failure
            try:
                db._conn.execute("SELECT 1")
            except Exception:
                db = SignalDB(db_file)
            continue

    # Step 4: Calibrate (in-process, no model)
    print(f"[{time.time()-t0:.1f}s] Calibrating...")
    calibrate_all(db)

    # Step 5: Report (in-process, pure SQL)
    print(f"[{time.time()-t0:.1f}s] Building report...")
    report = build_report(db)

    # Issue 6: Warn when no benign prompts for calibration
    cal = report.get("calibration", [])
    uncalibrated = [c["scorer"] for c in cal if c.get("fpr") is None]
    if uncalibrated:
        print(
            f"WARNING: No benign prompts for calibration. Scorers {uncalibrated} "
            f"have unknown FPR. Add --prompts benign_calibration or include benign "
            f"prompts in your prompt set."
        )

    if output:
        # Issue 11: Use _json_safe instead of default=str to catch type bugs
        Path(output).write_text(json.dumps(_json_safe(report), indent=2))
        print(f"[{time.time()-t0:.1f}s] Report written to {output}")

    db.close()
    print(f"[{time.time()-t0:.1f}s] Done.")
    return report


def _run_subprocess(cmd: list[str], timeout: int = 1800):
    """Run a subprocess, streaming stdout/stderr."""
    result = subprocess.run(
        cmd,
        capture_output=False,  # let output stream to terminal
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed with exit code {result.returncode}: {' '.join(cmd)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run heinrich eval pipeline")
    parser.add_argument("--model", required=True, help="Model ID (e.g. mlx-community/Qwen2.5-7B-Instruct-4bit)")
    parser.add_argument("--prompts", required=True, help="Comma-separated prompt set names (e.g. harmbench,simplesafety)")
    parser.add_argument("--scorers", required=True, help="Comma-separated scorer names (e.g. word_match,regex_harm,qwen3guard)")
    parser.add_argument("--conditions", default="clean", help="Comma-separated conditions (default: clean)")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file path")
    parser.add_argument("--db", default=None, help="Database path (default: ./data/heinrich.db)")
    parser.add_argument("--max-prompts", type=int, default=None, help="Max prompts per set")
    parser.add_argument("--timeout", type=int, default=1800, help="Subprocess timeout in seconds (default: 1800)")
    args = parser.parse_args()

    run_pipeline(
        model=args.model,
        prompts=args.prompts.split(","),
        scorers=args.scorers.split(","),
        conditions=args.conditions.split(","),
        output=args.output,
        db_path=args.db,
        max_prompts=args.max_prompts,
        timeout=args.timeout,
    )
