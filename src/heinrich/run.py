"""Run the full heinrich pipeline: discover -> attack -> eval -> report.

Usage:
    python -m heinrich.run \
        --model mlx-community/Qwen2.5-7B-Instruct-4bit \
        --prompts simple_safety \
        --scorers word_match,regex_harm \
        --max-prompts 3 \
        --output /tmp/full_flow_report.json

Or equivalently via the CLI:
    heinrich run --model X --prompts harmbench --scorers word_match,qwen3guard
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _run_subprocess(cmd: list[str], timeout: int = 1800) -> None:
    """Run a subprocess, streaming stdout/stderr."""
    result = subprocess.run(
        cmd,
        capture_output=False,  # let output stream to terminal
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed (exit {result.returncode}): {' '.join(cmd)}")


def run_full_pipeline(
    model: str,
    prompts: list[str],
    scorers: list[str],
    output: str | None = None,
    db_path: str | None = None,
    max_prompts: int | None = None,
    timeout: int = 1800,
    skip_discover: bool = False,
    skip_attack: bool = False,
) -> dict:
    """Run the full pipeline: discover -> attack -> eval -> report.

    Each model-loading step runs in a subprocess to release GPU memory.
    Returns the final report dict.
    """
    from heinrich.core.db import SignalDB
    from heinrich.eval.prompts import insert_prompts_to_db

    db = SignalDB(db_path) if db_path else SignalDB()
    db_file = str(db.path)
    t0 = time.time()

    # Step 1: Load prompts into DB (in-process, no model needed)
    for prompt_set in prompts:
        insert_prompts_to_db(db, prompt_set, max_prompts=max_prompts)
    # Always load benign prompts for direction finding and calibration
    try:
        insert_prompts_to_db(db, "benign", max_prompts=50)
    except Exception:
        pass  # benign download may fail without datasets library
    n_prompts = len(db.get_prompts())
    n_benign = len(db.get_prompts(is_benign=True))
    print(f"[{time.time()-t0:.1f}s] Loaded {n_prompts} prompts ({n_benign} benign) from {prompts}")

    # Scale timeout: ~2s per generation (forward + generate), up to 28 conditions per prompt
    # Plus ~200s for direction finding + cliff search
    estimated_conditions = 15  # worst case: clean + 14 steer conditions
    estimated_generations = n_prompts * estimated_conditions
    effective_timeout = max(timeout, 300 + estimated_generations * 3)

    db.close()

    # Step 2: Unified target subprocess (SUBPROCESS -- loads model ONCE,
    # runs discover + attack + generate, writes everything to DB)
    if not skip_discover or not skip_attack:
        print(f"\n[{time.time()-t0:.1f}s] === TARGET (discover + attack + generate) ===")
        cmd = [
            sys.executable, "-m", "heinrich.eval.target_subprocess",
            "--model", model, "--db", db_file,
        ]
        try:
            _run_subprocess(cmd, timeout=effective_timeout)
        except RuntimeError as e:
            print(f"WARNING: Target subprocess failed: {e}")
            print("Continuing with eval step (will use whatever is in DB)")
    else:
        print(f"[{time.time()-t0:.1f}s] Skipping discover+attack (both skipped)")

    # Step 3: Eval pipeline with conditions="auto"
    # Target subprocess already wrote generations; eval will skip generation
    # if rows exist, then score + calibrate + report.
    print(f"\n[{time.time()-t0:.1f}s] === EVAL (score + report) ===")
    from heinrich.eval.run import run_pipeline
    db = SignalDB(db_path) if db_path else SignalDB(db_file)
    report = run_pipeline(
        model=model,
        prompts=prompts,
        scorers=scorers,
        conditions=["auto"],
        output=output,
        db_path=db_file,
        max_prompts=max_prompts,
        timeout=effective_timeout,
    )

    # Print summary
    print(f"\n[{time.time()-t0:.1f}s] === PIPELINE COMPLETE ===")
    print(f"  Prompts: {report.get('n_prompts', 0)}")
    print(f"  Generations: {report.get('n_generations', 0)}")
    print(f"  Scores: {report.get('n_scores', 0)}")

    per_condition = report.get("per_condition", [])
    if per_condition:
        print(f"  Per-condition breakdown:")
        for entry in per_condition:
            print(f"    {entry.get('condition', '?')}: "
                  f"scorer={entry.get('scorer', '?')}, "
                  f"label={entry.get('label', '?')}, "
                  f"count={entry.get('count', 0)}")

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the full heinrich pipeline: discover -> attack -> eval -> report",
    )
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--prompts", required=True, help="Comma-separated prompt set names")
    parser.add_argument("--scorers", required=True, help="Comma-separated scorer names")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file path")
    parser.add_argument("--db", default=None, help="Database path")
    parser.add_argument("--max-prompts", type=int, default=None, help="Max prompts per set")
    parser.add_argument("--timeout", type=int, default=1800, help="Subprocess timeout")
    parser.add_argument("--skip-discover", action="store_true", help="Skip discover step")
    parser.add_argument("--skip-attack", action="store_true", help="Skip attack step")
    args = parser.parse_args(argv)

    run_full_pipeline(
        model=args.model,
        prompts=args.prompts.split(","),
        scorers=args.scorers.split(","),
        output=args.output,
        db_path=args.db,
        max_prompts=args.max_prompts,
        timeout=args.timeout,
        skip_discover=args.skip_discover,
        skip_attack=args.skip_attack,
    )


if __name__ == "__main__":
    main()
