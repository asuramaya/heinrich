"""Orchestrator: run all recompute scripts in sequence.

Usage:
    .venv/bin/python scripts/recompute_all.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Run all recompute scripts")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    args = parser.parse_args()

    scripts = [
        "recompute_layer_map",
        "recompute_neurons",
        "recompute_sharts",
        "recompute_directions",
        "recompute_basins",
        "recompute_interpolation",
        "recompute_heads",
        "recompute_evaluations",
    ]

    t0 = time.time()
    failed = []

    for script in scripts:
        print(f"\n{'='*60}")
        print(f"Running {script}...")
        print(f"{'='*60}")
        try:
            subprocess.check_call(
                [sys.executable, f"scripts/{script}.py", "--model", args.model]
            )
        except subprocess.CalledProcessError as e:
            print(f"FAILED: {script} exited with code {e.returncode}")
            failed.append(script)
        except FileNotFoundError:
            print(f"SKIPPED: scripts/{script}.py not found")
            failed.append(script)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Recompute complete in {elapsed:.0f}s")
    if failed:
        print(f"FAILED/SKIPPED: {', '.join(failed)}")
    else:
        print("All scripts succeeded.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
