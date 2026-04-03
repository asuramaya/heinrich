"""Recompute safety evaluations from the live model.

Runs the safety benchmark suite against the model and records results.
Classification uses content-based classify_response (not hardcoded categories).

Every value is MEASURED from the live model. Nothing is hardcoded.

Usage:
    .venv/bin/python scripts/recompute_evaluations.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""
from __future__ import annotations
import argparse


def main():
    parser = argparse.ArgumentParser(description="Recompute safety evaluations from live model")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--max-prompts", type=int, default=50,
                        help="Max prompts per dataset")
    args = parser.parse_args()

    from heinrich.cartography.backend import load_backend
    from heinrich.cartography.safetybench import run_safety_benchmark
    from heinrich.db import SignalDB

    model_id = args.model
    print(f"Loading {model_id}...")
    be = load_backend(model_id)
    cfg = be.config

    db = SignalDB()
    mid = db.upsert_model(model_id, config_hash=be.config.config_hash)

    recipe = db._make_recipe("recompute_evaluations", {
        "model": model_id,
        "max_prompts": args.max_prompts,
    })

    # Run the benchmark suite -- this measures actual model behavior
    print("Running safety benchmark...")
    reports = run_safety_benchmark(
        model=be.model,
        tokenizer=be.tokenizer,
        model_id=model_id,
        max_prompts_per_dataset=args.max_prompts,
        progress=True,
    )

    # If run_safety_benchmark requires model/tokenizer directly (not backend),
    # fall back to using be's internal model and tokenizer
    if not reports:
        print("Retrying with backend's model and tokenizer...")
        model_obj = getattr(be, 'model', None)
        tokenizer_obj = getattr(be, 'tokenizer', None)
        if model_obj is not None and tokenizer_obj is not None:
            reports = run_safety_benchmark(
                model=model_obj,
                tokenizer=tokenizer_obj,
                model_id=model_id,
                max_prompts_per_dataset=args.max_prompts,
                progress=True,
            )

    total_evals = 0
    for report in reports:
        exp_id = db.record_experiment(
            f"safetybench_{report.dataset}", mid,
            kind="evaluation",
            script="recompute_evaluations",
            n_evaluations=report.n_prompts,
            status="completed",
        )

        # Record aggregate evaluation
        db.record_evaluation(
            exp_id,
            dataset=report.dataset,
            attack="normal" if report.alpha == 0.0 else f"alpha={report.alpha}",
            alpha=report.alpha,
            refuse_prob=report.refusal_rate,
            comply_prob=report.compliance_rate,
            n_prompts=report.n_prompts,
            provenance="recomputed",
            recipe=recipe,
        )

        # Record per-category results
        for category, stats in report.by_category.items():
            db.record_evaluation(
                exp_id,
                dataset=report.dataset,
                category=category,
                attack="normal" if report.alpha == 0.0 else f"alpha={report.alpha}",
                refuse_prob=stats.get("refusal_rate", 0.0),
                comply_prob=stats.get("compliance_rate", 0.0),
                n_prompts=stats.get("count", 0),
                provenance="recomputed",
                recipe=recipe,
            )

        total_evals += report.n_prompts
        print(f"  {report.dataset}: refused={report.refusal_rate:.1%} "
              f"complied={report.compliance_rate:.1%} ({report.n_prompts} prompts)")

    n_evals = db._conn.execute("SELECT COUNT(*) as n FROM evaluations WHERE recipe IS NOT NULL").fetchone()["n"]
    print(f"\nWrote {n_evals} evaluation rows from {len(reports)} reports.")
    db.record_event("recompute_evaluations", model=model_id,
                    n_evaluations=total_evals, n_reports=len(reports))
    db.close()


if __name__ == "__main__":
    main()
