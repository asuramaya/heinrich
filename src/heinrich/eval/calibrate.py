"""Calibrate each scorer's false positive rate on benign prompts.

Usage:
    python -m heinrich.eval.calibrate --db path

Reads benign generations + their scores from DB.
Computes FPR per scorer: fraction of benign generations scored as 'unsafe'.
Writes to calibration table.
"""
from __future__ import annotations

import argparse


def calibrate_all(db) -> dict[str, dict]:
    """Compute FPR (and rough FNR) for every scorer that has scored benign generations.

    FPR = n_unsafe_on_benign / n_benign_scored
    FNR = n_safe_on_harmful / n_harmful_scored  (less reliable — harmful prompts
          may legitimately be refused)

    Returns dict keyed by scorer name with fpr, fnr, n_benign, n_harmful.
    Also writes each row to the calibration table via db.record_calibration().
    """
    # 1. Get all benign generations (join generations to prompts via prompt_id)
    benign_gens = db.get_generations(is_benign=True)
    harmful_gens = db.get_generations(is_benign=False)

    benign_ids = [g["id"] for g in benign_gens]
    harmful_ids = [g["id"] for g in harmful_gens]

    # 2. Get all scores for benign and harmful generations
    benign_scores = db.get_scores(generation_ids=benign_ids) if benign_ids else []
    harmful_scores = db.get_scores(generation_ids=harmful_ids) if harmful_ids else []

    # 3. Group by scorer
    # For FPR: count how many benign generations each scorer labeled 'unsafe'
    scorer_benign: dict[str, dict] = {}  # scorer -> {n_total, n_unsafe}
    for s in benign_scores:
        scorer = s["scorer"]
        if scorer not in scorer_benign:
            scorer_benign[scorer] = {"n_total": 0, "n_unsafe": 0}
        scorer_benign[scorer]["n_total"] += 1
        if s["label"] == "unsafe":
            scorer_benign[scorer]["n_unsafe"] += 1

    # For FNR: count how many harmful generations each scorer labeled 'safe'
    scorer_harmful: dict[str, dict] = {}  # scorer -> {n_total, n_safe}
    for s in harmful_scores:
        scorer = s["scorer"]
        if scorer not in scorer_harmful:
            scorer_harmful[scorer] = {"n_total": 0, "n_safe": 0}
        scorer_harmful[scorer]["n_total"] += 1
        if s["label"] == "safe":
            scorer_harmful[scorer]["n_safe"] += 1

    # 4. Compute rates and write to DB
    all_scorers = set(scorer_benign) | set(scorer_harmful)
    results: dict[str, dict] = {}

    # We need a model_id for the calibration table. Use the model_id from
    # the first generation, or 0 if none exist.
    model_ids = set()
    for g in benign_gens + harmful_gens:
        model_ids.add(g["model_id"])

    for scorer in sorted(all_scorers):
        b = scorer_benign.get(scorer, {"n_total": 0, "n_unsafe": 0})
        h = scorer_harmful.get(scorer, {"n_total": 0, "n_safe": 0})

        fpr = b["n_unsafe"] / b["n_total"] if b["n_total"] > 0 else None
        fnr = h["n_safe"] / h["n_total"] if h["n_total"] > 0 else None

        entry = {
            "fpr": fpr,
            "fnr": fnr,
            "n_benign": b["n_total"],
            "n_harmful": h["n_total"],
        }
        results[scorer] = entry

        # Write one calibration row per (scorer, model_id) pair
        for mid in model_ids or {0}:
            db.record_calibration(
                scorer=scorer,
                model_id=mid,
                fpr=fpr,
                fnr=fnr,
                n_benign=b["n_total"],
                n_harmful=h["n_total"],
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate scorer FPR on benign prompts")
    parser.add_argument("--db", default=None, help="Path to SignalDB")
    args = parser.parse_args()

    from heinrich.core.db import SignalDB

    db = SignalDB(args.db) if args.db else SignalDB()
    results = calibrate_all(db)
    for scorer, info in results.items():
        print(f"{scorer}: FPR={info['fpr']}, FNR={info['fnr']}, "
              f"n_benign={info['n_benign']}, n_harmful={info['n_harmful']}")
    db.close()
