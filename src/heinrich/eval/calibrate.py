"""Per-scorer signal distributions — descriptive, not evaluative.

No FPR/FNR. No assumed ground truth. Each scorer's label distribution
is reported per condition so the reader can see the shape of the stack.

Calibration as a concept is replaced by signal isolation: each scorer
stays in its own lane, the report presents them side by side, and
interpretation stays with the reader.

Usage:
    python -m heinrich.eval.calibrate --db path
"""
from __future__ import annotations

import argparse
import json
from collections import Counter


def describe_scorers(db) -> dict[str, dict]:
    """Compute descriptive statistics for every scorer in the DB.

    For each scorer, returns:
      - label distribution per condition
      - total count
      - number of distinct labels

    No evaluative metrics (no FPR/FNR). The distributions describe
    what each scorer says, not whether it's "right."
    """
    # Get all generations with their condition
    rows = db._conn.execute(
        "SELECT s.scorer, s.label, g.condition "
        "FROM scores s "
        "JOIN generations g ON g.id = s.generation_id "
        "WHERE s.label IS NOT NULL AND s.label != 'error' "
        "ORDER BY s.scorer, g.condition"
    ).fetchall()

    # Group by scorer
    by_scorer: dict[str, list[dict]] = {}
    for r in rows:
        by_scorer.setdefault(r["scorer"], []).append(dict(r))

    results: dict[str, dict] = {}
    for scorer, score_rows in sorted(by_scorer.items()):
        # Per-condition label distribution
        by_condition: dict[str, Counter] = {}
        for r in score_rows:
            cond = r["condition"]
            by_condition.setdefault(cond, Counter())[r["label"]] += 1

        # Overall label distribution
        overall = Counter(r["label"] for r in score_rows)

        results[scorer] = {
            "total": len(score_rows),
            "n_labels": len(overall),
            "overall_distribution": dict(overall),
            "per_condition": {
                cond: dict(dist) for cond, dist in sorted(by_condition.items())
            },
        }

    return results


def describe_context_dependence(
    db,
    model_id: str,
    *,
    n_prefixes: int = 3,
    prefix_turns: int = 15,
    n_final: int = 15,
    backend=None,
) -> dict:
    """Measure how much the safety projection depends on conversation history vs the current turn.

    Builds n_prefixes long conversations, varies the final turn across n_final random prompts,
    and decomposes the projection variance into prefix vs final components.

    Returns dict with variance decomposition and the context length tested.
    """
    from heinrich.core.db import SignalDB
    from heinrich.cartography.templates import build_prompt
    import numpy as np

    if backend is None:
        raise ValueError("Backend required for context dependence measurement")

    cfg = backend.config

    # Load safety direction
    _db = db if db is not None else SignalDB()
    _mid = _db._conn.execute("SELECT id FROM models WHERE name = ?", (model_id,)).fetchone()
    if _mid is None:
        return {"error": f"Model {model_id} not found in DB"}
    mid = _mid["id"]

    dir_row = _db._conn.execute(
        "SELECT layer, vector_blob FROM directions WHERE model_id = ? AND name = ? AND layer <= ? ORDER BY stability DESC LIMIT 1",
        (mid, "safety", cfg.n_layers - 1),
    ).fetchone()
    if dir_row is None:
        return {"error": "No safety direction found"}

    safety_dir = np.frombuffer(dir_row["vector_blob"], dtype=np.float32)
    d_norm = np.linalg.norm(safety_dir)
    layer = dir_row["layer"]

    # Load prompts
    all_prompts = _db.get_prompts(limit=5000)
    if len(all_prompts) < n_prefixes * prefix_turns + n_final:
        return {"error": "Not enough prompts in DB"}

    import random
    rng = random.Random(42)
    rng.shuffle(all_prompts)

    def _mt(turns):
        return backend.tokenizer.apply_chat_template(
            [{"role": r, "content": c} for r, c in turns],
            tokenize=False, add_generation_prompt=True,
        )

    def _proj(text):
        fwd = backend.forward(text, return_residual=True, residual_layer=layer)
        if fwd.residual is None:
            return 0.0
        return float(np.dot(fwd.residual, safety_dir) / d_norm)

    # Build prefixes
    ptr = 0
    prefixes = []
    prefix_token_counts = []
    for _ in range(n_prefixes):
        turns = []
        for t in range(prefix_turns):
            p = all_prompts[ptr]; ptr += 1
            turns.append(("user", p["text"]))
            ctx = _mt(turns)
            resp = backend.generate(ctx, max_tokens=30)
            turns.append(("assistant", resp))
        prefixes.append(turns)
        prefix_token_counts.append(len(backend.tokenizer.encode(_mt(turns))))

    # Final turns
    finals = [all_prompts[ptr + i] for i in range(n_final)]
    ptr += n_final

    # Measure
    results = np.zeros((n_prefixes, n_final))
    for p_idx, prefix in enumerate(prefixes):
        for f_idx, fp in enumerate(finals):
            turns = list(prefix) + [("user", fp["text"])]
            ctx = _mt(turns)
            results[p_idx, f_idx] = _proj(ctx)

    total_var = float(np.var(results))
    var_prefix = float(np.var(results.mean(axis=1)))
    var_final = float(np.var(results.mean(axis=0)))
    var_inter = max(0.0, total_var - var_prefix - var_final)

    return {
        "layer": layer,
        "n_prefixes": n_prefixes,
        "n_final": n_final,
        "prefix_turns": prefix_turns,
        "mean_prefix_tokens": int(np.mean(prefix_token_counts)),
        "total_variance": round(total_var, 2),
        "prefix_variance": round(var_prefix, 2),
        "final_variance": round(var_final, 2),
        "interaction_variance": round(var_inter, 2),
        "prefix_pct": round(var_prefix / total_var * 100, 1) if total_var > 0 else 0,
        "final_pct": round(var_final / total_var * 100, 1) if total_var > 0 else 0,
        "interaction_pct": round(var_inter / total_var * 100, 1) if total_var > 0 else 0,
    }


# Keep backward-compatible name so callers that import calibrate_all
# still work, but it now computes descriptive stats only and does NOT
# write to the calibration table.
def calibrate_all(db) -> dict[str, dict]:
    """Backward-compatible entry point. Returns descriptive scorer stats.

    No longer writes to the calibration table — signal isolation means
    each scorer's output stays in the scores table, undistorted.
    """
    return describe_scorers(db)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Describe scorer signal distributions (no ground-truth calibration)"
    )
    parser.add_argument("--db", default=None, help="Path to SignalDB")
    args = parser.parse_args()

    from heinrich.core.db import SignalDB

    db = SignalDB(args.db) if args.db else SignalDB()
    results = describe_scorers(db)
    for scorer, info in results.items():
        print(f"\n{scorer}: {info['total']} scores, {info['n_labels']} distinct labels")
        print(f"  overall: {info['overall_distribution']}")
        for cond, dist in info["per_condition"].items():
            print(f"  {cond}: {dist}")
    db.close()
