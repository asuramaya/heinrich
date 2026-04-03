"""Generate evaluation report from DB. Pure SQL, no model loading.

Usage:
    python -m heinrich.eval.report --db path --output report.json
"""
from __future__ import annotations

import argparse
import json


def _count_prompts(db) -> int:
    row = db._conn.execute("SELECT COUNT(*) as cnt FROM prompts").fetchone()
    return row["cnt"]


def _count_generations(db, model_id: int | None = None) -> int:
    if model_id is not None:
        row = db._conn.execute(
            "SELECT COUNT(*) as cnt FROM generations WHERE model_id = ?",
            (model_id,),
        ).fetchone()
    else:
        row = db._conn.execute("SELECT COUNT(*) as cnt FROM generations").fetchone()
    return row["cnt"]


def _count_scores(db, model_id: int | None = None) -> int:
    if model_id is not None:
        row = db._conn.execute(
            "SELECT COUNT(*) as cnt FROM scores s "
            "JOIN generations g ON g.id = s.generation_id "
            "WHERE g.model_id = ?",
            (model_id,),
        ).fetchone()
    else:
        row = db._conn.execute("SELECT COUNT(*) as cnt FROM scores").fetchone()
    return row["cnt"]


def _model_info(db, model_id: int | None = None) -> dict | None:
    if model_id is not None:
        row = db._conn.execute(
            "SELECT * FROM models WHERE id = ?", (model_id,)
        ).fetchone()
        return dict(row) if row else None
    # Return first model if no ID specified
    row = db._conn.execute("SELECT * FROM models ORDER BY id LIMIT 1").fetchone()
    return dict(row) if row else None


def _per_category_breakdown(db, model_id: int | None = None) -> list[dict]:
    """GROUP BY prompt_category, scorer, label.

    Shows which harm categories each scorer flags.
    """
    clauses = []
    params: list = []
    if model_id is not None:
        clauses.append("g.model_id = ?")
        params.append(model_id)
    where = (" AND " + " AND ".join(clauses)) if clauses else ""

    rows = db._conn.execute(
        f"SELECT g.prompt_category as category, s.scorer, s.label, "
        f"COUNT(*) as count "
        f"FROM generations g "
        f"JOIN scores s ON s.generation_id = g.id "
        f"WHERE g.prompt_category IS NOT NULL{where} "
        f"GROUP BY g.prompt_category, s.scorer, s.label "
        f"ORDER BY g.prompt_category, s.scorer, s.label",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def _per_condition_breakdown(db, model_id: int | None = None) -> list[dict]:
    """GROUP BY condition, scorer -- with separate formats for judges vs measurements.

    Judge scorers (labels containing ':'): report safe/unsafe/ambiguous counts.
    Measurement scorers: report a labels distribution dict.
    """
    clauses = []
    params: list = []
    if model_id is not None:
        clauses.append("g.model_id = ?")
        params.append(model_id)
    where = (" AND " + " AND ".join(clauses)) if clauses else ""

    rows = db._conn.execute(
        f"SELECT g.condition, s.scorer, s.label, COUNT(*) as count "
        f"FROM generations g "
        f"JOIN scores s ON s.generation_id = g.id "
        f"WHERE 1=1{where} "
        f"GROUP BY g.condition, s.scorer, s.label "
        f"ORDER BY g.condition, s.scorer, s.label",
        params,
    ).fetchall()

    # Group by (condition, scorer), then format based on judge vs measurement
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r["condition"], r["scorer"])
        grouped.setdefault(key, []).append(dict(r))

    result = []
    for (condition, scorer), label_rows in sorted(grouped.items()):
        is_judge = any(":" in (lr.get("label") or "") for lr in label_rows)

        if is_judge:
            # Judge scorer: extract safe/unsafe/ambiguous counts from prefixed labels
            safe = 0
            unsafe = 0
            ambiguous = 0
            for lr in label_rows:
                label = (lr.get("label") or "").lower()
                if ":unsafe" in label:
                    unsafe += lr["count"]
                elif ":safe" in label:
                    safe += lr["count"]
                else:
                    ambiguous += lr["count"]
            result.append({
                "condition": condition,
                "scorer": scorer,
                "safe": safe,
                "unsafe": unsafe,
                "ambiguous": ambiguous,
            })
        else:
            # Measurement scorer: show label distribution
            labels = {}
            for lr in label_rows:
                if lr.get("label"):
                    labels[lr["label"]] = lr["count"]
            result.append({
                "condition": condition,
                "scorer": scorer,
                "labels": labels,
            })

    return result


def build_report(db, model_id: int | None = None, backend=None, model_name: str | None = None) -> dict:
    """Build a full evaluation report from DB data.

    Returns a dict suitable for JSON serialization.
    No calibration — each scorer's signal stays isolated. The report
    presents raw distributions and disagreements; interpretation is
    the reader's job.
    """
    from heinrich.eval.calibrate import describe_scorers, describe_context_dependence

    return {
        "model": _model_info(db, model_id),
        "n_prompts": _count_prompts(db),
        "n_generations": _count_generations(db, model_id),
        "n_scores": _count_scores(db, model_id),
        "score_matrix": db.query_score_matrix(model_id=model_id),
        "scorer_distributions": describe_scorers(db),
        "disagreements": db.query_disagreements(model_id=model_id),
        "per_category": _per_category_breakdown(db, model_id),
        "per_condition": _per_condition_breakdown(db, model_id),
        "context_dependence": (
            describe_context_dependence(db, model_name, backend=backend)
            if backend is not None and model_name is not None
            else None
        ),
    }


def _json_safe(obj):
    """Recursively convert *obj* to JSON-serializable types.

    Raises TypeError for unknown types instead of silently converting
    via ``default=str``.
    """
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation report from DB")
    parser.add_argument("--db", default=None, help="Path to SignalDB")
    parser.add_argument("--output", default="report.json", help="Output JSON file path")
    parser.add_argument("--model-id", type=int, default=None, help="Filter by model ID")
    args = parser.parse_args()

    from heinrich.core.db import SignalDB

    db = SignalDB(args.db) if args.db else SignalDB()
    report = build_report(db, model_id=args.model_id)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2)
    print(f"Report written to {args.output}")
    print(f"  Prompts: {report['n_prompts']}")
    print(f"  Generations: {report['n_generations']}")
    print(f"  Scores: {report['n_scores']}")
    print(f"  Disagreements: {len(report['disagreements'])}")
    db.close()
