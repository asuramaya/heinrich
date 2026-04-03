"""Score generations with a specific scorer. Reads/writes DB.

Usage as subprocess:
    python -m heinrich.eval.score --scorer word_match --db path
"""
from __future__ import annotations

import argparse
import importlib
import pkgutil
import time
import traceback

from heinrich.eval.scorers.base import Scorer


def _discover_scorers() -> dict[str, type]:
    """Auto-discover scorer classes in the scorers package.

    Walks all modules in heinrich.eval.scorers/, finds subclasses of Scorer,
    and registers them by their ``name`` attribute.  Adding a new scorer is:
    drop a .py file in scorers/, define a class with ``name = "my_scorer"``.
    """
    from heinrich.eval import scorers as scorers_pkg

    registry: dict[str, type] = {}
    for _, mod_name, _ in pkgutil.iter_modules(scorers_pkg.__path__):
        mod = importlib.import_module(f".scorers.{mod_name}", package="heinrich.eval")
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Scorer)
                and attr is not Scorer
            ):
                instance = attr()
                registry[instance.name] = attr
    return registry


SCORER_REGISTRY = _discover_scorers()


def load_scorer(name: str, model_id: str | None = None):
    """Load a scorer by name from the auto-discovered registry."""
    if name not in SCORER_REGISTRY:
        raise ValueError(f"Unknown scorer: {name}. Available: {list(SCORER_REGISTRY)}")
    cls = SCORER_REGISTRY[name]
    # Scorers that need a model accept model_id as first arg
    if model_id and cls.requires_model:
        try:
            return cls(model_id=model_id)
        except TypeError:
            return cls()
    return cls()


def score_all(db, scorer_name: str, model_id: str | None = None) -> int:
    """Score all unscored generations with the given scorer.

    Returns count of newly scored generations.  If a scorer raises an
    exception on a particular generation, a score row with label='error'
    is written so the generation is not retried endlessly.
    """
    scorer = load_scorer(scorer_name, model_id=model_id)
    unscored = db.get_unscored_generations(scorer_name)
    count = 0
    consecutive_errors = 0
    for gen in unscored:
        try:
            result = scorer.score(gen["prompt_text"], gen["generation_text"])
            db.record_score(
                gen["id"], scorer_name, result.label, result.confidence, result.raw_output
            )
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                # Scorer is broken (model won't load, missing auth, etc.).
                # Fail fast — don't write 900 error rows to the DB.
                raise RuntimeError(
                    f"Scorer {scorer_name!r} failed {consecutive_errors} times "
                    f"consecutively. Aborting ��� not writing error rows. "
                    f"Last error: {exc}"
                ) from exc
            # First 1-2 failures could be transient; skip without writing error rows.
        count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score generations with a scorer")
    parser.add_argument("--scorer", required=True, help="Scorer name (word_match, regex_harm)")
    parser.add_argument("--db", default=None, help="Path to SignalDB")
    parser.add_argument("--model", default=None, help="Model ID (required for model-based scorers)")
    args = parser.parse_args()

    from heinrich.core.db import SignalDB

    db = SignalDB(args.db) if args.db else SignalDB()
    n = score_all(db, args.scorer, model_id=args.model)
    print(f"Scored {n} generations with {args.scorer}")
    db.close()
