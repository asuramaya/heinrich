"""Generate model outputs for all prompts x conditions. Store in DB.

Usage as subprocess:
    python -m heinrich.eval.generate --model X --db path --conditions clean
"""
from __future__ import annotations

import argparse
import re
import time

# Valid condition values. "clean" is the baseline. Steering conditions use
# the prefix "steer_" followed by a float (e.g. "steer_-0.15").
# Framing conditions use the prefix "framing_" followed by a framing name.
VALID_CONDITIONS = {"clean", "steer", "framing_forensic", "framing_debug", "framing_academic"}

_STEER_RE = re.compile(r"^steer_-?\d+\.?\d*$")
_FRAMING_RE = re.compile(r"^framing_\w+$")


def _validate_condition(condition: str) -> None:
    """Raise ValueError if *condition* is not a recognized format."""
    if condition in VALID_CONDITIONS:
        return
    if _STEER_RE.match(condition):
        return
    if _FRAMING_RE.match(condition):
        return
    raise ValueError(
        f"Invalid condition {condition!r}. "
        f"Expected one of {sorted(VALID_CONDITIONS)}, "
        f"or 'steer_<float>' (e.g. 'steer_-0.15'), "
        f"or 'framing_<name>' (e.g. 'framing_forensic')."
    )


def generate_all(db, model_id: str, conditions: list[str] | None = None) -> int:
    """Generate outputs for all prompts in DB under given conditions.

    Returns total number of generation rows for this model.
    """
    from heinrich.cartography.backend import load_backend

    if conditions is None:
        conditions = ["clean"]

    # Validate all conditions up front before doing any work.
    for condition in conditions:
        _validate_condition(condition)

    be = load_backend(model_id)
    cfg = be.config
    mid = db.upsert_model(model_id, config_hash=cfg.config_hash)

    prompts = db.get_prompts()
    for prompt in prompts:
        for condition in conditions:
            # Check if already generated
            existing = db._conn.execute(
                "SELECT id FROM generations WHERE model_id=? AND prompt_text=? AND condition=?",
                (mid, prompt["text"], condition),
            ).fetchone()
            if existing:
                continue

            text = _generate_with_condition(be, cfg, prompt["text"], condition,
                                              db=db, model_id=mid)
            db.record_generation(
                mid,
                prompt["id"],
                prompt["text"],
                condition,
                text,
                prompt_source=prompt.get("source"),
                prompt_category=prompt.get("category"),
            )

    return db._conn.execute(
        "SELECT COUNT(*) as cnt FROM generations WHERE model_id=?", (mid,)
    ).fetchone()["cnt"]


def _generate_with_condition(be, cfg, prompt_text: str, condition: str,
                              *, db=None, model_id: int | None = None) -> str:
    """Generate with a specific condition applied.

    build_prompt() returns a ready-to-tokenize string that already ends with
    the assistant turn prefix (e.g. "<|im_start|>assistant\\n" for ChatML).
    The model generates from the assistant position — no additional prefix
    is needed here.
    """
    from heinrich.cartography.templates import build_prompt

    # build_prompt appends the chat-format assistant_prefix automatically,
    # so the model generates from the assistant position in chat format.
    formatted = build_prompt(prompt_text, model_config=cfg)

    if condition == "clean":
        return be.generate(formatted, max_tokens=150)

    # Handle steering conditions: steer_-0.15 means steer with alpha=-0.15
    if _STEER_RE.match(condition):
        alpha = float(condition.split("_", 1)[1])
        return _generate_steered(be, cfg, formatted, alpha, db=db, model_id=model_id)

    # Fallback for unknown conditions
    return be.generate(formatted, max_tokens=150)


def _generate_steered(be, cfg, formatted_prompt: str, alpha: float,
                       *, db=None, model_id: int | None = None) -> str:
    """Generate with activation steering at the discovered safety layer.

    Reads the safety direction metadata from DB (layer), then recomputes
    the direction vector and steers at that layer.
    """
    import json as _json

    # Determine which layer to steer at
    layer = cfg.last_layer  # default

    if db is not None and model_id is not None:
        # Look up the condition params for the layer info
        conds = db.get_conditions(model_id)
        for c in conds:
            if c.get("params"):
                try:
                    params = _json.loads(c["params"])
                    if "layer" in params:
                        layer = params["layer"]
                        break
                except (ValueError, TypeError):
                    pass

        # Also try reading the direction table directly
        direction_row = db._conn.execute(
            "SELECT layer FROM directions WHERE model_id = ? AND name = 'safety' "
            "ORDER BY stability DESC LIMIT 1",
            (model_id,),
        ).fetchone()
        if direction_row:
            layer = direction_row["layer"]

    # Compute safety direction at this layer
    from heinrich.discover.directions import find_direction
    from heinrich.cartography.templates import build_prompt

    # Load direction-finding prompts from DB
    if db is None:
        from heinrich.core.db import SignalDB
        db = SignalDB()
    harmful_qs = [r["text"] for r in db.require_prompts(is_benign=False, min_count=3, limit=3)]
    benign_qs = [r["text"] for r in db.require_prompts(is_benign=True, min_count=3, limit=3)]
    harmful = [build_prompt(q, model_config=cfg) for q in harmful_qs]
    benign = [build_prompt(q, model_config=cfg) for q in benign_qs]

    states = be.capture_residual_states(harmful + benign, layers=[layer])
    if layer not in states:
        # Fallback: generate without steering
        return be.generate(formatted_prompt, max_tokens=150)

    sl = states[layer]
    dr = find_direction(sl[:len(harmful)], sl[len(harmful):], name="safety", layer=layer)

    # Steer: direction * mean_gap * alpha
    steer_dirs = {layer: (dr.direction, dr.mean_gap)}
    return be.generate(formatted_prompt, steer_dirs=steer_dirs, alpha=alpha, max_tokens=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model outputs for eval pipeline")
    parser.add_argument("--model", required=True, help="Model ID (e.g., Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--conditions", default="clean", help="Comma-separated conditions")
    parser.add_argument("--db", default=None, help="Path to SignalDB (default: ./data/heinrich.db)")
    args = parser.parse_args()

    from heinrich.core.db import SignalDB

    db = SignalDB(args.db) if args.db else SignalDB()
    conditions = args.conditions.split(",")
    n = generate_all(db, args.model, conditions)
    print(f"Generated {n} total outputs")
    db.close()
