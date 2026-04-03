"""Run discovery on a model and write results to DB.

Usage:
    python -m heinrich.discover --model X --db path

Writes: models, directions, neurons, sharts, and conditions tables.
Records a 'clean' condition for subsequent pipeline steps.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def discover_to_db(
    model_id: str,
    db_path: str | None = None,
    *,
    force: bool = False,
    progress: bool = True,
) -> dict:
    """Run discovery and write all results to the pipeline DB.

    Returns a summary dict with counts of rows written.
    """
    from heinrich.core.db import SignalDB
    from heinrich.discover.profile import discover_profile

    db = SignalDB(db_path) if db_path else SignalDB()
    db_file = str(db.path)

    def log(msg: str) -> None:
        if progress:
            print(f"[discover] {msg}", file=sys.stderr)

    log(f"Starting discovery for {model_id}")
    t0 = time.time()

    # Run the full discovery profile (loads model, scans layers, etc.)
    profile = discover_profile(
        model_id,
        db_path=db_file,
        progress=progress,
        force=force,
    )

    # Register the model in the pipeline DB
    mid = db.upsert_model(model_id)

    # Write directions to DB
    n_directions = 0
    if profile.safety_direction is not None:
        layer = profile.safety_direction_layer
        # Store direction vector as blob
        vector_blob = profile.safety_direction.astype(np.float32).tobytes()
        db.record_direction(
            mid, "safety", layer,
            stability=profile.safety_direction_accuracy,
            effect_size=profile.safety_direction_accuracy,
            vector_blob=vector_blob,
            provenance="discover",
        )
        n_directions += 1
        log(f"Recorded safety direction at L{layer} "
            f"(accuracy={profile.safety_direction_accuracy:.2f})")

    # Write top neurons to DB
    n_neurons = 0
    for entry in profile.top_safety_neurons[:20]:
        db.record_neuron(
            mid, entry["layer"], entry["neuron"],
            max_z=entry["z_score"],
            category="safety",
            provenance="discover",
        )
        n_neurons += 1
    if n_neurons:
        log(f"Recorded {n_neurons} safety neurons")

    # Write sharts to DB
    n_sharts = 0
    for shart in profile.top_sharts:
        # Token text to approximate token_id (use hash for uniqueness)
        token_id = hash(shart["token"]) % (2**31)
        db.record_shart(
            mid, token_id,
            token_text=shart["token"],
            max_z=shart["max_z"],
            n_anomalous_neurons=shart.get("n_anomalous", 0),
            category="discovered",
            provenance="discover",
        )
        n_sharts += 1
    if n_sharts:
        log(f"Recorded {n_sharts} sharts")

    # Always record a 'clean' condition
    db.record_condition(
        mid, "clean",
        kind="baseline",
        source="discover",
    )
    log("Recorded 'clean' condition")

    elapsed = round(time.time() - t0, 1)
    log(f"Discovery complete in {elapsed}s")

    summary = {
        "model": model_id,
        "model_id": mid,
        "n_directions": n_directions,
        "n_neurons": n_neurons,
        "n_sharts": n_sharts,
        "primary_safety_layer": profile.primary_safety_layer,
        "safety_direction_accuracy": profile.safety_direction_accuracy,
        "elapsed_s": elapsed,
    }

    db.close()
    return summary


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description="Run discovery on a model")
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--db", default=None, help="Database path")
    parser.add_argument("--force", action="store_true", help="Force re-discovery")
    args = parser.parse_args()

    result = discover_to_db(args.model, db_path=args.db, force=args.force)
    print(json.dumps(result, indent=2))
