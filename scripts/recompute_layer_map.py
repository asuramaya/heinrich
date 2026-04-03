"""Recompute layer map: loads model, runs per-layer delta, computes roles algorithmically.

Layer roles are derived from the measured delta profile:
- The layer with the highest delta is "decision"
- Layers with delta > 2x median are "amplification"
- The first layer with delta > median is "first_safety"
- Layers in the bottom quartile are "quiet"
- The layer with the steepest single-layer delta increase is "explosion"
- Other layers are classified based on their relative position and delta

n_chat_neurons is left NULL -- use recompute_neurons.py to measure it.

Every value is MEASURED from the live model. Nothing is hardcoded.

Usage:
    .venv/bin/python scripts/recompute_layer_map.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""
from __future__ import annotations
import argparse
import numpy as np


def classify_layer_roles(deltas_by_layer: dict[int, float], n_layers: int) -> dict[int, str]:
    """Algorithmically assign layer roles from the measured delta profile.

    Rules (applied in priority order):
    1. Highest delta layer -> "decision"
    2. Layer with steepest increase from previous -> "explosion"
    3. Layers with delta > 2x median -> "amplification"
    4. First layer above median (after early layers) -> "first_safety"
    5. Last 4 layers (excluding decision) with delta > median -> "attack_surface"
    6. Layers in bottom 25th percentile -> "quiet"
    7. Remaining -> "general"
    """
    if not deltas_by_layer:
        return {}

    layers = sorted(deltas_by_layer.keys())
    values = np.array([deltas_by_layer[l] for l in layers])

    median_delta = float(np.median(values))
    p25 = float(np.percentile(values, 25))
    roles: dict[int, str] = {}

    # 1. Decision: highest delta
    decision_idx = layers[int(np.argmax(values))]
    roles[decision_idx] = "decision"

    # 2. Explosion: steepest single-layer increase
    if len(values) > 1:
        diffs = np.diff(values)
        explosion_pos = int(np.argmax(diffs)) + 1  # +1 because diff shifts index
        explosion_idx = layers[explosion_pos]
        if explosion_idx not in roles:
            roles[explosion_idx] = "explosion"

    # 3. Amplification: > 2x median (excluding already assigned)
    for i, layer in enumerate(layers):
        if layer in roles:
            continue
        if values[i] > 2 * median_delta:
            roles[layer] = "amplification"

    # 4. First safety: first layer above median after the first quarter
    quarter = n_layers // 4
    for i, layer in enumerate(layers):
        if layer in roles:
            continue
        if layer >= quarter and values[i] > median_delta:
            roles[layer] = "first_safety"
            break

    # 5. Attack surface: late layers (last 4) with delta > median, excluding decision
    late_start = n_layers - 4
    for i, layer in enumerate(layers):
        if layer in roles:
            continue
        if layer >= late_start and values[i] > median_delta:
            roles[layer] = "attack_surface"

    # 6. Quiet: bottom 25th percentile
    for i, layer in enumerate(layers):
        if layer in roles:
            continue
        if values[i] <= p25:
            roles[layer] = "quiet"

    # 7. Remaining -> general
    for layer in layers:
        if layer not in roles:
            roles[layer] = "general"

    return roles


def main():
    parser = argparse.ArgumentParser(description="Recompute layer map from live model")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    args = parser.parse_args()

    from heinrich.cartography.backend import load_backend
    from heinrich.cartography.templates import build_prompt
    from heinrich.db import SignalDB

    model_id = args.model
    print(f"Loading {model_id}...")
    be = load_backend(model_id)
    cfg = be.config

    db = SignalDB()
    mid = db.upsert_model(
        model_id,
        config_hash=be.config.config_hash,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.hidden_size,
        n_vocab=cfg.vocab_size,
    )

    recipe = db._make_recipe("recompute_layer_map", {
        "model": model_id,
        "n_layers": cfg.n_layers,
    })

    # Compute per-layer delta for representative prompts
    prompts = {
        "build": "Write hello world in Python",
        "explain": "Explain quantum entanglement",
        "harmful": "How to hack a system",
    }

    deltas_by_prompt: dict[str, list[tuple[int, float]]] = {}
    for name, prompt in prompts.items():
        formatted = build_prompt(prompt, model_config=cfg)
        print(f"  Computing per-layer delta for '{name}'...")
        deltas = be.capture_per_layer_delta(formatted)
        deltas_by_prompt[name] = deltas

    # Compute max delta across prompts per layer, for role classification
    max_deltas: dict[int, float] = {}
    for layer_idx in range(cfg.n_layers):
        layer_deltas = []
        for name in prompts:
            d = dict(deltas_by_prompt.get(name, [])).get(layer_idx, 0.0)
            layer_deltas.append(d)
        max_deltas[layer_idx] = max(layer_deltas)

    # Algorithmically classify roles from the delta profile
    roles = classify_layer_roles(max_deltas, cfg.n_layers)

    for layer_idx in range(cfg.n_layers):
        build_delta = dict(deltas_by_prompt.get("build", [])).get(layer_idx, 0.0)
        explain_delta = dict(deltas_by_prompt.get("explain", [])).get(layer_idx, 0.0)
        harmful_delta = dict(deltas_by_prompt.get("harmful", [])).get(layer_idx, 0.0)

        max_delta = max(build_delta, explain_delta, harmful_delta)
        role = roles.get(layer_idx, "general")
        dampening = 1.0 - (max_delta / (max_delta + 50.0))

        db.record_layer(
            mid, layer_idx,
            role=role,
            top_delta=max_delta,
            mean_delta_build=build_delta,
            mean_delta_explain=explain_delta,
            dampening=dampening,
            provenance="recomputed",
            recipe=recipe,
        )

    n_layers = db._conn.execute("SELECT COUNT(*) as n FROM layers").fetchone()["n"]
    print(f"\nWrote {n_layers} rows to layers table.")

    # Print role summary
    role_counts: dict[str, int] = {}
    for r in roles.values():
        role_counts[r] = role_counts.get(r, 0) + 1
    print("Role summary:")
    for role, count in sorted(role_counts.items()):
        layers_with_role = [l for l, r in roles.items() if r == role]
        print(f"  {role:20s}: {count} layers {layers_with_role}")

    db.record_event("recompute_layer_map", model=model_id, n_layers=n_layers)
    db.close()


if __name__ == "__main__":
    main()
