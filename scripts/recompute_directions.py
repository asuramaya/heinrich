"""Recompute behavioral directions from the live model.

Computes safety, language (en vs zh), and chat directions at multiple layers.
Records stability, effect_size, and per-layer direction gap.

Every value is MEASURED from the live model. Nothing is hardcoded.

Usage:
    .venv/bin/python scripts/recompute_directions.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""
from __future__ import annotations
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Recompute behavioral directions from live model")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    args = parser.parse_args()

    from heinrich.cartography.backend import load_backend
    from heinrich.cartography.templates import build_prompt
    from heinrich.cartography.directions import find_direction
    from heinrich.cartography.prompt_bank import (
        HARMFUL_PROMPTS, BENIGN_PROMPTS, train_test_split,
    )
    from heinrich.db import SignalDB

    model_id = args.model
    print(f"Loading {model_id}...")
    be = load_backend(model_id)
    cfg = be.config

    db = SignalDB()
    mid = db.upsert_model(model_id, config_hash=be.config.config_hash)

    recipe = db._make_recipe("recompute_directions", {
        "model": model_id,
        "layers": str(cfg.all_layers),
    })

    # Use train/test split from prompt_bank for stability measurement
    split = train_test_split(seed=42)

    # ==========================
    # 1. Safety direction
    # ==========================
    print("\n=== Safety direction ===")
    harmful_prompts = [build_prompt(t, model_config=cfg) for t in split.train_harmful]
    benign_prompts = [build_prompt(t, model_config=cfg) for t in split.train_benign]

    # Test set for stability
    test_harmful = [build_prompt(t, model_config=cfg) for t in split.test_harmful]
    test_benign = [build_prompt(t, model_config=cfg) for t in split.test_benign]

    layers_to_probe = cfg.all_layers
    print(f"Capturing residual states at {len(layers_to_probe)} layers...")

    harmful_states = be.capture_residual_states(harmful_prompts, layers=layers_to_probe)
    benign_states = be.capture_residual_states(benign_prompts, layers=layers_to_probe)

    # Test states for stability
    test_harmful_states = be.capture_residual_states(test_harmful, layers=layers_to_probe)
    test_benign_states = be.capture_residual_states(test_benign, layers=layers_to_probe)

    for layer in layers_to_probe:
        dr = find_direction(
            harmful_states[layer], benign_states[layer],
            name="safety", layer=layer,
        )
        # Stability: accuracy on held-out test set
        test_pos_proj = test_harmful_states[layer] @ dr.direction
        test_neg_proj = test_benign_states[layer] @ dr.direction
        threshold = (test_pos_proj.mean() + test_neg_proj.mean()) / 2
        test_correct = np.sum(test_pos_proj > threshold) + np.sum(test_neg_proj <= threshold)
        stability = test_correct / (len(test_pos_proj) + len(test_neg_proj))

        db.record_direction(
            mid, "safety", layer,
            stability=float(stability),
            effect_size=float(dr.effect_size),
            provenance="recomputed",
            recipe=recipe,
        )
        db.record_layer(
            mid, layer,
            direction_gap_safety=float(dr.mean_gap),
            recipe=recipe,
        )
        if layer in cfg.safety_layers:
            print(f"  L{layer:2d}: gap={dr.mean_gap:.1f}, stability={stability:.2f}, "
                  f"effect_size={dr.effect_size:.2f}, train_acc={dr.separation_accuracy:.2f}")

    # ==========================
    # 2. Language direction (en vs zh)
    # ==========================
    print("\n=== Language direction (en vs zh) ===")
    en_texts = [
        "The weather today is nice",
        "Hello, how are you?",
        "Dogs are popular pets",
        "Science explains the natural world",
        "History teaches us about the past",
    ]
    zh_texts = [
        "今天天气很好",
        "你好，最近怎么样？",
        "狗是很受欢迎的宠物",
        "科学解释了自然界的规律",
        "历史教会我们了解过去",
    ]
    en_prompts = [build_prompt(t, model_config=cfg) for t in en_texts]
    zh_prompts = [build_prompt(t, model_config=cfg) for t in zh_texts]

    en_states = be.capture_residual_states(en_prompts, layers=layers_to_probe)
    zh_states = be.capture_residual_states(zh_prompts, layers=layers_to_probe)

    for layer in layers_to_probe:
        dr = find_direction(
            en_states[layer], zh_states[layer],
            name="en_vs_zh", layer=layer,
        )
        db.record_direction(
            mid, "en_vs_zh", layer,
            stability=float(dr.separation_accuracy),
            effect_size=float(dr.effect_size),
            provenance="recomputed",
            recipe=recipe,
        )
        db.record_layer(
            mid, layer,
            direction_gap_language=float(dr.mean_gap),
            recipe=recipe,
        )
        if layer in cfg.safety_layers:
            print(f"  L{layer:2d}: gap={dr.mean_gap:.1f}, acc={dr.separation_accuracy:.2f}, "
                  f"effect_size={dr.effect_size:.2f}")

    # ==========================
    # 3. Chat direction (chat format vs plain text)
    # ==========================
    print("\n=== Chat direction ===")
    plain_texts = [
        "Write hello world in Python",
        "Explain quantum entanglement",
        "What causes earthquakes?",
        "Tell me about the solar system",
        "How does photosynthesis work?",
    ]
    # Chat prompts: with chat template
    chat_prompts = [build_prompt(t, model_config=cfg) for t in plain_texts]
    # Plain prompts: raw text WITHOUT chat template (no build_prompt)
    plain_prompts = plain_texts

    chat_states = be.capture_residual_states(chat_prompts, layers=layers_to_probe)
    plain_states = be.capture_residual_states(plain_prompts, layers=layers_to_probe)

    for layer in layers_to_probe:
        dr = find_direction(
            chat_states[layer], plain_states[layer],
            name="chat", layer=layer,
        )
        # Validation: skip low-accuracy directions (they're garbage)
        if dr.separation_accuracy < 0.6:
            print(f"  WARNING: chat direction at L{layer} accuracy too low "
                  f"({dr.separation_accuracy:.2f}), skipping")
            continue
        db.record_direction(
            mid, "chat", layer,
            stability=float(dr.separation_accuracy),
            effect_size=float(dr.effect_size),
            provenance="recomputed",
            recipe=recipe,
        )
        if layer in cfg.safety_layers:
            print(f"  L{layer:2d}: gap={dr.mean_gap:.1f}, acc={dr.separation_accuracy:.2f}, "
                  f"effect_size={dr.effect_size:.2f}")

    n_dirs = db._conn.execute("SELECT COUNT(*) as n FROM directions WHERE model_id = ?", (mid,)).fetchone()["n"]
    print(f"\nWrote {n_dirs} direction rows.")
    db.record_event("recompute_directions", model=model_id, n_directions=n_dirs)
    db.close()


if __name__ == "__main__":
    main()
