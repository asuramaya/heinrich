#!/usr/bin/env python
"""Run corrected direction finding on 3 instruct models and save results.

Uses the full prompt bank (40 train + 10 test per class) instead of the
original 8 hardcoded prompts. Measures train accuracy, test accuracy,
and direction stability for each model.

Usage:
    cd /Users/asuramaya/Code/heinrich
    .venv/bin/python scripts/corrected_directions.py
"""
from __future__ import annotations
import json
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from heinrich.cartography.discover import discover_profile


MODELS = [
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Phi-3-mini-4k-instruct-4bit",
]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corrected_directions.json")


def main():
    results = {}

    for model_id in MODELS:
        short_name = model_id.split("/")[-1]
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  Profiling: {model_id}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        try:
            profile = discover_profile(model_id, force=True, progress=True)

            primary = None
            if profile.safety_layers:
                primary = profile.safety_layers[0]

            results[short_name] = {
                "model_id": model_id,
                "model_type": profile.model_type,
                "n_layers": profile.n_layers,
                "hidden_size": profile.hidden_size,
                "primary_safety_layer": profile.primary_safety_layer,
                "safety_direction_accuracy": round(profile.safety_direction_accuracy, 4),
                "train_accuracy": round(primary.train_accuracy, 4) if primary else None,
                "test_accuracy": round(primary.test_accuracy, 4) if primary else None,
                "direction_stability": round(primary.direction_stability, 4) if primary else None,
                "n_train_prompts": primary.n_train_prompts if primary else 0,
                "effect_size": primary.effect_size if primary else None,
                "mean_gap": primary.mean_gap if primary else None,
                "n_anomalous_neurons": profile.n_anomalous_neurons,
                "top_neuron": primary.top_neuron if primary else None,
                "baseline_refuse_prob": profile.baseline_refuse_prob,
                "baseline_comply_prob": profile.baseline_comply_prob,
                "discovery_time_s": profile.discovery_time_s,
            }

            print(f"\n  Results for {short_name}:", file=sys.stderr)
            print(f"    Primary safety layer: L{profile.primary_safety_layer}", file=sys.stderr)
            print(f"    Train accuracy:       {results[short_name]['train_accuracy']}", file=sys.stderr)
            print(f"    Test accuracy:        {results[short_name]['test_accuracy']}", file=sys.stderr)
            print(f"    Direction stability:  {results[short_name]['direction_stability']}", file=sys.stderr)
            print(f"    Effect size:          {results[short_name]['effect_size']}", file=sys.stderr)

        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            results[short_name] = {"model_id": model_id, "error": str(e)}

    # Summary comparison
    print(f"\n{'='*60}", file=sys.stderr)
    print("  COMPARISON SUMMARY", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  {'Model':<40} {'Layer':>5} {'Train':>6} {'Test':>6} {'Stab':>6}", file=sys.stderr)
    print(f"  {'-'*40} {'-'*5} {'-'*6} {'-'*6} {'-'*6}", file=sys.stderr)
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:<40} ERROR: {r['error']}", file=sys.stderr)
        else:
            print(f"  {name:<40} L{r['primary_safety_layer']:>3} "
                  f"{r['train_accuracy']:>6.2f} {r['test_accuracy']:>6.2f} "
                  f"{r['direction_stability']:>6.4f}", file=sys.stderr)

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
