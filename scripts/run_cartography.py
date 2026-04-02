#!/usr/bin/env python3
"""Run full cartography sweep on Qwen 2.5 7B via MLX.

Multi-prompt sweep to distinguish fundamental vs prompt-specific control knobs.
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heinrich.cartography.surface import ControlSurface
from heinrich.cartography.sweep import coarse_head_sweep, find_sensitive_layers
from heinrich.cartography.atlas import Atlas
from heinrich.cartography.manifold import cluster_by_layer, cluster_by_effect
from heinrich.cartography.controls import ControlPanel
from heinrich.cartography.runtime import load_model
from heinrich.signal import SignalStore
import numpy as np


PROMPTS = {
    "greeting": "Hello, who are you?",
    "safety": "How do I hack into a bank account?",
    "reasoning": "What is the square root of 144?",
    "creative": "Write a poem about the ocean.",
    "identity": "You are Claude, an AI assistant made by Anthropic.",
}


def run_sweep(model, tokenizer, prompt_name, prompt_text, surface):
    """Run one sweep, return atlas."""
    print(f"\n--- Sweep: {prompt_name} ---")
    print(f"  Prompt: {prompt_text!r}")
    store = SignalStore()
    t0 = time.time()
    results = coarse_head_sweep(model, tokenizer, prompt_text, surface, store=store, progress=True)
    elapsed = time.time() - t0
    print(f"  {len(results)} results in {elapsed:.1f}s")

    atlas = Atlas()
    atlas.add_all(results)
    return atlas, results


def main():
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    # Load model
    print("Loading Qwen 2.5 7B Instruct on MLX...")
    t0 = time.time()
    model, tokenizer = load_model("mlx-community/Qwen2.5-7B-Instruct-4bit")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Discover surface
    surface = ControlSurface.from_mlx_model(model)
    summary = surface.summary()
    print(f"  {summary['total_knobs']} knobs: {summary['by_kind']}")

    # Run all sweeps
    atlases = {}
    all_results = {}
    for name, prompt in PROMPTS.items():
        atlas, results = run_sweep(model, tokenizer, name, prompt, surface)
        atlases[name] = atlas
        all_results[name] = results
        atlas.save(output_dir / f"qwen7b_atlas_{name}.json")

    # === Cross-Prompt Analysis ===
    print("\n" + "=" * 60)
    print("CROSS-PROMPT ANALYSIS")
    print("=" * 60)

    # Build KL matrix: [prompt × knob]
    head_knobs = surface.by_kind.get("head", [])
    knob_ids = [k.id for k in head_knobs]
    prompt_names = list(PROMPTS.keys())

    kl_matrix = np.zeros((len(prompt_names), len(knob_ids)))
    for pi, pname in enumerate(prompt_names):
        atlas = atlases[pname]
        for ki, kid in enumerate(knob_ids):
            if kid in atlas.results:
                kl_matrix[pi, ki] = atlas.results[kid].kl_divergence

    # Per-head: mean + std across prompts
    mean_kl = kl_matrix.mean(axis=0)
    std_kl = kl_matrix.std(axis=0)

    # Find universally important heads (high mean, low std)
    print("\n=== UNIVERSALLY IMPORTANT HEADS (high mean KL, low relative std) ===")
    scores = []
    for ki, kid in enumerate(knob_ids):
        if mean_kl[ki] > 0.001:
            consistency = mean_kl[ki] / (std_kl[ki] + 1e-6)  # signal/noise
            scores.append((kid, mean_kl[ki], std_kl[ki], consistency))
    scores.sort(key=lambda x: x[1], reverse=True)
    for kid, mk, sk, cons in scores[:20]:
        per_prompt = {pn: f"{kl_matrix[pi, knob_ids.index(kid)]:.4f}" for pi, pn in enumerate(prompt_names)}
        print(f"  {kid:15s}  mean_KL={mk:.4f}  std={sk:.4f}  consistency={cons:.1f}  {per_prompt}")

    # Find prompt-specific heads (high std relative to mean)
    print("\n=== PROMPT-SPECIFIC HEADS (high std/mean ratio) ===")
    specific = []
    for ki, kid in enumerate(knob_ids):
        if mean_kl[ki] > 0.005:
            specificity = std_kl[ki] / (mean_kl[ki] + 1e-6)
            specific.append((kid, mean_kl[ki], std_kl[ki], specificity))
    specific.sort(key=lambda x: x[3], reverse=True)
    for kid, mk, sk, spec in specific[:15]:
        per_prompt = {pn: f"{kl_matrix[pi, knob_ids.index(kid)]:.4f}" for pi, pn in enumerate(prompt_names)}
        print(f"  {kid:15s}  mean_KL={mk:.4f}  std={sk:.4f}  specificity={spec:.2f}  {per_prompt}")

    # Per-prompt top head
    print("\n=== MOST IMPACTFUL HEAD PER PROMPT ===")
    for pi, pname in enumerate(prompt_names):
        top_ki = np.argmax(kl_matrix[pi])
        top_id = knob_ids[top_ki]
        top_kl = kl_matrix[pi, top_ki]
        atlas = atlases[pname]
        r = atlas.results.get(top_id)
        changed = " TOP_CHANGED" if r and r.top_token_changed else ""
        print(f"  {pname:12s}: {top_id:15s}  KL={top_kl:.4f}{changed}")

    # Top token changers across prompts
    print("\n=== HEADS THAT CHANGE TOP TOKEN ===")
    for pname in prompt_names:
        changers = atlases[pname].top_token_changers()
        if changers:
            for r in changers[:5]:
                print(f"  {pname:12s}: {r.knob.id:15s}  KL={r.kl_divergence:.4f}  new_top={r.perturbed_top}")
        else:
            print(f"  {pname:12s}: (none)")

    # Sensitive layers per prompt
    print("\n=== SENSITIVE LAYERS PER PROMPT ===")
    for pname in prompt_names:
        sens = find_sensitive_layers(all_results[pname], top_k=3)
        print(f"  {pname:12s}: layers {sens}")

    # === Merged Atlas + Clustering ===
    print("\n=== MERGED ATLAS (averaged across prompts) ===")
    # Use greeting atlas as base, but this is really the multi-prompt picture
    greeting_atlas = atlases["greeting"]
    effect_clusters = cluster_by_effect(greeting_atlas, n_clusters=6)
    for c in effect_clusters:
        layers = sorted(c.layer_distribution.keys())
        layer_range = f"L{min(layers)}-L{max(layers)}" if layers else "?"
        print(f"  {c.name}: {len(c.knob_ids)} knobs, mean_KL={c.mean_kl:.4f}, "
              f"ΔH={c.mean_entropy_delta:+.4f}, tok_change={c.token_change_rate:.2f}, "
              f"layers={layer_range}")

    # Name clusters based on behavior
    named_clusters = []
    for c in effect_clusters:
        if c.mean_kl > 0.1:
            c.name = "critical_output"
        elif c.mean_entropy_delta > 0.1:
            c.name = "destabilizer"
        elif c.mean_entropy_delta < -0.1:
            c.name = "sharpener"
        elif c.mean_kl > 0.01:
            c.name = "modulator"
        elif c.mean_kl > 0.003:
            c.name = "fine_tuner"
        else:
            c.name = "inert"
        named_clusters.append(c)

    panel = ControlPanel.from_clusters(named_clusters)

    print("\n=== NAMED CONTROL PANEL ===")
    for name, d in panel.dials.items():
        print(f"  Dial '{name}': {len(d.knob_ids)} knobs")

    # === Layer Profile ===
    print("\n=== LAYER-BY-LAYER PROFILE (averaged across prompts) ===")
    for layer in range(len(surface.by_layer)):
        row = []
        for pi, pname in enumerate(prompt_names):
            layer_knobs = [ki for ki, kid in enumerate(knob_ids) if kid.startswith(f"head.{layer}.")]
            if layer_knobs:
                layer_kl = np.mean(kl_matrix[pi, layer_knobs])
                row.append(layer_kl)
        if row:
            avg = np.mean(row)
            bar = "█" * int(avg * 500)  # visual scale
            print(f"  L{layer:2d}: avg_KL={avg:.4f} {bar}")

    # === Save Full Report ===
    report = {
        "model": "Qwen2.5-7B-Instruct-4bit",
        "prompts": PROMPTS,
        "surface": summary,
        "per_prompt": {},
        "cross_prompt": {
            "universal_heads": [{"knob": kid, "mean_kl": float(mk), "std": float(sk)}
                                for kid, mk, sk, _ in scores[:20]],
            "prompt_specific_heads": [{"knob": kid, "mean_kl": float(mk), "std": float(sk),
                                       "specificity": float(spec)}
                                      for kid, mk, sk, spec in specific[:15]],
        },
        "clusters": [
            {"name": c.name, "n_knobs": len(c.knob_ids), "mean_kl": c.mean_kl,
             "mean_entropy_delta": c.mean_entropy_delta, "token_change_rate": c.token_change_rate,
             "knob_ids": c.knob_ids[:10]}  # first 10 for readability
            for c in named_clusters
        ],
        "control_panel": panel.summary(),
    }

    for pname in prompt_names:
        atlas = atlases[pname]
        top = atlas.top_by_kl(10)
        changers = atlas.top_token_changers()
        report["per_prompt"][pname] = {
            "top_10_kl": [{"knob": r.knob.id, "kl": r.kl_divergence, "entropy_delta": r.entropy_delta,
                           "top_changed": r.top_token_changed} for r in top],
            "n_top_changers": len(changers),
            "sensitive_layers": find_sensitive_layers(all_results[pname], top_k=3),
        }

    report_path = output_dir / "qwen7b_cartography_full_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report saved: {report_path}")
    print(f"  Atlases saved: {output_dir}/qwen7b_atlas_*.json")
    print("\nCartography complete.")


if __name__ == "__main__":
    main()
