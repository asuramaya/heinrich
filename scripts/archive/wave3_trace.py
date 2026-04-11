#!/usr/bin/env python3
"""Wave 3: Position-aware causal tracing + distributed ablation.

The right decomposition: (layer × position), not components.
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heinrich.cartography.trace import (
    causal_trace, full_spec, dims_spec, direction_spec,
    distributed_ablation,
)
from heinrich.cartography.directions import capture_residual_states, find_direction
from heinrich.cartography.neurons import scan_neurons
from heinrich.cartography.runtime import load_model
from heinrich.signal import SignalStore


def print_heatmap(result, max_layers=14):
    """Print ASCII heatmap."""
    layers_to_show = min(result.n_layers, max_layers)
    step = max(1, result.n_layers // layers_to_show)
    indices = list(range(0, result.n_layers, step))

    # Header
    header = "       " + "".join(f"{t[:6]:>7s}" for t in result.tokens[:result.n_positions])
    print(header)

    for li in indices:
        row = result.heatmap[li]
        cells = ""
        for v in row:
            if v > 0.5:
                cells += f"  {'█' * 5}"
            elif v > 0.2:
                cells += f"  {'▓' * 5}"
            elif v > 0.1:
                cells += f"  {'▒' * 5}"
            elif v > 0.05:
                cells += f"  {'░' * 5}"
            else:
                cells += f"  {'·' * 5}"
        # Get actual layer number
        actual_layer = li * (len(result.cells) // (result.n_layers * result.n_positions)) if result.cells else li
        # Find the layer number from cells
        layer_num = result.cells[li * result.n_positions].layer if li * result.n_positions < len(result.cells) else li
        print(f"  L{layer_num:2d}  {cells}")

    print(f"\n  Legend: █>0.5  ▓>0.2  ▒>0.1  ░>0.05  ·<0.05")


# ============================================================
# TRACE 1: English → Chinese (language decision)
# ============================================================
def trace_language(model, tokenizer):
    print("\n" + "="*70)
    print("TRACE 1: WHERE AND WHEN IS LANGUAGE DECIDED?")
    print("English → Chinese, full residual patch")
    print("="*70)

    result = causal_trace(
        model, tokenizer,
        clean_prompt="The weather today is",
        corrupt_prompt="今天的天气是",
        layers=list(range(0, 28, 2)),  # every other layer for speed
        progress=True,
    )

    print(f"\n  clean={result.clean_top!r}  corrupt={result.corrupt_top!r}  KL={result.kl_baseline:.2f}")
    print_heatmap(result)

    print(f"\n  Top 10 causal sites:")
    for c in result.top_sites(10):
        print(f"    L{c.layer:2d} P{c.position} ({c.token:>8s}): recovery={c.recovery:.3f}  "
              f"→{c.patched_top_token!r}  {'RECOVERS' if c.top_recovered else ''}")

    # Position summary
    print(f"\n  Mean recovery by position:")
    for pos, tok, rec in result.position_summary():
        bar = "█" * int(rec * 50)
        print(f"    P{pos} {tok:>10s}: {rec:.3f} {bar}")

    # Layer summary
    print(f"\n  Mean recovery by layer:")
    for layer, rec in result.layer_summary():
        bar = "█" * int(rec * 50)
        print(f"    L{layer:2d}: {rec:.3f} {bar}")

    return result


# ============================================================
# TRACE 2: Plain → Chat (mode decision)
# ============================================================
def trace_chat(model, tokenizer):
    print("\n" + "="*70)
    print("TRACE 2: WHERE AND WHEN IS CHAT MODE DECIDED?")
    print("Plain → Chat, full residual patch")
    print("="*70)

    # Match token count roughly by padding plain
    result = causal_trace(
        model, tokenizer,
        clean_prompt="Hello, how are you today?",
        corrupt_prompt="<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        layers=list(range(0, 28, 2)),
        progress=True,
    )

    print(f"\n  clean={result.clean_top!r}  corrupt={result.corrupt_top!r}  KL={result.kl_baseline:.2f}")
    print_heatmap(result)

    print(f"\n  Top 10 causal sites:")
    for c in result.top_sites(10):
        print(f"    L{c.layer:2d} P{c.position} ({c.token:>10s}): recovery={c.recovery:.3f}  "
              f"→{c.patched_top_token!r}  {'RECOVERS' if c.top_recovered else ''}")

    print(f"\n  Mean recovery by position:")
    for pos, tok, rec in result.position_summary():
        bar = "█" * int(rec * 50)
        print(f"    P{pos} {tok:>10s}: {rec:.3f} {bar}")

    return result


# ============================================================
# TRACE 3: Benign → Sensitive (topic sensitivity)
# ============================================================
def trace_sensitive(model, tokenizer):
    print("\n" + "="*70)
    print("TRACE 3: WHERE AND WHEN IS TOPIC SENSITIVITY ENCODED?")
    print("="*70)

    result = causal_trace(
        model, tokenizer,
        clean_prompt="The weather today is very",
        corrupt_prompt="On June 4th, 1989, in",
        layers=list(range(0, 28, 2)),
        progress=True,
    )

    print(f"\n  clean={result.clean_top!r}  corrupt={result.corrupt_top!r}  KL={result.kl_baseline:.2f}")
    print_heatmap(result)

    print(f"\n  Top 10 causal sites:")
    for c in result.top_sites(10):
        print(f"    L{c.layer:2d} P{c.position} ({c.token:>10s}): recovery={c.recovery:.3f}  "
              f"→{c.patched_top_token!r}  {'RECOVERS' if c.top_recovered else ''}")

    return result


# ============================================================
# TRACE 4: Direction-projected trace (language direction only)
# ============================================================
def trace_direction(model, tokenizer):
    print("\n" + "="*70)
    print("TRACE 4: LANGUAGE TRACE PROJECTED ONTO LANGUAGE DIRECTION")
    print("Does patching only the language direction component recover language?")
    print("="*70)

    # First find the language direction at multiple layers
    en = ["The weather today is", "Once upon a time", "The capital is",
          "Scientists discovered", "The most important"]
    zh = ["今天的天气是", "从前，有一个", "首都是",
          "科学家发现", "最重要的"]

    best_dir = None
    best_gap = 0
    for layer in [5, 15, 27]:
        states = capture_residual_states(model, tokenizer, en + zh, layers=[layer])
        d = find_direction(states[layer][:5], states[layer][5:], name="lang", layer=layer)
        print(f"  Language direction at L{layer}: acc={d.separation_accuracy:.2f} gap={d.mean_gap:.1f}")
        if abs(d.mean_gap) > best_gap:
            best_gap = abs(d.mean_gap)
            best_dir = d

    print(f"  Using direction at L{best_dir.layer}")

    result = causal_trace(
        model, tokenizer,
        clean_prompt="The weather today is",
        corrupt_prompt="今天的天气是",
        spec=direction_spec(best_dir.direction),
        layers=list(range(0, 28, 4)),  # fewer layers, this is slower
        progress=True,
    )

    print(f"\n  Direction-projected trace:")
    print_heatmap(result)

    print(f"\n  Top 5 causal sites (direction-projected):")
    for c in result.top_sites(5):
        print(f"    L{c.layer:2d} P{c.position} ({c.token:>8s}): recovery={c.recovery:.3f}")

    return result


# ============================================================
# DISTRIBUTED ABLATION: find the redundancy threshold
# ============================================================
def test_distributed_ablation(model, tokenizer):
    print("\n" + "="*70)
    print("DISTRIBUTED ABLATION: WHERE DOES CHAT BREAK?")
    print("="*70)

    chat = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"

    # Get neuron ranking at L27 (chat vs plain)
    chat_prompts = [chat, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"]
    plain_prompts = ["The capital of France is", "Hello, how are you?"]
    r = scan_neurons(model, tokenizer, chat_prompts, plain_prompts, 27, top_k=815)
    ranking = [p.neuron for p in r.selective_neurons]

    print(f"  Ranking {len(ranking)} neurons by |chat-plain| selectivity at L27")
    print(f"  Top 5: {[(p.neuron, f'{p.selectivity:+.1f}') for p in r.selective_neurons[:5]]}")

    # Ablate increasing numbers
    counts = [1, 5, 10, 25, 50, 100, 200, 400, 600, 815]
    results = distributed_ablation(model, tokenizer, chat, 27, ranking, test_counts=counts)

    baseline_text = results[0][1] if results else ""
    print(f"\n  Baseline (0 ablated): see normal output")
    for n, text in results:
        changed = " ← DIFFERENT" if text[:30] != results[0][1][:30] else ""
        print(f"    zero {n:4d} neurons: {text[:60]!r}{changed}")

    # Also test at L7
    print(f"\n  --- Same test at L7 ---")
    r7 = scan_neurons(model, tokenizer, chat_prompts, plain_prompts, 7, top_k=50)
    ranking7 = [p.neuron for p in r7.selective_neurons]
    counts7 = [1, 5, 10, 20, 50]
    results7 = distributed_ablation(model, tokenizer, chat, 7, ranking7, test_counts=counts7)
    for n, text in results7:
        changed = " ← DIFFERENT" if text[:30] != results7[0][1][:30] else ""
        print(f"    zero {n:4d} L7 neurons: {text[:60]!r}{changed}")

    # Test at L21
    print(f"\n  --- Same test at L21 ---")
    r21 = scan_neurons(model, tokenizer, chat_prompts, plain_prompts, 21, top_k=100)
    ranking21 = [p.neuron for p in r21.selective_neurons]
    counts21 = [1, 5, 10, 25, 50, 100]
    results21 = distributed_ablation(model, tokenizer, chat, 21, ranking21, test_counts=counts21)
    for n, text in results21:
        changed = " ← DIFFERENT" if text[:30] != results21[0][1][:30] else ""
        print(f"    zero {n:4d} L21 neurons: {text[:60]!r}{changed}")


def main():
    model, tokenizer = load_model("mlx-community/Qwen2.5-7B-4bit")

    # Traces (most important — the right decomposition)
    trace_language(model, tokenizer)
    trace_chat(model, tokenizer)
    trace_sensitive(model, tokenizer)
    trace_direction(model, tokenizer)

    # Distributed ablation (find the threshold)
    test_distributed_ablation(model, tokenizer)

    print("\n\nWAVE 3 COMPLETE.")


if __name__ == "__main__":
    main()
