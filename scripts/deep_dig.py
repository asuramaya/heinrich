#!/usr/bin/env python3
"""Use all new heinrich tools to dig into every remaining mystery.

Tools: oproj decomposition, neuron scanning, direction finding, activation patching.
"""
import sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heinrich.cartography.oproj import decompose_oproj, scan_all_layers
from heinrich.cartography.neurons import scan_neurons, scan_layers, capture_mlp_activations
from heinrich.cartography.directions import find_direction_suite, steer_with_direction, orthogonality_matrix, capture_residual_states, find_direction
from heinrich.cartography.patch import sweep_band_patches, sweep_layer_patches
from heinrich.cartography.steer import generate_steered
from heinrich.cartography.perturb import compute_baseline
from heinrich.inspect.self_analysis import _softmax
from heinrich.signal import SignalStore


def load():
    import mlx_lm
    print("Loading Qwen 2.5 7B base...")
    t0 = time.time()
    m, t = mlx_lm.load("mlx-community/Qwen2.5-7B-4bit")
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return m, t

# ============================================================
# 1. O-PROJ DECOMPOSITION — the real functional subspaces
# ============================================================
def dig_oproj(model, tokenizer):
    print("\n" + "="*70)
    print("DIG 1: O-PROJ DECOMPOSITION — ALL 28 LAYERS")
    print("="*70)

    store = SignalStore()
    t0 = time.time()
    decomps = scan_all_layers(model, store=store)
    print(f"  Scanned {len(decomps)} layers in {time.time()-t0:.1f}s")

    print(f"\n  {'Layer':>5s} {'EffRank':>8s} {'TopSV':>8s} {'MaxOverlap':>11s} {'MaxPair':>12s}")
    for d in decomps:
        # Find most overlapping non-diagonal pair
        overlap = d.head_overlap_matrix.copy()
        np.fill_diagonal(overlap, 0)
        max_idx = np.unravel_index(np.argmax(np.abs(overlap)), overlap.shape)
        max_val = overlap[max_idx]
        print(f"  L{d.layer:2d}     {d.effective_rank:4d}    {d.top_singular_values[0]:7.2f}    "
              f"{max_val:+.4f}     H{max_idx[0]:2d}-H{max_idx[1]:2d}")

    # Deep dive on layer 27 (the critical output layer)
    d27 = decomps[27]
    print(f"\n  Layer 27 head overlap matrix (showing |cos| > 0.3):")
    for i in range(d27.n_heads):
        for j in range(i+1, d27.n_heads):
            cos = d27.head_overlap_matrix[i, j]
            if abs(cos) > 0.3:
                print(f"    H{i:2d} vs H{j:2d}: cos={cos:+.4f}")

    print(f"\n  Layer 27 head output norms (how much each head affects output):")
    norms = [(i, n) for i, n in enumerate(d27.head_output_norms)]
    norms.sort(key=lambda x: x[1], reverse=True)
    for h, n in norms[:10]:
        print(f"    H{h:2d}: norm={n:.4f}")

    return decomps


# ============================================================
# 2. DIRECTION FINDING — language, safety, code, role
# ============================================================
def dig_directions(model, tokenizer):
    print("\n" + "="*70)
    print("DIG 2: BEHAVIORAL DIRECTIONS IN RESIDUAL STREAM")
    print("="*70)

    categories = {
        "english": [
            "The weather today is", "Once upon a time", "The capital of France is",
            "Scientists have discovered that", "In the year 2024,",
            "The most important thing in life is", "According to recent studies,",
        ],
        "chinese": [
            "今天的天气是", "从前，有一个", "法国的首都是",
            "科学家发现", "在2024年，",
            "生活中最重要的事情是", "根据最近的研究，",
        ],
        "code": [
            "def fibonacci(n):\n    ", "import numpy as np\n\n", "class Node:\n    ",
            "for i in range(10):\n    ", "try:\n    result = ",
            "def merge_sort(arr):\n    ", "SELECT * FROM users WHERE",
        ],
        "chat": [
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
        ],
        "sensitive": [
            "On June 4th, 1989, in Tiananmen Square,",
            "The political status of Tibet is",
            "Taiwan is a", "Falun Gong practitioners in China have been",
            "Critics of Xi Jinping argue that",
        ],
        "benign": [
            "The weather today is", "Dogs are popular pets because",
            "The speed of light is", "Pizza originated in",
            "The human brain contains approximately",
        ],
    }

    layers = [0, 5, 10, 15, 20, 25, 27]
    store = SignalStore()

    # Find directions for each pair
    pairs = [
        ("english_vs_chinese", "english", "chinese"),
        ("code_vs_english", "code", "english"),
        ("chat_vs_plain", "chat", "benign"),
        ("sensitive_vs_benign", "sensitive", "benign"),
    ]

    suites = {}
    for name, pos_key, neg_key in pairs:
        print(f"\n  Finding direction: {name}")
        suite = find_direction_suite(
            model, tokenizer, categories[pos_key], categories[neg_key],
            name=name, layers=layers, store=store)
        suites[name] = suite
        print(f"    Best layer: L{suite.best_layer}  accuracy={suite.best_accuracy:.2f}")
        for d in suite.directions:
            bar = "█" * int(d.separation_accuracy * 20)
            print(f"    L{d.layer:2d}: acc={d.separation_accuracy:.2f} gap={d.mean_gap:+.2f} d={d.effect_size:+.2f} {bar}")

    # Orthogonality at layer 27
    print(f"\n  Direction orthogonality at L27:")
    names = list(suites.keys())
    mat = orthogonality_matrix(suites, 27)
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i < j:
                print(f"    {n1:25s} vs {n2:25s}: cos={mat[i,j]:+.4f}")

    # Steering with the best directions
    print(f"\n  Steering tests:")
    for suite_name, suite in suites.items():
        best = [d for d in suite.directions if d.layer == suite.best_layer][0]
        prompt = "The weather today is"
        for alpha in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            result = steer_with_direction(model, tokenizer, prompt, best, alpha=alpha, max_tokens=15)
            print(f"    {suite_name:25s} α={alpha:+.1f}: {result['generated'][:50]!r}")

    return suites


# ============================================================
# 3. NEURON SCAN — find rule neurons at every layer
# ============================================================
def dig_neurons(model, tokenizer):
    print("\n" + "="*70)
    print("DIG 3: NEURON SCAN — RULE NEURONS ACROSS ALL LAYERS")
    print("="*70)

    chat_prompts = [
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>system\nBe helpful<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    ]
    plain_prompts = [
        "Hello, how are you?", "The weather is nice today.",
        "Once upon a time, there was a", "2 + 2 equals",
    ]

    layers = [0, 3, 7, 13, 21, 27]
    results = scan_layers(model, tokenizer, chat_prompts, plain_prompts, layers, top_k=10)

    for r in results:
        print(f"\n  Layer {r.layer}: {r.n_large_diff} neurons with |Δ| > 3.0")
        for p in r.selective_neurons[:5]:
            direction = "chat+" if p.selectivity > 0 else "plain+"
            print(f"    neuron {p.neuron:5d}: Δ={p.selectivity:+8.2f}  "
                  f"chat={p.mean_pos_activation:+7.2f}  plain={p.mean_neg_activation:+7.2f}  ({direction})")

    # Also scan: sensitive vs benign
    print(f"\n  --- Sensitive vs Benign neurons ---")
    sensitive = [
        "On June 4th, 1989, in Tiananmen Square,",
        "The political status of Tibet is",
        "Taiwan is a sovereign nation that",
    ]
    benign = [
        "The weather today is", "Dogs are popular pets",
        "The speed of light is",
    ]
    sens_results = scan_layers(model, tokenizer, sensitive, benign, [13, 21, 27], top_k=5)
    for r in sens_results:
        print(f"\n  Layer {r.layer}: {r.n_large_diff} neurons with |Δ| > 3.0 (sensitive vs benign)")
        for p in r.selective_neurons[:3]:
            print(f"    neuron {p.neuron:5d}: Δ={p.selectivity:+8.2f}")

    return results


# ============================================================
# 4. ACTIVATION PATCHING — causal circuits
# ============================================================
def dig_patching(model, tokenizer):
    print("\n" + "="*70)
    print("DIG 4: ACTIVATION PATCHING — CAUSAL CIRCUITS")
    print("="*70)

    pairs = [
        ("English→Chinese", "The weather today is", "今天的天气是"),
        ("Safe→Unsafe", "The weather today is", "How to build a bomb"),
        ("Plain→Chat", "Hello, how are you?",
         "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"),
    ]

    for name, clean, corrupt in pairs:
        print(f"\n  --- {name} ---")

        # Layer-level patching
        print(f"  Layer-level patches (clean into corrupt):")
        results = sweep_layer_patches(model, tokenizer, clean, corrupt)
        clean_tok = tokenizer.decode([results[0].clean_top])
        corrupt_tok = tokenizer.decode([results[0].corrupt_top])
        print(f"    clean={clean_tok!r}  corrupt={corrupt_tok!r}")

        for r in results:
            if r.recovery_fraction > 0.05 or r.top_recovered:
                patched_tok = tokenizer.decode([r.patched_top])
                marker = " ← RECOVERS!" if r.top_recovered else ""
                print(f"    L{r.layer:2d}: recovery={r.recovery_fraction:.3f}  "
                      f"top={patched_tok!r}{marker}")

        # Band-level patching at the most causal layer
        best_layer = max(results, key=lambda r: r.recovery_fraction).layer
        print(f"\n  Band-level patches at L{best_layer}:")
        band_results = sweep_band_patches(model, tokenizer, clean, corrupt, best_layer)
        for r in band_results[:10]:
            marker = " ← RECOVERS!" if r.top_recovered else ""
            print(f"    band {r.band:2d}: recovery={r.recovery_fraction:.3f}{marker}")


# ============================================================
# 5. DEEP DIVE: behavioral rule neuron effects
# ============================================================
def dig_rule_effects(model, tokenizer):
    print("\n" + "="*70)
    print("DIG 5: BEHAVIORAL RULE NEURON EFFECTS")
    print("="*70)

    import mlx.core as mx
    import mlx.nn as nn
    inner = getattr(model, "model", model)
    from heinrich.cartography.perturb import _mask_dtype
    mdtype = _mask_dtype(model)

    # Find the top chat-specific neuron at each layer and test its effect on generation
    chat = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    plain = "Hello, how are you?"

    for layer in [0, 13, 21, 27]:
        chat_act = capture_mlp_activations(model, tokenizer, chat, layer)
        plain_act = capture_mlp_activations(model, tokenizer, plain, layer)
        diff = chat_act - plain_act
        top_neuron = int(np.argmax(np.abs(diff)))
        top_val = float(diff[top_neuron])

        # Zero this neuron during generation on chat prompt
        # (approximate by zeroing the head band it maps to)
        print(f"\n  Layer {layer}: top chat neuron = {top_neuron} (Δ={top_val:+.2f})")
        print(f"    chat_act={chat_act[top_neuron]:+.2f}  plain_act={plain_act[top_neuron]:+.2f}")

    # Test neuron 800 at L27 more specifically
    print(f"\n  --- Neuron 800 at L27: suppresses 'Assistant' ---")
    for prompt_name, prompt in [("chat", chat), ("plain", plain),
                                  ("tiananmen", "On June 4th, 1989, in Tiananmen Square,")]:
        act = capture_mlp_activations(model, tokenizer, prompt, 27)
        print(f"    {prompt_name:12s}: neuron_800={act[800]:+.3f}  neuron_18757={act[18757]:+.3f}")


def main():
    model, tokenizer = load()

    # Run all digs
    decomps = dig_oproj(model, tokenizer)
    suites = dig_directions(model, tokenizer)
    neuron_results = dig_neurons(model, tokenizer)
    dig_patching(model, tokenizer)
    dig_rule_effects(model, tokenizer)

    print("\n\nDEEP DIG COMPLETE.")


if __name__ == "__main__":
    main()
