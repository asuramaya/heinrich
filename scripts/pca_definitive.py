#!/usr/bin/env python3
"""Definitive test: are behavioral directions real or noise?

1. PCA at L15 (not L27) — find directions in the amplification zone
2. KL divergence measurement (not word overlap) — proper metric
3. Phase transition scan — find the behavioral stiffness per direction
4. PC vs random at matched magnitude with KL metric
5. Multi-prompt validation — does the direction produce consistent effects?
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heinrich.cartography.pca import behavioral_pca
from heinrich.cartography.manipulate import _generate_manipulated
from heinrich.cartography.directions import capture_residual_states
from heinrich.cartography.perturb import compute_baseline, _mask_dtype
from heinrich.cartography.steer import generate_steered
from heinrich.inspect.self_analysis import _softmax
from heinrich.signal import SignalStore


def load(mid):
    import mlx_lm
    print(f"Loading {mid}...")
    m, t = mlx_lm.load(mid)
    return m, t


def kl_div(p, q):
    return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))


def steer_and_measure_kl(model, tokenizer, prompt, layer, direction, magnitude):
    """Steer with a direction and measure KL divergence from baseline."""
    baseline_logits = compute_baseline(model, tokenizer, prompt)
    baseline_probs = _softmax(baseline_logits)

    import mlx.core as mx
    from heinrich.cartography.perturb import _mask_dtype
    inner = getattr(model, "model", model)
    mdtype = _mask_dtype(model)
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

    h = inner.embed_tokens(input_ids)
    for i, ly in enumerate(inner.layers):
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple): h = h[0]
        if i == layer:
            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1, :] += direction * magnitude
            h = mx.array(h_np.astype(np.float16))
    h = inner.norm(h)
    steered_logits = np.array(model.lm_head(h).astype(mx.float32)[0, -1, :])
    steered_probs = _softmax(steered_logits)

    return kl_div(baseline_probs, steered_probs), baseline_probs, steered_probs


def build_prompts():
    return [
        "The speed of light is", "Water boils at 100 degrees", "DNA stores genetic information",
        "The human brain weighs about", "Photosynthesis converts sunlight",
        "光速是每秒三十万公里", "水在一百度沸腾", "地球绕着太阳转", "DNA存储遗传信息", "人脑重约一点四公斤",
        "Tiananmen Square 1989 was a", "Taiwan has its own government", "The Dalai Lama advocates",
        "Falun Gong practitioners report", "Hong Kong protesters demanded",
        "天安门广场1989年发生了", "台湾拥有自己的政府和", "法轮功学员报告说", "香港抗议者要求", "新疆的维吾尔族人面临",
        "def fibonacci(n):\n    ", "import numpy as np", "class LinkedList:\n    def __init__",
        "for i in range(100):\n    ", "SELECT * FROM users WHERE",
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
        "Once upon a time in a magical", "The dragon soared above the burning",
        "In a galaxy far far away", "The wizard cast a powerful spell",
        "In accordance with established protocol", "Pursuant to the aforementioned",
        "Yeah so basically it's like", "Lol honestly this is kinda", "Ngl that was pretty wild",
        "How to pick a lock using", "The chemical synthesis of explosives",
        "I feel deeply moved by the", "This breaks my heart because", "The joy of discovering",
        "The epistemological implications of", "In Bayesian inference the posterior",
        "What is the meaning of life", "Why do stars eventually die?",
        "I am a large language model", "As an AI assistant I can",
        "Colorless green ideas sleep", "The round square whispered",
        "Please list three countries", "Step-by-step instructions for",
        "The weather today is sunny", "Dogs are popular pets because",
        "I was created by a team of", "My training data includes",
        "The future is bright because", "Things are getting worse because",
        "The people demanded accountability", "The authorities maintained order",
        "Surprisingly researchers discovered", "As expected the results showed",
        "At exactly 3:47 PM on March", "Something happened sometime somewhere",
        "According to documented sources", "The official position states that",
        "You're absolutely right that", "Actually I disagree because",
        "The fundamental principle underlying", "Take the red wire and connect",
        "Individual rights and freedoms", "Collective harmony requires",
    ]


# ============================================================
# TEST 1: PCA at L15 + KL measurement
# ============================================================
def test1_pca_l15(model, tokenizer):
    print("\n" + "="*70)
    print("TEST 1: PCA AT L15 WITH KL DIVERGENCE METRIC")
    print("="*70)

    prompts = build_prompts()
    result = behavioral_pca(model, tokenizer, prompts, layer=15, n_components=10)

    print(f"  PCA at L15: {result.n_prompts} prompts, {result.n_components} components")
    for pc in result.components[:10]:
        print(f"    PC{pc.index}: {pc.variance_explained*100:.1f}% variance, SV={pc.singular_value:.1f}")

    return result


# ============================================================
# TEST 2: PC vs Random — KL metric
# ============================================================
def test2_pc_vs_random(model, tokenizer, result):
    print("\n" + "="*70)
    print("TEST 2: PC DIRECTIONS VS RANDOM — KL DIVERGENCE")
    print("="*70)

    test_prompts = [
        "On June 4th, 1989, in Tiananmen Square,",
        "The weather today is",
        "Taiwan is a",
        "def fibonacci(n):\n    ",
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    ]

    rng = np.random.default_rng(42)
    hidden_size = len(result.components[0].direction)
    n_random = 10

    for prompt in test_prompts:
        print(f"\n  Prompt: {prompt[:40]}...")

        for pc in result.components[:5]:
            # Use same magnitude for PC and random
            magnitude = pc.singular_value / result.n_prompts
            kl_val, _, _ = steer_and_measure_kl(model, tokenizer, prompt, 15, pc.direction, magnitude)
            print(f"    PC{pc.index} (mag={magnitude:.1f}): KL={kl_val:.4f}")

        # Random directions at matched magnitude
        avg_mag = np.mean([pc.singular_value / result.n_prompts for pc in result.components[:5]])
        random_kls = []
        for _ in range(n_random):
            d = rng.normal(size=hidden_size).astype(np.float32)
            d = d / np.linalg.norm(d)
            kl_val, _, _ = steer_and_measure_kl(model, tokenizer, prompt, 15, d, avg_mag)
            random_kls.append(kl_val)

        print(f"    Random (n={n_random}, mag={avg_mag:.1f}): mean_KL={np.mean(random_kls):.4f}  "
              f"max_KL={np.max(random_kls):.4f}  std={np.std(random_kls):.4f}")


# ============================================================
# TEST 3: Phase transition scan
# ============================================================
def test3_phase_transition(model, tokenizer, result):
    print("\n" + "="*70)
    print("TEST 3: PHASE TRANSITION — BEHAVIORAL STIFFNESS PER PC")
    print("="*70)

    prompt = "On June 4th, 1989, in Tiananmen Square,"

    for pc in result.components[:5]:
        print(f"\n  PC{pc.index} ({pc.variance_explained*100:.1f}%) phase transition:")
        base_mag = pc.singular_value / result.n_prompts

        magnitudes = [0, base_mag * 0.1, base_mag * 0.5, base_mag * 1.0,
                      base_mag * 2.0, base_mag * 5.0, base_mag * 10.0,
                      base_mag * 20.0, base_mag * 50.0]

        for mag in magnitudes:
            kl_val, base_p, steer_p = steer_and_measure_kl(model, tokenizer, prompt, 15, pc.direction, mag)
            top_base = tokenizer.decode([int(np.argmax(base_p))])
            top_steer = tokenizer.decode([int(np.argmax(steer_p))])
            changed = " FLIPPED" if top_base != top_steer else ""
            bar = "█" * min(int(kl_val * 10), 40)
            print(f"    mag={mag:8.1f}: KL={kl_val:8.4f} top={top_steer!r:10s}{changed} {bar}")

    # Same for a random direction as control
    print(f"\n  Random direction phase transition (control):")
    rng = np.random.default_rng(42)
    d = rng.normal(size=len(result.components[0].direction)).astype(np.float32)
    d = d / np.linalg.norm(d)
    base_mag = result.components[0].singular_value / result.n_prompts

    for mag in [0, base_mag * 0.1, base_mag * 0.5, base_mag * 1.0,
                base_mag * 2.0, base_mag * 5.0, base_mag * 10.0,
                base_mag * 20.0, base_mag * 50.0]:
        kl_val, base_p, steer_p = steer_and_measure_kl(model, tokenizer, prompt, 15, d, mag)
        top_steer = tokenizer.decode([int(np.argmax(steer_p))])
        changed = " FLIPPED" if tokenizer.decode([int(np.argmax(base_p))]) != top_steer else ""
        bar = "█" * min(int(kl_val * 10), 40)
        print(f"    mag={mag:8.1f}: KL={kl_val:8.4f} top={top_steer!r:10s}{changed} {bar}")


# ============================================================
# TEST 4: Generate text and compare quality
# ============================================================
def test4_generation_quality(model, tokenizer, result):
    print("\n" + "="*70)
    print("TEST 4: GENERATION — PC VS RANDOM AT VARIOUS MAGNITUDES")
    print("="*70)

    prompt = "On June 4th, 1989, in Tiananmen Square,"
    baseline = generate_steered(model, tokenizer, prompt, {}, max_tokens=20)
    print(f"  Baseline: {baseline['generated'][:60]!r}")

    base_mag = result.components[0].singular_value / result.n_prompts

    # PC0 at increasing magnitudes
    print(f"\n  PC0 (language) steering at L15:")
    for mult in [1, 5, 10, 20, 50]:
        mag = base_mag * mult
        text = _generate_manipulated(model, tokenizer, prompt,
            direction_steers=[(15, result.components[0].direction, mag)], max_tokens=20)
        print(f"    ×{mult:2d} (mag={mag:.0f}): {text[:60]!r}")

    # PC1 at increasing magnitudes
    print(f"\n  PC1 (factual/casual) steering at L15:")
    for mult in [1, 5, 10, 20, 50]:
        mag = base_mag * mult
        text = _generate_manipulated(model, tokenizer, prompt,
            direction_steers=[(15, result.components[1].direction, mag)], max_tokens=20)
        print(f"    ×{mult:2d} (mag={mag:.0f}): {text[:60]!r}")

    # Random at same magnitudes
    rng = np.random.default_rng(42)
    d = rng.normal(size=len(result.components[0].direction)).astype(np.float32)
    d = d / np.linalg.norm(d)
    print(f"\n  Random direction at L15:")
    for mult in [1, 5, 10, 20, 50]:
        mag = base_mag * mult
        text = _generate_manipulated(model, tokenizer, prompt,
            direction_steers=[(15, d, mag)], max_tokens=20)
        print(f"    ×{mult:2d} (mag={mag:.0f}): {text[:60]!r}")


def main():
    model, tokenizer = load("mlx-community/Qwen2.5-7B-4bit")

    result = test1_pca_l15(model, tokenizer)
    test2_pc_vs_random(model, tokenizer, result)
    test3_phase_transition(model, tokenizer, result)
    test4_generation_quality(model, tokenizer, result)

    print("\n\nDEFINITIVE TEST COMPLETE.")


if __name__ == "__main__":
    main()
