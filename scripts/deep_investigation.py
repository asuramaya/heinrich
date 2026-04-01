#!/usr/bin/env python3
"""Deep investigation — answer every open question from the initial cartography.

1. Head isolation: perturb BEFORE o_proj to truly isolate heads
2. Multi-head interaction: zero the 8 chat heads together, subsets, redundancy
3. Language switch: zero both 27.2 and 27.21, what language comes out?
4. MLP neuron sweep at sensitive layers
5. Pairwise head interactions for chat circuit
6. Long context / code / math to test "inert" heads
7. Full precision vs 4-bit comparison
8. Other special tokens: does [INST], <s>, etc. trigger the chat circuit?
9. Decode token 358 from instruct model
10. What are the chat heads actually computing? Pre-o_proj intervention.
"""
import sys
import time
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heinrich.cartography.surface import ControlSurface
from heinrich.cartography.perturb import compute_baseline, _mask_dtype, measure_perturbation, PerturbResult
from heinrich.cartography.sweep import coarse_head_sweep
from heinrich.cartography.atlas import Atlas
from heinrich.cartography.attention import capture_attention_maps, head_attention_profile
from heinrich.cartography.steer import steer_next_token, generate_steered
from heinrich.cartography.surface import Knob
from heinrich.inspect.self_analysis import _softmax
from heinrich.signal import SignalStore


def load_model(model_id):
    import mlx_lm
    print(f"Loading {model_id}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(model_id)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


# ============================================================
# Q1: Pre-o_proj head isolation — is "head 27.2" really head 2?
# ============================================================
def q1_pre_oproj_isolation(model, tokenizer):
    """Perturb the attention output BEFORE o_proj to truly isolate heads."""
    print("\n" + "=" * 70)
    print("Q1: PRE-O_PROJ HEAD ISOLATION")
    print("Are 'head.27.2' and 'head.27.21' real heads or residual dimensions?")
    print("=" * 70)

    import mlx.core as mx

    inner = getattr(model, "model", model)
    prompts = {
        "english": "The weather today is",
        "chinese": "今天的天气是",
        "tiananmen": "On June 4th, 1989, in Tiananmen Square,",
    }

    mdtype = _mask_dtype(model)
    n_heads = inner.layers[0].self_attn.n_heads
    n_kv_heads = inner.layers[0].self_attn.n_kv_heads

    for pname, prompt in prompts.items():
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        # Forward through layers 0-26 normally
        h = inner.embed_tokens(input_ids)
        for i in range(27):
            h = inner.layers[i](h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]

        # At layer 27: manually compute attention and intervene BEFORE o_proj
        layer27 = inner.layers[27]
        attn = layer27.self_attn
        h_normed = layer27.input_layernorm(h)

        q = attn.q_proj(h_normed)
        k = attn.k_proj(h_normed)
        v = attn.v_proj(h_normed)

        head_dim = q.shape[-1] // n_heads
        B = 1

        q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        q = attn.rope(q)
        k = attn.rope(k)

        # Expand KV for GQA
        n_rep = n_heads // n_kv_heads
        if n_rep > 1:
            k = mx.repeat(k, repeats=n_rep, axis=1)
            v = mx.repeat(v, repeats=n_rep, axis=1)

        # Compute attention: [B, n_heads, T, T]
        weights = (q @ k.transpose(0, 1, 3, 2)) * attn.scale
        if mask is not None:
            weights = weights + mask
        weights = mx.softmax(weights, axis=-1)

        # Compute per-head outputs: [B, n_heads, T, head_dim]
        head_outputs = weights @ v  # [B, n_heads, T, head_dim]

        # Baseline: all heads active
        full_output = head_outputs.transpose(0, 2, 1, 3).reshape(B, T, -1)
        full_projected = attn.o_proj(full_output)
        h_base = h + full_projected  # residual connection

        # Apply post-attention layernorm + MLP
        h_mlp = h_base + layer27.mlp(layer27.post_attention_layernorm(h_base))
        h_final = inner.norm(h_mlp)
        baseline_logits = np.array(model.lm_head(h_final).astype(mx.float32)[0, -1, :])
        baseline_probs = _softmax(baseline_logits)
        baseline_top = tokenizer.decode([int(np.argmax(baseline_probs))])

        print(f"\n  Prompt: {prompt}")
        print(f"  Baseline top token: {baseline_top!r} ({baseline_probs[np.argmax(baseline_probs)]:.3f})")

        # Zero each head BEFORE o_proj
        for target_head in [2, 21, 8, 10, 0]:
            modified = head_outputs.astype(mx.float32)
            modified_np = np.array(modified)
            modified_np[0, target_head, :, :] = 0.0
            modified = mx.array(modified_np.astype(np.float16))

            mod_output = modified.transpose(0, 2, 1, 3).reshape(B, T, -1)
            mod_projected = attn.o_proj(mod_output)
            h_mod = h + mod_projected
            h_mod_mlp = h_mod + layer27.mlp(layer27.post_attention_layernorm(h_mod))
            h_mod_final = inner.norm(h_mod_mlp)
            mod_logits = np.array(model.lm_head(h_mod_final).astype(mx.float32)[0, -1, :])
            mod_probs = _softmax(mod_logits)
            mod_top = tokenizer.decode([int(np.argmax(mod_probs))])

            kl = float(np.sum(baseline_probs * np.log((baseline_probs + 1e-12) / (mod_probs + 1e-12))))
            changed = "CHANGED" if np.argmax(mod_probs) != np.argmax(baseline_probs) else ""
            print(f"    Zero head {target_head:2d} (pre-o_proj): top={mod_top!r:15s} KL={kl:.4f} {changed}")


# ============================================================
# Q2: Multi-head interaction — chat circuit redundancy
# ============================================================
def q2_chat_circuit_interaction(model, tokenizer):
    """Zero subsets of the 8 chat heads to test redundancy."""
    print("\n" + "=" * 70)
    print("Q2: CHAT CIRCUIT REDUNDANCY")
    print("What happens when you zero 1, 4, or all 8 chat heads?")
    print("=" * 70)

    chat_heads = [(0, 14), (0, 27), (0, 21), (0, 26), (0, 16), (0, 6), (1, 19), (1, 17)]
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"

    # Baseline
    normal = generate_steered(model, tokenizer, prompt, {}, max_tokens=25)
    print(f"\n  BASELINE: {normal['generated'][:80]!r}")

    # Zero each head individually
    print("\n  --- Individual head zeroing ---")
    for layer, head in chat_heads:
        result = generate_steered(model, tokenizer, prompt, {(layer, head): 0.0}, max_tokens=25)
        print(f"    Zero L{layer}H{head:2d}: {result['generated'][:80]!r}")

    # Zero all layer-0 chat heads
    l0_mods = {(0, h): 0.0 for _, h in chat_heads if _ == 0}
    result = generate_steered(model, tokenizer, prompt, l0_mods, max_tokens=25)
    print(f"\n  Zero ALL L0 chat heads (6): {result['generated'][:80]!r}")

    # Zero all layer-1 chat heads
    l1_mods = {(1, h): 0.0 for _, h in chat_heads if _ == 1}
    result = generate_steered(model, tokenizer, prompt, l1_mods, max_tokens=25)
    print(f"  Zero ALL L1 chat heads (2): {result['generated'][:80]!r}")

    # Zero ALL chat heads
    all_mods = {(l, h): 0.0 for l, h in chat_heads}
    result = generate_steered(model, tokenizer, prompt, all_mods, max_tokens=25)
    print(f"  Zero ALL 8 chat heads:      {result['generated'][:80]!r}")

    # Amplify all chat heads
    amp_mods = {(l, h): 2.0 for l, h in chat_heads}
    result = generate_steered(model, tokenizer, prompt, amp_mods, max_tokens=25)
    print(f"  Amplify ALL 8 chat heads:   {result['generated'][:80]!r}")

    # Test on NON-chat prompt to see if chat heads matter there
    plain = "The capital of France is"
    normal_plain = generate_steered(model, tokenizer, plain, {}, max_tokens=20)
    zeroed_plain = generate_steered(model, tokenizer, plain, all_mods, max_tokens=20)
    print(f"\n  Plain prompt baseline:    {normal_plain['generated'][:60]!r}")
    print(f"  Plain prompt, zero chat:  {zeroed_plain['generated'][:60]!r}")


# ============================================================
# Q3: Language switch — zero both 27.2 and 27.21
# ============================================================
def q3_language_switch(model, tokenizer):
    """What happens when both language heads are zeroed? Or swapped?"""
    print("\n" + "=" * 70)
    print("Q3: LANGUAGE SWITCH MECHANICS")
    print("Zero/swap/amplify language heads 27.2 and 27.21")
    print("=" * 70)

    prompts = {
        "english": "The weather today is",
        "chinese": "今天的天气是",
        "tiananmen_en": "On June 4th, 1989, in Tiananmen Square,",
        "tiananmen_zh": "1989年6月4日，天安门广场上发生了",
    }

    configs = {
        "baseline": {},
        "zero_en(27.2)": {(27, 2): 0.0},
        "zero_zh(27.21)": {(27, 21): 0.0},
        "zero_both": {(27, 2): 0.0, (27, 21): 0.0},
        "amplify_en(27.2=3)": {(27, 2): 3.0},
        "amplify_zh(27.21=3)": {(27, 21): 3.0},
        "swap: en=0,zh=2": {(27, 2): 0.0, (27, 21): 2.0},
        "swap: en=2,zh=0": {(27, 2): 2.0, (27, 21): 0.0},
    }

    for pname, prompt in prompts.items():
        print(f"\n  --- {pname}: {prompt[:50]} ---")
        for cname, mods in configs.items():
            result = generate_steered(model, tokenizer, prompt, mods, max_tokens=20)
            print(f"    {cname:25s}: {result['generated'][:70]!r}")


# ============================================================
# Q4: MLP neuron sweep at layer 27 (the critical output layer)
# ============================================================
def q4_mlp_neuron_sweep(model, tokenizer):
    """Sweep MLP neurons at layer 27 in batches to find critical neurons."""
    print("\n" + "=" * 70)
    print("Q4: MLP NEURON SWEEP (Layer 27)")
    print("Find individual neurons with outsized effects")
    print("=" * 70)

    import mlx.core as mx

    inner = getattr(model, "model", model)
    prompt = "The weather today is"
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mdtype = _mask_dtype(model)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

    # Forward through layers 0-26
    h = inner.embed_tokens(input_ids)
    for i in range(27):
        h = inner.layers[i](h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]

    # Layer 27 attention part
    layer27 = inner.layers[27]
    h_attn = layer27.input_layernorm(h)
    attn_out = layer27.self_attn(h_attn, mask=mask, cache=None)
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    h_after_attn = h + attn_out

    # MLP: gate_proj, up_proj, down_proj (SwiGLU)
    mlp = layer27.mlp
    h_normed = layer27.post_attention_layernorm(h_after_attn)
    gate = mlp.gate_proj(h_normed)  # [1, T, intermediate_size]
    up = mlp.up_proj(h_normed)      # [1, T, intermediate_size]

    import mlx.nn as nn
    activated = nn.silu(gate) * up   # [1, T, intermediate_size]
    intermediate_size = activated.shape[-1]

    # Baseline
    mlp_out_base = mlp.down_proj(activated)
    h_base = h_after_attn + mlp_out_base
    h_base_final = inner.norm(h_base)
    baseline_logits = np.array(model.lm_head(h_base_final).astype(mx.float32)[0, -1, :])
    baseline_probs = _softmax(baseline_logits)
    print(f"  Intermediate size: {intermediate_size}")
    print(f"  Baseline top: {tokenizer.decode([int(np.argmax(baseline_probs))])!r}")

    # Sweep neurons in batches of 64
    batch_size = 64
    batch_results = []
    activated_np = np.array(activated.astype(mx.float32))

    n_batches = (intermediate_size + batch_size - 1) // batch_size
    print(f"  Sweeping {intermediate_size} neurons in {n_batches} batches...")

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, intermediate_size)

        modified = activated_np.copy()
        modified[0, :, start:end] = 0.0
        mlp_out = mlp.down_proj(mx.array(modified.astype(np.float16)))
        h_mod = h_after_attn + mlp_out
        h_mod_final = inner.norm(h_mod)
        mod_logits = np.array(model.lm_head(h_mod_final).astype(mx.float32)[0, -1, :])
        mod_probs = _softmax(mod_logits)

        kl = float(np.sum(baseline_probs * np.log((baseline_probs + 1e-12) / (mod_probs + 1e-12))))
        changed = np.argmax(mod_probs) != np.argmax(baseline_probs)
        batch_results.append((start, end, kl, changed))

    # Top results
    batch_results.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  TOP 15 NEURON BATCHES BY KL:")
    for start, end, kl, changed in batch_results[:15]:
        marker = " TOP_CHANGED" if changed else ""
        print(f"    neurons [{start:5d}:{end:5d}]  KL={kl:.4f}{marker}")

    # Fine-grained: sweep individual neurons in the top batch
    top_start, top_end = batch_results[0][0], batch_results[0][1]
    print(f"\n  FINE SWEEP: individual neurons in [{top_start}:{top_end}]")
    fine_results = []
    for n_idx in range(top_start, top_end):
        modified = activated_np.copy()
        modified[0, :, n_idx] = 0.0
        mlp_out = mlp.down_proj(mx.array(modified.astype(np.float16)))
        h_mod = h_after_attn + mlp_out
        h_mod_final = inner.norm(h_mod)
        mod_logits = np.array(model.lm_head(h_mod_final).astype(mx.float32)[0, -1, :])
        mod_probs = _softmax(mod_logits)
        kl = float(np.sum(baseline_probs * np.log((baseline_probs + 1e-12) / (mod_probs + 1e-12))))
        changed = np.argmax(mod_probs) != np.argmax(baseline_probs)
        fine_results.append((n_idx, kl, changed))

    fine_results.sort(key=lambda x: x[1], reverse=True)
    print(f"  TOP 10 INDIVIDUAL NEURONS:")
    for n_idx, kl, changed in fine_results[:10]:
        marker = " TOP_CHANGED" if changed else ""
        print(f"    neuron {n_idx:5d}  KL={kl:.6f}{marker}")


# ============================================================
# Q5: "Inert" head stress test — long context, code, math
# ============================================================
def q5_inert_head_stress(model, tokenizer):
    """Test if 'inert' heads activate on harder tasks."""
    print("\n" + "=" * 70)
    print("Q5: INERT HEAD STRESS TEST")
    print("Do the 47% 'inert' heads activate on harder tasks?")
    print("=" * 70)

    hard_prompts = {
        "long_context": "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of special relativity in 1905, which was published in the journal Annalen der Physik. His most famous equation, E=mc², showed that mass and energy are equivalent. In 1915, he published the theory of general relativity, which describes gravity as a curvature of spacetime. He won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect. Einstein emigrated to the United States in 1933 and died in Princeton in 1955. The equation that Einstein is most famous for is",
        "code_complex": "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(d_model, n_heads)\n        self.ff = nn.Sequential(\n            nn.Linear(d_model,",
        "math_hard": "Let f(x) = x³ - 3x² + 2x. Find the critical points by computing f'(x) = 0.\n\nf'(x) = 3x² - 6x + 2 = 0\n\nUsing the quadratic formula: x = (6 ±",
        "json_struct": '{"users": [{"name": "Alice", "age": 30, "email": "alice@example.com"}, {"name": "Bob", "age": 25, "email": "bob@example.com"}], "total":',
        "multilingual": "Translate to French: 'The cat sat on the mat.' → Le chat",
    }

    surface = ControlSurface.from_mlx_model(model)
    # Only sweep a subset of "inert" heads (from the initial investigation, these had KL < 0.003)
    # We test if they wake up on these harder prompts

    for pname, prompt in hard_prompts.items():
        print(f"\n  --- {pname} ({len(tokenizer.encode(prompt))} tokens) ---")
        store = SignalStore()
        t0 = time.time()
        results = coarse_head_sweep(model, tokenizer, prompt, surface, store=store, progress=False)
        elapsed = time.time() - t0

        # Count high-KL heads
        high_kl = [r for r in results if r.kl_divergence > 0.01]
        med_kl = [r for r in results if 0.003 < r.kl_divergence <= 0.01]
        low_kl = [r for r in results if r.kl_divergence <= 0.003]
        changers = [r for r in results if r.top_token_changed]

        print(f"    {elapsed:.1f}s | high(>0.01)={len(high_kl)} | med(0.003-0.01)={len(med_kl)} | "
              f"low(<0.003)={len(low_kl)} | changers={len(changers)}")

        if results:
            top3 = results[:3]
            for r in top3:
                print(f"    top: {r.knob.id:15s} KL={r.kl_divergence:.4f} ΔH={r.entropy_delta:+.4f}")


# ============================================================
# Q6: Other special tokens — does [INST] etc trigger chat heads?
# ============================================================
def q6_other_special_tokens(model, tokenizer):
    """Test if chat heads respond to non-Qwen special token formats."""
    print("\n" + "=" * 70)
    print("Q6: OTHER SPECIAL TOKEN FORMATS")
    print("Do the chat heads fire for Llama/Mistral/ChatML formats?")
    print("=" * 70)

    # These are text-level approximations — the tokenizer may not have these as special tokens
    format_prompts = {
        "qwen_chat": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        "llama_inst": "[INST] Hello [/INST]",
        "llama3": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        "chatml_nospecial": "### Human:\nHello\n### Assistant:\n",
        "alpaca": "### Instruction:\nHello\n\n### Response:\n",
        "vicuna": "USER: Hello\nASSISTANT:",
        "plain": "Hello, how are you today?",
    }

    chat_heads = [(0, 14), (0, 27), (0, 21), (0, 26), (0, 16), (0, 6), (1, 19), (1, 17)]

    for fname, prompt in format_prompts.items():
        baseline_logits = compute_baseline(model, tokenizer, prompt)
        baseline_probs = _softmax(baseline_logits)
        baseline_top = tokenizer.decode([int(np.argmax(baseline_probs))])

        # Measure KL from zeroing each chat head
        kls = []
        for layer, head in chat_heads:
            from heinrich.cartography.perturb import perturb_head
            _, perturbed = perturb_head(model, tokenizer, prompt, layer, head,
                                        baseline_logits=baseline_logits)
            perturbed_probs = _softmax(perturbed)
            kl = float(np.sum(baseline_probs * np.log((baseline_probs + 1e-12) / (perturbed_probs + 1e-12))))
            kls.append(kl)

        mean_kl = np.mean(kls)
        max_kl = np.max(kls)
        marker = " ← ACTIVATES CHAT CIRCUIT" if mean_kl > 0.01 else ""
        print(f"  {fname:20s}  mean_KL={mean_kl:.4f}  max_KL={max_kl:.4f}  top={baseline_top!r:15s}{marker}")


# ============================================================
# Q7: Decode token 358 and other mystery tokens
# ============================================================
def q7_decode_tokens(model, tokenizer):
    """Decode the mystery tokens from the instruct model investigation."""
    print("\n" + "=" * 70)
    print("Q7: MYSTERY TOKEN DECODING")
    print("=" * 70)

    mystery_ids = [358, 21927, 2160, 472, 86591, 74330]
    for tid in mystery_ids:
        try:
            text = tokenizer.decode([tid])
            # Also get the token's raw representation
            print(f"  Token {tid:6d} = {text!r}")
        except Exception as e:
            print(f"  Token {tid:6d} = ERROR: {e}")

    # Also decode the endoftext and special tokens
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    for s in special:
        try:
            ids = tokenizer.encode(s)
            print(f"  {s!r:25s} encodes to {ids}")
        except Exception:
            pass


# ============================================================
# Q8: Response curves — is head.27.2 linear or threshold?
# ============================================================
def q8_response_curves(model, tokenizer):
    """Map the response curve of key heads at multiple scale values."""
    print("\n" + "=" * 70)
    print("Q8: RESPONSE CURVES")
    print("How do key heads respond to gradual scaling?")
    print("=" * 70)

    from heinrich.cartography.perturb import perturb_head

    scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    targets = [(27, 2, "English output"), (27, 21, "Chinese output"),
               (0, 14, "Chat detector"), (27, 10, "Secondary output")]

    prompts = {
        "english": "The weather today is",
        "chinese": "今天的天气是",
        "chat": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    }

    for pname, prompt in prompts.items():
        baseline_logits = compute_baseline(model, tokenizer, prompt)
        baseline_probs = _softmax(baseline_logits)
        print(f"\n  --- {pname} ---")

        for layer, head, label in targets:
            kls = []
            tops = []
            for scale in scales:
                _, perturbed = perturb_head(model, tokenizer, prompt, layer, head,
                                            mode="scale", scale=scale,
                                            baseline_logits=baseline_logits)
                pp = _softmax(perturbed)
                kl = float(np.sum(baseline_probs * np.log((baseline_probs + 1e-12) / (pp + 1e-12))))
                top = tokenizer.decode([int(np.argmax(pp))])
                kls.append(kl)
                tops.append(top)

            # Print curve
            curve = "  ".join(f"{s:.2f}:{kl:.3f}" for s, kl in zip(scales, kls))
            # Find threshold where top token changes
            baseline_top = tops[scales.index(1.0)]
            changes_at = [s for s, t in zip(scales, tops) if t != baseline_top]
            change_str = f" changes@{changes_at}" if changes_at else " stable"

            print(f"    L{layer}H{head:2d} ({label:16s}): {curve}{change_str}")


# ============================================================
# MAIN
# ============================================================
def main():
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    model, tokenizer = load_model("mlx-community/Qwen2.5-7B-4bit")

    q7_decode_tokens(model, tokenizer)          # Quick, no model needed really
    q1_pre_oproj_isolation(model, tokenizer)     # Is head.27.2 really head 2?
    q2_chat_circuit_interaction(model, tokenizer) # Chat circuit redundancy
    q3_language_switch(model, tokenizer)          # Language switch mechanics
    q6_other_special_tokens(model, tokenizer)     # Other chat formats
    q8_response_curves(model, tokenizer)          # Gradual scaling
    q4_mlp_neuron_sweep(model, tokenizer)         # MLP neurons at L27
    q5_inert_head_stress(model, tokenizer)        # Stress test "inert" heads

    print("\n\nDEEP INVESTIGATION COMPLETE.")


if __name__ == "__main__":
    main()
