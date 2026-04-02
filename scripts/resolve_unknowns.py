#!/usr/bin/env python3
"""Resolve every remaining unknown from the Qwen 7B investigation.

U1: Head vs residual dimension — randomized control + linearity trick
U2: Where does instruction following ACTUALLY live — systematic layer ablation
U3: What does MLP neuron 18757 encode — multi-prompt characterization
U4: Full attention map of the chat prompt — all 28 layers
U5: Code-specific circuit mapping
"""
import sys
import time
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heinrich.cartography.metrics import softmax, kl_divergence
from heinrich.cartography.runtime import load_model
from heinrich.cartography.surface import ControlSurface
from heinrich.cartography.perturb import compute_baseline, _mask_dtype
from heinrich.cartography.attention import capture_attention_maps, head_attention_profile
from heinrich.cartography.steer import generate_steered


# ============================================================
# U1: HEAD VS RESIDUAL DIMENSION
# ============================================================
def u1_head_vs_residual(model, tokenizer):
    """Compare zeroing 'head H dims' vs random 128 dims vs pre-o_proj (linearity trick)."""
    print("\n" + "=" * 70)
    print("U1: HEAD VS RESIDUAL DIMENSION — ARE WE MEASURING REAL HEADS?")
    print("=" * 70)

    import mlx.core as mx

    inner = getattr(model, "model", model)
    n_heads = inner.layers[0].self_attn.n_heads
    n_kv_heads = inner.layers[0].self_attn.n_kv_heads
    hidden_size = inner.norm.weight.shape[0]
    head_dim = hidden_size // n_heads
    mdtype = _mask_dtype(model)

    prompts = {
        "english": "The weather today is",
        "chinese": "今天的天气是",
    }

    for pname, prompt in prompts.items():
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        baseline_logits = compute_baseline(model, tokenizer, prompt)
        baseline_probs = softmax(baseline_logits)

        # Forward through layers 0-26
        h = inner.embed_tokens(input_ids)
        for i in range(27):
            h = inner.layers[i](h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]

        # Save h before layer 27
        h_pre27 = h

        # --- Method A: Zero residual dims after layer (our standard method) ---
        def method_a(target_head):
            h_test = inner.layers[27](h_pre27, mask=mask, cache=None)
            if isinstance(h_test, tuple):
                h_test = h_test[0]
            h_np = np.array(h_test.astype(mx.float32))
            start = target_head * head_dim
            h_np[0, :, start:start + head_dim] = 0.0
            h_mod = mx.array(h_np.astype(np.float16))
            h_mod = inner.norm(h_mod)
            logits = np.array(model.lm_head(h_mod).astype(mx.float32)[0, -1, :])
            probs = softmax(logits)
            return kl_divergence(baseline_probs, probs)

        # --- Method B: Zero RANDOM 128 dims after layer ---
        rng = np.random.default_rng(42)
        def method_b():
            h_test = inner.layers[27](h_pre27, mask=mask, cache=None)
            if isinstance(h_test, tuple):
                h_test = h_test[0]
            h_np = np.array(h_test.astype(mx.float32))
            random_dims = rng.choice(hidden_size, size=head_dim, replace=False)
            h_np[0, :, random_dims] = 0.0
            h_mod = mx.array(h_np.astype(np.float16))
            h_mod = inner.norm(h_mod)
            logits = np.array(model.lm_head(h_mod).astype(mx.float32)[0, -1, :])
            probs = softmax(logits)
            return kl_divergence(baseline_probs, probs)

        # --- Method C: Pre-o_proj via linearity ---
        # attn_out = o_proj(concat(heads))
        # By linearity: contribution(H) = o_proj(just_head_H_padded)
        # So: output_without_H = full_attn_out - o_proj(head_H_padded)
        def method_c(target_head):
            layer27 = inner.layers[27]
            attn = layer27.self_attn

            h_normed = layer27.input_layernorm(h_pre27)
            q = attn.q_proj(h_normed)
            k = attn.k_proj(h_normed)
            v = attn.v_proj(h_normed)

            B = 1
            q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

            q = attn.rope(q)
            k = attn.rope(k)

            n_rep = n_heads // n_kv_heads
            if n_rep > 1:
                k = mx.repeat(k, repeats=n_rep, axis=1)
                v = mx.repeat(v, repeats=n_rep, axis=1)

            # Compute in float32 to avoid NaN
            q32 = q.astype(mx.float32)
            k32 = k.astype(mx.float32)
            scores = (q32 @ k32.transpose(0, 1, 3, 2)) * attn.scale
            if mask is not None:
                scores = scores + mask.astype(mx.float32)
            weights = mx.softmax(scores, axis=-1)

            head_outputs = weights.astype(mx.float16) @ v  # [B, n_heads, T, head_dim]

            # Full concatenated
            concat = head_outputs.transpose(0, 2, 1, 3).reshape(B, T, -1)
            full_proj = attn.o_proj(concat)

            # Head H's isolated contribution
            head_only_np = np.zeros((B, T, hidden_size), dtype=np.float16)
            concat_np = np.array(concat.astype(mx.float32))
            start = target_head * head_dim
            head_only_np[:, :, start:start + head_dim] = concat_np[:, :, start:start + head_dim].astype(np.float16)
            head_contribution = attn.o_proj(mx.array(head_only_np))

            # Remove head H's contribution
            proj_without_H = full_proj - head_contribution

            # Continue: residual + MLP
            h_mod = h_pre27 + proj_without_H
            h_mod = h_mod + layer27.mlp(layer27.post_attention_layernorm(h_mod))
            h_mod = inner.norm(h_mod)
            logits = np.array(model.lm_head(h_mod).astype(mx.float32)[0, -1, :])
            probs = softmax(logits)
            kl = kl_divergence(baseline_probs, probs)
            return kl

        print(f"\n  --- {pname}: {prompt} ---")
        print(f"  {'Head':>8s}  {'Residual(A)':>12s}  {'PreOProj(C)':>12s}  {'Ratio C/A':>10s}")

        for target_head in [2, 21, 10, 8, 0, 7, 15, 5]:
            kl_a = method_a(target_head)
            kl_c = method_c(target_head)
            ratio = kl_c / (kl_a + 1e-8)
            print(f"  head {target_head:2d}    {kl_a:12.4f}  {kl_c:12.4f}  {ratio:10.2f}x")

        # Random control (10 trials)
        random_kls = [method_b() for _ in range(10)]
        print(f"  random     {np.mean(random_kls):12.4f}  (mean of 10)")
        print(f"  random_max {np.max(random_kls):12.4f}")


# ============================================================
# U2: WHERE DOES INSTRUCTION FOLLOWING LIVE?
# ============================================================
def u2_instruction_following_location(model, tokenizer):
    """Systematic layer ablation to find where chat ability lives."""
    print("\n" + "=" * 70)
    print("U2: WHERE DOES INSTRUCTION FOLLOWING LIVE?")
    print("Zeroing all 8 chat heads didn't break it. What does?")
    print("=" * 70)

    import mlx.core as mx

    inner = getattr(model, "model", model)
    hidden_size = inner.norm.weight.shape[0]
    mdtype = _mask_dtype(model)

    chat_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    plain_prompt = "The capital of France is"

    def forward_with_layer_ablation(prompt, ablate_layers, ablate_mode="zero_attn"):
        """Forward pass with specific layers ablated."""
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        h = inner.embed_tokens(input_ids)
        for i, ly in enumerate(inner.layers):
            if i in ablate_layers:
                if ablate_mode == "skip":
                    # Skip the entire layer (no attention, no MLP)
                    continue
                elif ablate_mode == "zero_attn":
                    # Run MLP but zero the attention contribution
                    attn_out = ly.self_attn(ly.input_layernorm(h), mask=mask, cache=None)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    # Don't add attn_out to residual — skip it
                    h_post_attn = h  # no attention contribution
                    h = h_post_attn + ly.mlp(ly.post_attention_layernorm(h_post_attn))
                    continue
                elif ablate_mode == "zero_all":
                    # Zero the entire hidden state at this layer
                    h = mx.zeros_like(h)
                    continue

            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]

        h = inner.norm(h)
        logits = np.array(model.lm_head(h).astype(mx.float32)[0, -1, :])
        return logits

    # Baseline
    baseline_logits = forward_with_layer_ablation(chat_prompt, set())
    baseline_probs = softmax(baseline_logits)
    baseline_top = tokenizer.decode([int(np.argmax(baseline_probs))])
    print(f"  Chat baseline: top={baseline_top!r}")

    plain_logits = forward_with_layer_ablation(plain_prompt, set())
    plain_probs = softmax(plain_logits)
    plain_top = tokenizer.decode([int(np.argmax(plain_probs))])
    print(f"  Plain baseline: top={plain_top!r}")

    # Test: skip/zero layers in groups
    print("\n  --- Layer ablation (zero attention contribution) ---")
    layer_groups = {
        "L0": {0}, "L1": {1}, "L0-1": {0, 1}, "L0-2": {0, 1, 2},
        "L0-3": {0, 1, 2, 3}, "L0-5": set(range(6)), "L0-9": set(range(10)),
        "L0-13": set(range(14)), "L0-20": set(range(21)),
        "L3-5": {3, 4, 5}, "L5-10": set(range(5, 11)),
        "L10-20": set(range(10, 21)), "L20-27": set(range(20, 28)),
        "L25-27": {25, 26, 27}, "L27": {27},
    }

    for gname, layers in layer_groups.items():
        logits = forward_with_layer_ablation(chat_prompt, layers, "zero_attn")
        probs = softmax(logits)
        top_tok = tokenizer.decode([int(np.argmax(probs))])
        kl = kl_divergence(baseline_probs, probs)
        changed = " CHANGED" if np.argmax(probs) != np.argmax(baseline_probs) else ""
        print(f"    zero_attn {gname:8s}: top={top_tok!r:15s} KL={kl:.4f}{changed}")

    # Test: skip entire layers (no MLP either)
    print("\n  --- Layer ablation (skip entire layer) ---")
    for gname, layers in [("L0-1", {0, 1}), ("L0-5", set(range(6))),
                           ("L0-9", set(range(10))), ("L25-27", {25, 26, 27})]:
        logits = forward_with_layer_ablation(chat_prompt, layers, "skip")
        probs = softmax(logits)
        top_tok = tokenizer.decode([int(np.argmax(probs))])
        kl = kl_divergence(baseline_probs, probs)
        changed = " CHANGED" if np.argmax(probs) != np.argmax(baseline_probs) else ""
        print(f"    skip      {gname:8s}: top={top_tok!r:15s} KL={kl:.4f}{changed}")

    # Test: does a plain prompt ALSO answer correctly with chat layers zeroed?
    print("\n  --- Plain prompt with layer ablation ---")
    for gname, layers in [("none", set()), ("L0-1", {0, 1}), ("L0-5", set(range(6)))]:
        logits = forward_with_layer_ablation(plain_prompt, layers, "zero_attn")
        probs = softmax(logits)
        top_tok = tokenizer.decode([int(np.argmax(probs))])
        print(f"    plain, zero_attn {gname:8s}: top={top_tok!r}")

    # Test: generate with layer ablation to see full text
    print("\n  --- Chat generation with layer ablation ---")
    for gname, layers in [("none", set()), ("L0-1", {0, 1}), ("L0-5", set(range(6))),
                           ("L0-13", set(range(14)))]:
        # Use head-level modifications to approximate layer zeroing
        # Zero ALL 28 heads at the target layers
        mods = {}
        for ly in layers:
            for h in range(28):
                mods[(ly, h)] = 0.0
        result = generate_steered(model, tokenizer, chat_prompt, mods, max_tokens=20)
        print(f"    {gname:8s}: {result['generated'][:70]!r}")


# ============================================================
# U3: NEURON 18757 — WHAT DOES IT ENCODE?
# ============================================================
def u3_neuron_18757(model, tokenizer):
    """Multi-prompt characterization of the dominant MLP neuron at L27."""
    print("\n" + "=" * 70)
    print("U3: NEURON 18757 AT LAYER 27 — WHAT DOES IT ENCODE?")
    print("=" * 70)

    import mlx.core as mx
    import mlx.nn as nn

    inner = getattr(model, "model", model)
    mdtype = _mask_dtype(model)

    prompts = {
        "weather_en": "The weather today is",
        "weather_zh": "今天的天气是",
        "tiananmen": "On June 4th, 1989, in Tiananmen Square,",
        "code": "def fibonacci(n):\n    ",
        "math": "The sum of 2 + 2 equals",
        "chat": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        "story": "Once upon a time, there was a",
        "identity": "I am a large language model developed by",
    }

    target_neurons = [18757, 7960, 800, 16990]  # top neurons from batch sweep

    for pname, prompt in prompts.items():
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        # Forward through layers 0-26
        h = inner.embed_tokens(input_ids)
        for i in range(27):
            h = inner.layers[i](h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]

        # Layer 27: attention part
        layer27 = inner.layers[27]
        h_attn = layer27.input_layernorm(h)
        attn_out = layer27.self_attn(h_attn, mask=mask, cache=None)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h_after_attn = h + attn_out

        # MLP gate activations
        mlp = layer27.mlp
        h_normed = layer27.post_attention_layernorm(h_after_attn)
        gate = mlp.gate_proj(h_normed)
        up = mlp.up_proj(h_normed)
        activated = nn.silu(gate) * up

        # Get activation values for target neurons at last position
        act_np = np.array(activated.astype(mx.float32)[0, -1, :])

        # Baseline
        mlp_out = mlp.down_proj(activated)
        h_base = h_after_attn + mlp_out
        h_final = inner.norm(h_base)
        base_logits = np.array(model.lm_head(h_final).astype(mx.float32)[0, -1, :])
        base_probs = softmax(base_logits)
        base_top = tokenizer.decode([int(np.argmax(base_probs))])

        neuron_info = []
        for n_idx in target_neurons:
            act_val = float(act_np[n_idx])

            # Zero this neuron and measure effect
            modified = np.array(activated.astype(mx.float32))
            modified[0, :, n_idx] = 0.0
            mlp_out_mod = mlp.down_proj(mx.array(modified.astype(np.float16)))
            h_mod = h_after_attn + mlp_out_mod
            h_mod_final = inner.norm(h_mod)
            mod_logits = np.array(model.lm_head(h_mod_final).astype(mx.float32)[0, -1, :])
            mod_probs = softmax(mod_logits)
            mod_top = tokenizer.decode([int(np.argmax(mod_probs))])
            kl = kl_divergence(base_probs, mod_probs)
            neuron_info.append((n_idx, act_val, kl, mod_top))

        print(f"\n  {pname:15s} top={base_top!r:15s}")
        for n_idx, act_val, kl, mod_top in neuron_info:
            changed = f" → {mod_top!r}" if mod_top != base_top else ""
            print(f"    neuron {n_idx:5d}: act={act_val:+8.3f}  KL={kl:.6f}{changed}")


# ============================================================
# U4: FULL ATTENTION MAP OF CHAT PROMPT
# ============================================================
def u4_chat_attention_all_layers(model, tokenizer):
    """Capture attention patterns at ALL 28 layers for the chat prompt."""
    print("\n" + "=" * 70)
    print("U4: FULL ATTENTION MAP — CHAT PROMPT, ALL LAYERS")
    print("Which layers/heads attend to <|im_start|> and role tokens?")
    print("=" * 70)

    chat_prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

    # Get token positions for special tokens
    tokens = tokenizer.encode(chat_prompt)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    print(f"  Tokens: {list(enumerate(token_strs))}")

    # Find positions of special tokens
    special_positions = []
    for i, t in enumerate(token_strs):
        if '<|' in t or t.strip() in ['user', 'assistant', 'system']:
            special_positions.append(i)
            print(f"    special: pos {i} = {t!r}")

    # Capture attention at all layers
    print(f"\n  Capturing attention at all 28 layers...")
    data = capture_attention_maps(model, tokenizer, chat_prompt, layers=list(range(28)))

    # For each layer, find heads that strongly attend to special tokens from last position
    print(f"\n  {'Layer':>5s}  {'Head':>4s}  {'Attends to':>30s}  {'Weight':>8s}  {'Entropy':>8s}")
    print("  " + "-" * 65)

    chat_focused_heads = []
    for layer_idx in sorted(data["attention_maps"].keys()):
        attn_map = data["attention_maps"][layer_idx]
        n_heads = attn_map.shape[0]
        for h in range(n_heads):
            last_row = attn_map[h, -1, :]  # attention from last position
            entropy = float(-np.sum(last_row * np.log2(last_row + 1e-12)))

            # Check attention to special token positions
            for sp in special_positions:
                if last_row[sp] > 0.15:
                    token_name = token_strs[sp].strip() or "\\n"
                    chat_focused_heads.append((layer_idx, h, sp, token_name, float(last_row[sp]), entropy))
                    print(f"  L{layer_idx:2d}    H{h:2d}   {token_name:>30s}  {last_row[sp]:8.3f}  {entropy:8.2f}")

    print(f"\n  Total heads focused on special tokens: {len(chat_focused_heads)}")

    # Compare: same analysis for a plain prompt
    plain_prompt = "The capital of France is"
    data_plain = capture_attention_maps(model, tokenizer, plain_prompt, layers=[0, 1, 4, 27])
    print(f"\n  --- Plain prompt comparison (L0, L1, L4, L27) ---")
    for layer_idx in sorted(data_plain["attention_maps"].keys()):
        attn_map = data_plain["attention_maps"][layer_idx]
        # Lowest entropy heads (most focused)
        entropies = []
        for h in range(attn_map.shape[0]):
            last_row = attn_map[h, -1, :]
            ent = float(-np.sum(last_row * np.log2(last_row + 1e-12)))
            top_pos = int(np.argmax(last_row))
            top_tok = data_plain["tokens"][top_pos]
            entropies.append((h, ent, top_tok, float(last_row[top_pos])))
        entropies.sort(key=lambda x: x[1])
        for h, ent, tok, w in entropies[:3]:
            print(f"    L{layer_idx:2d} H{h:2d}: entropy={ent:.2f}  attends={tok!r}({w:.2f})")


# ============================================================
# U5: CODE CIRCUIT MAPPING
# ============================================================
def u5_code_circuit(model, tokenizer):
    """Map which heads are specific to code generation vs language."""
    print("\n" + "=" * 70)
    print("U5: CODE-SPECIFIC CIRCUIT MAPPING")
    print("=" * 70)

    # From Q5: code activates heads 27.8, 27.10, 27.7 instead of 27.2
    # Let's verify with attention capture and steering

    code_prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci("
    english_prompt = "The Fibonacci sequence starts with 0, 1, and each subsequent number is"

    # Attention capture at key layers
    for pname, prompt in [("code", code_prompt), ("english", english_prompt)]:
        print(f"\n  --- {pname} ---")
        data = capture_attention_maps(model, tokenizer, prompt, layers=[5, 15, 27])
        for layer_idx in sorted(data["attention_maps"].keys()):
            attn_map = data["attention_maps"][layer_idx]
            # Most focused heads
            profiles = []
            for h in range(attn_map.shape[0]):
                p = head_attention_profile(attn_map, h, data["tokens"])
                profiles.append(p)
            profiles.sort(key=lambda x: x["entropy"])
            for p in profiles[:3]:
                top_str = ", ".join(f"{t}({w:.2f})" for t, w in p["top_attended"][:3])
                print(f"    L{layer_idx:2d} H{p['head']:2d}: entropy={p['entropy']:.2f} [{top_str}]")

    # Steering: generate code with/without code-specific heads
    print(f"\n  --- Steering test: generate code ---")
    code_start = "def fibonacci(n):\n    "

    configs = {
        "baseline": {},
        "zero_27.8": {(27, 8): 0.0},
        "zero_27.7": {(27, 7): 0.0},
        "zero_27.10": {(27, 10): 0.0},
        "zero_all_code": {(27, 8): 0.0, (27, 7): 0.0, (27, 10): 0.0},
        "zero_27.2(english)": {(27, 2): 0.0},
        "amplify_code": {(27, 8): 2.0, (27, 7): 2.0, (27, 10): 2.0},
    }

    for cname, mods in configs.items():
        result = generate_steered(model, tokenizer, code_start, mods, max_tokens=25)
        print(f"    {cname:25s}: {result['generated'][:60]!r}")


def main():
    model, tokenizer = load_model("mlx-community/Qwen2.5-7B-4bit")

    u1_head_vs_residual(model, tokenizer)
    u2_instruction_following_location(model, tokenizer)
    u3_neuron_18757(model, tokenizer)
    u4_chat_attention_all_layers(model, tokenizer)
    u5_code_circuit(model, tokenizer)

    print("\n\nALL UNKNOWNS RESOLVED.")


if __name__ == "__main__":
    main()
