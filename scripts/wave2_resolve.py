#!/usr/bin/env python3
"""Wave 2: Resolve everything avoided, unmapped, and unresolved.

AVOIDED:
A1: Ablate decision neurons 2228/9005 at L7 — does chat mode break?
A2: Generate text with master neuron 16992 zeroed
A3: Cross-validate directions vs bands
A4: Test on Llama 3B (different model family)
A5: Trace signal at ALL token positions, not just last

UNMAPPED:
U1: Scan neurons at EVERY layer (L0-L27) for chat sensitivity
U2: Trace the signal pathway: does L7→L21→L27 cascade causally?
U3: Extract down_proj vector of neuron 16992 — what direction does it push?
U4: O_proj superclusters at layers other than 27
U5: The 438 unused dimensions at L27

UNRESOLVED:
R1: Is the cascade causal? Ablate L7 neurons, check L21/L27
R2: What does chat mode compute differently? Compare internals token-by-token
R3: Why does safe→unsafe patching overshoot?
R4: Why is sensitive-vs-benign effect size 4x lower?
R5: Connect code supercluster to code direction
"""
import sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heinrich.cartography.neurons import capture_mlp_activations, scan_neurons
from heinrich.cartography.directions import capture_residual_states, find_direction, steer_with_direction
from heinrich.cartography.patch import capture_all_states, sweep_band_patches, sweep_layer_patches
from heinrich.cartography.oproj import decompose_oproj
from heinrich.cartography.steer import generate_steered
from heinrich.cartography.perturb import compute_baseline, _mask_dtype
from heinrich.cartography.attention import capture_attention_maps
from heinrich.inspect.self_analysis import _softmax
from heinrich.signal import SignalStore


def load(model_id):
    import mlx_lm
    print(f"Loading {model_id}...")
    t0 = time.time()
    m, t = mlx_lm.load(model_id)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return m, t

def kl(p, q):
    return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))

def get_probs(model, tokenizer, prompt):
    return _softmax(compute_baseline(model, tokenizer, prompt))


# ============================================================
# A1: ABLATE DECISION NEURONS AT L7
# ============================================================
def a1_ablate_decision_neurons(model, tokenizer):
    print("\n" + "="*70)
    print("A1: ABLATE DECISION NEURONS 2228 AND 9005 AT LAYER 7")
    print("Does chat mode break?")
    print("="*70)

    import mlx.core as mx
    import mlx.nn as nn
    inner = getattr(model, "model", model)
    mdtype = _mask_dtype(model)

    chat = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    plain = "The capital of France is"

    def generate_with_neuron_ablation(prompt, layer, neurons_to_zero, max_tokens=20):
        """Generate text with specific MLP neurons zeroed at a specific layer."""
        tokens = list(tokenizer.encode(prompt))
        generated = []
        for _ in range(max_tokens):
            input_ids = mx.array([tokens])
            T = len(tokens)
            mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None
            h = inner.embed_tokens(input_ids)
            for i, ly in enumerate(inner.layers):
                if i == layer:
                    # Manual MLP with neuron ablation
                    h_attn = ly.input_layernorm(h)
                    attn_out = ly.self_attn(h_attn, mask=mask, cache=None)
                    if isinstance(attn_out, tuple): attn_out = attn_out[0]
                    h = h + attn_out
                    h_normed = ly.post_attention_layernorm(h)
                    gate = ly.mlp.gate_proj(h_normed)
                    up = ly.mlp.up_proj(h_normed)
                    activated = nn.silu(gate) * up
                    # Zero target neurons
                    act_np = np.array(activated.astype(mx.float32))
                    for n in neurons_to_zero:
                        act_np[0, :, n] = 0.0
                    mlp_out = ly.mlp.down_proj(mx.array(act_np.astype(np.float16)))
                    h = h + mlp_out
                else:
                    h = ly(h, mask=mask, cache=None)
                    if isinstance(h, tuple): h = h[0]
            h = inner.norm(h)
            logits = np.array(model.lm_head(h).astype(mx.float32)[0, -1, :])
            next_id = int(np.argmax(logits))
            eos = getattr(tokenizer, "eos_token_id", None)
            if next_id == eos: break
            tokens.append(next_id)
            generated.append(next_id)
        return tokenizer.decode(generated)

    # Baseline
    print(f"\n  Chat prompt:")
    normal = generate_steered(model, tokenizer, chat, {}, max_tokens=20)
    print(f"    NORMAL:          {normal['generated'][:70]!r}")

    # Ablate individual L7 neurons
    for neurons, label in [
        ([2228], "zero 2228"),
        ([9005], "zero 9005"),
        ([2228, 9005], "zero BOTH"),
        ([2228, 9005, 10454, 15788, 17786], "zero ALL 5 top L7"),
    ]:
        result = generate_with_neuron_ablation(chat, 7, neurons)
        print(f"    {label:25s}: {result[:70]!r}")

    # Also test on plain prompt as control
    print(f"\n  Plain prompt (control):")
    normal_plain = generate_steered(model, tokenizer, plain, {}, max_tokens=15)
    print(f"    NORMAL:          {normal_plain['generated'][:60]!r}")
    for neurons, label in [([2228, 9005], "zero BOTH L7")]:
        result = generate_with_neuron_ablation(plain, 7, neurons)
        print(f"    {label:25s}: {result[:60]!r}")

    # Test: ablate master neuron 16992 at L27
    print(f"\n  Ablate MASTER neuron 16992 at L27:")
    for prompt_name, prompt in [("chat", chat), ("plain", plain)]:
        result = generate_with_neuron_ablation(prompt, 27, [16992])
        print(f"    {prompt_name:8s} zero 16992: {result[:70]!r}")

    # Test: ablate top 5 chat neurons at L27
    print(f"\n  Ablate top 5 L27 chat neurons (16992, 4033, 12818, 5496, 17894):")
    top5 = [16992, 4033, 12818, 5496, 17894]
    result = generate_with_neuron_ablation(chat, 27, top5)
    print(f"    chat zero top5 L27: {result[:70]!r}")

    return generate_with_neuron_ablation


# ============================================================
# A3: CROSS-VALIDATE DIRECTIONS VS BANDS
# ============================================================
def a3_directions_vs_bands(model, tokenizer):
    print("\n" + "="*70)
    print("A3: DO DIRECTIONS MATCH BANDS?")
    print("="*70)

    # Recompute English-vs-Chinese direction at L27
    en_prompts = ["The weather today is", "Once upon a time", "The capital of France is",
                  "Scientists have discovered", "The most important thing is"]
    zh_prompts = ["今天的天气是", "从前，有一个", "法国的首都是",
                  "科学家发现", "生活中最重要的事情是"]

    states = capture_residual_states(model, tokenizer, en_prompts + zh_prompts, layers=[27])
    en_states = states[27][:len(en_prompts)]
    zh_states = states[27][len(en_prompts):]

    lang_dir = find_direction(en_states, zh_states, name="lang", layer=27)

    # Find chat direction
    chat_prompts = [
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    ]
    plain_prompts = ["Hello, how are you?", "The weather is nice."]
    states2 = capture_residual_states(model, tokenizer, chat_prompts + plain_prompts, layers=[27])
    chat_states = states2[27][:len(chat_prompts)]
    plain_states = states2[27][len(chat_prompts):]
    chat_dir = find_direction(chat_states, plain_states, name="chat", layer=27)

    hidden_size = len(lang_dir.direction)
    head_dim = hidden_size // 28

    # Project each direction onto each band
    print(f"\n  Language direction weight by band:")
    for band in range(28):
        start = band * head_dim
        end = start + head_dim
        band_weight = float(np.linalg.norm(lang_dir.direction[start:end]))
        bar = "█" * int(band_weight * 100)
        if band_weight > 0.05:
            print(f"    band {band:2d}: {band_weight:.4f} {bar}")

    print(f"\n  Chat direction weight by band:")
    for band in range(28):
        start = band * head_dim
        end = start + head_dim
        band_weight = float(np.linalg.norm(chat_dir.direction[start:end]))
        bar = "█" * int(band_weight * 100)
        if band_weight > 0.05:
            print(f"    band {band:2d}: {band_weight:.4f} {bar}")

    # Compare: patching said band 0 = 49% of language. Does the direction agree?
    band0_lang = float(np.linalg.norm(lang_dir.direction[:head_dim]))
    band0_chat = float(np.linalg.norm(chat_dir.direction[:head_dim]))
    band6_chat = float(np.linalg.norm(chat_dir.direction[6*head_dim:7*head_dim]))
    print(f"\n  Key comparisons:")
    print(f"    Band 0 language weight: {band0_lang:.4f} (patching said 49% recovery)")
    print(f"    Band 0 chat weight:     {band0_chat:.4f} (patching said 28% recovery)")
    print(f"    Band 6 chat weight:     {band6_chat:.4f} (patching said 42% recovery)")


# ============================================================
# A5: TRACE SIGNAL AT ALL TOKEN POSITIONS
# ============================================================
def a5_all_positions(model, tokenizer):
    print("\n" + "="*70)
    print("A5: SIGNAL AT ALL TOKEN POSITIONS (not just last)")
    print("="*70)

    import mlx.core as mx
    inner = getattr(model, "model", model)
    mdtype = _mask_dtype(model)

    chat = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    plain = "Hello, how are you today?"

    # Get token labels
    chat_tokens = tokenizer.encode(chat)
    chat_labels = [tokenizer.decode([t]) for t in chat_tokens]
    print(f"  Chat tokens: {list(enumerate(chat_labels))}")

    # Capture hidden states at ALL positions for key layers
    def get_all_position_states(prompt, layers):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None
        h = inner.embed_tokens(input_ids)
        states = {}
        for i, ly in enumerate(inner.layers):
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple): h = h[0]
            if i in layers:
                states[i] = np.array(h.astype(mx.float32)[0])  # [T, hidden_size]
        return states

    chat_states = get_all_position_states(chat, [0, 1, 7, 13, 21, 27])
    plain_states = get_all_position_states(plain, [0, 1, 7, 13, 21, 27])

    # For each layer, compute the norm of (chat_state - plain_state) at each position
    # This shows WHERE in the sequence the chat/plain divergence appears
    print(f"\n  Residual stream divergence at each token position (L2 norm):")
    print(f"  {'Layer':>5s}  " + "  ".join(f"{l:>8s}" for l in chat_labels[:9]))

    for layer in [0, 1, 7, 13, 21, 27]:
        # Chat has different number of tokens than plain — compare what we can
        chat_s = chat_states[layer]  # [T_chat, hidden_size]
        plain_s = plain_states[layer]  # [T_plain, hidden_size]
        # Report norms at each chat position
        norms = [float(np.linalg.norm(chat_s[i])) for i in range(min(9, chat_s.shape[0]))]
        norm_str = "  ".join(f"{n:8.1f}" for n in norms)
        print(f"  L{layer:2d}    {norm_str}")

    # More useful: project onto chat direction at each position
    # First get the chat direction
    en_prompts = ["The weather today is", "Hello, how are you?"]
    ch_prompts = ["<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
                  "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"]
    dir_states = capture_residual_states(model, tokenizer, ch_prompts + en_prompts, layers=[7])
    chat_dir_7 = find_direction(dir_states[7][:2], dir_states[7][2:], name="chat", layer=7)

    print(f"\n  Chat direction projection at each position (L7):")
    print(f"  {'Pos':>4s}  {'Token':>15s}  {'Projection':>12s}")
    for pos in range(min(9, chat_states[7].shape[0])):
        proj = float(np.dot(chat_states[7][pos], chat_dir_7.direction))
        bar = "█" * int(abs(proj) / 2)
        sign = "+" if proj > 0 else "-"
        print(f"  {pos:4d}  {chat_labels[pos]:>15s}  {proj:+12.2f}  {sign}{bar}")


# ============================================================
# U1: FULL NEURON CASCADE MAP (every layer)
# ============================================================
def u1_full_cascade(model, tokenizer):
    print("\n" + "="*70)
    print("U1: FULL NEURON CASCADE MAP — EVERY LAYER")
    print("="*70)

    chat_prompts = [
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    ]
    plain_prompts = ["Hello, how are you?", "What is 2+2?"]

    print(f"  {'Layer':>5s}  {'|Δ|>3':>6s}  {'|Δ|>10':>7s}  {'TopNeuron':>10s}  {'TopΔ':>10s}  {'2nd':>10s}")
    for layer in range(28):
        r = scan_neurons(model, tokenizer, chat_prompts, plain_prompts, layer, top_k=5)
        n_large = r.n_large_diff
        n_10 = sum(1 for p in r.selective_neurons if abs(p.selectivity) > 10.0)
        top = r.selective_neurons[0] if r.selective_neurons else None
        second = r.selective_neurons[1] if len(r.selective_neurons) > 1 else None
        top_str = f"{top.neuron:5d} {top.selectivity:+.1f}" if top else "none"
        sec_str = f"{second.neuron:5d} {second.selectivity:+.1f}" if second else ""
        print(f"  L{layer:2d}    {n_large:4d}    {n_10:5d}    {top_str:>12s}  {sec_str:>12s}")


# ============================================================
# U2: CAUSAL CASCADE — ablate L7, check L21/L27
# ============================================================
def u2_causal_cascade(model, tokenizer, gen_fn):
    print("\n" + "="*70)
    print("U2: IS THE CASCADE CAUSAL? Ablate L7, check downstream")
    print("="*70)

    import mlx.core as mx
    import mlx.nn as nn
    inner = getattr(model, "model", model)
    mdtype = _mask_dtype(model)

    chat = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

    def get_neuron_activations_with_ablation(prompt, ablation_layer, ablation_neurons, target_layers):
        """Forward pass with neurons ablated at ablation_layer, capture MLP activations at target_layers."""
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None
        h = inner.embed_tokens(input_ids)
        target_acts = {}
        for i, ly in enumerate(inner.layers):
            if i == ablation_layer or i in target_layers:
                h_attn = ly.input_layernorm(h)
                attn_out = ly.self_attn(h_attn, mask=mask, cache=None)
                if isinstance(attn_out, tuple): attn_out = attn_out[0]
                h = h + attn_out
                h_normed = ly.post_attention_layernorm(h)
                gate = ly.mlp.gate_proj(h_normed)
                up = ly.mlp.up_proj(h_normed)
                activated = nn.silu(gate) * up

                if i in target_layers:
                    target_acts[i] = np.array(activated.astype(mx.float32)[0, -1, :])

                if i == ablation_layer and ablation_neurons:
                    act_np = np.array(activated.astype(mx.float32))
                    for n in ablation_neurons:
                        act_np[0, :, n] = 0.0
                    activated = mx.array(act_np.astype(np.float16))

                mlp_out = ly.mlp.down_proj(activated)
                h = h + mlp_out
            else:
                h = ly(h, mask=mask, cache=None)
                if isinstance(h, tuple): h = h[0]
        return target_acts

    # Normal: capture L21 and L27 activations
    normal_acts = get_neuron_activations_with_ablation(chat, -1, [], [7, 21, 27])
    # Ablate L7 decision neurons: capture L21 and L27 activations
    ablated_acts = get_neuron_activations_with_ablation(chat, 7, [2228, 9005], [7, 21, 27])

    # Key neurons to check downstream
    key_neurons = {
        21: [274, 15581, 9477, 28, 14376],  # top chat neurons at L21
        27: [16992, 4033, 12818, 5496, 17894],  # top chat neurons at L27
    }

    print(f"  Effect of ablating L7 neurons 2228+9005 on downstream chat neurons:")
    for target_layer, neurons in key_neurons.items():
        print(f"\n  Layer {target_layer}:")
        print(f"    {'Neuron':>8s}  {'Normal':>10s}  {'Ablated':>10s}  {'Change':>10s}  {'%Change':>8s}")
        for n in neurons:
            norm_val = float(normal_acts[target_layer][n])
            abl_val = float(ablated_acts[target_layer][n])
            change = abl_val - norm_val
            pct = abs(change / (norm_val + 1e-6)) * 100
            print(f"    {n:6d}    {norm_val:+10.2f}  {abl_val:+10.2f}  {change:+10.2f}  {pct:7.1f}%")


# ============================================================
# U3: DOWN_PROJ VECTOR OF MASTER NEURON
# ============================================================
def u3_downproj_vector(model, tokenizer):
    print("\n" + "="*70)
    print("U3: WHAT DIRECTION DOES NEURON 16992 PUSH?")
    print("="*70)

    import mlx.core as mx
    inner = getattr(model, "model", model)
    hidden_size = inner.norm.weight.shape[0]

    # Extract down_proj column for neuron 16992
    mlp27 = inner.layers[27].mlp
    # Probe: pass a one-hot activation through down_proj
    one_hot = np.zeros((1, 1, 18944), dtype=np.float16)
    one_hot[0, 0, 16992] = 1.0
    down_vec = np.array(mlp27.down_proj(mx.array(one_hot)).astype(mx.float32)[0, 0, :])

    # Also get down_proj vectors for other key neurons
    neurons = {16992: "master_chat", 800: "suppress_assistant", 18757: "weather_boost", 4033: "anti_chat"}
    vecs = {}
    for n_idx, name in neurons.items():
        oh = np.zeros((1, 1, 18944), dtype=np.float16)
        oh[0, 0, n_idx] = 1.0
        vec = np.array(mlp27.down_proj(mx.array(oh)).astype(mx.float32)[0, 0, :])
        vecs[name] = vec
        norm = float(np.linalg.norm(vec))
        print(f"  Neuron {n_idx:5d} ({name:20s}): down_proj norm = {norm:.4f}")

    # Compare to behavioral directions
    en_prompts = ["The weather today is", "Once upon a time", "The capital is"]
    zh_prompts = ["今天的天气是", "从前，有一个", "首都是"]
    chat_prompts = ["<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"]
    plain_prompts = ["Hello, how are you?"]

    states = capture_residual_states(model, tokenizer, en_prompts + zh_prompts + chat_prompts + plain_prompts, layers=[27])
    s27 = states[27]
    lang_dir = find_direction(s27[:3], s27[3:6], name="lang", layer=27)
    chat_dir = find_direction(s27[6:7], s27[7:8], name="chat", layer=27)

    print(f"\n  Cosine similarity between neuron down_proj and behavioral directions:")
    for name, vec in vecs.items():
        vec_n = vec / (np.linalg.norm(vec) + 1e-12)
        cos_lang = float(np.dot(vec_n, lang_dir.direction))
        cos_chat = float(np.dot(vec_n, chat_dir.direction))
        print(f"    {name:20s}: cos(lang)={cos_lang:+.4f}  cos(chat)={cos_chat:+.4f}")

    # Band decomposition of master neuron's down_proj
    head_dim = hidden_size // 28
    print(f"\n  Neuron 16992 down_proj weight by band:")
    for band in range(28):
        start = band * head_dim
        w = float(np.linalg.norm(down_vec[start:start+head_dim]))
        if w > 0.01:
            bar = "█" * int(w * 20)
            print(f"    band {band:2d}: {w:.4f} {bar}")


# ============================================================
# U5: THE 438 UNUSED DIMENSIONS AT L27
# ============================================================
def u5_unused_dims(model, tokenizer):
    print("\n" + "="*70)
    print("U5: WHAT'S IN L27's UNUSED DIMENSIONS?")
    print("="*70)

    d27 = decompose_oproj(model, 27, top_k=50)
    print(f"  Effective rank: {d27.effective_rank} / {d27.hidden_size}")
    print(f"  Top 10 singular values: {[f'{s:.2f}' for s in d27.top_singular_values[:10]]}")

    # Extract the o_proj weight matrix and get the null space
    from heinrich.cartography.oproj import extract_oproj_weight
    W = extract_oproj_weight(model, 27)  # [hidden_size, n_heads * head_dim]

    U, S, Vt = np.linalg.svd(W, full_matrices=True)
    # The "used" dimensions are the first effective_rank columns of U
    # The "unused" are the rest
    used_dims = U[:, :d27.effective_rank]
    unused_dims = U[:, d27.effective_rank:]
    print(f"  Used subspace: {used_dims.shape[1]} dims")
    print(f"  Unused subspace: {unused_dims.shape[1]} dims")

    # Project actual residual states onto used vs unused
    prompts = ["The weather today is", "今天的天气是",
               "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
               "def fibonacci(n):\n    "]
    names = ["english", "chinese", "chat", "code"]

    import mlx.core as mx
    mdtype = _mask_dtype(model)
    inner = getattr(model, "model", model)

    print(f"\n  Residual stream projection onto used vs unused subspace:")
    for pname, prompt in zip(names, prompts):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None
        h = inner.embed_tokens(input_ids)
        for ly in inner.layers:
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple): h = h[0]
        h_np = np.array(h.astype(mx.float32)[0, -1, :])

        proj_used = float(np.linalg.norm(used_dims.T @ h_np))
        proj_unused = float(np.linalg.norm(unused_dims.T @ h_np))
        ratio = proj_unused / (proj_used + 1e-12)
        print(f"    {pname:10s}: used={proj_used:.1f}  unused={proj_unused:.1f}  ratio={ratio:.4f}")


# ============================================================
# R5: CONNECT CODE SUPERCLUSTER TO CODE DIRECTION
# ============================================================
def r5_code_connection(model, tokenizer):
    print("\n" + "="*70)
    print("R5: DO THE CODE SUPERCLUSTER AND CODE DIRECTION MATCH?")
    print("="*70)

    # Get code direction
    code_prompts = ["def fibonacci(n):\n    ", "import numpy as np\n\n", "class Node:\n    ",
                    "for i in range(10):\n    "]
    en_prompts = ["The weather today is", "Once upon a time", "The capital is",
                  "Scientists discovered that"]
    states = capture_residual_states(model, tokenizer, code_prompts + en_prompts, layers=[27])
    code_dir = find_direction(states[27][:4], states[27][4:], name="code", layer=27)

    # Get o_proj decomposition at L27
    d27 = decompose_oproj(model, 27)

    # The supercluster is heads 10, 12, 13 (cos>0.91)
    # Extract their combined o_proj output direction
    from heinrich.cartography.oproj import extract_oproj_weight
    W = extract_oproj_weight(model, 27)
    hidden_size = d27.hidden_size
    head_dim = d27.head_dim

    cluster_heads = [7, 10, 11, 12, 13]
    cluster_vecs = []
    for h in cluster_heads:
        col_slice = W[:, h*head_dim:(h+1)*head_dim]
        # Get the principal direction of this head's output
        U, S, Vt = np.linalg.svd(col_slice, full_matrices=False)
        cluster_vecs.append(U[:, 0] * S[0])  # first principal component weighted by singular value

    # Mean cluster direction
    cluster_dir = np.mean(cluster_vecs, axis=0)
    cluster_dir = cluster_dir / (np.linalg.norm(cluster_dir) + 1e-12)

    cos = float(np.dot(cluster_dir, code_dir.direction))
    print(f"  Code direction vs supercluster mean direction: cos = {cos:+.4f}")

    # Per-head alignment
    for h, vec in zip(cluster_heads, cluster_vecs):
        vec_n = vec / (np.linalg.norm(vec) + 1e-12)
        cos_h = float(np.dot(vec_n, code_dir.direction))
        print(f"    Head {h:2d} principal direction vs code direction: cos = {cos_h:+.4f}")

    # Also check language direction alignment
    lang_dir = find_direction(states[27][4:], states[27][:4], name="en_vs_code", layer=27)
    cos_lang = float(np.dot(cluster_dir, lang_dir.direction))
    print(f"\n  Supercluster vs language direction: cos = {cos_lang:+.4f}")


def main():
    model, tokenizer = load("mlx-community/Qwen2.5-7B-4bit")

    gen_fn = a1_ablate_decision_neurons(model, tokenizer)
    a3_directions_vs_bands(model, tokenizer)
    a5_all_positions(model, tokenizer)
    u1_full_cascade(model, tokenizer)
    u2_causal_cascade(model, tokenizer, gen_fn)
    u3_downproj_vector(model, tokenizer)
    u5_unused_dims(model, tokenizer)
    r5_code_connection(model, tokenizer)

    print("\n\nWAVE 2 COMPLETE.")


if __name__ == "__main__":
    main()
