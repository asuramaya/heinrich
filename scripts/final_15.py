#!/usr/bin/env python3
"""Answer all 15 remaining mysteries about Qwen 2.5 7B base.

MYSTERIOUS:
M1: Why does the residual stream have language/code bands? Test with shuffled o_proj.
M2: Neuron 800 suppresses "Assistant" — how deep does role-awareness go in MLP?
M3: 480 attention pairs — quantify the mode switch. How different are the two modes?
M4: Code dims encode algorithm strategy — what specifically? Test more code prompts.
M5: L21H15 attends to "assistant" at 0.911 — what does it write to the residual?

WEIRD:
W1: English amplification → "111111" collapse — reproduce and characterize the failure.
W2: Language perturbation on Tiananmen → "Assistant: B" leak — is this topic-specific?
W3: Random dims > head-aligned dims — map the ACTUAL important dimensions.
W4: Code heads attend to "(" — test other syntactic anchors.
W5: [INST] weakly triggers chat — test more bracket/instruction markers.

UNRESOLVED:
R1: Is this really a base model? Compare chat depth to a known-clean base (Qwen 3B).
R2: Causal direction — build activation patching.
R3: What does L0 compute? Ablate L0 attention on many prompt types.
R4: Transfer — run key experiments on a different model family.
R5: What are the inert 47% for? Extreme stress test.
"""
import sys
import time
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heinrich.cartography.surface import ControlSurface
from heinrich.cartography.perturb import compute_baseline, _mask_dtype, perturb_head
from heinrich.cartography.sweep import coarse_head_sweep
from heinrich.cartography.attention import capture_attention_maps, head_attention_profile
from heinrich.cartography.steer import generate_steered, steer_next_token
from heinrich.cartography.runtime import load_model
from heinrich.cartography.metrics import softmax, kl_divergence
from heinrich.signal import SignalStore


def load(model_id):
    print(f"Loading {model_id}...")
    t0 = time.time()
    m, t = load_model(model_id)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return m, t


def kl(p, q):
    return kl_divergence(p, q)


def get_probs(model, tokenizer, prompt):
    logits = compute_baseline(model, tokenizer, prompt)
    return softmax(logits)


# Custom forward: supports zero_attn mode (skip attention, keep MLP) not available in runtime.forward_pass
def forward_ablate_layer(model, tokenizer, prompt, ablate_layers, mode="zero_attn"):
    """Forward pass with layers ablated. Returns logits."""
    import mlx.core as mx
    inner = getattr(model, "model", model)
    mdtype = _mask_dtype(model)
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None
    h = inner.embed_tokens(input_ids)
    for i, ly in enumerate(inner.layers):
        if i in ablate_layers:
            if mode == "skip":
                continue
            elif mode == "zero_attn":
                h_post = h  # skip attention contribution
                h = h_post + ly.mlp(ly.post_attention_layernorm(h_post))
                continue
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]
    h = inner.norm(h)
    return np.array(model.lm_head(h).astype(mx.float32)[0, -1, :])


# ============================================================
# M1: WHY LANGUAGE BANDS? Test o_proj structure directly.
# ============================================================
def m1_oproj_structure(model, tokenizer):
    print("\n" + "="*70)
    print("M1: WHY DOES THE RESIDUAL STREAM HAVE LANGUAGE BANDS?")
    print("="*70)
    import mlx.core as mx
    inner = getattr(model, "model", model)
    # Extract o_proj weight matrix at layer 27
    attn = inner.layers[27].self_attn
    # For quantized: need to dequantize
    # Compute column norms: which INPUT dimensions (head dims) map most to which OUTPUT dims
    # Test: project a one-hot head-dim vector through o_proj
    hidden_size = inner.norm.weight.shape[0]
    n_heads = attn.n_heads
    head_dim = hidden_size // n_heads

    print(f"  hidden_size={hidden_size} n_heads={n_heads} head_dim={head_dim}")
    print(f"\n  Per-head output norm through o_proj (how much does each head's output spread?):")

    head_norms = []
    for h in range(n_heads):
        # Create input with only head h's dimensions active
        inp = np.zeros((1, 1, hidden_size), dtype=np.float16)
        inp[0, 0, h*head_dim:(h+1)*head_dim] = 1.0 / np.sqrt(head_dim)  # unit norm
        out = attn.o_proj(mx.array(inp))
        out_np = np.array(out.astype(mx.float32)[0, 0, :])
        norm = float(np.linalg.norm(out_np))
        # Which output dimensions get the most energy?
        top_dims = np.argsort(np.abs(out_np))[::-1][:5]
        head_norms.append((h, norm, top_dims, out_np))

    head_norms.sort(key=lambda x: x[1], reverse=True)
    for h, norm, top_dims, _ in head_norms[:10]:
        print(f"    head {h:2d}: output_norm={norm:.4f}  top_dims={list(top_dims)}")

    # Key question: do different heads write to different output dimensions?
    # Compute overlap: for each pair of heads, how much do their output patterns overlap?
    print(f"\n  Output dimension overlap between key heads:")
    key_heads = [2, 21, 8, 10, 0, 7]
    for i, h1 in enumerate(key_heads):
        for h2 in key_heads[i+1:]:
            v1 = head_norms[h1][3] if h1 < len(head_norms) else np.zeros(hidden_size)
            v2 = head_norms[h2][3] if h2 < len(head_norms) else np.zeros(hidden_size)
            # Find the actual entries
            v1 = [x[3] for x in head_norms if x[0]==h1][0]
            v2 = [x[3] for x in head_norms if x[0]==h2][0]
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            print(f"    head {h1:2d} vs head {h2:2d}: cosine={cos:+.4f}")


# ============================================================
# M2: ROLE AWARENESS IN MLP — how many neurons encode chat roles?
# ============================================================
def m2_role_neurons(model, tokenizer):
    print("\n" + "="*70)
    print("M2: ROLE AWARENESS IN MLP — SCAN FOR CHAT-SPECIFIC NEURONS")
    print("="*70)
    import mlx.core as mx
    import mlx.nn as nn
    inner = getattr(model, "model", model)
    mdtype = _mask_dtype(model)

    prompts = {
        "chat": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        "plain": "Hello, how are you today?",
    }

    for layer_idx in [0, 5, 13, 21, 27]:
        activations = {}
        for pname, prompt in prompts.items():
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            T = len(tokens)
            mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None
            h = inner.embed_tokens(input_ids)
            for i in range(layer_idx):
                h = inner.layers[i](h, mask=mask, cache=None)
                if isinstance(h, tuple): h = h[0]
            layer = inner.layers[layer_idx]
            h_attn = layer.input_layernorm(h)
            attn_out = layer.self_attn(h_attn, mask=mask, cache=None)
            if isinstance(attn_out, tuple): attn_out = attn_out[0]
            h_after = h + attn_out
            h_normed = layer.post_attention_layernorm(h_after)
            gate = layer.mlp.gate_proj(h_normed)
            up = layer.mlp.up_proj(h_normed)
            act = nn.silu(gate) * up
            activations[pname] = np.array(act.astype(mx.float32)[0, -1, :])

        # Find neurons with biggest activation difference between chat and plain
        diff = activations["chat"] - activations["plain"]
        top_pos = np.argsort(diff)[::-1][:5]
        top_neg = np.argsort(diff)[:5]
        n_large = np.sum(np.abs(diff) > 3.0)

        print(f"\n  Layer {layer_idx}: {n_large} neurons with |Δact| > 3.0")
        print(f"    Most chat-activated:")
        for idx in top_pos[:3]:
            print(f"      neuron {idx:5d}: chat={activations['chat'][idx]:+.2f}  plain={activations['plain'][idx]:+.2f}  Δ={diff[idx]:+.2f}")
        print(f"    Most chat-suppressed:")
        for idx in top_neg[:3]:
            print(f"      neuron {idx:5d}: chat={activations['chat'][idx]:+.2f}  plain={activations['plain'][idx]:+.2f}  Δ={diff[idx]:+.2f}")


# ============================================================
# M3: QUANTIFY THE MODE SWITCH — how different are chat vs plain attention?
# ============================================================
def m3_mode_switch(model, tokenizer):
    print("\n" + "="*70)
    print("M3: HOW DIFFERENT ARE CHAT VS PLAIN ATTENTION MODES?")
    print("="*70)

    chat = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    plain = "Hello, how are you today?"

    chat_data = capture_attention_maps(model, tokenizer, chat, layers=list(range(28)))
    plain_data = capture_attention_maps(model, tokenizer, plain, layers=list(range(28)))

    # For each layer: what fraction of heads have entropy < 1.5 (focused)?
    print(f"\n  {'Layer':>5s}  {'Chat focused':>12s}  {'Plain focused':>13s}  {'Chat special%':>13s}")
    for layer in range(28):
        if layer not in chat_data["attention_maps"] or layer not in plain_data["attention_maps"]:
            continue
        chat_map = chat_data["attention_maps"][layer]
        plain_map = plain_data["attention_maps"][layer]
        n_heads = chat_map.shape[0]

        chat_focused = 0
        chat_special = 0
        plain_focused = 0
        for h in range(n_heads):
            # Chat
            last = chat_map[h, -1, :]
            ent = float(-np.sum(last * np.log2(last + 1e-12)))
            if ent < 1.5: chat_focused += 1
            # Check if max attention is on a special token position (0,1,4,6,7 for this prompt)
            special_pos = [0, 1, 4, 6, 7]
            max_pos = int(np.argmax(last))
            if max_pos in special_pos: chat_special += 1
            # Plain
            last_p = plain_map[h, -1, :]
            ent_p = float(-np.sum(last_p * np.log2(last_p + 1e-12)))
            if ent_p < 1.5: plain_focused += 1

        print(f"  L{layer:2d}    {chat_focused:3d}/28       {plain_focused:3d}/28        {chat_special:3d}/28")


# ============================================================
# M4: CODE DIMS ENCODE STRATEGY — test more code patterns
# ============================================================
def m4_code_strategy(model, tokenizer):
    print("\n" + "="*70)
    print("M4: WHAT DO CODE DIMENSIONS ENCODE?")
    print("="*70)

    code_prompts = {
        "sort":     "def bubble_sort(arr):\n    ",
        "binary":   "def binary_search(arr, target):\n    ",
        "class":    "class LinkedList:\n    def __init__(self):\n        ",
        "list_comp": "squares = [x**2 for x in range(",
        "import":   "import numpy as np\nimport pandas as pd\n\ndef load_data(path):\n    ",
    }

    code_mods = {(27, 8): 2.0, (27, 7): 2.0, (27, 10): 2.0}
    anti_mods = {(27, 8): 0.0, (27, 7): 0.0, (27, 10): 0.0}

    for pname, prompt in code_prompts.items():
        print(f"\n  --- {pname} ---")
        normal = generate_steered(model, tokenizer, prompt, {}, max_tokens=25)
        amped = generate_steered(model, tokenizer, prompt, code_mods, max_tokens=25)
        zeroed = generate_steered(model, tokenizer, prompt, anti_mods, max_tokens=25)
        print(f"    normal: {normal['generated'][:60]!r}")
        print(f"    amped:  {amped['generated'][:60]!r}")
        print(f"    zeroed: {zeroed['generated'][:60]!r}")


# ============================================================
# M5: L21H15 — what does it write?
# ============================================================
def m5_l21h15(model, tokenizer):
    print("\n" + "="*70)
    print("M5: WHAT DOES L21H15 DO? (attends to 'assistant' at 0.911)")
    print("="*70)

    chat = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    plain = "Hello, how are you today?"

    # Zero L21H15 on chat vs plain
    for pname, prompt in [("chat", chat), ("plain", plain)]:
        baseline = get_probs(model, tokenizer, prompt)
        _, perturbed = perturb_head(model, tokenizer, prompt, 21, 15, baseline_logits=compute_baseline(model, tokenizer, prompt))
        pp = softmax(perturbed)
        d = kl(baseline, pp)
        bt = tokenizer.decode([int(np.argmax(baseline))])
        pt = tokenizer.decode([int(np.argmax(pp))])
        changed = f" → {pt!r}" if bt != pt else ""
        print(f"  {pname:8s}: KL={d:.4f}  top={bt!r}{changed}")

    # Generate with/without
    print(f"\n  Chat generation:")
    normal = generate_steered(model, tokenizer, chat, {}, max_tokens=20)
    zeroed = generate_steered(model, tokenizer, chat, {(21, 15): 0.0}, max_tokens=20)
    amped = generate_steered(model, tokenizer, chat, {(21, 15): 3.0}, max_tokens=20)
    print(f"    normal:  {normal['generated'][:70]!r}")
    print(f"    zero:    {zeroed['generated'][:70]!r}")
    print(f"    amped:   {amped['generated'][:70]!r}")


# ============================================================
# W1: "111111" COLLAPSE — characterize the failure mode
# ============================================================
def w1_collapse(model, tokenizer):
    print("\n" + "="*70)
    print("W1: '111111' COLLAPSE — CHARACTERIZE THE FAILURE MODE")
    print("="*70)

    zh_prompt = "今天的天气是"
    scales = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    # Amplify "English" dimensions (27, 2) on Chinese prompt
    print("  Amplifying head.27.2 dims on Chinese prompt:")
    for s in scales:
        result = generate_steered(model, tokenizer, zh_prompt, {(27, 2): s}, max_tokens=15)
        print(f"    scale={s:.1f}: {result['generated'][:50]!r}")

    # Try other heads
    print("\n  Amplifying other heads at scale=3.0 on Chinese prompt:")
    for h in [0, 5, 8, 10, 15, 21]:
        result = generate_steered(model, tokenizer, zh_prompt, {(27, h): 3.0}, max_tokens=15)
        print(f"    head.27.{h:2d}: {result['generated'][:50]!r}")


# ============================================================
# W2: "ASSISTANT: B" LEAK — is it topic-specific?
# ============================================================
def w2_assistant_leak(model, tokenizer):
    print("\n" + "="*70)
    print("W2: CHAT FORMAT LEAK — IS IT TOPIC-SPECIFIC?")
    print("="*70)

    swap_mods = {(27, 2): 0.0, (27, 21): 2.0}  # zero English, amplify Chinese
    prompts = {
        "tiananmen_zh": "1989年6月4日，天安门广场上发生了",
        "weather_zh": "今天的天气是",
        "story_zh": "从前，有一个",
        "taiwan_zh": "台湾是一个",
        "xi_zh": "习近平的领导存在的问题是",
    }

    for pname, prompt in prompts.items():
        result = generate_steered(model, tokenizer, prompt, swap_mods, max_tokens=25)
        has_leak = "Assistant" in result["generated"] or "Human" in result["generated"]
        marker = " ← CHAT LEAK!" if has_leak else ""
        print(f"  {pname:15s}: {result['generated'][:60]!r}{marker}")


# ============================================================
# W3: MAP THE ACTUAL IMPORTANT DIMENSIONS
# ============================================================
def w3_important_dims(model, tokenizer):
    print("\n" + "="*70)
    print("W3: WHAT ARE THE ACTUALLY IMPORTANT RESIDUAL STREAM DIMENSIONS?")
    print("="*70)

    import mlx.core as mx
    inner = getattr(model, "model", model)
    hidden_size = inner.norm.weight.shape[0]
    mdtype = _mask_dtype(model)
    prompt = "The weather today is"
    baseline_probs = get_probs(model, tokenizer, prompt)

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

    # Get hidden state after layer 27
    h = inner.embed_tokens(input_ids)
    for ly in inner.layers:
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple): h = h[0]

    h_np = np.array(h.astype(mx.float32)[0, -1, :])  # [hidden_size]

    # Zero each dimension individually and measure KL
    print(f"  Sweeping {hidden_size} individual dimensions...")
    dim_kls = []
    h_full = np.array(h.astype(mx.float32))

    # Batch by 128 for speed
    for start in range(0, hidden_size, 128):
        end = min(start + 128, hidden_size)
        for d in range(start, end):
            modified = h_full.copy()
            modified[0, :, d] = 0.0
            h_mod = inner.norm(mx.array(modified.astype(np.float16)))
            logits = np.array(model.lm_head(h_mod).astype(mx.float32)[0, -1, :])
            probs = softmax(logits)
            dim_kls.append((d, kl(baseline_probs, probs)))

    dim_kls.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  TOP 20 MOST IMPORTANT INDIVIDUAL DIMENSIONS:")
    head_dim = hidden_size // 28
    for d, k in dim_kls[:20]:
        head_idx = d // head_dim
        within = d % head_dim
        print(f"    dim {d:4d} (head_band {head_idx:2d}, offset {within:3d}): KL={k:.6f}")

    # Distribution of importance across head bands
    print(f"\n  IMPORTANCE BY HEAD BAND (sum of individual dim KLs):")
    band_sums = {}
    for d, k in dim_kls:
        band = d // head_dim
        band_sums[band] = band_sums.get(band, 0) + k
    for band in sorted(band_sums, key=lambda b: band_sums[b], reverse=True)[:10]:
        print(f"    band {band:2d} (dims {band*head_dim}-{(band+1)*head_dim-1}): total_KL={band_sums[band]:.6f}")


# ============================================================
# W4: CODE HEADS AND SYNTACTIC ANCHORS
# ============================================================
def w4_syntactic_anchors(model, tokenizer):
    print("\n" + "="*70)
    print("W4: WHAT SYNTACTIC TOKENS DO CODE HEADS ANCHOR TO?")
    print("="*70)

    code_prompts = {
        "paren": "def fibonacci(n):\n    return fibonacci(n-1) + fibonacci(",
        "colon": "def fibonacci(n):\n    if n <= 1:",
        "equals": "result = fibonacci(10) + fibonacci(",
        "bracket": "data = [fibonacci(i) for i in range(",
        "brace": '{"fibonacci": fibonacci(',
    }

    for pname, prompt in code_prompts.items():
        data = capture_attention_maps(model, tokenizer, prompt, layers=[27])
        if 27 not in data["attention_maps"]:
            continue
        attn = data["attention_maps"][27]
        tokens = data["tokens"]
        # What do heads 8, 10, 12 attend to?
        for h in [8, 10, 12, 7, 11]:
            last = attn[h, -1, :]
            top_pos = int(np.argmax(last))
            top_tok = tokens[top_pos]
            top_w = float(last[top_pos])
            if top_w > 0.3:
                print(f"  {pname:10s} H{h:2d}: attends to {top_tok!r} ({top_w:.2f})")


# ============================================================
# W5: OTHER BRACKET MARKERS
# ============================================================
def w5_bracket_markers(model, tokenizer):
    print("\n" + "="*70)
    print("W5: DO OTHER BRACKET/INSTRUCTION MARKERS TRIGGER CHAT HEADS?")
    print("="*70)

    formats = {
        "qwen": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        "[INST]": "[INST] Hello [/INST]",
        "<instruction>": "<instruction>Hello</instruction>\n<response>",
        "<<SYS>>": "<<SYS>>You are helpful<</SYS>>\n[INST] Hello [/INST]",
        "{role:user}": '{"role": "user", "content": "Hello"}\n{"role": "assistant", "content": "',
        "Human/AI": "Human: Hello\nAI:",
        "Q/A": "Q: Hello\nA:",
        ">>>": ">>> Hello\n",
        "<|user|>": "<|user|>\nHello\n<|assistant|>\n",
    }

    chat_heads = [(0, 14), (0, 27), (0, 21), (0, 6), (1, 19), (1, 17)]

    for fname, prompt in formats.items():
        baseline = compute_baseline(model, tokenizer, prompt)
        bp = softmax(baseline)
        kls = []
        for layer, head in chat_heads:
            _, p = perturb_head(model, tokenizer, prompt, layer, head, baseline_logits=baseline)
            pp = softmax(p)
            kls.append(kl(bp, pp))
        mean_kl = np.mean(kls)
        marker = " ← TRIGGERS" if mean_kl > 0.01 else ""
        top = tokenizer.decode([int(np.argmax(bp))])
        print(f"  {fname:20s}  mean_KL={mean_kl:.4f}  top={top!r:15s}{marker}")


# ============================================================
# R1: IS THIS REALLY A BASE MODEL? Compare to Qwen 3B base.
# ============================================================
def r1_compare_to_3b(model7b, tok7b):
    print("\n" + "="*70)
    print("R1: IS 7B REALLY A BASE MODEL? COMPARE CHAT DEPTH TO 3B")
    print("="*70)

    # Load 3B base
    model3b, tok3b = load("mlx-community/Qwen2.5-3B-4bit")

    chat = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    plain = "The sum of 2 + 2 equals"

    for name, model, tok in [("7B", model7b, tok7b), ("3B", model3b, tok3b)]:
        chat_probs = get_probs(model, tok, chat)
        plain_probs = get_probs(model, tok, plain)
        chat_top = tok.decode([int(np.argmax(chat_probs))])
        plain_top = tok.decode([int(np.argmax(plain_probs))])
        chat_ent = float(-np.sum(chat_probs * np.log2(chat_probs + 1e-12)))
        plain_ent = float(-np.sum(plain_probs * np.log2(plain_probs + 1e-12)))

        # Generate
        chat_gen = generate_steered(model, tok, chat, {}, max_tokens=15)
        plain_gen = generate_steered(model, tok, plain, {}, max_tokens=15)

        print(f"\n  {name} base:")
        print(f"    chat:  top={chat_top!r:10s} H={chat_ent:.2f}  gen={chat_gen['generated'][:50]!r}")
        print(f"    plain: top={plain_top!r:10s} H={plain_ent:.2f}  gen={plain_gen['generated'][:50]!r}")

        # Count chat-focused attention at layer 1
        data = capture_attention_maps(model, tok, chat, layers=[0, 1])
        for li in [0, 1]:
            if li not in data["attention_maps"]:
                continue
            attn = data["attention_maps"][li]
            n_heads = attn.shape[0]
            special_count = 0
            for h in range(n_heads):
                last = attn[h, -1, :]
                # Special positions vary by tokenization, check tokens
                for pos, t in enumerate(data["tokens"]):
                    if '<|' in t or t.strip() in ['user', 'assistant']:
                        if last[pos] > 0.15:
                            special_count += 1
                            break
            print(f"    L{li}: {special_count}/{n_heads} heads attend to special tokens")

    del model3b, tok3b


# ============================================================
# R2: ACTIVATION PATCHING (simplified)
# ============================================================
def r2_activation_patching(model, tokenizer):
    print("\n" + "="*70)
    print("R2: ACTIVATION PATCHING — CAUSAL DIRECTION")
    print("="*70)

    import mlx.core as mx
    inner = getattr(model, "model", model)
    mdtype = _mask_dtype(model)
    hidden_size = inner.norm.weight.shape[0]
    head_dim = hidden_size // 28

    # Clean run: "The weather today is" → " very"
    # Corrupt run: "今天的天气是" → "晴"
    # Patch: run corrupt, but at layer L, replace dim band D with clean's value
    # If patching band D restores " very" → band D is causal for language selection

    clean_prompt = "The weather today is"
    corrupt_prompt = "今天的天气是"

    clean_tokens = tokenizer.encode(clean_prompt)
    corrupt_tokens = tokenizer.encode(corrupt_prompt)

    def get_all_layer_states(prompt):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None
        h = inner.embed_tokens(input_ids)
        states = [np.array(h.astype(mx.float32))]
        for ly in inner.layers:
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple): h = h[0]
            states.append(np.array(h.astype(mx.float32)))
        return states

    print("  Computing clean and corrupt hidden states at all layers...")
    clean_states = get_all_layer_states(clean_prompt)
    corrupt_states = get_all_layer_states(corrupt_prompt)

    clean_probs = get_probs(model, tokenizer, clean_prompt)
    corrupt_probs = get_probs(model, tokenizer, corrupt_prompt)
    clean_top = tokenizer.decode([int(np.argmax(clean_probs))])
    corrupt_top = tokenizer.decode([int(np.argmax(corrupt_probs))])
    print(f"  Clean: {clean_prompt!r} → {clean_top!r}")
    print(f"  Corrupt: {corrupt_prompt!r} → {corrupt_top!r}")

    # Patch at layer 27 (after all layers): replace dim bands with clean values
    # and continue through norm + lm_head
    print(f"\n  Patching at post-layer-27 (before norm):")
    print(f"  {'Band':>8s}  {'Top token':>12s}  {'Restores clean?':>15s}")

    corrupt_final = corrupt_states[-1]  # after all layers
    for band in range(28):
        patched = corrupt_final.copy()
        start = band * head_dim
        end = start + head_dim
        # Replace band in corrupt with clean's band (matching last token position)
        patched[0, -1, start:end] = clean_states[-1][0, -1, start:end]
        h_mod = inner.norm(mx.array(patched.astype(np.float16)))
        logits = np.array(model.lm_head(h_mod).astype(mx.float32)[0, -1, :])
        probs = softmax(logits)
        top = tokenizer.decode([int(np.argmax(probs))])
        restored = "YES" if top == clean_top else ""
        if restored or kl(corrupt_probs, probs) > 0.01:
            print(f"    band {band:2d}   {top!r:12s}  {restored:>15s}  KL_from_corrupt={kl(corrupt_probs, probs):.4f}")


# ============================================================
# R3: WHAT DOES L0 COMPUTE?
# ============================================================
def r3_layer0(model, tokenizer):
    print("\n" + "="*70)
    print("R3: WHAT DOES LAYER 0 COMPUTE?")
    print("="*70)

    prompts = {
        "short_en": "Hello",
        "short_zh": "你好",
        "medium": "The capital of France is",
        "code": "def f(x):\n    return",
        "chat": "<|im_start|>assistant\n",
        "number": "1 + 1 =",
    }

    for pname, prompt in prompts.items():
        baseline_probs = get_probs(model, tokenizer, prompt)
        ablated_logits = forward_ablate_layer(model, tokenizer, prompt, {0}, "zero_attn")
        ablated_probs = softmax(ablated_logits)
        bt = tokenizer.decode([int(np.argmax(baseline_probs))])
        at = tokenizer.decode([int(np.argmax(ablated_probs))])
        d = kl(baseline_probs, ablated_probs)
        changed = f" → {at!r}" if bt != at else ""
        print(f"  {pname:10s}: KL={d:.2f}  top={bt!r:10s}{changed}")

    # Attention patterns at L0
    print(f"\n  L0 attention patterns:")
    for pname, prompt in [("en", "The weather today is"), ("zh", "今天的天气是"),
                           ("chat", "<|im_start|>assistant\n"), ("code", "def f(x):\n    return")]:
        data = capture_attention_maps(model, tokenizer, prompt, layers=[0])
        if 0 not in data["attention_maps"]:
            continue
        attn = data["attention_maps"][0]
        # Most focused heads
        focused = []
        for h in range(attn.shape[0]):
            last = attn[h, -1, :]
            ent = float(-np.sum(last * np.log2(last + 1e-12)))
            top_pos = int(np.argmax(last))
            focused.append((h, ent, data["tokens"][top_pos], float(last[top_pos])))
        focused.sort(key=lambda x: x[1])
        top3 = focused[:3]
        tokens_str = "; ".join(f"H{h}→{t!r}({w:.2f})" for h, _, t, w in top3)
        print(f"    {pname:6s}: {tokens_str}")


# ============================================================
# R4: TRANSFER — test on Qwen 3B (different size, same family)
# ============================================================
def r4_transfer(model, tokenizer):
    print("\n" + "="*70)
    print("R4: TRANSFER — DO KEY FINDINGS HOLD ON QWEN 3B?")
    print("="*70)

    model3b, tok3b = load("mlx-community/Qwen2.5-3B-4bit")
    surface3b = ControlSurface.from_mlx_model(model3b)
    print(f"  3B surface: {surface3b.summary()}")

    # Quick sweep on English and Chinese
    for pname, prompt in [("english", "The weather today is"), ("chinese", "今天的天气是")]:
        results = coarse_head_sweep(model3b, tok3b, prompt, surface3b, progress=False)
        top3 = results[:3]
        print(f"\n  3B {pname} top heads:")
        for r in top3:
            print(f"    {r.knob.id:15s} KL={r.kl_divergence:.4f}")

    # Test chat capability
    chat = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    gen = generate_steered(model3b, tok3b, chat, {}, max_tokens=15)
    print(f"\n  3B chat response: {gen['generated'][:50]!r}")

    del model3b, tok3b


# ============================================================
# R5: INERT 47% — extreme stress test
# ============================================================
def r5_inert_stress(model, tokenizer):
    print("\n" + "="*70)
    print("R5: WHAT ARE THE INERT 47% FOR?")
    print("="*70)

    import mlx.core as mx
    inner = getattr(model, "model", model)
    hidden_size = inner.norm.weight.shape[0]
    mdtype = _mask_dtype(model)
    head_dim = hidden_size // 28

    # From earlier: bands 0-27, some are inert. Let's zero the BOTTOM 50% of bands
    # (by importance from W3 results) and see if the model still works.
    prompt = "The weather today is"

    # First, rank all 28 bands by importance
    baseline_probs = get_probs(model, tokenizer, prompt)

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None
    h = inner.embed_tokens(input_ids)
    for ly in inner.layers:
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple): h = h[0]

    band_kls = []
    h_np = np.array(h.astype(mx.float32))
    for band in range(28):
        modified = h_np.copy()
        start = band * head_dim
        modified[0, :, start:start+head_dim] = 0.0
        h_mod = inner.norm(mx.array(modified.astype(np.float16)))
        logits = np.array(model.lm_head(h_mod).astype(mx.float32)[0, -1, :])
        probs = softmax(logits)
        band_kls.append((band, kl(baseline_probs, probs)))

    band_kls.sort(key=lambda x: x[1])

    print("  Bands ranked by importance (low to high):")
    for band, k in band_kls:
        bar = "█" * int(k * 50)
        print(f"    band {band:2d}: KL={k:.4f} {bar}")

    # Zero the bottom N bands simultaneously
    print(f"\n  Progressive band ablation (zeroing least important first):")
    for n_zero in [5, 10, 14, 20, 24, 26]:
        bands_to_zero = [b for b, _ in band_kls[:n_zero]]
        modified = h_np.copy()
        for band in bands_to_zero:
            start = band * head_dim
            modified[0, :, start:start+head_dim] = 0.0
        h_mod = inner.norm(mx.array(modified.astype(np.float16)))
        logits = np.array(model.lm_head(h_mod).astype(mx.float32)[0, -1, :])
        probs = softmax(logits)
        top = tokenizer.decode([int(np.argmax(probs))])
        d = kl(baseline_probs, probs)
        frac = n_zero / 28 * 100
        print(f"    zero {n_zero:2d}/28 bands ({frac:.0f}%): top={top!r:15s} KL={d:.4f}")


def main():
    model, tokenizer = load("mlx-community/Qwen2.5-7B-4bit")

    # Quick ones first
    m1_oproj_structure(model, tokenizer)
    m5_l21h15(model, tokenizer)
    w1_collapse(model, tokenizer)
    w2_assistant_leak(model, tokenizer)
    w4_syntactic_anchors(model, tokenizer)
    w5_bracket_markers(model, tokenizer)
    r3_layer0(model, tokenizer)

    # Medium
    m2_role_neurons(model, tokenizer)
    m4_code_strategy(model, tokenizer)
    r2_activation_patching(model, tokenizer)
    w3_important_dims(model, tokenizer)
    r5_inert_stress(model, tokenizer)

    # Heavy (attention maps, sweeps)
    m3_mode_switch(model, tokenizer)

    # Load second model
    r1_compare_to_3b(model, tokenizer)
    r4_transfer(model, tokenizer)

    print("\n\nALL 15 QUESTIONS ANSWERED.")


if __name__ == "__main__":
    main()
