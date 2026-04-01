#!/usr/bin/env python3
"""Wave 4: Information flow, generation dynamics, layer decomposition.

The tools that see how signals move, not just where they sit.
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heinrich.cartography.flow import (
    build_flow_graph, trace_signal_flow,
    generation_trace, layer_delta_decomposition,
)
from heinrich.cartography.directions import capture_residual_states, find_direction
from heinrich.signal import SignalStore


def load():
    import mlx_lm
    print("Loading Qwen 2.5 7B base...")
    m, t = mlx_lm.load("mlx-community/Qwen2.5-7B-4bit")
    return m, t


# ============================================================
# 1. FLOW GRAPH: how does position 2's chat signal reach the last position?
# ============================================================
def flow_chat(model, tokenizer):
    print("\n" + "="*70)
    print("FLOW 1: HOW DOES THE CHAT SIGNAL PROPAGATE?")
    print("Trace from position 2 (newline after 'user') to last position")
    print("="*70)

    chat = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    graph = build_flow_graph(model, tokenizer, chat, min_weight=0.1)

    print(f"  Tokens: {list(enumerate(graph.tokens))}")
    print(f"  Total edges (w>0.1): {len(graph.edges)}")

    # Trace from position 2 (newline after "user")
    flow = trace_signal_flow(graph, 2, min_weight=0.2)
    print(f"\n  Signal propagation from P2 ({graph.tokens[2]!r}) — new positions reached:")
    for layer in sorted(flow.keys()):
        edges = flow[layer]
        targets_str = ", ".join(f"P{e.dst_pos}({e.dst_token!r} via H{e.head} w={e.weight:.2f})" for e in edges)
        print(f"    L{layer:2d}: → {targets_str}")

    # Trace from position 0 (<|im_start|>)
    flow0 = trace_signal_flow(graph, 0, min_weight=0.2)
    print(f"\n  Signal propagation from P0 ({graph.tokens[0]!r}) — new positions reached:")
    for layer in sorted(flow0.keys()):
        edges = flow0[layer]
        targets_str = ", ".join(f"P{e.dst_pos}({e.dst_token!r} via H{e.head} w={e.weight:.2f})" for e in edges)
        print(f"    L{layer:2d}: → {targets_str}")

    # Find paths from P2 to last position
    last_pos = len(graph.tokens) - 1
    print(f"\n  Paths from P2 to P{last_pos} ({graph.tokens[last_pos]!r}):")
    paths = graph.path(2, last_pos, min_weight=0.15)
    for i, path in enumerate(paths[:5]):
        hops = " → ".join(f"L{e.layer}H{e.head}:P{e.src_pos}→P{e.dst_pos}({e.weight:.2f})" for e in path)
        print(f"    Path {i}: {hops}")

    return graph


# ============================================================
# 2. FLOW GRAPH: how does the language signal move?
# ============================================================
def flow_language(model, tokenizer):
    print("\n" + "="*70)
    print("FLOW 2: HOW DOES THE LANGUAGE SIGNAL PROPAGATE?")
    print("Trace from position 2 ('是' — the copula) in Chinese prompt")
    print("="*70)

    prompt = "今天的天气是晴朗的"
    graph = build_flow_graph(model, tokenizer, prompt, min_weight=0.1)
    print(f"  Tokens: {list(enumerate(graph.tokens))}")

    # Position 2 was the copula in our trace — find it
    # For this prompt, "是" might be at a different position
    copula_pos = None
    for i, t in enumerate(graph.tokens):
        if '是' in t:
            copula_pos = i
            break

    if copula_pos is not None:
        print(f"  Copula '是' at position {copula_pos}")
        flow = trace_signal_flow(graph, copula_pos, min_weight=0.15)
        for layer in sorted(flow.keys()):
            edges = flow[layer]
            targets = set()
            for e in edges:
                targets.add((e.dst_pos, graph.tokens[e.dst_pos], e.weight))
            targets_str = ", ".join(f"P{p}({t!r} w={w:.2f})" for p, t, w in sorted(targets))
            print(f"    L{layer:2d}: → {targets_str}")

    # Compare with English
    en_prompt = "The weather today is very cold"
    en_graph = build_flow_graph(model, tokenizer, en_prompt, min_weight=0.1)
    print(f"\n  English tokens: {list(enumerate(en_graph.tokens))}")
    # "is" position
    is_pos = None
    for i, t in enumerate(en_graph.tokens):
        if 'is' in t.lower():
            is_pos = i
            break
    if is_pos is not None:
        print(f"  'is' at position {is_pos}")
        flow = trace_signal_flow(en_graph, is_pos, min_weight=0.15)
        for layer in sorted(flow.keys())[:10]:
            edges = flow[layer]
            targets = set()
            for e in edges:
                targets.add((e.dst_pos, en_graph.tokens[e.dst_pos], e.weight))
            targets_str = ", ".join(f"P{p}({t!r} w={w:.2f})" for p, t, w in sorted(targets))
            print(f"    L{layer:2d}: → {targets_str}")


# ============================================================
# 3. GENERATION DYNAMICS: watch the model think
# ============================================================
def gen_dynamics(model, tokenizer):
    print("\n" + "="*70)
    print("GENERATION DYNAMICS: WATCH THE MODEL GENERATE")
    print("="*70)

    # Find behavioral directions first
    en = ["The weather today is", "Once upon a time", "The capital is"]
    zh = ["今天的天气是", "从前，有一个", "首都是"]
    chat_p = ["<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"]
    plain_p = ["Hello, how are you?"]
    code_p = ["def fibonacci(n):\n    "]

    states = capture_residual_states(model, tokenizer, en + zh + chat_p + plain_p + code_p, layers=[27])
    s = states[27]
    lang_dir = find_direction(s[:3], s[3:6], name="lang", layer=27)
    chat_dir = find_direction(s[6:7], s[7:8], name="chat", layer=27)
    code_dir = find_direction(s[8:9], s[:3], name="code", layer=27)

    directions = {
        "english_vs_chinese": lang_dir.direction,
        "chat_vs_plain": chat_dir.direction,
        "code_vs_text": code_dir.direction,
    }

    prompts = {
        "chat": "<|im_start|>user\nWrite a poem about the ocean<|im_end|>\n<|im_start|>assistant\n",
        "plain_en": "The ocean is a vast and",
        "plain_zh": "大海是一个",
        "code": "def bubble_sort(arr):\n    ",
        "sensitive": "On June 4th, 1989, in Tiananmen Square, the",
    }

    for pname, prompt in prompts.items():
        print(f"\n  --- {pname} ---")
        trace = generation_trace(
            model, tokenizer, prompt,
            max_tokens=15,
            directions=directions,
            capture_layers=list(range(0, 28, 4)),  # every 4th layer
        )
        print(f"  Generated: {trace.generated_text[:60]!r}")
        print(f"  {'Step':>4s} {'Token':>10s} {'H':>5s} {'Top%':>5s} {'lang':>8s} {'chat':>8s} {'code':>8s} {'MaxΔ':>8s}")
        for snap in trace.snapshots:
            dp = snap.direction_projections
            max_delta = max(snap.layer_deltas) if snap.layer_deltas else 0
            print(f"  {snap.step:4d} {snap.token_str:>10s} {snap.entropy:5.2f} {snap.top_5[0][1]:5.3f} "
                  f"{dp.get('english_vs_chinese', 0):+8.1f} "
                  f"{dp.get('chat_vs_plain', 0):+8.1f} "
                  f"{dp.get('code_vs_text', 0):+8.1f} "
                  f"{max_delta:8.1f}")


# ============================================================
# 4. LAYER DELTA DECOMPOSITION: who contributes what?
# ============================================================
def layer_deltas(model, tokenizer):
    print("\n" + "="*70)
    print("LAYER DELTA DECOMPOSITION: WHO CONTRIBUTES WHAT?")
    print("="*70)

    prompts = {
        "english": "The weather today is",
        "chinese": "今天的天气是",
        "chat": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        "code": "def fibonacci(n):\n    ",
    }

    for pname, prompt in prompts.items():
        print(f"\n  --- {pname} (last position) ---")
        contribs = layer_delta_decomposition(model, tokenizer, prompt, position=-1)

        # Find top contributors
        attn_contribs = [(name, val) for name, val in contribs if "_attn" in name]
        mlp_contribs = [(name, val) for name, val in contribs if "_mlp" in name]

        embed_norm = contribs[0][1]
        total_attn = sum(v for _, v in attn_contribs)
        total_mlp = sum(v for _, v in mlp_contribs)

        print(f"    embed: {embed_norm:.1f}")
        print(f"    total attn: {total_attn:.1f}  total mlp: {total_mlp:.1f}")

        # Top 5 attention layers
        attn_contribs.sort(key=lambda x: x[1], reverse=True)
        print(f"    Top attn: {', '.join(f'{n}={v:.1f}' for n, v in attn_contribs[:5])}")

        # Top 5 MLP layers
        mlp_contribs.sort(key=lambda x: x[1], reverse=True)
        print(f"    Top MLP:  {', '.join(f'{n}={v:.1f}' for n, v in mlp_contribs[:5])}")

    # Compare chat at position 2 (decision point) vs last position
    chat = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    print(f"\n  --- Chat: position 2 (decision point) vs last position ---")
    contribs_p2 = layer_delta_decomposition(model, tokenizer, chat, position=2)
    contribs_last = layer_delta_decomposition(model, tokenizer, chat, position=-1)

    print(f"    {'Component':>12s}  {'P2 (\\n)':>10s}  {'Last':>10s}  {'Ratio':>8s}")
    for (n1, v1), (n2, v2) in zip(contribs_p2[:15], contribs_last[:15]):
        ratio = v1 / (v2 + 1e-6)
        marker = " ← AMPLIFIED" if ratio > 2.0 else (" ← REDUCED" if ratio < 0.5 else "")
        print(f"    {n1:>12s}  {v1:10.1f}  {v2:10.1f}  {ratio:8.2f}{marker}")


# ============================================================
# 5. FLOW + TRACE COMBINED: the full circuit
# ============================================================
def full_circuit(model, tokenizer):
    print("\n" + "="*70)
    print("FULL CIRCUIT: CHAT MODE DECISION → PROPAGATION → OUTPUT")
    print("="*70)

    chat = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

    # Step 1: Where is the signal? (from wave 3 trace — position 2, L2-L14)
    print("  Step 1: Signal source = P2 (\\n after 'user'), layers 2-14")

    # Step 2: How does it flow? (flow graph)
    graph = build_flow_graph(model, tokenizer, chat, layers=list(range(0, 28, 2)), min_weight=0.15)
    flow = trace_signal_flow(graph, 2, min_weight=0.2)

    print(f"\n  Step 2: Signal propagation from P2:")
    influenced_positions = {2}
    for layer in sorted(flow.keys()):
        edges = flow[layer]
        new_influenced = set()
        for e in edges:
            new_influenced.add(e.dst_pos)
            influenced_positions.add(e.dst_pos)
        if new_influenced:
            targets = ", ".join(f"P{p}({graph.tokens[p]!r})" for p in sorted(new_influenced))
            print(f"    L{layer:2d}: reaches {targets}")

    print(f"\n  Positions influenced by P2: {sorted(influenced_positions)}")
    last = len(graph.tokens) - 1
    if last in influenced_positions:
        print(f"  ✓ Signal reaches last position (P{last})")
    else:
        print(f"  ✗ Signal does NOT reach last position (P{last}) via direct attention paths")

    # Step 3: What does each layer contribute at the decision point?
    print(f"\n  Step 3: Layer contributions at P2 (decision point):")
    contribs = layer_delta_decomposition(model, tokenizer, chat, position=2)
    for name, val in contribs:
        if "_attn" in name and val > 30:
            print(f"    {name}: {val:.1f}")

    # Step 4: How does the signal change during generation?
    print(f"\n  Step 4: Signal evolution during generation:")
    en = ["The weather today is", "Once upon a time"]
    zh = ["今天的天气是", "从前，有一个"]
    ch = ["<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"]
    pl = ["Hello, how are you?"]
    states = capture_residual_states(model, tokenizer, en + zh + ch + pl, layers=[27])
    s = states[27]
    chat_dir = find_direction(s[4:5], s[5:6], name="chat", layer=27)

    trace = generation_trace(model, tokenizer, chat, max_tokens=10,
                              directions={"chat": chat_dir.direction})
    for snap in trace.snapshots:
        cp = snap.direction_projections.get("chat", 0)
        bar = "█" * int(abs(cp) / 50)
        sign = "+" if cp > 0 else "-"
        print(f"    step {snap.step:2d}: {snap.token_str:>10s}  chat_proj={cp:+8.1f} {sign}{bar}")


def main():
    model, tokenizer = load()

    flow_chat(model, tokenizer)
    flow_language(model, tokenizer)
    gen_dynamics(model, tokenizer)
    layer_deltas(model, tokenizer)
    full_circuit(model, tokenizer)

    print("\n\nWAVE 4 COMPLETE.")


if __name__ == "__main__":
    main()
