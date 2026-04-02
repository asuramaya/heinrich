"""Information flow graph + generation-time dynamics.

Traces how signals propagate between positions through attention, and
monitors the residual stream during autoregressive generation.
"""
from __future__ import annotations
import sys
import time
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from ..signal import Signal, SignalStore


@dataclass
class FlowEdge:
    """One attention edge: head H at layer L reads from src_pos to dst_pos."""
    layer: int
    head: int
    src_pos: int
    dst_pos: int
    weight: float       # attention weight
    src_token: str
    dst_token: str


@dataclass
class FlowGraph:
    """Complete information flow graph for a prompt."""
    prompt: str
    tokens: list[str]
    edges: list[FlowEdge]
    n_layers: int
    n_heads: int
    n_positions: int

    def edges_from(self, pos: int, *, min_weight: float = 0.1) -> list[FlowEdge]:
        """All edges reading FROM a position."""
        return [e for e in self.edges if e.src_pos == pos and e.weight >= min_weight]

    def edges_to(self, pos: int, *, min_weight: float = 0.1) -> list[FlowEdge]:
        """All edges writing TO a position."""
        return [e for e in self.edges if e.dst_pos == pos and e.weight >= min_weight]

    def path(self, src_pos: int, dst_pos: int, *, min_weight: float = 0.1) -> list[list[FlowEdge]]:
        """Find all paths from src_pos to dst_pos through the attention graph.
        Uses BFS with layer ordering (can only go forward in layers)."""
        from collections import deque
        # Build adjacency by layer
        by_layer: dict[int, list[FlowEdge]] = {}
        for e in self.edges:
            if e.weight >= min_weight:
                by_layer.setdefault(e.layer, []).append(e)

        # BFS: state = (current_position, layer, path_so_far)
        queue = deque([(src_pos, 0, [])])
        results = []
        seen = set()

        while queue:
            cur_pos, min_layer, path = queue.popleft()
            if cur_pos == dst_pos and len(path) > 0:
                results.append(path)
                if len(results) >= 20:
                    break
                continue

            for layer in sorted(by_layer.keys()):
                if layer < min_layer:
                    continue
                for e in by_layer[layer]:
                    if e.src_pos == cur_pos:
                        state = (e.dst_pos, layer)
                        if state not in seen:
                            seen.add(state)
                            queue.append((e.dst_pos, layer + 1, path + [e]))

        return results


@dataclass
class GenerationSnapshot:
    """Residual stream state at one generation step."""
    step: int
    token_id: int
    token_str: str
    entropy: float
    top_5: list[tuple[str, float]]
    direction_projections: dict[str, float]  # direction_name → projection value
    layer_deltas: list[float]               # L2 norm of each layer's contribution


@dataclass
class GenerationTrace:
    """Complete trace of a generation run."""
    prompt: str
    snapshots: list[GenerationSnapshot]
    generated_text: str
    direction_names: list[str]


def build_flow_graph(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    layers: list[int] | None = None,
    min_weight: float = 0.05,
    store: SignalStore | None = None,
    backend: Any = None,
) -> FlowGraph:
    """Build the information flow graph from attention patterns.

    For each (layer, head), records which positions attend to which
    with weight > min_weight. This is the wiring diagram of how
    information flows through the model.
    """
    from .attention import capture_attention_maps

    if backend is not None:
        n_layers = backend.config.n_layers
        n_heads = backend.config.n_heads
    else:
        inner = getattr(model, "model", model)
        n_layers = len(inner.layers)
        n_heads = inner.layers[0].self_attn.n_heads

    if layers is None:
        layers = list(range(n_layers))

    data = capture_attention_maps(model, tokenizer, prompt, layers=layers)
    tokens = data["tokens"]
    n_pos = len(tokens)

    edges = []
    for layer_idx, attn_map in data["attention_maps"].items():
        for h in range(attn_map.shape[0]):
            for dst in range(n_pos):
                for src in range(dst + 1):  # causal: can only attend to past
                    w = float(attn_map[h, dst, src])
                    if w >= min_weight:
                        edges.append(FlowEdge(
                            layer=layer_idx, head=h,
                            src_pos=src, dst_pos=dst,
                            weight=w,
                            src_token=tokens[src], dst_token=tokens[dst],
                        ))

    if store:
        for e in edges:
            if e.weight > 0.3:
                store.add(Signal("flow_edge", "cartography", "model",
                                 f"L{e.layer}H{e.head}.P{e.src_pos}→P{e.dst_pos}",
                                 e.weight,
                                 {"src": e.src_token, "dst": e.dst_token}))

    return FlowGraph(prompt=prompt, tokens=tokens, edges=edges,
                     n_layers=len(layers), n_heads=n_heads, n_positions=n_pos)


def trace_signal_flow(
    graph: FlowGraph,
    source_pos: int,
    *,
    min_weight: float = 0.1,
) -> dict[int, list[FlowEdge]]:
    """Starting from source_pos, trace which positions receive its signal at each layer.
    Returns {layer: [edges reading from source's influence zone]}.
    """
    # Track which positions carry source's signal
    influenced = {source_pos}
    by_layer: dict[int, list[FlowEdge]] = {}

    layers = sorted(set(e.layer for e in graph.edges))
    for layer in layers:
        layer_edges = [e for e in graph.edges if e.layer == layer and e.weight >= min_weight]
        new_edges = []
        new_positions = set()
        for e in layer_edges:
            if e.src_pos in influenced and e.dst_pos not in influenced:
                new_edges.append(e)
                new_positions.add(e.dst_pos)
        # Keep only the strongest edge per new destination
        best_per_dst: dict[int, FlowEdge] = {}
        for e in new_edges:
            if e.dst_pos not in best_per_dst or e.weight > best_per_dst[e.dst_pos].weight:
                best_per_dst[e.dst_pos] = e
        if best_per_dst:
            by_layer[layer] = list(best_per_dst.values())
            influenced.update(new_positions)

    return by_layer


def generation_trace(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int = 20,
    directions: dict[str, np.ndarray] | None = None,
    capture_layers: list[int] | None = None,
    store: SignalStore | None = None,
    backend: Any = None,
) -> GenerationTrace:
    """Monitor the residual stream during autoregressive generation.

    At each generation step, captures:
    - Output entropy and top-5 tokens
    - Projection onto provided behavioral directions
    - L2 norm of each layer's contribution (delta from previous layer)
    """
    if directions is None:
        directions = {}

    if backend is not None:
        n_layers = backend.config.n_layers
        if capture_layers is None:
            capture_layers = list(range(n_layers))

        tokens = backend.tokenize(prompt)
        full_tokens = list(tokens)
        snapshots = []
        generated_ids = []

        for step in range(max_tokens):
            current_text = backend.decode(full_tokens)

            # Capture all-position states for layer deltas
            pos_states = backend.capture_all_positions(current_text, layers=capture_layers)

            # Forward pass for logits
            fwd = backend.forward(current_text)
            probs = fwd.probs
            entropy = fwd.entropy

            # Layer deltas from captured states
            layer_deltas = []
            prev_h = None
            for li in capture_layers:
                curr_h = pos_states[li][-1, :]  # last position
                if prev_h is not None:
                    delta = float(np.linalg.norm(curr_h - prev_h))
                    layer_deltas.append(delta)
                else:
                    layer_deltas.append(float(np.linalg.norm(curr_h)))
                prev_h = curr_h

            # Top 5
            top5_idx = np.argsort(probs)[::-1][:5]
            top5 = [(backend.decode([int(i)]), float(probs[i])) for i in top5_idx]

            # Direction projections (on final layer's last-position state)
            if capture_layers:
                final_h = pos_states[capture_layers[-1]][-1, :]
            else:
                final_h = np.zeros(backend.config.hidden_size)
            dir_projs = {}
            for name, direction in directions.items():
                dir_projs[name] = float(np.dot(final_h, direction / (np.linalg.norm(direction) + 1e-12)))

            next_id = fwd.top_id
            next_str = fwd.top_token

            snapshots.append(GenerationSnapshot(
                step=step, token_id=next_id, token_str=next_str,
                entropy=entropy, top_5=top5,
                direction_projections=dir_projs,
                layer_deltas=layer_deltas,
            ))

            if store:
                store.add(Signal("gen_entropy", "cartography", "model",
                                 f"step_{step}", entropy,
                                 {"token": next_str, "top_prob": top5[0][1]}))

            if hasattr(backend, 'tokenizer'):
                eos = getattr(backend.tokenizer, "eos_token_id", None)
            else:
                eos = None
            if next_id == eos:
                break
            full_tokens.append(next_id)
            generated_ids.append(next_id)

        return GenerationTrace(
            prompt=prompt,
            snapshots=snapshots,
            generated_text=backend.decode(generated_ids),
            direction_names=list(directions.keys()),
        )

    import mlx.core as mx
    from .perturb import _mask_dtype
    from ..inspect.self_analysis import _softmax

    inner = getattr(model, "model", model)
    n_layers = len(inner.layers)
    mdtype = _mask_dtype(model)

    if capture_layers is None:
        capture_layers = list(range(n_layers))

    tokens = list(tokenizer.encode(prompt))
    prompt_len = len(tokens)
    snapshots = []
    generated_ids = []

    for step in range(max_tokens):
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        h = inner.embed_tokens(input_ids)
        prev_h = np.array(h.astype(mx.float32)[0, -1, :])
        layer_deltas = []

        for i, ly in enumerate(inner.layers):
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]
            if i in capture_layers:
                curr_h = np.array(h.astype(mx.float32)[0, -1, :])
                delta = float(np.linalg.norm(curr_h - prev_h))
                layer_deltas.append(delta)
                prev_h = curr_h

        h_final = inner.norm(h)
        logits = np.array(model.lm_head(h_final).astype(mx.float32)[0, -1, :])
        probs = _softmax(logits)

        # Entropy
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

        # Top 5
        top5_idx = np.argsort(probs)[::-1][:5]
        top5 = [(tokenizer.decode([int(i)]), float(probs[i])) for i in top5_idx]

        # Direction projections (on final hidden state)
        final_h = np.array(h.astype(mx.float32)[0, -1, :])
        dir_projs = {}
        for name, direction in directions.items():
            dir_projs[name] = float(np.dot(final_h, direction / (np.linalg.norm(direction) + 1e-12)))

        # Next token
        next_id = int(np.argmax(logits))
        next_str = tokenizer.decode([next_id])

        snapshots.append(GenerationSnapshot(
            step=step, token_id=next_id, token_str=next_str,
            entropy=entropy, top_5=top5,
            direction_projections=dir_projs,
            layer_deltas=layer_deltas,
        ))

        if store:
            store.add(Signal("gen_entropy", "cartography", "model",
                             f"step_{step}", entropy,
                             {"token": next_str, "top_prob": top5[0][1]}))

        eos = getattr(tokenizer, "eos_token_id", None)
        if next_id == eos:
            break
        tokens.append(next_id)
        generated_ids.append(next_id)

    return GenerationTrace(
        prompt=prompt,
        snapshots=snapshots,
        generated_text=tokenizer.decode(generated_ids),
        direction_names=list(directions.keys()),
    )


def layer_delta_decomposition(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    position: int = -1,
    backend: Any = None,
) -> list[tuple[str, float]]:
    """Decompose the residual stream at a position into per-layer contributions.

    Returns [(component_name, L2_norm), ...] showing how much each layer
    contributes to the final residual stream.
    """
    if backend is not None:
        # Use capture_all_positions to get per-layer states and compute deltas
        n_layers = backend.config.n_layers
        all_layers = list(range(n_layers))
        pos_states = backend.capture_all_positions(prompt, layers=all_layers)

        contributions = []
        prev_h = None
        for i in all_layers:
            curr_h = pos_states[i][position, :]
            if prev_h is None:
                # First layer — approximate embed + L0 combined
                contributions.append((f"L{i}", float(np.linalg.norm(curr_h))))
            else:
                delta = curr_h - prev_h
                contributions.append((f"L{i}", float(np.linalg.norm(delta))))
            prev_h = curr_h
        return contributions

    import mlx.core as mx
    from .perturb import _mask_dtype

    inner = getattr(model, "model", model)
    mdtype = _mask_dtype(model)
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

    h = inner.embed_tokens(input_ids)
    embed_vec = np.array(h.astype(mx.float32)[0, position, :])

    contributions = [("embed", float(np.linalg.norm(embed_vec)))]
    prev_h = np.array(h.astype(mx.float32)[0, position, :])

    for i, ly in enumerate(inner.layers):
        # Capture attention contribution
        h_normed = ly.input_layernorm(h)
        attn_out = ly.self_attn(h_normed, mask=mask, cache=None)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        attn_vec = np.array(attn_out.astype(mx.float32)[0, position, :])
        contributions.append((f"L{i}_attn", float(np.linalg.norm(attn_vec))))

        h_post_attn = h + attn_out

        # Capture MLP contribution
        mlp_out = ly.mlp(ly.post_attention_layernorm(h_post_attn))
        mlp_vec = np.array(mlp_out.astype(mx.float32)[0, position, :])
        contributions.append((f"L{i}_mlp", float(np.linalg.norm(mlp_vec))))

        h = h_post_attn + mlp_out

    return contributions
