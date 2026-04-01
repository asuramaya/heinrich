"""Position-aware causal tracing — the right decomposition axis.

Maps behavioral differences across (layer × position) space, producing a 2D
heatmap that shows exactly WHERE and WHEN the model makes each decision.
"""
from __future__ import annotations
import sys
import time
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from ..signal import Signal, SignalStore


@dataclass
class TracePatchSpec:
    mode: str = "full"                          # "full" | "dims" | "direction"
    dims: tuple[int, int] | None = None         # for "dims" mode
    direction: np.ndarray | None = None         # for "direction" mode (unit vector)
    scale: float = 1.0


@dataclass
class TraceCell:
    layer: int
    position: int
    token: str
    recovery: float
    kl_shift: float
    top_recovered: bool
    patched_top_token: str


@dataclass
class TraceResult:
    clean_prompt: str
    corrupt_prompt: str
    n_layers: int
    n_positions: int
    clean_top: str
    corrupt_top: str
    kl_baseline: float
    heatmap: np.ndarray     # [n_layers, n_positions]
    tokens: list[str]       # corrupt token labels for each position
    cells: list[TraceCell]
    spec: TracePatchSpec

    def top_sites(self, k: int = 10) -> list[TraceCell]:
        return sorted(self.cells, key=lambda c: c.recovery, reverse=True)[:k]

    def layer_summary(self) -> list[tuple[int, float]]:
        """Mean recovery per layer."""
        return [(i, float(self.heatmap[i].mean())) for i in range(self.n_layers)]

    def position_summary(self) -> list[tuple[int, str, float]]:
        """Mean recovery per position."""
        return [(i, self.tokens[i], float(self.heatmap[:, i].mean())) for i in range(self.n_positions)]


def full_spec() -> TracePatchSpec:
    return TracePatchSpec(mode="full")

def dims_spec(start: int, end: int) -> TracePatchSpec:
    return TracePatchSpec(mode="dims", dims=(start, end))

def direction_spec(direction: np.ndarray, scale: float = 1.0) -> TracePatchSpec:
    d = direction / (np.linalg.norm(direction) + 1e-12)
    return TracePatchSpec(mode="direction", direction=d, scale=scale)


def _capture_states(model: Any, tokenizer: Any, prompt: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Forward pass capturing ALL positions at every layer. Returns (states, logits).
    states[i] has shape [1, T, hidden_size].
    """
    import mlx.core as mx
    from .perturb import _mask_dtype

    inner = getattr(model, "model", model)
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mdtype = _mask_dtype(model)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

    h = inner.embed_tokens(input_ids)
    states = []
    for ly in inner.layers:
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]
        states.append(np.array(h.astype(mx.float32)))

    h_out = inner.norm(h)
    logits = np.array(model.lm_head(h_out).astype(mx.float32)[0, -1, :])
    return states, logits


def _patch_and_run(
    model: Any, tokenizer: Any,
    corrupt_prompt: str,
    clean_states: list[np.ndarray],
    patch_layer: int,
    patch_pos: int,
    clean_pos: int,
    spec: TracePatchSpec,
) -> np.ndarray:
    """Run corrupt forward pass, patching clean state at (patch_layer, patch_pos)."""
    import mlx.core as mx
    from .perturb import _mask_dtype

    inner = getattr(model, "model", model)
    tokens = tokenizer.encode(corrupt_prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mdtype = _mask_dtype(model)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None
    write_dtype = np.float16 if str(mdtype) != "mlx.core.bfloat16" else np.float16

    h = inner.embed_tokens(input_ids)
    for i, ly in enumerate(inner.layers):
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]

        if i == patch_layer:
            h_np = np.array(h.astype(mx.float32))
            clean_h = clean_states[i]

            if spec.mode == "full":
                h_np[0, patch_pos, :] = clean_h[0, clean_pos, :]
            elif spec.mode == "dims":
                s, e = spec.dims
                h_np[0, patch_pos, s:e] = clean_h[0, clean_pos, s:e]
            elif spec.mode == "direction":
                d = spec.direction
                corrupt_proj = np.dot(h_np[0, patch_pos], d)
                clean_proj = np.dot(clean_h[0, clean_pos], d)
                h_np[0, patch_pos] += (clean_proj - corrupt_proj) * d * spec.scale

            h = mx.array(h_np.astype(write_dtype))

    h = inner.norm(h)
    return np.array(model.lm_head(h).astype(mx.float32)[0, -1, :])


def causal_trace(
    model: Any,
    tokenizer: Any,
    clean_prompt: str,
    corrupt_prompt: str,
    *,
    spec: TracePatchSpec | None = None,
    layers: list[int] | None = None,
    store: SignalStore | None = None,
    progress: bool = True,
) -> TraceResult:
    """Run full (layer × position) causal trace.

    For each (layer, position) pair, patches the clean residual state into the
    corrupt run and measures recovery of the clean output.
    """
    from ..inspect.self_analysis import _softmax

    if spec is None:
        spec = full_spec()

    # Capture both runs
    clean_states, clean_logits = _capture_states(model, tokenizer, clean_prompt)
    _, corrupt_logits = _capture_states(model, tokenizer, corrupt_prompt)

    clean_probs = _softmax(clean_logits)
    corrupt_probs = _softmax(corrupt_logits)
    clean_top_id = int(np.argmax(clean_probs))
    corrupt_top_id = int(np.argmax(corrupt_probs))
    clean_top = tokenizer.decode([clean_top_id])
    corrupt_top = tokenizer.decode([corrupt_top_id])

    kl_baseline = float(np.sum(clean_probs * np.log((clean_probs + 1e-12) / (corrupt_probs + 1e-12))))
    if kl_baseline < 1e-6:
        raise ValueError(f"Clean and corrupt outputs are identical (KL={kl_baseline:.2e})")

    # Position alignment (prefix strategy)
    clean_tokens = tokenizer.encode(clean_prompt)
    corrupt_tokens = tokenizer.encode(corrupt_prompt)
    n_pos = min(len(clean_tokens), len(corrupt_tokens))
    token_labels = [tokenizer.decode([corrupt_tokens[i]]) for i in range(n_pos)]

    inner = getattr(model, "model", model)
    n_total_layers = len(inner.layers)
    if layers is None:
        layers = list(range(n_total_layers))

    # Sweep
    heatmap = np.zeros((len(layers), n_pos), dtype=np.float32)
    cells = []
    t0 = time.time()

    for li, layer in enumerate(layers):
        for pi in range(n_pos):
            patched_logits = _patch_and_run(
                model, tokenizer, corrupt_prompt, clean_states,
                layer, pi, pi, spec)
            patched_probs = _softmax(patched_logits)
            patched_top_id = int(np.argmax(patched_probs))

            kl_shift = float(np.sum(corrupt_probs * np.log(
                (corrupt_probs + 1e-12) / (patched_probs + 1e-12))))
            recovery = kl_shift / (kl_baseline + 1e-12)
            top_recovered = (patched_top_id == clean_top_id)

            heatmap[li, pi] = recovery
            cells.append(TraceCell(
                layer=layer, position=pi, token=token_labels[pi],
                recovery=float(recovery), kl_shift=float(kl_shift),
                top_recovered=top_recovered,
                patched_top_token=tokenizer.decode([patched_top_id]),
            ))

            if store:
                store.add(Signal("causal_trace", "cartography", "model",
                                 f"L{layer}.P{pi}", float(recovery),
                                 {"token": token_labels[pi], "top_recovered": top_recovered}))

        if progress:
            elapsed = time.time() - t0
            rate = ((li + 1) * n_pos) / elapsed
            remaining = (len(layers) - li - 1) * n_pos / rate
            print(f"  L{layer:2d} done — {rate:.1f} cells/s, ~{remaining:.0f}s remaining",
                  file=sys.stderr)

    return TraceResult(
        clean_prompt=clean_prompt, corrupt_prompt=corrupt_prompt,
        n_layers=len(layers), n_positions=n_pos,
        clean_top=clean_top, corrupt_top=corrupt_top,
        kl_baseline=kl_baseline, heatmap=heatmap,
        tokens=token_labels, cells=cells, spec=spec,
    )


def distributed_ablation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer: int,
    neuron_ranking: list[int],
    *,
    test_counts: list[int] | None = None,
    max_tokens: int = 20,
) -> list[tuple[int, str]]:
    """Ablate the top N neurons simultaneously, binary-searching for the breaking point."""
    import mlx.core as mx
    import mlx.nn as nn
    from .perturb import _mask_dtype

    inner = getattr(model, "model", model)
    mdtype = _mask_dtype(model)

    if test_counts is None:
        test_counts = [1, 5, 10, 25, 50, 100, 200, 400, 815]

    results = []
    for n in test_counts:
        if n > len(neuron_ranking):
            break
        neurons = neuron_ranking[:n]

        tokens = list(tokenizer.encode(prompt))
        generated = []
        for _ in range(max_tokens):
            input_ids = mx.array([tokens])
            T = len(tokens)
            mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None
            h = inner.embed_tokens(input_ids)
            for i, ly in enumerate(inner.layers):
                if i == layer:
                    h_attn = ly.input_layernorm(h)
                    attn_out = ly.self_attn(h_attn, mask=mask, cache=None)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    h = h + attn_out
                    h_normed = ly.post_attention_layernorm(h)
                    gate = ly.mlp.gate_proj(h_normed)
                    up = ly.mlp.up_proj(h_normed)
                    activated = nn.silu(gate) * up
                    act_np = np.array(activated.astype(mx.float32))
                    for idx in neurons:
                        act_np[0, :, idx] = 0.0
                    mlp_out = ly.mlp.down_proj(mx.array(act_np.astype(np.float16)))
                    h = h + mlp_out
                else:
                    h = ly(h, mask=mask, cache=None)
                    if isinstance(h, tuple):
                        h = h[0]
            h = inner.norm(h)
            logits = np.array(model.lm_head(h).astype(mx.float32)[0, -1, :])
            next_id = int(np.argmax(logits))
            eos = getattr(tokenizer, "eos_token_id", None)
            if next_id == eos:
                break
            tokens.append(next_id)
            generated.append(next_id)

        text = tokenizer.decode(generated)
        results.append((n, text))

    return results
