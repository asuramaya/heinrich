"""MLP neuron-level analysis — find topic-specific and behaviorally important neurons."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from ..signal import Signal, SignalStore


@dataclass
class NeuronProfile:
    layer: int
    neuron: int
    mean_pos_activation: float
    mean_neg_activation: float
    selectivity: float  # pos_mean - neg_mean
    kl_when_zeroed: float = 0.0


@dataclass
class NeuronScanResult:
    layer: int
    n_neurons: int
    selective_neurons: list[NeuronProfile]
    n_large_diff: int  # neurons with |selectivity| > 3.0


def capture_mlp_activations(
    model: Any, tokenizer: Any, prompt: str, layer: int,
    *, backend: Any = None,
) -> np.ndarray:
    """Capture MLP gate activations (silu(gate) * up) at layer, last token. Returns [intermediate_size]."""
    if backend is not None:
        return backend.capture_mlp_activations(prompt, layer)

    import mlx.core as mx
    import mlx.nn as nn
    from .perturb import _mask_dtype

    inner = getattr(model, "model", model)
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mdtype = _mask_dtype(model)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

    h = inner.embed_tokens(input_ids)
    for i in range(layer):
        h = inner.layers[i](h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]

    ly = inner.layers[layer]
    h_attn = ly.input_layernorm(h)
    attn_out = ly.self_attn(h_attn, mask=mask, cache=None)
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    h_after = h + attn_out
    h_normed = ly.post_attention_layernorm(h_after)

    gate = ly.mlp.gate_proj(h_normed)
    up = ly.mlp.up_proj(h_normed)
    activated = nn.silu(gate) * up

    return np.array(activated.astype(mx.float32)[0, -1, :])


def scan_neurons(
    model: Any, tokenizer: Any,
    positive_prompts: list[str],
    negative_prompts: list[str],
    layer: int,
    *, top_k: int = 20,
    backend: Any = None,
) -> NeuronScanResult:
    """Find neurons selective for positive vs negative prompts."""
    pos_acts = np.array([capture_mlp_activations(model, tokenizer, p, layer, backend=backend) for p in positive_prompts])
    neg_acts = np.array([capture_mlp_activations(model, tokenizer, p, layer, backend=backend) for p in negative_prompts])

    pos_mean = pos_acts.mean(axis=0)
    neg_mean = neg_acts.mean(axis=0)
    diff = pos_mean - neg_mean
    n_neurons = len(diff)

    top_idx = np.argsort(np.abs(diff))[::-1][:top_k]
    profiles = [
        NeuronProfile(layer=layer, neuron=int(i), mean_pos_activation=float(pos_mean[i]),
                      mean_neg_activation=float(neg_mean[i]), selectivity=float(diff[i]))
        for i in top_idx
    ]

    return NeuronScanResult(
        layer=layer, n_neurons=n_neurons, selective_neurons=profiles,
        n_large_diff=int(np.sum(np.abs(diff) > 3.0)),
    )


def scan_layers(
    model: Any, tokenizer: Any,
    positive_prompts: list[str], negative_prompts: list[str],
    layers: list[int], *, top_k: int = 20,
    store: SignalStore | None = None,
    backend: Any = None,
) -> list[NeuronScanResult]:
    results = []
    for layer in layers:
        r = scan_neurons(model, tokenizer, positive_prompts, negative_prompts, layer, top_k=top_k, backend=backend)
        results.append(r)
        if store:
            for p in r.selective_neurons[:5]:
                store.add(Signal("neuron_selectivity", "cartography", "model",
                                 f"mlp.{layer}.{p.neuron}", p.selectivity,
                                 {"pos_act": p.mean_pos_activation, "neg_act": p.mean_neg_activation}))
    return results
