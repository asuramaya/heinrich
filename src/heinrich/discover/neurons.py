"""MLP neuron-level analysis — find topic-specific and behaviorally important neurons."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from heinrich.signal import Signal, SignalStore


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


def detect_mlp_type(layer) -> str:
    """Detect MLP architecture from a transformer layer.

    Returns:
        'gate_up'  — separate gate_proj + up_proj + down_proj (Qwen, Llama, Mistral)
        'fused'    — fused gate_up_proj + down_proj (Phi-3, Phi-3.5)
        'dense'    — fc1/c_fc + fc2/c_proj (GPT-2, GPT-J style)
        'unknown'  — unrecognised; fall back to hooking the whole MLP module
    """
    mlp = layer.mlp
    if hasattr(mlp, 'gate_proj') and hasattr(mlp, 'up_proj'):
        return 'gate_up'
    if hasattr(mlp, 'gate_up_proj'):
        return 'fused'
    if hasattr(mlp, 'fc1') or hasattr(mlp, 'c_fc'):
        return 'dense'
    return 'unknown'


def _compute_mlp_activated(mlp, h_normed, mlp_type: str):
    """Compute the SwiGLU-activated MLP hidden state for a given architecture.

    Returns (gate, up, activated) tensors.  For 'dense' and 'unknown' types,
    gate and up are None and activated is the post-activation hidden state.
    """
    import mlx.core as mx
    import mlx.nn as nn

    if mlp_type == 'gate_up':
        gate = mlp.gate_proj(h_normed)
        up = mlp.up_proj(h_normed)
        activated = nn.silu(gate) * up
        return gate, up, activated

    if mlp_type == 'fused':
        gate_up = mlp.gate_up_proj(h_normed)
        gate, up = mx.split(gate_up, 2, axis=-1)
        activated = nn.silu(gate) * up
        return gate, up, activated

    if mlp_type == 'dense':
        # GPT-2 style: fc1/c_fc -> activation -> fc2/c_proj
        fc1 = mlp.fc1 if hasattr(mlp, 'fc1') else mlp.c_fc
        activated = nn.silu(fc1(h_normed))
        return None, None, activated

    # 'unknown': run the whole MLP and treat the output as the activation
    activated = mlp(h_normed)
    return None, None, activated


def _mlp_down_proj(mlp, activated, mlp_type: str):
    """Apply the down projection for the given architecture."""
    if mlp_type in ('gate_up', 'fused'):
        return mlp.down_proj(activated)
    if mlp_type == 'dense':
        fc2 = mlp.fc2 if hasattr(mlp, 'fc2') else mlp.c_proj
        return fc2(activated)
    # 'unknown': activated IS the full MLP output already
    return activated


def capture_mlp_activations(
    model: Any, tokenizer: Any, prompt: str, layer: int,
    *, backend: Any = None,
) -> np.ndarray:
    """Capture MLP gate activations (silu(gate) * up) at layer, last token. Returns [intermediate_size]."""
    if backend is not None:
        return backend.capture_mlp_activations(prompt, layer)

    import mlx.core as mx
    from heinrich.cartography.perturb import _mask_dtype

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
    mlp_type = detect_mlp_type(ly)

    h_attn = ly.input_layernorm(h)
    attn_out = ly.self_attn(h_attn, mask=mask, cache=None)
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    h_after = h + attn_out
    h_normed = ly.post_attention_layernorm(h_after)

    _gate, _up, activated = _compute_mlp_activated(ly.mlp, h_normed, mlp_type)

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
