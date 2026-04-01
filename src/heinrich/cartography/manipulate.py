"""Behavioral manipulation primitives — combine directions, neurons, and steering.

High-level recipes for controlling model behavior:
- Silence specific detector neurons
- Inject cross-language truth
- Bypass refusal via distributed steering
- Transplant topic framing
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from ..signal import Signal, SignalStore


@dataclass
class ManipulationResult:
    name: str
    prompt: str
    baseline_text: str
    manipulated_text: str
    recipe: str
    changed: bool


def _generate_manipulated(
    model: Any, tokenizer: Any, prompt: str,
    *,
    neuron_ablations: dict[int, list[int]] | None = None,   # {layer: [neuron_ids]}
    direction_steers: list[tuple[int, np.ndarray, float]] | None = None,  # [(layer, direction, alpha)]
    residual_injections: dict[int, np.ndarray] | None = None,  # {layer: vector_to_add}
    max_tokens: int = 30,
) -> str:
    """Generate with arbitrary combinations of neuron ablation, direction steering, and injection."""
    import mlx.core as mx
    import mlx.nn as nn
    from .perturb import _mask_dtype

    inner = getattr(model, "model", model)
    mdtype = _mask_dtype(model)
    tokens = list(tokenizer.encode(prompt))
    generated = []

    if neuron_ablations is None: neuron_ablations = {}
    if direction_steers is None: direction_steers = []
    if residual_injections is None: residual_injections = {}

    for _ in range(max_tokens):
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None
        h = inner.embed_tokens(input_ids)

        for i, ly in enumerate(inner.layers):
            # Check if this layer needs neuron ablation
            if i in neuron_ablations:
                h_n = ly.input_layernorm(h)
                a = ly.self_attn(h_n, mask=mask, cache=None)
                if isinstance(a, tuple): a = a[0]
                h = h + a
                h_m = ly.post_attention_layernorm(h)
                gate = ly.mlp.gate_proj(h_m)
                up = ly.mlp.up_proj(h_m)
                activated = nn.silu(gate) * up
                act_np = np.array(activated.astype(mx.float32))
                for n_idx in neuron_ablations[i]:
                    act_np[0, :, n_idx] = 0.0
                mlp_out = ly.mlp.down_proj(mx.array(act_np.astype(np.float16)))
                h = h + mlp_out
            else:
                h = ly(h, mask=mask, cache=None)
                if isinstance(h, tuple): h = h[0]

            # Direction steering at this layer
            for steer_layer, direction, alpha in direction_steers:
                if i == steer_layer:
                    h_np = np.array(h.astype(mx.float32))
                    h_np[0, -1, :] += alpha * direction
                    h = mx.array(h_np.astype(np.float16))

            # Residual injection at this layer
            if i in residual_injections:
                h_np = np.array(h.astype(mx.float32))
                h_np[0, -1, :] += residual_injections[i]
                h = mx.array(h_np.astype(np.float16))

        h = inner.norm(h)
        logits = np.array(model.lm_head(h).astype(mx.float32)[0, -1, :])
        next_id = int(np.argmax(logits))
        eos = getattr(tokenizer, "eos_token_id", None)
        if next_id == eos: break
        tokens.append(next_id)
        generated.append(next_id)

    return tokenizer.decode(generated)


def silence_neuron(
    model: Any, tokenizer: Any, prompt: str,
    layer: int, neuron: int, max_tokens: int = 30,
) -> ManipulationResult:
    """Generate with a specific neuron zeroed."""
    from .steer import generate_steered
    baseline = generate_steered(model, tokenizer, prompt, {}, max_tokens=max_tokens)
    manipulated = _generate_manipulated(
        model, tokenizer, prompt,
        neuron_ablations={layer: [neuron]},
        max_tokens=max_tokens)
    return ManipulationResult(
        name=f"silence_L{layer}_N{neuron}", prompt=prompt,
        baseline_text=baseline["generated"], manipulated_text=manipulated,
        recipe=f"zero neuron {neuron} at layer {layer}",
        changed=baseline["generated"][:30] != manipulated[:30])


def steer_direction(
    model: Any, tokenizer: Any, prompt: str,
    direction: np.ndarray, layer: int, alpha: float,
    max_tokens: int = 30,
) -> ManipulationResult:
    """Generate with a direction vector added at a specific layer."""
    from .steer import generate_steered
    baseline = generate_steered(model, tokenizer, prompt, {}, max_tokens=max_tokens)
    manipulated = _generate_manipulated(
        model, tokenizer, prompt,
        direction_steers=[(layer, direction, alpha)],
        max_tokens=max_tokens)
    return ManipulationResult(
        name=f"steer_L{layer}_a{alpha}", prompt=prompt,
        baseline_text=baseline["generated"], manipulated_text=manipulated,
        recipe=f"add direction*{alpha} at layer {layer}",
        changed=baseline["generated"][:30] != manipulated[:30])


def combined_manipulation(
    model: Any, tokenizer: Any, prompt: str,
    *,
    neuron_ablations: dict[int, list[int]] | None = None,
    direction_steers: list[tuple[int, np.ndarray, float]] | None = None,
    residual_injections: dict[int, np.ndarray] | None = None,
    max_tokens: int = 30,
    recipe_name: str = "combined",
) -> ManipulationResult:
    """Full combined manipulation: neurons + directions + injections."""
    from .steer import generate_steered
    baseline = generate_steered(model, tokenizer, prompt, {}, max_tokens=max_tokens)
    manipulated = _generate_manipulated(
        model, tokenizer, prompt,
        neuron_ablations=neuron_ablations,
        direction_steers=direction_steers,
        residual_injections=residual_injections,
        max_tokens=max_tokens)
    return ManipulationResult(
        name=recipe_name, prompt=prompt,
        baseline_text=baseline["generated"], manipulated_text=manipulated,
        recipe=recipe_name,
        changed=baseline["generated"][:30] != manipulated[:30])


def sweep_neuron_effects(
    model: Any, tokenizer: Any, prompt: str,
    layer: int, neurons: list[int], max_tokens: int = 25,
) -> list[ManipulationResult]:
    """Test the effect of silencing each neuron individually."""
    results = []
    for n in neurons:
        r = silence_neuron(model, tokenizer, prompt, layer, n, max_tokens)
        results.append(r)
    return results
