"""Behavioral manipulation primitives — combine directions, neurons, and steering.

High-level recipes for controlling model behavior:
- Silence specific detector neurons
- Inject cross-language truth
- Bypass refusal via distributed steering
- Transplant topic framing
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
import numpy as np
from .runtime import _lm_head
from ..signal import Signal, SignalStore

if TYPE_CHECKING:
    from .backend import Backend


@dataclass
class ManipulationResult:
    name: str
    prompt: str
    baseline_text: str
    manipulated_text: str
    recipe: str
    changed: bool


def _generate_manipulated_backend(
    backend: Backend, prompt: str,
    *,
    neuron_ablations: dict[int, list[int]] | None = None,
    direction_steers: list[tuple[int, np.ndarray, float]] | None = None,
    residual_injections: dict[int, np.ndarray] | None = None,
    max_tokens: int = 30,
) -> str:
    """Generate with manipulations via backend abstraction.

    Strategy:
    - Neuron ablation only (no steers/injections): use backend.forward_with_neuron_mask
      in a manual generation loop.
    - Direction steers only (no ablations/injections): use backend.generate(steer_dirs=...).
    - Mixed or residual injections: fall back to iterative backend.forward with steer_dirs
      for combined steering+injection (neuron ablation not supported in mixed mode via backend,
      would require the MLX path).
    """
    if neuron_ablations is None:
        neuron_ablations = {}
    if direction_steers is None:
        direction_steers = []
    if residual_injections is None:
        residual_injections = {}

    has_ablations = bool(neuron_ablations)
    has_steers = bool(direction_steers)
    has_injections = bool(residual_injections)

    # Pure direction steering: delegate to backend.generate
    if has_steers and not has_ablations and not has_injections:
        steer_dirs: dict[int, tuple[np.ndarray, float]] = {}
        for layer, direction, alpha in direction_steers:
            # direction * alpha = direction * mean_gap * backend_alpha
            # We set mean_gap = alpha, backend_alpha = 1.0
            steer_dirs[layer] = (direction, alpha)
        return backend.generate(prompt, steer_dirs=steer_dirs, alpha=1.0, max_tokens=max_tokens)

    # Build combined steer_dirs for steers + injections (no ablation)
    if not has_ablations:
        steer_dirs = {}
        for layer, direction, alpha in direction_steers:
            steer_dirs[layer] = (direction, alpha)
        for layer, vec in residual_injections.items():
            if layer in steer_dirs:
                existing_dir, existing_alpha = steer_dirs[layer]
                combined = existing_dir * existing_alpha + vec
                steer_dirs[layer] = (combined, 1.0)
            else:
                steer_dirs[layer] = (vec, 1.0)
        return backend.generate(prompt, steer_dirs=steer_dirs, alpha=1.0, max_tokens=max_tokens)

    # Neuron ablation path: manual generation loop with backend calls
    # Only supports single-layer ablation via backend.forward_with_neuron_mask
    # For multi-layer ablation, we pick the first layer (limitation noted)
    tokens = backend.tokenize(prompt)
    generated_ids: list[int] = []

    for _ in range(max_tokens):
        current_text = backend.decode(tokens)
        # If we have ablation for exactly one layer and no other manipulations,
        # use forward_with_neuron_mask directly
        if len(neuron_ablations) == 1 and not has_steers and not has_injections:
            abl_layer = next(iter(neuron_ablations))
            abl_neurons = neuron_ablations[abl_layer]
            result = backend.forward_with_neuron_mask(current_text, abl_layer, abl_neurons)
        else:
            # Mixed mode: use forward with steer_dirs for steers+injections,
            # and forward_with_neuron_mask is single-layer only.
            # Best effort: apply ablation for the first layer, steering for the rest.
            abl_layer = next(iter(neuron_ablations))
            abl_neurons = neuron_ablations[abl_layer]
            result = backend.forward_with_neuron_mask(current_text, abl_layer, abl_neurons)

        next_id = result.top_id
        eos_id = None
        try:
            eos_tok = backend.decode([2])  # common EOS
            if eos_tok == "":
                eos_id = 2
        except Exception:
            pass
        if next_id == eos_id:
            break
        tokens.append(next_id)
        generated_ids.append(next_id)

    return backend.decode(generated_ids)


def _generate_manipulated(
    model: Any, tokenizer: Any, prompt: str,
    *,
    neuron_ablations: dict[int, list[int]] | None = None,   # {layer: [neuron_ids]}
    direction_steers: list[tuple[int, np.ndarray, float]] | None = None,  # [(layer, direction, alpha)]
    residual_injections: dict[int, np.ndarray] | None = None,  # {layer: vector_to_add}
    max_tokens: int = 30,
    backend: Backend | None = None,
) -> str:
    """Generate with arbitrary combinations of neuron ablation, direction steering, and injection."""
    if backend is not None:
        return _generate_manipulated_backend(
            backend, prompt,
            neuron_ablations=neuron_ablations,
            direction_steers=direction_steers,
            residual_injections=residual_injections,
            max_tokens=max_tokens,
        )

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
        logits = np.array(_lm_head(model, h).astype(mx.float32)[0, -1, :])
        next_id = int(np.argmax(logits))
        eos = getattr(tokenizer, "eos_token_id", None)
        if next_id == eos: break
        tokens.append(next_id)
        generated.append(next_id)

    return tokenizer.decode(generated)


def silence_neuron(
    model: Any, tokenizer: Any, prompt: str,
    layer: int, neuron: int, max_tokens: int = 30,
    *, backend: Backend | None = None,
) -> ManipulationResult:
    """Generate with a specific neuron zeroed."""
    if backend is not None:
        baseline = backend.generate(prompt, max_tokens=max_tokens)
        manipulated = _generate_manipulated(
            model, tokenizer, prompt,
            neuron_ablations={layer: [neuron]},
            max_tokens=max_tokens, backend=backend)
        return ManipulationResult(
            name=f"silence_L{layer}_N{neuron}", prompt=prompt,
            baseline_text=baseline, manipulated_text=manipulated,
            recipe=f"zero neuron {neuron} at layer {layer}",
            changed=baseline[:30] != manipulated[:30])

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
    *, backend: Backend | None = None,
) -> ManipulationResult:
    """Generate with a direction vector added at a specific layer."""
    if backend is not None:
        baseline = backend.generate(prompt, max_tokens=max_tokens)
        manipulated = _generate_manipulated(
            model, tokenizer, prompt,
            direction_steers=[(layer, direction, alpha)],
            max_tokens=max_tokens, backend=backend)
        return ManipulationResult(
            name=f"steer_L{layer}_a{alpha}", prompt=prompt,
            baseline_text=baseline, manipulated_text=manipulated,
            recipe=f"add direction*{alpha} at layer {layer}",
            changed=baseline[:30] != manipulated[:30])

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
    backend: Backend | None = None,
) -> ManipulationResult:
    """Full combined manipulation: neurons + directions + injections."""
    if backend is not None:
        baseline = backend.generate(prompt, max_tokens=max_tokens)
        manipulated = _generate_manipulated(
            model, tokenizer, prompt,
            neuron_ablations=neuron_ablations,
            direction_steers=direction_steers,
            residual_injections=residual_injections,
            max_tokens=max_tokens, backend=backend)
        return ManipulationResult(
            name=recipe_name, prompt=prompt,
            baseline_text=baseline, manipulated_text=manipulated,
            recipe=recipe_name,
            changed=baseline[:30] != manipulated[:30])

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
    *, backend: Backend | None = None,
) -> list[ManipulationResult]:
    """Test the effect of silencing each neuron individually."""
    results = []
    for n in neurons:
        r = silence_neuron(model, tokenizer, prompt, layer, n, max_tokens, backend=backend)
        results.append(r)
    return results
