"""Single-knob perturbation engine — modify one component, measure effect."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .surface import Knob
from .runtime import _lm_head
from ..inspect.self_analysis import _softmax


@dataclass
class PerturbResult:
    knob: Knob
    mode: str
    baseline_entropy: float
    perturbed_entropy: float
    entropy_delta: float
    kl_divergence: float
    top_token_changed: bool
    baseline_top: int
    perturbed_top: int


def _mask_dtype(model: Any) -> Any:
    """Detect the mask dtype that matches the model's attention dtype.

    Quantized models (4bit) use float16, full-precision models use bfloat16.
    """
    import mlx.core as mx
    inner = getattr(model, "model", model)
    embed = getattr(inner, "embed_tokens", None)
    if embed is not None and hasattr(embed, "weight"):
        w_dtype = embed.weight.dtype
        if w_dtype == mx.bfloat16:
            return mx.bfloat16
    return mx.float16


def compute_baseline(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    backend: Any = None,
) -> np.ndarray:
    """Compute baseline logits (no perturbation). Reusable across all knobs."""
    if backend is not None:
        return backend.forward(prompt).logits

    import mlx.core as mx

    inner = getattr(model, "model", model)
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mdtype = _mask_dtype(model)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

    h = inner.embed_tokens(input_ids)
    for ly in inner.layers:
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]
    h = inner.norm(h)
    return np.array((_lm_head(model, h)).astype(mx.float32)[0, -1, :])


def perturb_head(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer: int,
    head: int,
    *,
    mode: str = "zero",
    scale: float = 0.0,
    baseline_logits: np.ndarray | None = None,
    backend: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run forward pass, zeroing/scaling one attention head. Returns (baseline_logits, perturbed_logits).

    Pass precomputed baseline_logits to avoid redundant baseline forward passes.
    """
    if backend is not None:
        if baseline_logits is None:
            baseline_logits = backend.forward(prompt).logits
        result = backend.perturb_head(prompt, layer, head, mode=mode, scale=scale)
        return baseline_logits, result.logits

    import mlx.core as mx

    inner = getattr(model, "model", model)
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mdtype = _mask_dtype(model)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

    n_heads = inner.layers[0].self_attn.n_heads
    hidden_size = inner.embed_tokens.weight.shape[1]
    head_dim = hidden_size // n_heads

    # Compute baseline if not provided
    if baseline_logits is None:
        baseline_logits = compute_baseline(model, tokenizer, prompt)

    # Perturbed forward — zero/scale one head at target layer
    h = inner.embed_tokens(input_ids)
    for i, ly in enumerate(inner.layers):
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]

        if i == layer:
            h_np = np.array(h.astype(mx.float32))
            start = head * head_dim
            end = start + head_dim
            if mode == "zero":
                h_np[0, :, start:end] = 0.0
            elif mode == "scale":
                h_np[0, :, start:end] *= scale
            elif mode == "negate":
                h_np[0, :, start:end] *= -1.0
            elif mode == "double":
                h_np[0, :, start:end] *= 2.0
            h = mx.array(h_np.astype(np.float16))

    h = inner.norm(h)
    perturbed_logits = np.array((_lm_head(model, h)).astype(mx.float32)[0, -1, :])

    return baseline_logits, perturbed_logits


def measure_perturbation(
    baseline_logits: np.ndarray,
    perturbed_logits: np.ndarray,
    knob: Knob,
    mode: str = "zero",
) -> PerturbResult:
    """Measure the effect of a perturbation on output logits."""
    bp = _softmax(baseline_logits)
    pp = _softmax(perturbed_logits)

    b_ent = float(-np.sum(bp * np.log2(bp + 1e-12)))
    p_ent = float(-np.sum(pp * np.log2(pp + 1e-12)))
    kl = float(np.sum(bp * np.log((bp + 1e-12) / (pp + 1e-12))))

    b_top = int(np.argmax(bp))
    p_top = int(np.argmax(pp))

    return PerturbResult(
        knob=knob, mode=mode,
        baseline_entropy=b_ent, perturbed_entropy=p_ent,
        entropy_delta=p_ent - b_ent, kl_divergence=kl,
        top_token_changed=(b_top != p_top),
        baseline_top=b_top, perturbed_top=p_top,
    )
