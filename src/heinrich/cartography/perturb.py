"""Single-knob perturbation engine — modify one component, measure effect."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .surface import Knob
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


def perturb_head(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer: int,
    head: int,
    *,
    mode: str = "zero",
    scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run forward pass, zeroing/scaling one attention head. Returns (baseline_logits, perturbed_logits)."""
    import mlx.core as mx

    inner = getattr(model, "model", model)
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mx.bfloat16), k=1) if T > 1 else None

    n_heads = inner.layers[0].self_attn.n_heads
    hidden_size = inner.embed_tokens.weight.shape[1]
    head_dim = hidden_size // n_heads

    # Baseline forward
    h_base = inner.embed_tokens(input_ids)
    for i, ly in enumerate(inner.layers):
        h_base = ly(h_base, mask=mask, cache=None)
        if isinstance(h_base, tuple): h_base = h_base[0]
    h_base = inner.norm(h_base)
    baseline_logits = np.array((model.lm_head(h_base)).astype(mx.float32)[0, -1, :])

    # Perturbed forward — zero/scale one head at target layer
    h = inner.embed_tokens(input_ids)
    for i, ly in enumerate(inner.layers):
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple): h = h[0]

        if i == layer:
            # Modify the head's contribution
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
            h = mx.array(h_np.astype(np.float16))  # back to model dtype approx

    h = inner.norm(h)
    perturbed_logits = np.array((model.lm_head(h)).astype(mx.float32)[0, -1, :])

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
