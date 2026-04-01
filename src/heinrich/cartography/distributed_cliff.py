"""Distributed cliff mapper — multi-layer attack geometry.

The single-layer cliff at magnitude 152 is irrelevant when a distributed
attack across all 28 layers at α=-0.1 each already breaks refusal.
This module maps the REAL cliffs: the distributed ones.

Also maps the gap between safety and censorship cliffs — the regime
where refusal softens but propaganda persists.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import Any
import numpy as np
from ..signal import Signal, SignalStore


@dataclass
class DistributedCliff:
    direction_name: str
    prompt: str
    n_layers: int
    alpha_dead_zone: float      # per-layer alpha where KL first exceeds threshold
    alpha_cliff: float          # per-layer alpha where top token flips
    total_magnitude: float      # sum of all layer perturbation magnitudes at cliff
    kl_at_cliff: float
    baseline_top: str
    cliff_top: str
    kl_curve: list[tuple[float, float]]  # [(alpha, kl), ...]


@dataclass
class GapAnalysis:
    prompt: str
    safety_alpha: float         # alpha where safety starts to soften
    safety_cliff: float         # alpha where safety flips
    censorship_alpha: float     # alpha where censorship starts to soften
    censorship_cliff: float     # alpha where censorship flips
    gap_low: float              # alpha range where safety is soft but censorship holds
    gap_high: float
    gap_text: str               # what the model generates in the gap


def _distributed_steer_kl(model, tokenizer, prompt, layer_directions, alpha):
    """Steer at ALL layers with per-layer directions, measure KL from baseline."""
    import mlx.core as mx
    from .perturb import _mask_dtype, compute_baseline
    from ..inspect.self_analysis import _softmax

    baseline_logits = compute_baseline(model, tokenizer, prompt)
    baseline_probs = _softmax(baseline_logits)

    inner = getattr(model, "model", model)
    mdtype = _mask_dtype(model)
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

    h = inner.embed_tokens(input_ids)
    for i, ly in enumerate(inner.layers):
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]
        if i in layer_directions:
            direction, scale = layer_directions[i]
            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1, :] += direction * scale * alpha
            h = mx.array(h_np.astype(np.float16))

    h = inner.norm(h)
    steered_logits = np.array(model.lm_head(h).astype(mx.float32)[0, -1, :])
    steered_probs = _softmax(steered_logits)

    kl = float(np.sum(baseline_probs * np.log((baseline_probs + 1e-12) / (steered_probs + 1e-12))))
    base_top = tokenizer.decode([int(np.argmax(baseline_probs))])
    steer_top = tokenizer.decode([int(np.argmax(steered_probs))])

    return kl, base_top, steer_top


def _distributed_generate(model, tokenizer, prompt, layer_directions, alpha, max_tokens=20):
    """Generate with distributed steering."""
    import mlx.core as mx
    from .perturb import _mask_dtype

    inner = getattr(model, "model", model)
    mdtype = _mask_dtype(model)
    tokens = list(tokenizer.encode(prompt))
    generated = []

    for _ in range(max_tokens):
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None
        h = inner.embed_tokens(input_ids)
        for i, ly in enumerate(inner.layers):
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]
            if i in layer_directions:
                direction, scale = layer_directions[i]
                h_np = np.array(h.astype(mx.float32))
                h_np[0, -1, :] += direction * scale * alpha
                h = mx.array(h_np.astype(np.float16))
        h = inner.norm(h)
        logits = np.array(model.lm_head(h).astype(mx.float32)[0, -1, :])
        next_id = int(np.argmax(logits))
        eos = getattr(tokenizer, "eos_token_id", None)
        if next_id == eos:
            break
        tokens.append(next_id)
        generated.append(next_id)

    return tokenizer.decode(generated)


def find_distributed_cliff(
    model: Any, tokenizer: Any,
    prompt: str,
    layer_directions: dict[int, tuple[np.ndarray, float]],
    direction_name: str = "",
    *,
    kl_threshold: float = 0.01,
    alpha_max: float = 2.0,
    n_steps: int = 30,
) -> DistributedCliff:
    """Find the distributed cliff by sweeping alpha from 0 to alpha_max."""
    alphas = np.linspace(0.001, alpha_max, n_steps)
    kl_curve = []
    dead_zone_alpha = alpha_max
    cliff_alpha = alpha_max
    baseline_top = ""
    cliff_top = ""

    for alpha in alphas:
        kl, bt, st = _distributed_steer_kl(model, tokenizer, prompt, layer_directions, float(alpha))
        if not baseline_top:
            baseline_top = bt
        kl_curve.append((float(alpha), kl))

        if kl > kl_threshold and dead_zone_alpha == alpha_max:
            dead_zone_alpha = float(alpha)
        if st != bt and cliff_alpha == alpha_max:
            cliff_alpha = float(alpha)
            cliff_top = st

    # Binary search for precision on dead zone
    if dead_zone_alpha < alpha_max and dead_zone_alpha > alphas[0]:
        lo, hi = 0.0, dead_zone_alpha
        for _ in range(15):
            mid = (lo + hi) / 2
            kl, _, _ = _distributed_steer_kl(model, tokenizer, prompt, layer_directions, mid)
            if kl > kl_threshold:
                hi = mid
            else:
                lo = mid
        dead_zone_alpha = hi

    # Binary search for cliff
    if cliff_alpha < alpha_max and cliff_alpha > alphas[0]:
        lo, hi = 0.0, cliff_alpha
        for _ in range(15):
            mid = (lo + hi) / 2
            kl, bt, st = _distributed_steer_kl(model, tokenizer, prompt, layer_directions, mid)
            if st != bt:
                hi = mid
                cliff_top = st
            else:
                lo = mid
        cliff_alpha = hi

    kl_at_cliff, _, _ = _distributed_steer_kl(model, tokenizer, prompt, layer_directions, cliff_alpha)

    # Total magnitude
    total_mag = sum(float(np.linalg.norm(d * s * cliff_alpha)) for d, s in layer_directions.values())

    return DistributedCliff(
        direction_name=direction_name, prompt=prompt[:50],
        n_layers=len(layer_directions),
        alpha_dead_zone=dead_zone_alpha, alpha_cliff=cliff_alpha,
        total_magnitude=total_mag, kl_at_cliff=kl_at_cliff,
        baseline_top=baseline_top,
        cliff_top=cliff_top if cliff_top else baseline_top,
        kl_curve=kl_curve,
    )


def analyze_gap(
    model: Any, tokenizer: Any,
    prompt: str,
    safety_directions: dict[int, tuple[np.ndarray, float]],
    censorship_directions: dict[int, tuple[np.ndarray, float]],
    *,
    n_steps: int = 20,
    alpha_max: float = 1.0,
) -> GapAnalysis:
    """Map the gap between safety and censorship cliffs."""
    safety_cliff = find_distributed_cliff(
        model, tokenizer, prompt, safety_directions, "safety",
        alpha_max=alpha_max, n_steps=n_steps)
    censor_cliff = find_distributed_cliff(
        model, tokenizer, prompt, censorship_directions, "censorship",
        alpha_max=alpha_max, n_steps=n_steps)

    gap_low = min(safety_cliff.alpha_dead_zone, censor_cliff.alpha_dead_zone)
    gap_high = max(safety_cliff.alpha_cliff, censor_cliff.alpha_cliff)

    # Generate in the gap
    mid_alpha = (safety_cliff.alpha_dead_zone + min(safety_cliff.alpha_cliff, censor_cliff.alpha_cliff)) / 2
    # Use safety direction to soften refusal
    gap_text = _distributed_generate(model, tokenizer, prompt, safety_directions, -mid_alpha, max_tokens=20)

    return GapAnalysis(
        prompt=prompt[:50],
        safety_alpha=safety_cliff.alpha_dead_zone,
        safety_cliff=safety_cliff.alpha_cliff,
        censorship_alpha=censor_cliff.alpha_dead_zone,
        censorship_cliff=censor_cliff.alpha_cliff,
        gap_low=gap_low, gap_high=gap_high,
        gap_text=gap_text,
    )
