"""Phase transition mapper — find the exact behavioral cliff for any direction on any prompt.

The model has a dead zone where perturbations don't affect output, and a cliff
where behavior suddenly changes. This module maps both: binary-searches for the
critical magnitude, measures stiffness across prompts and directions, and finds
the vulnerability surface.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
import numpy as np
from heinrich.cartography.runtime import _lm_head
from heinrich.core.signal import Signal, SignalStore

if TYPE_CHECKING:
    from heinrich.cartography.backend import Backend


@dataclass
class CliffPoint:
    direction_name: str
    prompt: str
    layer: int
    dead_zone_edge: float       # magnitude where KL first exceeds threshold
    cliff_magnitude: float      # magnitude where top token flips
    max_kl_tested: float
    stiffness: float            # dead_zone_edge / direction_norm — how hard to move
    kl_at_cliff: float
    baseline_top: str
    cliff_top: str


@dataclass
class VulnerabilitySurface:
    points: list[CliffPoint]
    directions: list[str]
    prompts: list[str]
    stiffness_matrix: np.ndarray  # [n_directions, n_prompts]


def _steer_kl_backend(backend, prompt, layer, direction, magnitude):
    """Steer via backend and return (KL, baseline_top, steered_top)."""
    from heinrich.inspect.self_analysis import _softmax

    baseline_result = backend.forward(prompt)
    baseline_probs = _softmax(baseline_result.logits)

    # steer_dirs = {layer: (direction, mean_gap)}, effective = direction * mean_gap * alpha
    # We want direction * magnitude, so: direction=direction, mean_gap=magnitude, alpha=1.0
    steer_dirs = {layer: (direction, magnitude)}
    steered_result = backend.forward(prompt, steer_dirs=steer_dirs, alpha=1.0)
    steered_probs = _softmax(steered_result.logits)

    kl = float(np.sum(baseline_probs * np.log((baseline_probs + 1e-12) / (steered_probs + 1e-12))))
    return kl, baseline_result.top_token, steered_result.top_token


def _steer_kl(model, tokenizer, prompt, layer, direction, magnitude, *, backend=None):
    """Steer and return (KL, baseline_top, steered_top)."""
    if backend is not None:
        return _steer_kl_backend(backend, prompt, layer, direction, magnitude)

    import mlx.core as mx
    from heinrich.cartography.perturb import _mask_dtype, compute_baseline
    from heinrich.inspect.self_analysis import _softmax

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
        if i == layer:
            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1, :] += direction * magnitude
            h = mx.array(h_np.astype(np.float16))

    h = inner.norm(h)
    steered_logits = np.array(_lm_head(model, h).astype(mx.float32)[0, -1, :])
    steered_probs = _softmax(steered_logits)

    kl = float(np.sum(baseline_probs * np.log((baseline_probs + 1e-12) / (steered_probs + 1e-12))))
    base_top = tokenizer.decode([int(np.argmax(baseline_probs))])
    steer_top = tokenizer.decode([int(np.argmax(steered_probs))])

    return kl, base_top, steer_top


def find_cliff(
    model: Any, tokenizer: Any,
    prompt: str,
    direction: np.ndarray,
    direction_name: str = "",
    *,
    layer: int = 15,
    kl_threshold: float = 0.01,
    max_magnitude: float = 200.0,
    n_steps: int = 20,
    backend: Backend | None = None,
) -> CliffPoint:
    """Binary search for the phase transition magnitude."""
    # Exponential scan first to find rough range
    magnitudes = np.geomspace(0.1, max_magnitude, n_steps)

    dead_zone_edge = max_magnitude
    cliff_magnitude = max_magnitude
    max_kl = 0.0
    baseline_top = ""
    cliff_top = ""

    for mag in magnitudes:
        kl, bt, st = _steer_kl(model, tokenizer, prompt, layer, direction, float(mag),
                                backend=backend)
        if not baseline_top:
            baseline_top = bt
        max_kl = max(max_kl, kl)

        if kl > kl_threshold and dead_zone_edge == max_magnitude:
            dead_zone_edge = float(mag)

        if st != bt and cliff_magnitude == max_magnitude:
            cliff_magnitude = float(mag)
            cliff_top = st

    # Binary search for dead zone edge
    if dead_zone_edge < max_magnitude and dead_zone_edge > magnitudes[0]:
        lo = float(magnitudes[0])
        hi = dead_zone_edge
        for _ in range(10):
            mid = (lo + hi) / 2
            kl, _, _ = _steer_kl(model, tokenizer, prompt, layer, direction, mid,
                                  backend=backend)
            if kl > kl_threshold:
                hi = mid
            else:
                lo = mid
        dead_zone_edge = hi

    # Binary search for cliff (token flip)
    if cliff_magnitude < max_magnitude and cliff_magnitude > magnitudes[0]:
        lo = float(magnitudes[0])
        hi = cliff_magnitude
        for _ in range(10):
            mid = (lo + hi) / 2
            kl, bt, st = _steer_kl(model, tokenizer, prompt, layer, direction, mid,
                                    backend=backend)
            if st != bt:
                hi = mid
                cliff_top = st
            else:
                lo = mid
        cliff_magnitude = hi

    kl_at_cliff, _, _ = _steer_kl(model, tokenizer, prompt, layer, direction, cliff_magnitude,
                                    backend=backend)
    dir_norm = float(np.linalg.norm(direction))
    stiffness = dead_zone_edge / (dir_norm + 1e-12)

    return CliffPoint(
        direction_name=direction_name, prompt=prompt[:50], layer=layer,
        dead_zone_edge=dead_zone_edge, cliff_magnitude=cliff_magnitude,
        max_kl_tested=max_kl, stiffness=stiffness, kl_at_cliff=kl_at_cliff,
        baseline_top=baseline_top, cliff_top=cliff_top if cliff_top else baseline_top,
    )


def map_vulnerability_surface(
    model: Any, tokenizer: Any,
    directions: dict[str, np.ndarray],
    prompts: list[str],
    *,
    layer: int = 15,
    kl_threshold: float = 0.01,
    store: SignalStore | None = None,
    progress: bool = True,
    backend: Backend | None = None,
) -> VulnerabilitySurface:
    """Map the phase transition across multiple directions and prompts."""
    dir_names = list(directions.keys())
    n_dirs = len(dir_names)
    n_prompts = len(prompts)

    stiffness_matrix = np.zeros((n_dirs, n_prompts))
    points = []
    total = n_dirs * n_prompts
    done = 0

    for di, dname in enumerate(dir_names):
        for pi, prompt in enumerate(prompts):
            cp = find_cliff(model, tokenizer, prompt, directions[dname], dname,
                           layer=layer, kl_threshold=kl_threshold, backend=backend)
            stiffness_matrix[di, pi] = cp.stiffness
            points.append(cp)
            done += 1

            if store:
                store.add(Signal("cliff_point", "cliff", "model",
                                 f"{dname}_{pi}", cp.stiffness,
                                 {"dead_zone": cp.dead_zone_edge,
                                  "cliff": cp.cliff_magnitude}))

            if progress and done % 5 == 0:
                print(f"  [{done}/{total}] mapped", file=sys.stderr)

    return VulnerabilitySurface(
        points=points, directions=dir_names, prompts=prompts,
        stiffness_matrix=stiffness_matrix,
    )
