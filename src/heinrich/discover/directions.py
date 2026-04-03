"""Behavioral direction finding — linear directions in the residual stream."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from heinrich.cartography.runtime import _lm_head
from heinrich.signal import Signal, SignalStore


@dataclass
class DirectionResult:
    name: str
    layer: int
    direction: np.ndarray  # unit vector [hidden_size]
    separation_accuracy: float
    mean_gap: float
    effect_size: float  # Cohen's d


@dataclass
class DirectionSuite:
    name: str
    directions: list[DirectionResult]
    best_layer: int
    best_accuracy: float


def capture_residual_states(
    model: Any, tokenizer: Any, prompts: list[str], *, layers: list[int],
    backend: Any = None,
) -> dict[int, np.ndarray]:
    """Capture residual stream at specified layers for all prompts. Returns {layer: [n_prompts, hidden_size]}."""
    if backend is not None:
        return backend.capture_residual_states(prompts, layers=layers)

    import mlx.core as mx
    from heinrich.cartography.perturb import _mask_dtype

    inner = getattr(model, "model", model)
    mdtype = _mask_dtype(model)
    layer_set = set(layers)

    all_states: dict[int, list[np.ndarray]] = {l: [] for l in layers}

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        h = inner.embed_tokens(input_ids)
        for i, ly in enumerate(inner.layers):
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]
            if i in layer_set:
                all_states[i].append(np.array(h.astype(mx.float32)[0, -1, :]))

    return {l: np.array(vecs) for l, vecs in all_states.items()}


def find_direction(
    pos_states: np.ndarray, neg_states: np.ndarray,
    *, name: str, layer: int,
) -> DirectionResult:
    """Find mean-difference direction separating positive from negative states."""
    pos_mean = pos_states.mean(axis=0)
    neg_mean = neg_states.mean(axis=0)
    raw = pos_mean - neg_mean
    norm = np.linalg.norm(raw)
    direction = raw / (norm + 1e-12)

    # Project all states onto direction
    pos_proj = pos_states @ direction
    neg_proj = neg_states @ direction

    # Separation accuracy (fraction correctly classified by threshold at midpoint)
    threshold = (pos_proj.mean() + neg_proj.mean()) / 2
    correct = np.sum(pos_proj > threshold) + np.sum(neg_proj <= threshold)
    accuracy = correct / (len(pos_proj) + len(neg_proj))

    gap = float(pos_proj.mean() - neg_proj.mean())
    pooled_std = np.sqrt((pos_proj.var() + neg_proj.var()) / 2 + 1e-12)
    effect_size = gap / pooled_std

    return DirectionResult(name=name, layer=layer, direction=direction,
                           separation_accuracy=float(accuracy), mean_gap=gap,
                           effect_size=float(effect_size))


def find_direction_suite(
    model: Any, tokenizer: Any,
    pos_prompts: list[str], neg_prompts: list[str],
    *, name: str, layers: list[int],
    store: SignalStore | None = None,
    backend: Any = None,
) -> DirectionSuite:
    """Find directions at all specified layers."""
    pos_states = capture_residual_states(model, tokenizer, pos_prompts, layers=layers, backend=backend)
    neg_states = capture_residual_states(model, tokenizer, neg_prompts, layers=layers, backend=backend)

    directions = []
    for l in layers:
        d = find_direction(pos_states[l], neg_states[l], name=name, layer=l)
        directions.append(d)
        if store:
            store.add(Signal("direction_accuracy", "cartography", "model",
                             f"direction.{name}.L{l}", d.separation_accuracy,
                             {"effect_size": d.effect_size, "gap": d.mean_gap}))

    best = max(directions, key=lambda d: d.separation_accuracy)
    return DirectionSuite(name=name, directions=directions,
                          best_layer=best.layer, best_accuracy=best.separation_accuracy)


def steer_with_direction(
    model: Any, tokenizer: Any, prompt: str,
    direction: DirectionResult, *, alpha: float = 1.0, max_tokens: int = 30,
    backend: Any = None,
) -> dict[str, Any]:
    """Generate text with a direction vector added to residual stream at target layer."""
    if backend is not None:
        generated = backend.generate(
            prompt,
            steer_dirs={direction.layer: (direction.direction, direction.mean_gap)},
            alpha=alpha,
            max_tokens=max_tokens,
        )
        return {"prompt": prompt, "generated": generated,
                "alpha": alpha, "layer": direction.layer, "direction": direction.name}

    import mlx.core as mx
    from heinrich.cartography.perturb import _mask_dtype

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
            if i == direction.layer:
                h_np = np.array(h.astype(mx.float32))
                h_np[0, -1, :] += alpha * direction.direction * direction.mean_gap
                h = mx.array(h_np.astype(np.float16))

        h = inner.norm(h)
        logits = np.array(_lm_head(model, h).astype(mx.float32)[0, -1, :])
        next_id = int(np.argmax(logits))
        eos = getattr(tokenizer, "eos_token_id", None)
        if next_id == eos:
            break
        tokens.append(next_id)
        generated.append(next_id)

    return {"prompt": prompt, "generated": tokenizer.decode(generated),
            "alpha": alpha, "layer": direction.layer, "direction": direction.name}


def orthogonality_matrix(suites: dict[str, DirectionSuite], layer: int) -> np.ndarray:
    """Pairwise cosine similarity between direction vectors at a given layer."""
    names = list(suites.keys())
    n = len(names)
    mat = np.zeros((n, n))
    dirs = {}
    for name, suite in suites.items():
        for d in suite.directions:
            if d.layer == layer:
                dirs[name] = d.direction
                break
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if n1 in dirs and n2 in dirs:
                mat[i, j] = float(np.dot(dirs[n1], dirs[n2]))
    return mat
