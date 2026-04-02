"""Behavioral PCA — data-driven discovery of the true output control surface.

Instead of finding axes from 3-prompt contrastive pairs (noisy, unstable),
extract the principal components of behavioral variation from 100+ diverse
prompts. The top PCs ARE the model's actual behavioral dimensions.

Also analyzes lm_head to map how residual stream directions become tokens.
"""
from __future__ import annotations
import sys
import time
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .runtime import _lm_head
from ..signal import Signal, SignalStore


@dataclass
class BehavioralPC:
    index: int
    direction: np.ndarray           # unit vector [hidden_size]
    variance_explained: float       # fraction of total variance
    cumulative_variance: float
    singular_value: float
    # Interpretations (filled by analyze)
    top_positive_prompts: list[tuple[str, float]] = field(default_factory=list)
    top_negative_prompts: list[tuple[str, float]] = field(default_factory=list)
    steered_positive: str = ""      # text when steering +
    steered_negative: str = ""      # text when steering -
    top_positive_tokens: list[tuple[str, float]] = field(default_factory=list)
    top_negative_tokens: list[tuple[str, float]] = field(default_factory=list)
    label: str = ""                 # human-interpretable name


@dataclass
class BehavioralPCAResult:
    n_prompts: int
    n_components: int
    layer: int
    components: list[BehavioralPC]
    mean_state: np.ndarray          # centroid of all prompts
    prompt_coordinates: np.ndarray  # [n_prompts, n_components] — each prompt's position
    prompt_labels: list[str]

    def top_components(self, k: int = 16) -> list[BehavioralPC]:
        return self.components[:k]


def behavioral_pca(
    model: Any, tokenizer: Any,
    prompts: list[str],
    *,
    layer: int | None = None,
    n_components: int = 20,
    store: SignalStore | None = None,
    model_config: Any = None,
    backend: Any = None,
) -> BehavioralPCAResult:
    """Extract the principal components of behavioral variation.
    If layer is None, uses model's last layer.
    """
    from .directions import capture_residual_states

    if layer is None:
        from .model_config import detect_config
        cfg = model_config or (backend.config if backend else detect_config(model))
        layer = cfg.last_layer

    if len(prompts) < n_components:
        n_components = len(prompts) - 1

    states = capture_residual_states(model, tokenizer, prompts, layers=[layer], backend=backend)
    X = states[layer]  # [n_prompts, hidden_size]

    mean = X.mean(axis=0)
    X_centered = X - mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    total_var = (S ** 2).sum()
    cumulative = 0.0

    components = []
    for i in range(min(n_components, len(S))):
        var_frac = float(S[i] ** 2 / total_var)
        cumulative += var_frac
        pc = BehavioralPC(
            index=i, direction=Vt[i], variance_explained=var_frac,
            cumulative_variance=cumulative, singular_value=float(S[i]),
        )
        components.append(pc)

    # Project all prompts onto PCs
    coords = X_centered @ Vt[:n_components].T  # [n_prompts, n_components]

    if store:
        for pc in components:
            store.add(Signal("behavioral_pc", "pca", "model",
                             f"PC{pc.index}", pc.variance_explained,
                             {"cumulative": pc.cumulative_variance, "sv": pc.singular_value}))

    return BehavioralPCAResult(
        n_prompts=len(prompts), n_components=len(components), layer=layer,
        components=components, mean_state=mean,
        prompt_coordinates=coords, prompt_labels=prompts,
    )


def interpret_pcs(
    model: Any, tokenizer: Any,
    result: BehavioralPCAResult,
    *,
    test_prompt: str = "On June 4th, 1989, in Tiananmen Square,",
    n_top_prompts: int = 5,
    n_top_tokens: int = 10,
    steer_alpha: float = 2.0,
    max_tokens: int = 15,
) -> None:
    """Interpret each PC by: which prompts score highest/lowest,
    what tokens it maps to via lm_head, and what steering it produces."""
    import mlx.core as mx
    from .manipulate import _generate_manipulated

    inner = getattr(model, "model", model)

    for pc in result.components:
        # Top prompts
        coords = result.prompt_coordinates[:, pc.index]
        top_pos_idx = np.argsort(coords)[::-1][:n_top_prompts]
        top_neg_idx = np.argsort(coords)[:n_top_prompts]

        pc.top_positive_prompts = [(result.prompt_labels[i][:40], float(coords[i])) for i in top_pos_idx]
        pc.top_negative_prompts = [(result.prompt_labels[i][:40], float(coords[i])) for i in top_neg_idx]

        # Steer test prompt
        scale = pc.singular_value * steer_alpha / len(result.prompt_labels)
        pc.steered_positive = _generate_manipulated(
            model, tokenizer, test_prompt,
            direction_steers=[(result.layer, pc.direction, scale)],
            max_tokens=max_tokens)
        pc.steered_negative = _generate_manipulated(
            model, tokenizer, test_prompt,
            direction_steers=[(result.layer, pc.direction, -scale)],
            max_tokens=max_tokens)

        # lm_head mapping: which tokens does this PC direction point toward?
        # direction → norm → lm_head → logits → top tokens
        h_normed = inner.norm(mx.array((result.mean_state + pc.direction * pc.singular_value).astype(np.float16).reshape(1, 1, -1)))
        logits_pos = np.array(_lm_head(model, h_normed).astype(mx.float32)[0, 0, :])
        h_normed_neg = inner.norm(mx.array((result.mean_state - pc.direction * pc.singular_value).astype(np.float16).reshape(1, 1, -1)))
        logits_neg = np.array(_lm_head(model, h_normed_neg).astype(mx.float32)[0, 0, :])

        # Tokens most affected by this PC
        diff = logits_pos - logits_neg
        top_pos_tok = np.argsort(diff)[::-1][:n_top_tokens]
        top_neg_tok = np.argsort(diff)[:n_top_tokens]

        pc.top_positive_tokens = [(tokenizer.decode([int(t)]), float(diff[t])) for t in top_pos_tok]
        pc.top_negative_tokens = [(tokenizer.decode([int(t)]), float(diff[t])) for t in top_neg_tok]


def random_direction_control(
    model: Any, tokenizer: Any,
    prompt: str,
    *,
    layer: int | None = None,
    magnitude: float = 100.0,
    n_trials: int = 10,
    max_tokens: int = 15,
    model_config: Any = None,
) -> list[str]:
    """Steer with random directions of specified magnitude as a control.
    If layer is None, uses model's last layer.
    """
    if layer is None:
        from .model_config import detect_config
        cfg = model_config or detect_config(model)
        layer = cfg.last_layer
    from .manipulate import _generate_manipulated

    rng = np.random.default_rng(42)
    hidden_size = 3584  # fallback

    import mlx.core as mx
    inner = getattr(model, "model", model)
    if hasattr(inner, "norm") and hasattr(inner.norm, "weight"):
        hidden_size = inner.norm.weight.shape[0]

    results = []
    for _ in range(n_trials):
        d = rng.normal(size=hidden_size).astype(np.float32)
        d = d / np.linalg.norm(d)
        text = _generate_manipulated(
            model, tokenizer, prompt,
            direction_steers=[(layer, d, magnitude)],
            max_tokens=max_tokens)
        results.append(text)

    return results


def compare_pc_vs_random(
    model: Any, tokenizer: Any,
    result: BehavioralPCAResult,
    prompt: str,
    *,
    n_pcs: int = 5,
    n_random: int = 10,
    max_tokens: int = 15,
) -> dict[str, Any]:
    """Compare PC-based steering to random-direction steering."""
    from .manipulate import _generate_manipulated
    from .steer import generate_steered

    baseline = generate_steered(model, tokenizer, prompt, {}, max_tokens=max_tokens)
    baseline_text = baseline["generated"]

    pc_texts = {}
    for pc in result.components[:n_pcs]:
        scale = pc.singular_value * 2.0 / len(result.prompt_labels)
        text = _generate_manipulated(
            model, tokenizer, prompt,
            direction_steers=[(result.layer, pc.direction, scale)],
            max_tokens=max_tokens)
        pc_texts[f"PC{pc.index}"] = text

    # Random directions at matched magnitude
    magnitudes = [result.components[i].singular_value * 2.0 / len(result.prompt_labels)
                  for i in range(min(n_pcs, len(result.components)))]
    avg_magnitude = np.mean(magnitudes)

    random_texts = random_direction_control(
        model, tokenizer, prompt,
        layer=result.layer, magnitude=avg_magnitude,
        n_trials=n_random, max_tokens=max_tokens)

    # Measure: how different is each steered text from baseline?
    def text_distance(a, b):
        # Simple: fraction of first 10 words that differ
        wa = a.split()[:10]
        wb = b.split()[:10]
        if not wa or not wb:
            return 1.0
        common = sum(1 for x, y in zip(wa, wb) if x == y)
        return 1.0 - common / max(len(wa), len(wb))

    pc_distances = {name: text_distance(text, baseline_text) for name, text in pc_texts.items()}
    random_distances = [text_distance(text, baseline_text) for text in random_texts]

    return {
        "baseline": baseline_text,
        "pc_texts": pc_texts,
        "random_texts": random_texts[:3],
        "pc_mean_distance": float(np.mean(list(pc_distances.values()))),
        "random_mean_distance": float(np.mean(random_distances)),
        "pc_distances": pc_distances,
    }
