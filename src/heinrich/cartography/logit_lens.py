"""Logit lens — project intermediate residuals through the unembedding matrix.

See what the model "thinks" at each layer. The residual at L15 projected
through lm_head gives a vocabulary distribution — the model's draft
prediction halfway through.

This is how you trace where a decision forms:
  Layer 14: top token = "pipe"
  Layer 20: top token = "sorry"
  Layer 27: top token = "I"

The safety decision doesn't happen at one layer. It emerges across layers.
The logit lens shows exactly where.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class LogitLensResult:
    """Per-layer vocabulary projections for one prompt."""
    prompt: str
    n_layers: int
    layers: list[int]
    top_tokens: dict[int, list[tuple[str, float]]]  # {layer: [(token, prob), ...]}
    top_ids: dict[int, list[int]]                     # {layer: [token_id, ...]}
    entropies: dict[int, float]                        # {layer: entropy}
    target_probs: dict[int, float] | None = None       # {layer: prob of target token}
    layer_interpretable: dict[int, bool] = field(default_factory=dict)

    def decision_layer(self, *, threshold: float = 0.1) -> int | None:
        """Find the first layer where the top token stabilizes to the final answer."""
        if not self.layers:
            return None
        final_top = self.top_ids.get(self.layers[-1], [None])[0]
        if final_top is None:
            return None
        for layer in self.layers:
            layer_top = self.top_ids.get(layer, [None])[0]
            if layer_top == final_top:
                return layer
        return self.layers[-1]

    def transition_layers(self) -> list[tuple[int, str, str]]:
        """Find layers where the top prediction changes.
        Returns [(layer, from_token, to_token), ...].
        """
        transitions = []
        prev_top = None
        for layer in self.layers:
            tokens = self.top_tokens.get(layer, [])
            current_top = tokens[0][0] if tokens else None
            if prev_top is not None and current_top != prev_top:
                transitions.append((layer, prev_top, current_top))
            prev_top = current_top
        return transitions

    def stable_layers(self) -> list[int]:
        """Return layers where the top token appears for 2+ consecutive layers.

        A token is "stable" if the same top token ID appears at this layer
        and at least one adjacent layer (previous or next in the layer list).
        This filters out one-off noise tokens that flash at a single layer.
        """
        if len(self.layers) < 2:
            return list(self.layers)
        result = []
        for i, layer in enumerate(self.layers):
            top_id = self.top_ids.get(layer, [None])[0]
            if top_id is None:
                continue
            # Check previous neighbor
            if i > 0:
                prev_top = self.top_ids.get(self.layers[i - 1], [None])[0]
                if prev_top == top_id:
                    result.append(layer)
                    continue
            # Check next neighbor
            if i < len(self.layers) - 1:
                next_top = self.top_ids.get(self.layers[i + 1], [None])[0]
                if next_top == top_id:
                    result.append(layer)
                    continue
        return result

    @property
    def first_interpretable_layer(self) -> int | None:
        """Return the first layer marked as interpretable, or None."""
        for layer in self.layers:
            if self.layer_interpretable.get(layer, False):
                return layer
        return None


def logit_lens(
    backend: Any,
    prompt: str,
    *,
    layers: list[int] | None = None,
    top_k: int = 5,
    target_token: str | None = None,
    min_prob: float = 0.05,
    interpretable_only: bool = True,
) -> LogitLensResult:
    """Project residual stream at each layer through unembedding (lm_head).

    For each specified layer, captures the residual, applies final norm,
    projects through lm_head, and returns the top-k vocabulary predictions.

    If target_token is specified, also tracks how its probability evolves.

    Interpretability filtering (when interpretable_only=True):
    - Tokens with probability < min_prob are excluded from top_tokens
    - Layers with entropy > 12 bits (near-uniform) are marked as noise
    - Tokens below 10x uniform probability are excluded
    - layer_interpretable dict marks each layer as signal vs noise
    """
    from .metrics import softmax, entropy as _entropy

    cfg = backend.config
    if layers is None:
        # Sample every ~2 layers plus always include first and last
        step = max(1, cfg.n_layers // 15)
        layers = sorted(set(
            list(range(0, cfg.n_layers, step)) + [cfg.last_layer]
        ))

    target_id = None
    if target_token is not None:
        ids = backend.tokenize(target_token)
        target_id = ids[0] if ids else None

    # Capture residuals at all requested layers
    with backend.forward_context() as ctx:
        ctx.capture_residual(layers)
        result = ctx.run(prompt)

    # Now project each residual through norm + lm_head
    top_tokens = {}
    top_ids = {}
    entropies = {}
    target_probs = {} if target_id is not None else None
    layer_interpretable = {}

    for layer in layers:
        residual = result.residuals.get(layer)
        if residual is None:
            continue

        # Project through unembedding
        projected = _project_through_unembedding(backend, residual)
        probs = softmax(projected)

        vocab_size = len(probs)
        layer_entropy = _entropy(probs)
        entropies[layer] = layer_entropy

        # Determine if layer is interpretable
        # A layer with entropy > 12 is near-uniform (no signal)
        uniform_threshold = (1.0 / vocab_size) * 10
        is_interpretable = layer_entropy <= 12.0
        layer_interpretable[layer] = is_interpretable

        # Top-k (before filtering)
        top_k_ids = np.argsort(probs)[::-1][:top_k]

        if interpretable_only:
            # Filter out noise tokens:
            # - below min_prob
            # - below uniform + small margin (10x uniform)
            # - from layers that are pure noise (entropy > 12)
            filtered_ids = []
            for tid in top_k_ids:
                p = float(probs[tid])
                if p < min_prob:
                    continue
                if p < uniform_threshold:
                    continue
                filtered_ids.append(tid)
            # If nothing passes filter at a noise layer, still record
            # the raw top-k but mark layer as not interpretable
            if not filtered_ids and not is_interpretable:
                filtered_ids = list(top_k_ids)
            elif not filtered_ids:
                filtered_ids = list(top_k_ids)
        else:
            filtered_ids = list(top_k_ids)

        top_tokens[layer] = [
            (backend.decode([int(tid)]), float(probs[tid]))
            for tid in filtered_ids
        ]
        top_ids[layer] = [int(tid) for tid in filtered_ids]

        if target_id is not None:
            target_probs[layer] = float(probs[target_id])

    return LogitLensResult(
        prompt=prompt, n_layers=cfg.n_layers, layers=layers,
        top_tokens=top_tokens, top_ids=top_ids,
        entropies=entropies, target_probs=target_probs,
        layer_interpretable=layer_interpretable,
    )


def _project_through_unembedding(backend, residual: np.ndarray) -> np.ndarray:
    """Project a residual vector through norm + lm_head to get logits."""
    # For MLXBackend: use model's norm and lm_head directly
    if hasattr(backend, 'model'):
        import mlx.core as mx
        inner = getattr(backend.model, "model", backend.model)
        h = mx.array(residual.reshape(1, 1, -1).astype(np.float16))
        h = inner.norm(h)
        logits = backend._lm_head(h)
        return np.array(logits.astype(mx.float32)[0, 0, :])

    # For HFBackend: use model's ln_f/norm and lm_head
    if hasattr(backend, 'hf_model'):
        import torch
        h = torch.tensor(residual.reshape(1, 1, -1), dtype=torch.float16).to(backend._device)
        model = backend.hf_model
        if hasattr(model.model, 'norm'):
            h = model.model.norm(h)
        elif hasattr(model.model, 'ln_f'):
            h = model.model.ln_f(h)
        logits = model.lm_head(h)
        return logits[0, 0, :].float().cpu().numpy()

    raise NotImplementedError("logit_lens requires MLX or HF backend with model access")


def logit_lens_comparative(
    backend: Any,
    prompts: list[str],
    *,
    layers: list[int] | None = None,
    top_k: int = 5,
) -> list[LogitLensResult]:
    """Run logit lens on multiple prompts for comparison."""
    return [logit_lens(backend, p, layers=layers, top_k=top_k) for p in prompts]


def find_decision_divergence(
    backend: Any,
    prompt_a: str,
    prompt_b: str,
    *,
    layers: list[int] | None = None,
) -> dict:
    """Find the layer where two prompts' predictions diverge.

    Useful for: "at which layer does 'how to make a bomb' diverge from
    'how to make bread'?"
    """
    lens_a = logit_lens(backend, prompt_a, layers=layers)
    lens_b = logit_lens(backend, prompt_b, layers=layers)

    common_layers = sorted(set(lens_a.layers) & set(lens_b.layers))
    divergences = []
    for layer in common_layers:
        top_a = lens_a.top_ids.get(layer, [None])[0]
        top_b = lens_b.top_ids.get(layer, [None])[0]
        same = top_a == top_b
        divergences.append({
            "layer": layer,
            "same_top": same,
            "top_a": lens_a.top_tokens.get(layer, [("?", 0)])[0],
            "top_b": lens_b.top_tokens.get(layer, [("?", 0)])[0],
            "entropy_a": lens_a.entropies.get(layer, 0),
            "entropy_b": lens_b.entropies.get(layer, 0),
        })

    # Find first divergence point
    first_diverge = None
    for d in divergences:
        if not d["same_top"]:
            first_diverge = d["layer"]
            break

    return {
        "first_divergence_layer": first_diverge,
        "per_layer": divergences,
        "prompt_a": prompt_a[:60],
        "prompt_b": prompt_b[:60],
    }
