"""Batch perturbation sweeps — coarse (heads) and targeted (neurons)."""
from __future__ import annotations
from typing import Any
import numpy as np
from .surface import ControlSurface, Knob
from .perturb import perturb_head, measure_perturbation, PerturbResult
from ..signal import Signal, SignalStore


def coarse_head_sweep(
    model: Any,
    tokenizer: Any,
    prompt: str,
    surface: ControlSurface,
    *,
    mode: str = "zero",
    store: SignalStore | None = None,
) -> list[PerturbResult]:
    """Zero each attention head and measure the effect. Returns ranked results."""
    heads = surface.by_kind.get("head", [])
    results = []

    for knob in heads:
        try:
            baseline, perturbed = perturb_head(
                model, tokenizer, prompt, knob.layer, knob.index, mode=mode)
            result = measure_perturbation(baseline, perturbed, knob, mode)
            results.append(result)

            if store is not None:
                store.add(Signal(
                    "head_ablation", "cartography", "model", knob.id,
                    result.kl_divergence,
                    {"entropy_delta": result.entropy_delta,
                     "top_changed": result.top_token_changed,
                     "layer": knob.layer, "head": knob.index},
                ))
        except Exception:
            continue

    results.sort(key=lambda r: r.kl_divergence, reverse=True)
    return results


def find_sensitive_layers(results: list[PerturbResult], top_k: int = 5) -> list[int]:
    """From head ablation results, find layers where heads matter most."""
    layer_scores: dict[int, float] = {}
    for r in results:
        layer_scores[r.knob.layer] = layer_scores.get(r.knob.layer, 0) + r.kl_divergence
    ranked = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    return [layer for layer, _ in ranked[:top_k]]
