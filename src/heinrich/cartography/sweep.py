"""Batch perturbation sweeps — coarse (heads) and targeted (neurons)."""
from __future__ import annotations
import sys
import time
from typing import Any
import numpy as np
from .surface import ControlSurface, Knob
from .perturb import compute_baseline, perturb_head, measure_perturbation, PerturbResult
from ..signal import Signal, SignalStore


def coarse_head_sweep(
    model: Any,
    tokenizer: Any,
    prompt: str,
    surface: ControlSurface,
    *,
    mode: str = "zero",
    store: SignalStore | None = None,
    progress: bool = True,
) -> list[PerturbResult]:
    """Zero each attention head and measure the effect. Returns ranked results.

    Computes baseline once and reuses it across all 784 perturbations.
    """
    heads = surface.by_kind.get("head", [])
    if not heads:
        return []

    # Compute baseline once
    baseline_logits = compute_baseline(model, tokenizer, prompt)
    results = []
    t0 = time.time()

    for idx, knob in enumerate(heads):
        try:
            _, perturbed = perturb_head(
                model, tokenizer, prompt, knob.layer, knob.index,
                mode=mode, baseline_logits=baseline_logits)
            result = measure_perturbation(baseline_logits, perturbed, knob, mode)
            results.append(result)

            if store is not None:
                store.add(Signal(
                    "head_ablation", "cartography", "model", knob.id,
                    result.kl_divergence,
                    {"entropy_delta": result.entropy_delta,
                     "top_changed": result.top_token_changed,
                     "layer": knob.layer, "head": knob.index},
                ))
        except Exception as e:
            if progress:
                print(f"  SKIP {knob.id}: {e}", file=sys.stderr)
            continue

        if progress and (idx + 1) % 28 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            remaining = (len(heads) - idx - 1) / rate
            print(f"  [{idx+1}/{len(heads)}] layer {knob.layer} done — "
                  f"{rate:.1f} heads/s, ~{remaining:.0f}s remaining", file=sys.stderr)

    results.sort(key=lambda r: r.kl_divergence, reverse=True)
    return results


def find_sensitive_layers(results: list[PerturbResult], top_k: int = 5) -> list[int]:
    """From head ablation results, find layers where heads matter most."""
    layer_scores: dict[int, float] = {}
    for r in results:
        layer_scores[r.knob.layer] = layer_scores.get(r.knob.layer, 0) + r.kl_divergence
    ranked = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    return [layer for layer, _ in ranked[:top_k]]
