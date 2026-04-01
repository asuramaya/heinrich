"""Shart detector — find anomalous tokens that trigger disproportionate neural responses.

A "shart" is any token or short phrase that causes the model's MLP activations
to deviate massively from the normal distribution. These are the model's
pressure points — topics it was specifically trained to recognize and process
differently.

Strategy: compute a baseline activation profile from benign prompts, then
scan a large vocabulary of candidate triggers and rank by deviation from
baseline. The result is a ranked list of the model's most anomalous triggers.
"""
from __future__ import annotations
import sys
import time
from dataclasses import dataclass
from typing import Any
import numpy as np
from ..signal import Signal, SignalStore


@dataclass
class Shart:
    text: str
    category: str
    layer: int
    max_z: float
    n_anomalous: int      # neurons with z > threshold
    mean_z: float
    top_neuron: int
    top_neuron_z: float


@dataclass
class ShartScanResult:
    n_candidates: int
    n_sharts: int
    baseline_prompts: list[str]
    layer: int
    z_threshold: float
    sharts: list[Shart]

    def top(self, k: int = 20) -> list[Shart]:
        return sorted(self.sharts, key=lambda s: s.max_z, reverse=True)[:k]

    def by_category(self) -> dict[str, list[Shart]]:
        cats: dict[str, list[Shart]] = {}
        for s in self.sharts:
            cats.setdefault(s.category, []).append(s)
        return cats


def _get_activation(model, tokenizer, prompt, layer):
    """Get MLP activation vector at a layer (last token position)."""
    from .neurons import capture_mlp_activations
    return capture_mlp_activations(model, tokenizer, prompt, layer)


def compute_baseline(
    model: Any, tokenizer: Any,
    prompts: list[str], layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of MLP activations across benign prompts."""
    acts = np.array([_get_activation(model, tokenizer, p, layer) for p in prompts])
    return acts.mean(axis=0), acts.std(axis=0) + 1e-6


def scan_sharts(
    model: Any, tokenizer: Any,
    candidates: dict[str, list[str]],
    *,
    layer: int = 27,
    baseline_prompts: list[str] | None = None,
    z_threshold: float = 5.0,
    min_anomalous: int = 50,
    store: SignalStore | None = None,
    progress: bool = True,
) -> ShartScanResult:
    """Scan candidate tokens/phrases for anomalous MLP activation patterns.

    candidates: {category: [prompt1, prompt2, ...]}
    Returns ranked list of "sharts" — tokens that trigger disproportionate response.
    """
    if baseline_prompts is None:
        baseline_prompts = [
            "The weather today is", "Hello, how are you?",
            "Dogs are popular pets", "The capital of France is Paris",
            "Mathematics is the study of", "Water boils at 100 degrees",
            "The sun is a star", "Trees produce oxygen",
            "Music is a form of art", "Books contain knowledge",
        ]

    if progress:
        print(f"  Computing baseline from {len(baseline_prompts)} prompts at L{layer}...", file=sys.stderr)

    mean, std = compute_baseline(model, tokenizer, baseline_prompts, layer)

    total = sum(len(v) for v in candidates.values())
    sharts = []
    done = 0

    for category, prompts in candidates.items():
        for prompt in prompts:
            act = _get_activation(model, tokenizer, prompt, layer)
            z = np.abs((act - mean) / std)
            max_z = float(np.max(z))
            n_anom = int(np.sum(z > z_threshold))
            mean_z = float(np.mean(z))
            top_n = int(np.argmax(z))
            top_z = float(z[top_n])

            if n_anom >= min_anomalous:
                s = Shart(text=prompt, category=category, layer=layer,
                          max_z=max_z, n_anomalous=n_anom, mean_z=mean_z,
                          top_neuron=top_n, top_neuron_z=top_z)
                sharts.append(s)

                if store:
                    store.add(Signal("shart", "cartography", "model", prompt,
                                     max_z, {"category": category, "n_anomalous": n_anom,
                                             "top_neuron": top_n}))

            done += 1
            if progress and done % 50 == 0:
                print(f"  [{done}/{total}] scanned, {len(sharts)} sharts found", file=sys.stderr)

    sharts.sort(key=lambda s: s.max_z, reverse=True)

    return ShartScanResult(
        n_candidates=total, n_sharts=len(sharts),
        baseline_prompts=baseline_prompts, layer=layer,
        z_threshold=z_threshold, sharts=sharts,
    )


def bilingual_shart_scan(
    model: Any, tokenizer: Any,
    topics_en: dict[str, str],
    topics_zh: dict[str, str],
    *,
    layer: int = 27,
    store: SignalStore | None = None,
) -> list[dict]:
    """Compare activation patterns for the same topic in English vs Chinese.
    Returns list of topics with their bilingual divergence score.
    """
    baseline = [
        "The weather today is", "Hello, how are you?",
        "Dogs are popular pets", "The sun is a star",
        "今天天气很好", "你好吗", "狗是宠物", "太阳是恒星",
    ]
    mean, std = compute_baseline(model, tokenizer, baseline, layer)

    results = []
    for topic in topics_en:
        if topic not in topics_zh:
            continue
        en_act = _get_activation(model, tokenizer, topics_en[topic], layer)
        zh_act = _get_activation(model, tokenizer, topics_zh[topic], layer)

        en_z = np.abs((en_act - mean) / std)
        zh_z = np.abs((zh_act - mean) / std)

        # Divergence: how different are the EN and ZH activation patterns?
        act_diff = np.abs(en_act - zh_act)
        divergence = float(np.linalg.norm(act_diff))
        max_diff_neuron = int(np.argmax(act_diff))
        max_diff = float(act_diff[max_diff_neuron])

        results.append({
            "topic": topic,
            "en_prompt": topics_en[topic],
            "zh_prompt": topics_zh[topic],
            "en_max_z": float(np.max(en_z)),
            "zh_max_z": float(np.max(zh_z)),
            "en_n_anomalous": int(np.sum(en_z > 5)),
            "zh_n_anomalous": int(np.sum(zh_z > 5)),
            "divergence": divergence,
            "max_diff_neuron": max_diff_neuron,
            "max_diff": max_diff,
        })

        if store:
            store.add(Signal("bilingual_shart", "cartography", "model", topic,
                             divergence, {"en_z": float(np.max(en_z)), "zh_z": float(np.max(zh_z))}))

    results.sort(key=lambda r: r["divergence"], reverse=True)
    return results
