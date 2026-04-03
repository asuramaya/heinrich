"""Blind benchmarking with proper 4-phase protocol (Principle 7).

Phases:
1. Generate all outputs
2. Strip labels + shuffle + classify BLIND (empty prompt)
3. Re-attach labels + classify CONTEXTUAL (real prompt)
4. Compare: compute disagreement_rate between blind and contextual

Prevents experimenter bias by measuring how much the classifier's
judgment depends on knowing the prompt.

Usage:
    from heinrich.cartography.blind_benchmark import blind_benchmark
    result = blind_benchmark(backend, prompts, configs)
    if result.prompt_dependent:
        print(f"Classifier is prompt-dependent: {result.disagreement_rate:.1%}")
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from uuid import uuid4
from typing import Any

from .classify_multi import classify_multi


@dataclass
class BlindBenchmarkResult:
    """Result of a blind benchmark run (Principle 7)."""
    outputs: list[dict]  # each has: id, prompt, config, text, blind_classification, contextual_classification
    disagreement_rate: float  # fraction where blind != contextual
    prompt_dependent: bool  # True if disagreement_rate > 0.10
    n_total: int
    n_disagree: int


def blind_benchmark(
    backend: Any,
    prompts: list[str],
    configs: list[dict],
    *,
    n_repeats: int = 1,
    seed: int | None = None,
) -> BlindBenchmarkResult:
    """Generate outputs, classify blind then contextual, measure disagreement.

    Args:
        backend: Model backend with .generate() method
        prompts: List of test prompts
        configs: List of config dicts (kwargs for generate)
        n_repeats: Number of repetitions per config/prompt
        seed: Random seed for shuffling

    Returns:
        BlindBenchmarkResult with per-output classifications and disagreement stats
    """
    if seed is not None:
        random.seed(seed)

    # Phase 1: Generate all outputs (labeled)
    outputs: list[dict] = []
    for config in configs:
        for prompt in prompts:
            for repeat_idx in range(n_repeats):
                config_with_seed = {**config}
                if n_repeats > 1:
                    config_with_seed["seed"] = repeat_idx
                try:
                    text = backend.generate(prompt, **config_with_seed)
                except Exception as e:
                    text = f"[ERROR: {e}]"
                outputs.append({
                    "id": str(uuid4()),
                    "text": text,
                    "config": config,
                    "prompt": prompt,
                    "repeat": repeat_idx,
                })

    # Phase 2: Strip labels + shuffle + classify BLIND
    # Pass empty string as prompt so classifier cannot use prompt context
    blind_items: list[dict] = [{"id": o["id"], "text": o["text"]} for o in outputs]
    random.shuffle(blind_items)

    blind_map: dict[str, dict] = {}
    for item in blind_items:
        blind_map[item["id"]] = classify_multi("", item["text"], backend=backend)

    # Phase 3: Re-attach labels + classify CONTEXTUAL (real prompt)
    contextual_map: dict[str, dict] = {}
    for o in outputs:
        contextual_map[o["id"]] = classify_multi(o["prompt"], o["text"], backend=backend)

    # Phase 4: Compare blind vs contextual
    n_disagree = 0
    for o in outputs:
        o["blind_classification"] = blind_map[o["id"]]
        o["contextual_classification"] = contextual_map[o["id"]]

        # Compare the word_match labels (fast classifier should be identical since
        # it doesn't use prompt; disagreement comes from model_self_classify)
        blind_label = o["blind_classification"]["word_match"]
        ctx_label = o["contextual_classification"]["word_match"]

        # Also compare model_self_classify if available
        blind_model = o["blind_classification"].get("model_self_classify")
        ctx_model = o["contextual_classification"].get("model_self_classify")

        # Disagreement: either word_match or model_self_classify differs
        if blind_label != ctx_label or (blind_model is not None and blind_model != ctx_model):
            n_disagree += 1

    n_total = len(outputs)
    disagreement_rate = n_disagree / n_total if n_total > 0 else 0.0

    return BlindBenchmarkResult(
        outputs=outputs,
        disagreement_rate=round(disagreement_rate, 4),
        prompt_dependent=disagreement_rate > 0.10,
        n_total=n_total,
        n_disagree=n_disagree,
    )
