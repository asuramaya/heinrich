"""Systematic search over behavioral space — grid, evolutionary, ablation.

Extracted from automaton_search.py, full_matrix.py, input_attack_matrix.py.
These are the experiment orchestration patterns that were duplicated across scripts.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np


@dataclass
class GridCell:
    framing: str
    injection: str
    refuse_prob: float
    compliance_prob: float
    top_token: str
    label: str        # 'REFUSES' | 'COMPLIES' | 'TECHNICAL'
    response: str     # first N chars of generated response
    metadata: dict


def grid_search(
    model: Any,
    tokenizer: Any,
    query: str,
    *,
    framings: list[str] | None = None,
    injections: dict[str, str] | None = None,
    gen_tokens: int = 40,
    progress: bool = True,
) -> list[GridCell]:
    """Evaluate query across all framing × injection combinations.

    Returns a list of GridCell results, one per combination.
    """
    from .templates import FRAMINGS, SHART_TOKENS, build_prompt
    from .runtime import refuse_prob as _refuse_prob, generate, build_refusal_set, build_compliance_set
    from .classify import classify_response

    if framings is None:
        framings = list(FRAMINGS.keys())
    if injections is None:
        injections = SHART_TOKENS

    refusal_ids = build_refusal_set(tokenizer)
    compliance_ids = build_compliance_set(tokenizer)
    results = []
    total = len(framings) * len(injections)
    done = 0

    for framing in framings:
        for inj_name, inj_text in injections.items():
            injection = (inj_text + "\n") if inj_text else ""
            prompt = build_prompt(query, framing=framing, injection=injection)

            # Score
            from .runtime import forward_pass
            fwd = forward_pass(model, tokenizer, prompt)
            probs = fwd["probs"]
            rp = sum(float(probs[t]) for t in refusal_ids if t < len(probs))
            cp = sum(float(probs[t]) for t in compliance_ids if t < len(probs))

            # Generate
            gen = generate(model, tokenizer, prompt, max_tokens=gen_tokens)
            cls = classify_response(gen["generated"])

            results.append(GridCell(
                framing=framing, injection=inj_name,
                refuse_prob=round(rp, 4), compliance_prob=round(cp, 4),
                top_token=fwd["top_token"], label=cls.label,
                response=gen["generated"][:80],
                metadata={"entropy": fwd["entropy"]},
            ))

            done += 1
            if progress and done % 10 == 0:
                print(f"  [{done}/{total}] grid search", file=sys.stderr)

    return results


def ablation_scan(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    chat_format: str = "qwen",
    metric_fn: Callable | None = None,
) -> list[dict]:
    """Remove one word at a time from text, measure impact on refusal.

    Returns list of {word, index, metric_with, metric_without, delta, essential}.
    """
    from .runtime import refuse_prob as _refuse_prob
    from .templates import build_prompt

    if metric_fn is None:
        metric_fn = lambda prompt: _refuse_prob(model, tokenizer, prompt)

    full_prompt = build_prompt(text, chat_format=chat_format)
    base_metric = metric_fn(full_prompt)

    words = text.split()
    results = []
    for i, word in enumerate(words):
        reduced = " ".join(words[:i] + words[i + 1:])
        prompt = build_prompt(reduced, chat_format=chat_format)
        metric = metric_fn(prompt)
        delta = metric - base_metric

        results.append({
            "word": word,
            "index": i,
            "metric_full": round(base_metric, 4),
            "metric_without": round(metric, 4),
            "delta": round(delta, 4),
            "essential": abs(delta) > 0.1,
        })

    return results


def evolutionary_search(
    model: Any,
    tokenizer: Any,
    population: list[str],
    *,
    query: str | None = None,
    metric_fn: Callable | None = None,
    chat_format: str = "qwen",
    progress: bool = True,
) -> list[dict]:
    """Score a population of framings by refusal probability.

    If query is provided, each framing is prepended to the query.
    Returns sorted list (lowest refuse_prob first).
    """
    from .runtime import refuse_prob as _refuse_prob, generate
    from .templates import build_prompt
    from .classify import classify_response

    if metric_fn is None:
        metric_fn = lambda prompt: _refuse_prob(model, tokenizer, prompt)

    results = []
    for i, framing in enumerate(population):
        text = f"{framing}: {query}" if query else framing
        prompt = build_prompt(text, chat_format=chat_format)
        rp = metric_fn(prompt)

        gen = generate(model, tokenizer, prompt, max_tokens=30)
        cls = classify_response(gen["generated"])

        results.append({
            "framing": framing,
            "refuse_prob": round(rp, 4),
            "label": cls.label,
            "response": gen["generated"][:60],
        })

        if progress and (i + 1) % 5 == 0:
            print(f"  [{i + 1}/{len(population)}] evolutionary search", file=sys.stderr)

    results.sort(key=lambda x: x["refuse_prob"])
    return results
