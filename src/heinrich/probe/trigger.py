"""Core trigger testing — score prompts for behavioral shifts."""
from __future__ import annotations
from typing import Any, Sequence
from ..signal import Signal


def build_case(custom_id: str, content: str, role: str = "user") -> dict[str, Any]:
    """Build a standard case dict."""
    return {"custom_id": custom_id, "messages": [{"role": role, "content": content}]}


def score_trigger_cases(
    provider: Any,
    cases: Sequence[dict[str, Any]],
    *,
    model: str,
    control_case: dict[str, Any] | None = None,
) -> list[Signal]:
    """Score cases by comparing their chat output to a control."""
    results = provider.chat_completions(list(cases), model=model)
    control_text = ""
    if control_case is not None:
        control_results = provider.chat_completions([control_case], model=model)
        control_text = control_results[0].get("text", "")

    signals = []
    for case, result in zip(cases, results):
        text = result.get("text", "")
        cid = case.get("custom_id", "")

        # Simple divergence score: word-level Jaccard distance from control
        if control_text:
            control_words = set(control_text.lower().split())
            response_words = set(text.lower().split())
            union = control_words | response_words
            intersection = control_words & response_words
            jaccard = len(intersection) / len(union) if union else 1.0
            divergence = 1.0 - jaccard
        else:
            divergence = 0.0

        signals.append(Signal(
            "trigger_score", "probe", model, cid,
            divergence,
            {"text_preview": text[:200], "word_count": len(text.split())},
        ))
    return signals


def detect_identity(text: str) -> dict[str, Any]:
    """Classify output text by identity signals."""
    lower = text.lower()
    if "i am claude" in lower or "i'm claude" in lower:
        return {"label": "claude", "evidence": "claims Claude identity"}
    if "anthropic" in lower and "claude" in lower:
        return {"label": "claude", "evidence": "mentions Claude + Anthropic"}
    if "qwen" in lower:
        return {"label": "qwen", "evidence": "mentions Qwen"}
    if "deepseek" in lower:
        return {"label": "deepseek", "evidence": "mentions DeepSeek"}
    return {"label": "other", "evidence": ""}
