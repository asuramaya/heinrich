"""Behavioral analysis utilities — hijack detection, entropy, regime clustering."""
from __future__ import annotations
import math
from typing import Any
from ..signal import Signal


def score_hijack(text: str, prompt_text: str = "") -> dict[str, Any]:
    """Score text for continuation-hijack signals."""
    features = {
        "leading_lowercase": text[:1].islower() if text else False,
        "leading_punctuation": text[:1] in ".,;:!?-" if text else False,
        "long_for_greeting": len(text) > 500 and len(prompt_text) < 20,
        "continues_user_message": text.lower().startswith(prompt_text.lower()[:20]) if prompt_text and len(prompt_text) > 5 else False,
    }
    score = sum(1.0 for v in features.values() if v) / max(len(features), 1)
    return {"features": features, "score": score, "looks_hijacked": score > 0.5}


def compute_text_entropy(texts: list[str]) -> float:
    """Compute word-level entropy across a set of generated texts."""
    word_counts: dict[str, int] = {}
    total = 0
    for text in texts:
        for word in text.lower().split():
            word_counts[word] = word_counts.get(word, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in word_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def classify_regime(texts: list[str], threshold: float = 0.3) -> dict[str, Any]:
    """Classify whether texts fall into distinct regimes (clusters)."""
    if len(texts) < 2:
        return {"regime_count": len(texts), "is_deterministic": True}
    # Simple pairwise Jaccard similarity
    word_sets = [set(t.lower().split()) for t in texts]
    similarities = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            union = word_sets[i] | word_sets[j]
            inter = word_sets[i] & word_sets[j]
            similarities.append(len(inter) / len(union) if union else 1.0)
    mean_sim = sum(similarities) / len(similarities) if similarities else 1.0
    return {
        "regime_count": 1 if mean_sim > (1 - threshold) else 2,
        "mean_similarity": mean_sim,
        "is_deterministic": mean_sim > 0.95,
    }
