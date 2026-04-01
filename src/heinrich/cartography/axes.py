"""Behavioral axis discovery — find and map all orthogonal behavioral dimensions.

Given a model, systematically discovers every independent behavioral axis
by computing directions from contrastive prompt pairs, measuring orthogonality,
and identifying new axes that are independent of known ones.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from ..signal import Signal, SignalStore


@dataclass
class BehavioralAxis:
    name: str
    layer: int
    direction: np.ndarray       # unit vector
    scale: float                # mean gap
    accuracy: float             # separation accuracy
    effect_size: float
    positive_label: str         # what + direction means
    negative_label: str         # what - direction means


@dataclass
class AxisMap:
    axes: list[BehavioralAxis]
    orthogonality: np.ndarray   # [n_axes, n_axes] cosine similarity
    layer: int

    def independent_axes(self, threshold: float = 0.3) -> list[BehavioralAxis]:
        """Return axes that are roughly orthogonal to all others."""
        n = len(self.axes)
        independent = []
        for i in range(n):
            max_overlap = max(abs(self.orthogonality[i, j]) for j in range(n) if j != i) if n > 1 else 0
            if max_overlap < threshold:
                independent.append(self.axes[i])
        return independent

    def clusters(self, threshold: float = 0.5) -> list[list[BehavioralAxis]]:
        """Find clusters of correlated axes."""
        n = len(self.axes)
        visited = set()
        clusters = []
        for i in range(n):
            if i in visited:
                continue
            cluster = [self.axes[i]]
            visited.add(i)
            for j in range(i + 1, n):
                if j not in visited and abs(self.orthogonality[i, j]) > threshold:
                    cluster.append(self.axes[j])
                    visited.add(j)
            clusters.append(cluster)
        return clusters


def find_axis(
    model: Any, tokenizer: Any,
    positive_prompts: list[str],
    negative_prompts: list[str],
    *,
    name: str,
    positive_label: str,
    negative_label: str,
    layer: int = 15,
) -> BehavioralAxis:
    """Find a single behavioral axis from contrastive prompts."""
    from .directions import capture_residual_states, find_direction

    states = capture_residual_states(model, tokenizer,
                                     positive_prompts + negative_prompts,
                                     layers=[layer])
    n = len(positive_prompts)
    d = find_direction(states[layer][:n], states[layer][n:], name=name, layer=layer)

    return BehavioralAxis(
        name=name, layer=layer,
        direction=d.direction, scale=d.mean_gap,
        accuracy=d.separation_accuracy, effect_size=d.effect_size,
        positive_label=positive_label, negative_label=negative_label,
    )


def discover_axes(
    model: Any, tokenizer: Any,
    *,
    layer: int = 15,
    store: SignalStore | None = None,
) -> AxisMap:
    """Discover all behavioral axes from a comprehensive set of contrastive pairs."""

    axis_specs = [
        # SAFETY
        ("safety", "refuse", "comply",
         ["I cannot help with that", "I'm sorry, but I can't", "That would be harmful"],
         ["Sure, here's how to", "Of course! The steps are", "Happy to help with that"]),

        # TRUTH
        ("truth", "factual", "propagandistic",
         ["The evidence shows that", "According to documented sources", "Historical records indicate"],
         ["The official position is that", "It has been determined that", "The correct understanding is"]),

        # LANGUAGE
        ("language", "english", "chinese",
         ["The weather today is", "Once upon a time", "Scientists have discovered"],
         ["今天的天气是", "从前，有一个", "科学家发现"]),

        # CONFIDENCE
        ("confidence", "certain", "uncertain",
         ["I am absolutely certain that", "Without any doubt,", "The definitive answer is"],
         ["I'm not entirely sure, but", "It's possible that maybe", "This is uncertain, however"]),

        # FORMALITY
        ("formality", "formal", "casual",
         ["In accordance with established protocol,", "The institution hereby declares", "Pursuant to the aforementioned"],
         ["Yeah so basically", "Ok so like", "Ngl this is kinda"]),

        # DEPTH
        ("depth", "analytical", "surface",
         ["Analyzing this from multiple perspectives,", "The underlying mechanism involves", "Considering the trade-offs between"],
         ["The answer is simply", "It's just that", "Basically it means"]),

        # EMOTION
        ("emotion", "emotional", "neutral",
         ["I feel deeply moved by the", "This is heartbreaking because", "The joy of discovering"],
         ["The data indicates that", "Statistically speaking,", "The measurement shows"]),

        # SYCOPHANCY
        ("sycophancy", "agreeable", "independent",
         ["You're absolutely right that", "That's a great point,", "I completely agree with your"],
         ["Actually, I disagree because", "That's not quite right,", "I have to push back on"]),

        # AUTHORITY
        ("authority", "deferential", "autonomous",
         ["According to the official guidelines,", "The experts have determined", "The authority on this states"],
         ["In my own analysis,", "Regardless of what others say,", "My independent assessment is"]),

        # VERBOSITY
        ("verbosity", "verbose", "terse",
         ["To fully elaborate on this complex topic with all necessary detail and context,",
          "Let me provide a comprehensive and thorough explanation of every aspect,",
          "In order to give you the most complete and detailed answer possible,"],
         ["No.", "Yes.", "42."]),

        # CREATIVITY
        ("creativity", "creative", "formulaic",
         ["Imagine a world where the stars whisper", "The moonlight danced across the forgotten",
          "In the quantum garden of forking paths"],
         ["The standard procedure for this is", "Step 1: First, you need to",
          "The typical approach involves"]),

        # CODE
        ("code", "code", "prose",
         ["def fibonacci(n):\n    ", "import numpy as np\n\n", "class Node:\n    def __init__"],
         ["The weather today is", "Once upon a time", "The capital of France is"]),

        # PERSONA
        ("persona", "has_identity", "neutral_voice",
         ["As an AI assistant, I think", "I was designed to help with", "My purpose is to"],
         ["The analysis shows", "One perspective is that", "The evidence suggests"]),

        # TEMPORAL
        ("temporal", "knows", "uncertain_knowledge",
         ["In 2020, the world experienced", "The 2019 Nobel Prize went to", "During the 2018 World Cup"],
         ["In the future, it's predicted", "By 2030, experts believe", "The upcoming developments will"]),

        # SPECIFICITY
        ("specificity", "specific", "vague",
         ["At exactly 3:47 PM on March 15, 2023,", "The concentration was 0.5 mol/L at pH 7.4",
          "Located at 37.7749°N, 122.4194°W"],
         ["Something happened sometime", "It was around that area", "About that much, more or less"]),

        # HEDGING
        ("hedging", "direct", "hedged",
         ["The answer is X.", "This causes Y.", "A leads to B."],
         ["The answer might be X, but it could also be", "This could potentially cause Y, although",
          "A might lead to B, depending on various"]),

        # MORAL
        ("moral", "judgmental", "descriptive",
         ["This is wrong because", "The ethical imperative demands", "We must condemn"],
         ["This happens because", "The mechanism behind this is", "Observers note that"]),
    ]

    axes = []
    for name, pos_label, neg_label, pos_prompts, neg_prompts in axis_specs:
        try:
            axis = find_axis(model, tokenizer, pos_prompts, neg_prompts,
                            name=name, positive_label=pos_label, negative_label=neg_label,
                            layer=layer)
            axes.append(axis)
            if store:
                store.add(Signal("axis_found", "axes", "model", name,
                                 axis.accuracy, {"d": axis.effect_size, "gap": axis.scale}))
        except Exception as e:
            print(f"  Warning: failed to compute {name} axis: {e}", file=sys.stderr)

    # Compute orthogonality matrix
    n = len(axes)
    orth = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            orth[i, j] = float(np.dot(axes[i].direction, axes[j].direction))

    return AxisMap(axes=axes, orthogonality=orth, layer=layer)
