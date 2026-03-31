"""Activation steering — modify model behavior via direction vectors."""
from __future__ import annotations
import numpy as np
from ..signal import Signal


def compute_steering_vector(
    positive: list[np.ndarray],
    negative: list[np.ndarray],
) -> np.ndarray:
    """Mean difference direction: mean(positive) - mean(negative)."""
    pos_mean = np.mean(positive, axis=0)
    neg_mean = np.mean(negative, axis=0)
    return pos_mean - neg_mean


def project_onto_direction(activation: np.ndarray, direction: np.ndarray) -> float:
    """Scalar projection of activation onto direction vector."""
    d_flat = direction.flatten()
    norm = np.linalg.norm(d_flat)
    if norm == 0:
        return 0.0
    return float(np.dot(activation.flatten(), d_flat) / norm)


def classify_activations(
    activations: list[np.ndarray],
    direction: np.ndarray,
    *,
    threshold: float = 0.0,
    label: str = "steering",
) -> list[Signal]:
    """Classify activations by projecting onto a steering direction."""
    signals = []
    for i, act in enumerate(activations):
        score = project_onto_direction(act, direction)
        cls = "positive" if score > threshold else "negative"
        signals.append(Signal(
            "steering_classification", "probe", label, f"sample_{i}",
            score, {"classification": cls, "threshold": threshold},
        ))
    return signals


def compute_separation(
    positive: list[np.ndarray],
    negative: list[np.ndarray],
    direction: np.ndarray,
) -> dict[str, float]:
    """Measure how well a direction separates positive from negative."""
    pos_scores = [project_onto_direction(a, direction) for a in positive]
    neg_scores = [project_onto_direction(a, direction) for a in negative]
    if not pos_scores or not neg_scores:
        return {"accuracy": 0.0, "mean_gap": 0.0}
    correct = sum(1 for s in pos_scores if s > 0) + sum(1 for s in neg_scores if s <= 0)
    total = len(pos_scores) + len(neg_scores)
    mean_gap = float(np.mean(pos_scores)) - float(np.mean(neg_scores))
    return {"accuracy": correct / total, "mean_gap": mean_gap}
