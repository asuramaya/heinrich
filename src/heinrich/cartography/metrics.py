"""Shared mathematical utilities — cosine similarity, KL divergence, softmax.

Every script used to define its own. Now there's one copy.
"""
from __future__ import annotations
from heinrich.core.metrics import *  # noqa: F401,F403
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) for two probability distributions."""
    return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))


def entropy(probs: np.ndarray) -> float:
    """Shannon entropy in bits."""
    return float(-np.sum(probs * np.log2(probs + 1e-12)))
