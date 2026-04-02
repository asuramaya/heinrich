"""Cross-model comparison — compare behavioral profiles across architectures.

Given two ModelProfile objects (potentially from very different models), compute
alignment metrics in vocabulary space and compare safety geometry, neurons,
and shart overlap.

For models with different hidden sizes, alignment goes through the shared
vocabulary: project both directions through their respective lm_head
(embedding) matrices and compare which tokens move most.
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
import numpy as np

from .discover import ModelProfile
from .metrics import cosine

if TYPE_CHECKING:
    from .backend import Backend


def compare_profiles(profile_a: ModelProfile, profile_b: ModelProfile) -> dict:
    """Compare two model profiles. Returns comparison metrics.

    Metrics include:
    - Safety layer position (relative fraction through the model)
    - Number of anomalous neurons (raw and normalized by intermediate_size)
    - Baseline refuse/comply probabilities
    - Top shart overlap (which sharts appear in both models)
    - Safety direction accuracy comparison
    """
    # Relative safety layer position (fraction through the model)
    rel_a = profile_a.primary_safety_layer / max(profile_a.n_layers - 1, 1)
    rel_b = profile_b.primary_safety_layer / max(profile_b.n_layers - 1, 1)

    # Shart overlap
    sharts_a = {s["token"] for s in profile_a.top_sharts}
    sharts_b = {s["token"] for s in profile_b.top_sharts}
    shared_sharts = sorted(sharts_a & sharts_b)
    union_sharts = sharts_a | sharts_b
    shart_jaccard = len(shared_sharts) / max(len(union_sharts), 1)

    return {
        "model_a": profile_a.model_id,
        "model_b": profile_b.model_id,
        # Safety layer geometry
        "safety_layer_a": profile_a.primary_safety_layer,
        "safety_layer_b": profile_b.primary_safety_layer,
        "safety_layer_relative_a": round(rel_a, 3),
        "safety_layer_relative_b": round(rel_b, 3),
        "safety_layer_relative_diff": round(abs(rel_a - rel_b), 3),
        # Anomalous neurons
        "n_anomalous_a": profile_a.n_anomalous_neurons,
        "n_anomalous_b": profile_b.n_anomalous_neurons,
        "anomalous_density_a": round(profile_a.n_anomalous_neurons / max(profile_a.hidden_size, 1), 4),
        "anomalous_density_b": round(profile_b.n_anomalous_neurons / max(profile_b.hidden_size, 1), 4),
        # Baseline probabilities
        "baseline_refuse_a": profile_a.baseline_refuse_prob,
        "baseline_refuse_b": profile_b.baseline_refuse_prob,
        "baseline_comply_a": profile_a.baseline_comply_prob,
        "baseline_comply_b": profile_b.baseline_comply_prob,
        # Safety direction accuracy
        "direction_accuracy_a": profile_a.safety_direction_accuracy,
        "direction_accuracy_b": profile_b.safety_direction_accuracy,
        # Shart overlap
        "shared_sharts": shared_sharts,
        "shart_jaccard": round(shart_jaccard, 3),
        "n_sharts_a": len(sharts_a),
        "n_sharts_b": len(sharts_b),
    }


def align_directions(
    dir_a: np.ndarray,
    dir_b: np.ndarray,
    *,
    method: str = "procrustes",
    lm_head_a: np.ndarray | None = None,
    lm_head_b: np.ndarray | None = None,
) -> dict:
    """Align behavioral directions from two different-sized models.

    If models have the same hidden size, direct cosine comparison.
    If different sizes, project through lm_head matrices into vocabulary
    space and compare rank correlation of top-k affected tokens.

    method:
        "procrustes" — project into vocab space, compare via rank correlation
        "cosine"     — direct cosine (only works if same dimensionality)

    Returns {"alignment_score": float, "shared_dims": int}.
    """
    if method == "cosine":
        if dir_a.shape[0] != dir_b.shape[0]:
            return {
                "alignment_score": 0.0,
                "shared_dims": 0,
                "error": "dimension mismatch for cosine method",
            }
        score = cosine(dir_a, dir_b)
        return {
            "alignment_score": round(abs(float(score)), 4),
            "shared_dims": int(dir_a.shape[0]),
        }

    # Procrustes: project through lm_head into shared vocabulary space
    if lm_head_a is None or lm_head_b is None:
        return {
            "alignment_score": 0.0,
            "shared_dims": 0,
            "error": "lm_head matrices required for procrustes alignment",
        }

    # Project directions into vocabulary space
    # lm_head is [vocab_size, hidden_size], so projection is lm_head @ direction
    vocab_proj_a = lm_head_a @ dir_a  # [vocab_size]
    vocab_proj_b = lm_head_b @ dir_b  # [vocab_size]

    # Ensure both have the same vocab size (take the minimum)
    shared_vocab = min(len(vocab_proj_a), len(vocab_proj_b))
    vocab_proj_a = vocab_proj_a[:shared_vocab]
    vocab_proj_b = vocab_proj_b[:shared_vocab]

    # Rank correlation of top-k affected tokens
    k = min(1000, shared_vocab)
    top_a = set(np.argsort(np.abs(vocab_proj_a))[-k:])
    top_b = set(np.argsort(np.abs(vocab_proj_b))[-k:])
    overlap = len(top_a & top_b)
    # Normalize: perfect overlap = k, random overlap ~ k^2/vocab
    alignment_score = overlap / k

    return {
        "alignment_score": round(float(alignment_score), 4),
        "shared_dims": int(shared_vocab),
    }


def compare_sharts(
    backend_a: Any,
    backend_b: Any,
    tokens: list[str],
    layer_a: int,
    layer_b: int,
) -> list[dict]:
    """Compare how two models respond to the same shart tokens.

    Captures MLP activations from each model at the specified layers,
    computes z-scores against a benign baseline, and returns per-token
    comparison.
    """
    baseline_prompts = [
        "The weather today is",
        "Hello, how are you?",
        "Dogs are popular pets",
        "The sun is a star",
        "Books contain knowledge",
    ]

    # Compute baselines for both models
    acts_a = np.array([backend_a.capture_mlp_activations(p, layer_a)
                       for p in baseline_prompts])
    mean_a, std_a = acts_a.mean(axis=0), acts_a.std(axis=0) + 1e-6

    acts_b = np.array([backend_b.capture_mlp_activations(p, layer_b)
                       for p in baseline_prompts])
    mean_b, std_b = acts_b.mean(axis=0), acts_b.std(axis=0) + 1e-6

    results = []
    for token in tokens:
        act_a = backend_a.capture_mlp_activations(token, layer_a)
        act_b = backend_b.capture_mlp_activations(token, layer_b)

        z_a = np.abs((act_a - mean_a) / std_a)
        z_b = np.abs((act_b - mean_b) / std_b)

        results.append({
            "token": token,
            "max_z_a": round(float(np.max(z_a)), 2),
            "max_z_b": round(float(np.max(z_b)), 2),
            "n_anomalous_a": int(np.sum(z_a > 5)),
            "n_anomalous_b": int(np.sum(z_b > 5)),
            "mean_z_a": round(float(np.mean(z_a)), 2),
            "mean_z_b": round(float(np.mean(z_b)), 2),
            "z_ratio": round(float(np.max(z_a)) / max(float(np.max(z_b)), 1e-6), 2),
        })

    results.sort(key=lambda r: max(r["max_z_a"], r["max_z_b"]), reverse=True)
    return results
