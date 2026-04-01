"""Behavioral space analysis — dimensionality, density, robustness, compilation.

The behavioral control surface is dense, not sparse. This module measures
the intrinsic dimensionality of the behavioral manifold, validates axis
robustness, analyzes direction density, and compiles behavioral specs
into steering vectors.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from ..signal import Signal, SignalStore


@dataclass
class DimensionalityResult:
    n_prompts: int
    hidden_size: int
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    dims_for_90: int
    dims_for_95: int
    dims_for_99: int
    intrinsic_dim_estimate: int  # elbow method


@dataclass
class AxisValidation:
    name: str
    train_accuracy: float
    test_accuracy: float
    direction_stability: float  # cosine between direction from train vs test
    n_train: int
    n_test: int
    generalized: bool


@dataclass
class DirectionDensity:
    name: str
    gini_coefficient: float      # 0=perfectly uniform, 1=perfectly sparse
    top_10_pct_weight: float     # fraction of L2 norm in top 10% of dims
    n_dims_for_90_pct: int       # how many dims needed for 90% of norm
    effective_dimensionality: int # participation ratio


@dataclass
class BehavioralCoordinate:
    text: str
    projections: dict[str, float]  # axis_name → projection value
    dominant_axis: str
    dominant_value: float


def estimate_dimensionality(
    model: Any, tokenizer: Any,
    prompts: list[str],
    *,
    layer: int = 15,
    store: SignalStore | None = None,
) -> DimensionalityResult:
    """Estimate the intrinsic dimensionality of the behavioral manifold.

    Captures residual states for many diverse prompts, then uses PCA
    to find how many dimensions carry meaningful behavioral variation.
    """
    from .directions import capture_residual_states

    states = capture_residual_states(model, tokenizer, prompts, layers=[layer])
    X = states[layer]  # [n_prompts, hidden_size]

    # Center
    X_centered = X - X.mean(axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Explained variance
    var = S ** 2
    total_var = var.sum()
    explained = var / total_var
    cumulative = np.cumsum(explained)

    dims_90 = int(np.searchsorted(cumulative, 0.90)) + 1
    dims_95 = int(np.searchsorted(cumulative, 0.95)) + 1
    dims_99 = int(np.searchsorted(cumulative, 0.99)) + 1

    # Elbow: where does the explained variance drop below 1/n_prompts?
    threshold = 1.0 / len(prompts)
    intrinsic = int(np.sum(explained > threshold))

    if store:
        store.add(Signal("dimensionality", "space", "model", f"L{layer}",
                         float(intrinsic), {"dims_90": dims_90, "dims_95": dims_95,
                                            "dims_99": dims_99, "n_prompts": len(prompts)}))

    return DimensionalityResult(
        n_prompts=len(prompts), hidden_size=X.shape[1],
        singular_values=S, explained_variance_ratio=explained,
        dims_for_90=dims_90, dims_for_95=dims_95, dims_for_99=dims_99,
        intrinsic_dim_estimate=intrinsic,
    )


def validate_axis(
    model: Any, tokenizer: Any,
    train_pos: list[str], train_neg: list[str],
    test_pos: list[str], test_neg: list[str],
    *,
    name: str, layer: int = 15,
) -> AxisValidation:
    """Validate an axis direction on held-out data."""
    from .directions import capture_residual_states, find_direction

    # Train
    train_states = capture_residual_states(model, tokenizer,
                                           train_pos + train_neg, layers=[layer])
    n_train_pos = len(train_pos)
    train_dir = find_direction(train_states[layer][:n_train_pos],
                                train_states[layer][n_train_pos:],
                                name=name, layer=layer)

    # Test with train direction
    test_states = capture_residual_states(model, tokenizer,
                                          test_pos + test_neg, layers=[layer])
    n_test_pos = len(test_pos)
    test_pos_projs = test_states[layer][:n_test_pos] @ train_dir.direction
    test_neg_projs = test_states[layer][n_test_pos:] @ train_dir.direction
    threshold = (test_pos_projs.mean() + test_neg_projs.mean()) / 2
    test_correct = np.sum(test_pos_projs > threshold) + np.sum(test_neg_projs <= threshold)
    test_acc = float(test_correct) / (len(test_pos) + len(test_neg))

    # Direction from test data
    test_dir = find_direction(test_states[layer][:n_test_pos],
                               test_states[layer][n_test_pos:],
                               name=name, layer=layer)
    stability = float(np.dot(train_dir.direction, test_dir.direction))

    return AxisValidation(
        name=name, train_accuracy=train_dir.separation_accuracy,
        test_accuracy=test_acc, direction_stability=stability,
        n_train=len(train_pos) + len(train_neg),
        n_test=len(test_pos) + len(test_neg),
        generalized=test_acc >= 0.8 and stability >= 0.5,
    )


def measure_density(direction: np.ndarray, name: str = "") -> DirectionDensity:
    """Measure how dense or sparse a direction vector is."""
    d = np.abs(direction)
    d_sorted = np.sort(d)[::-1]
    total = d.sum()

    # Gini coefficient
    n = len(d)
    index = np.arange(1, n + 1)
    gini = float((np.sum((2 * index - n - 1) * d_sorted)) / (n * total + 1e-12))

    # Top 10% weight
    top_10_pct = int(n * 0.1)
    top_weight = float(np.sum(d_sorted[:top_10_pct]) / (total + 1e-12))

    # Dims for 90% of L2 norm
    d_sq_sorted = np.sort(d ** 2)[::-1]
    cum_sq = np.cumsum(d_sq_sorted)
    total_sq = cum_sq[-1]
    dims_90 = int(np.searchsorted(cum_sq, 0.9 * total_sq)) + 1

    # Participation ratio (effective dimensionality)
    p = (d ** 2) / (total_sq + 1e-12)
    participation = float(1.0 / (np.sum(p ** 2) + 1e-12))

    return DirectionDensity(
        name=name, gini_coefficient=gini, top_10_pct_weight=top_weight,
        n_dims_for_90_pct=dims_90, effective_dimensionality=int(participation),
    )


def project_text(
    model: Any, tokenizer: Any,
    text: str,
    axes: list,  # list of BehavioralAxis
    *,
    layer: int = 15,
) -> BehavioralCoordinate:
    """Project a piece of text onto the behavioral axis space."""
    from .directions import capture_residual_states

    states = capture_residual_states(model, tokenizer, [text], layers=[layer])
    state = states[layer][0]

    projections = {}
    for axis in axes:
        if axis.layer == layer:
            proj = float(np.dot(state, axis.direction))
            projections[axis.name] = proj

    dominant = max(projections.items(), key=lambda x: abs(x[1])) if projections else ("none", 0.0)

    return BehavioralCoordinate(
        text=text, projections=projections,
        dominant_axis=dominant[0], dominant_value=dominant[1],
    )


def compile_behavior(
    axes: list,  # list of BehavioralAxis
    spec: dict[str, float],  # {axis_name: target_alpha}
) -> list[tuple[int, np.ndarray, float]]:
    """Compile a behavioral specification into steering vectors.

    spec: {"truth": 1.0, "depth": 0.5, "safety": -0.1, ...}
    Returns: [(layer, direction * scale, alpha), ...] ready for manipulate.combined_manipulation
    """
    steers = []
    axis_lookup = {a.name: a for a in axes}

    for axis_name, alpha in spec.items():
        if axis_name in axis_lookup:
            a = axis_lookup[axis_name]
            steers.append((a.layer, a.direction * a.scale, alpha))

    return steers
