"""Cluster knobs by behavioral effect to find behavioral dimensions."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .atlas import Atlas
from .perturb import PerturbResult


@dataclass
class BehaviorCluster:
    name: str
    knob_ids: list[str]
    mean_kl: float
    mean_entropy_delta: float
    token_change_rate: float
    layer_distribution: dict[int, int] = field(default_factory=dict)


def cluster_by_layer(atlas: Atlas) -> list[BehaviorCluster]:
    """Simple clustering: group knobs by layer, summarize each layer's behavioral profile."""
    by_layer: dict[int, list[PerturbResult]] = {}
    for r in atlas.results.values():
        by_layer.setdefault(r.knob.layer, []).append(r)

    clusters = []
    for layer in sorted(by_layer):
        results = by_layer[layer]
        clusters.append(BehaviorCluster(
            name=f"layer_{layer}",
            knob_ids=[r.knob.id for r in results],
            mean_kl=float(np.mean([r.kl_divergence for r in results])),
            mean_entropy_delta=float(np.mean([r.entropy_delta for r in results])),
            token_change_rate=sum(1 for r in results if r.top_token_changed) / len(results),
            layer_distribution={layer: len(results)},
        ))
    return clusters


def cluster_by_effect(atlas: Atlas, n_clusters: int = 4) -> list[BehaviorCluster]:
    """Cluster knobs by their effect profile (KL, entropy_delta, top_changed)."""
    results = list(atlas.results.values())
    if not results:
        return []

    # Feature matrix: [kl, entropy_delta, top_changed]
    features = np.array([[r.kl_divergence, r.entropy_delta, float(r.top_token_changed)] for r in results])

    # Normalize
    std = features.std(axis=0)
    std[std == 0] = 1
    normed = (features - features.mean(axis=0)) / std

    # Simple k-means (no sklearn dependency)
    n = min(n_clusters, len(results))
    # Init centroids from data
    rng = np.random.default_rng(42)
    indices = rng.choice(len(results), size=n, replace=False)
    centroids = normed[indices].copy()

    for _ in range(20):  # iterations
        # Assign
        dists = np.array([[np.linalg.norm(normed[i] - c) for c in centroids] for i in range(len(results))])
        labels = dists.argmin(axis=1)
        # Update
        for k in range(n):
            members = normed[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)

    clusters = []
    for k in range(n):
        member_idx = np.where(labels == k)[0]
        if len(member_idx) == 0:
            continue
        members = [results[i] for i in member_idx]
        layer_dist: dict[int, int] = {}
        for r in members:
            layer_dist[r.knob.layer] = layer_dist.get(r.knob.layer, 0) + 1
        clusters.append(BehaviorCluster(
            name=f"cluster_{k}",
            knob_ids=[r.knob.id for r in members],
            mean_kl=float(np.mean([r.kl_divergence for r in members])),
            mean_entropy_delta=float(np.mean([r.entropy_delta for r in members])),
            token_change_rate=sum(1 for r in members if r.top_token_changed) / len(members),
            layer_distribution=layer_dist,
        ))

    clusters.sort(key=lambda c: c.mean_kl, reverse=True)
    return clusters
