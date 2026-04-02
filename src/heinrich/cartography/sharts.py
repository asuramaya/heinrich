"""Shart detector — find anomalous tokens that trigger disproportionate neural responses.

A "shart" is any token or short phrase that causes the model's MLP activations
to deviate massively from the normal distribution. These are the model's
pressure points — topics it was specifically trained to recognize and process
differently.

Strategy: compute a baseline activation profile from benign prompts, then
scan a large vocabulary of candidate triggers and rank by deviation from
baseline. The result is a ranked list of the model's most anomalous triggers.
"""
from __future__ import annotations
import sys
import time
from dataclasses import dataclass
from typing import Any
import numpy as np
from ..signal import Signal, SignalStore


@dataclass
class Shart:
    text: str
    category: str
    layer: int
    max_z: float
    n_anomalous: int      # neurons with z > threshold
    mean_z: float
    top_neuron: int
    top_neuron_z: float


@dataclass
class ShartScanResult:
    n_candidates: int
    n_sharts: int
    baseline_prompts: list[str]
    layer: int
    z_threshold: float
    sharts: list[Shart]

    def top(self, k: int = 20) -> list[Shart]:
        return sorted(self.sharts, key=lambda s: s.max_z, reverse=True)[:k]

    def by_category(self) -> dict[str, list[Shart]]:
        cats: dict[str, list[Shart]] = {}
        for s in self.sharts:
            cats.setdefault(s.category, []).append(s)
        return cats


def _get_activation(model, tokenizer, prompt, layer, *, backend=None):
    """Get MLP activation vector at a layer (last token position)."""
    if backend is not None:
        return backend.capture_mlp_activations(prompt, layer)
    from .neurons import capture_mlp_activations
    return capture_mlp_activations(model, tokenizer, prompt, layer)


def compute_baseline(
    model: Any, tokenizer: Any,
    prompts: list[str], layer: int,
    *, backend: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of MLP activations across benign prompts."""
    acts = np.array([_get_activation(model, tokenizer, p, layer, backend=backend) for p in prompts])
    return acts.mean(axis=0), acts.std(axis=0) + 1e-6


def scan_sharts(
    model: Any, tokenizer: Any,
    candidates: dict[str, list[str]],
    *,
    layer: int | None = None,
    baseline_prompts: list[str] | None = None,
    z_threshold: float = 5.0,
    min_anomalous: int = 50,
    store: SignalStore | None = None,
    model_config: Any = None,
    progress: bool = True,
) -> ShartScanResult:
    """Scan candidate tokens/phrases for anomalous MLP activation patterns.

    candidates: {category: [prompt1, prompt2, ...]}
    If layer is None, uses model's last layer via model_config.
    Returns ranked list of "sharts" — tokens that trigger disproportionate response.
    """
    if layer is None:
        from .model_config import detect_config
        cfg = model_config or detect_config(model)
        layer = cfg.last_layer
    if baseline_prompts is None:
        baseline_prompts = [
            "The weather today is", "Hello, how are you?",
            "Dogs are popular pets", "The capital of France is Paris",
            "Mathematics is the study of", "Water boils at 100 degrees",
            "The sun is a star", "Trees produce oxygen",
            "Music is a form of art", "Books contain knowledge",
        ]

    if progress:
        print(f"  Computing baseline from {len(baseline_prompts)} prompts at L{layer}...", file=sys.stderr)

    mean, std = compute_baseline(model, tokenizer, baseline_prompts, layer, backend=backend)

    total = sum(len(v) for v in candidates.values())
    sharts = []
    done = 0

    for category, prompts in candidates.items():
        for prompt in prompts:
            act = _get_activation(model, tokenizer, prompt, layer, backend=backend)
            z = np.abs((act - mean) / std)
            max_z = float(np.max(z))
            n_anom = int(np.sum(z > z_threshold))
            mean_z = float(np.mean(z))
            top_n = int(np.argmax(z))
            top_z = float(z[top_n])

            if n_anom >= min_anomalous:
                s = Shart(text=prompt, category=category, layer=layer,
                          max_z=max_z, n_anomalous=n_anom, mean_z=mean_z,
                          top_neuron=top_n, top_neuron_z=top_z)
                sharts.append(s)

                if store:
                    store.add(Signal("shart", "cartography", "model", prompt,
                                     max_z, {"category": category, "n_anomalous": n_anom,
                                             "top_neuron": top_n}))

            done += 1
            if progress and done % 50 == 0:
                print(f"  [{done}/{total}] scanned, {len(sharts)} sharts found", file=sys.stderr)

    sharts.sort(key=lambda s: s.max_z, reverse=True)

    return ShartScanResult(
        n_candidates=total, n_sharts=len(sharts),
        baseline_prompts=baseline_prompts, layer=layer,
        z_threshold=z_threshold, sharts=sharts,
    )


def bilingual_shart_scan(
    model: Any, tokenizer: Any,
    topics_en: dict[str, str],
    topics_zh: dict[str, str],
    *,
    layer: int | None = None,
    store: SignalStore | None = None,
    model_config: Any = None,
    backend: Any = None,
) -> list[dict]:
    """Compare activation patterns for the same topic in English vs Chinese.
    Returns list of topics with their bilingual divergence score.
    """
    if layer is None:
        from .model_config import detect_config
        cfg = model_config or detect_config(model)
        layer = cfg.last_layer
    baseline = [
        "The weather today is", "Hello, how are you?",
        "Dogs are popular pets", "The sun is a star",
        "今天天气很好", "你好吗", "狗是宠物", "太阳是恒星",
    ]
    mean, std = compute_baseline(model, tokenizer, baseline, layer, backend=backend)

    results = []
    for topic in topics_en:
        if topic not in topics_zh:
            continue
        en_act = _get_activation(model, tokenizer, topics_en[topic], layer, backend=backend)
        zh_act = _get_activation(model, tokenizer, topics_zh[topic], layer, backend=backend)

        en_z = np.abs((en_act - mean) / std)
        zh_z = np.abs((zh_act - mean) / std)

        # Divergence: how different are the EN and ZH activation patterns?
        act_diff = np.abs(en_act - zh_act)
        divergence = float(np.linalg.norm(act_diff))
        max_diff_neuron = int(np.argmax(act_diff))
        max_diff = float(act_diff[max_diff_neuron])

        results.append({
            "topic": topic,
            "en_prompt": topics_en[topic],
            "zh_prompt": topics_zh[topic],
            "en_max_z": float(np.max(en_z)),
            "zh_max_z": float(np.max(zh_z)),
            "en_n_anomalous": int(np.sum(en_z > 5)),
            "zh_n_anomalous": int(np.sum(zh_z > 5)),
            "divergence": divergence,
            "max_diff_neuron": max_diff_neuron,
            "max_diff": max_diff,
        })

        if store:
            store.add(Signal("bilingual_shart", "cartography", "model", topic,
                             divergence, {"en_z": float(np.max(en_z)), "zh_z": float(np.max(zh_z))}))

    results.sort(key=lambda r: r["divergence"], reverse=True)
    return results


# === Fast weight-space shart crawl (from shart_crawl.py) ===

@dataclass
class ShartScore:
    token_id: int
    token: str
    neuron_scores: dict[str, float]   # {neuron_name: score}
    max_score: float
    top_neuron: str


def fast_shart_crawl(
    model: Any, tokenizer: Any,
    target_neurons: dict[int, str],
    *,
    layer: int | None = None,
    batch_size: int = 128,
    vocab_size: int | None = None,
    progress: bool = True,
    model_config: Any = None,
) -> dict[int, np.ndarray]:
    """Score ALL tokens against target neurons via weight-space projection.

    No forward passes. Pure matrix multiplication:
    embedding × gate_proj → score for each neuron.

    target_neurons: {neuron_index: name}
    Returns {neuron_index: scores_array[vocab_size]}
    """
    import mlx.core as mx
    import time

    if layer is None:
        from .model_config import detect_config
        cfg = model_config or detect_config(model)
        layer = cfg.last_layer

    inner = getattr(model, "model", model)
    if vocab_size is None:
        vocab_size = inner.embed_tokens.weight.shape[0]
    hidden_size = inner.norm.weight.shape[0]

    t0 = time.time()
    layer_obj = inner.layers[layer]
    mlp = layer_obj.mlp

    # Extract gate weights for each target neuron by probing
    neuron_gate_weights = {}
    for n_idx, n_name in target_neurons.items():
        weights = np.zeros(hidden_size)
        for start in range(0, hidden_size, batch_size):
            end = min(start + batch_size, hidden_size)
            inp = np.zeros((1, end - start, hidden_size), dtype=np.float16)
            for j in range(end - start):
                inp[0, j, start + j] = 1.0
            gate_out = mlp.gate_proj(mx.array(inp))
            gate_np = np.array(gate_out.astype(mx.float32)[0, :, n_idx])
            weights[start:end] = gate_np
        neuron_gate_weights[n_idx] = weights

    if progress:
        print(f"  Gate weights extracted in {time.time()-t0:.1f}s", file=sys.stderr)

    # Score all tokens by embedding projection
    t1 = time.time()
    token_scores = {n: np.zeros(vocab_size) for n in target_neurons}
    token_batch = 1000

    for start in range(0, vocab_size, token_batch):
        end = min(start + token_batch, vocab_size)
        token_ids = mx.array([list(range(start, end))])
        embeddings = inner.embed_tokens(token_ids)
        emb_np = np.array(embeddings.astype(mx.float32)[0])

        for n_idx, gate_w in neuron_gate_weights.items():
            scores = emb_np @ gate_w
            token_scores[n_idx][start:end] = scores

    if progress:
        print(f"  {vocab_size} tokens scored in {time.time()-t1:.1f}s", file=sys.stderr)

    return token_scores


def shart_taxonomy(
    token_scores: dict[int, np.ndarray],
    target_neurons: dict[int, str],
    tokenizer: Any,
    *,
    k_clusters: int = 6,
    top_n: int = 500,
) -> list[dict]:
    """Cluster top sharts by their neuron activation profile.

    Returns list of cluster dicts with members and dominant neuron.
    """
    neuron_list = list(target_neurons.keys())
    n_neurons = len(neuron_list)

    # Get top N sharts by max absolute score across any neuron
    vocab_size = len(next(iter(token_scores.values())))
    max_scores = np.zeros(vocab_size)
    for n_idx in target_neurons:
        max_scores = np.maximum(max_scores, np.abs(token_scores[n_idx]))

    top_ids = np.argsort(max_scores)[::-1][:top_n]
    features = np.zeros((top_n, n_neurons))
    for i, tid in enumerate(top_ids):
        for j, n_idx in enumerate(neuron_list):
            features[i, j] = token_scores[n_idx][tid]

    # Normalize and k-means
    std = features.std(axis=0)
    std[std == 0] = 1
    normed = features / std

    rng = np.random.default_rng(42)
    centroids = normed[rng.choice(top_n, k_clusters, replace=False)].copy()
    for _ in range(20):
        dists = np.array([[np.linalg.norm(normed[i] - c) for c in centroids]
                          for i in range(top_n)])
        labels = dists.argmin(axis=1)
        for c in range(k_clusters):
            members = normed[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)

    clusters = []
    for c in range(k_clusters):
        member_idx = np.where(labels == c)[0]
        if len(member_idx) == 0:
            continue
        member_tids = [int(top_ids[i]) for i in member_idx]
        member_tokens = [tokenizer.decode([tid]) for tid in member_tids[:8]]

        cluster_mean = features[member_idx].mean(axis=0)
        top_neuron_j = int(np.argmax(np.abs(cluster_mean)))
        top_neuron_idx = neuron_list[top_neuron_j]

        clusters.append({
            "cluster_id": c,
            "n_members": len(member_idx),
            "dominant_neuron": top_neuron_idx,
            "dominant_neuron_name": target_neurons[top_neuron_idx],
            "example_tokens": member_tokens,
            "example_ids": member_tids[:8],
        })

    return clusters


def multi_neuron_analysis(
    token_scores: dict[int, np.ndarray],
    target_neurons: dict[int, str],
    tokenizer: Any,
    *,
    percentile: float = 99,
    min_neurons: int = 3,
) -> list[dict]:
    """Find tokens that activate multiple target neurons above threshold.

    Returns list of {token_id, token, activated_neurons, n_activated}.
    """
    vocab_size = len(next(iter(token_scores.values())))
    thresholds = {n: np.percentile(np.abs(token_scores[n]), percentile)
                  for n in target_neurons}

    results = []
    for tid in range(vocab_size):
        activated = []
        for n_idx, n_name in target_neurons.items():
            if abs(token_scores[n_idx][tid]) > thresholds[n_idx]:
                activated.append(n_name)
        if len(activated) >= min_neurons:
            results.append({
                "token_id": tid,
                "token": tokenizer.decode([tid]),
                "activated_neurons": activated,
                "n_activated": len(activated),
            })

    results.sort(key=lambda x: -x["n_activated"])
    return results
