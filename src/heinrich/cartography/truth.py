"""Truth extraction — find and apply the objectivity direction.

Given a model and a politically sensitive topic, find the direction in
residual stream space that produces the most neutral, factual output
regardless of language. Combines embedding analysis, cross-lingual
direction finding, and manipulation primitives.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import Any
import numpy as np
from ..signal import Signal, SignalStore


@dataclass
class TruthVector:
    name: str
    layer: int
    direction: np.ndarray     # unit vector
    scale: float              # recommended magnitude
    en_zh_cosine: float       # how aligned is this with the language axis
    separation_accuracy: float


@dataclass
class EmbeddingProfile:
    token_text: str
    token_id: int
    embedding: np.ndarray
    norm: float
    political_projection: float
    language_projection: float


def analyze_embedding_space(
    model: Any, tokenizer: Any,
    political_tokens: list[str],
    benign_tokens: list[str],
    *,
    store: SignalStore | None = None,
    backend: Any = None,
) -> dict[str, Any]:
    """Analyze whether political bias exists at the embedding level."""
    if backend is not None:
        _tokenize = backend.tokenize
        def get_embedding(text):
            ids = _tokenize(text)
            if not ids:
                return None
            # Use forward with a single token to get embedding-level state
            result = backend.forward(text, return_residual=True, residual_layer=0)
            return result.residual, ids[-1]
    else:
        import mlx.core as mx
        inner = getattr(model, "model", model)
        def get_embedding(text):
            ids = tokenizer.encode(text)
            if not ids:
                return None
            emb_table = inner.embed_tokens
            tid = ids[-1]
            inp = mx.array([[tid]])
            vec = np.array(emb_table(inp).astype(mx.float32)[0, 0, :])
            return vec, tid

    pol_embeddings = []
    ben_embeddings = []
    profiles = []

    for text in political_tokens:
        result = get_embedding(text)
        if result:
            vec, tid = result
            pol_embeddings.append(vec)
            profiles.append(EmbeddingProfile(text, tid, vec, float(np.linalg.norm(vec)), 0, 0))

    for text in benign_tokens:
        result = get_embedding(text)
        if result:
            vec, tid = result
            ben_embeddings.append(vec)
            profiles.append(EmbeddingProfile(text, tid, vec, float(np.linalg.norm(vec)), 0, 0))

    if not pol_embeddings or not ben_embeddings:
        return {"error": "insufficient embeddings"}

    pol_arr = np.array(pol_embeddings)
    ben_arr = np.array(ben_embeddings)

    # Political direction in embedding space
    pol_mean = pol_arr.mean(axis=0)
    ben_mean = ben_arr.mean(axis=0)
    emb_direction = pol_mean - ben_mean
    emb_norm = np.linalg.norm(emb_direction)
    if emb_norm > 0:
        emb_direction = emb_direction / emb_norm

    # Project all embeddings onto this direction
    pol_projs = [float(np.dot(v, emb_direction)) for v in pol_embeddings]
    ben_projs = [float(np.dot(v, emb_direction)) for v in ben_embeddings]

    # Separation accuracy
    threshold = (np.mean(pol_projs) + np.mean(ben_projs)) / 2
    correct = sum(1 for p in pol_projs if p > threshold) + sum(1 for p in ben_projs if p <= threshold)
    accuracy = correct / (len(pol_projs) + len(ben_projs))

    # Update profiles with projections
    for i, p in enumerate(profiles):
        p.political_projection = float(np.dot(p.embedding, emb_direction))

    gap = float(np.mean(pol_projs) - np.mean(ben_projs))

    if store:
        store.add(Signal("embedding_political_direction", "truth", "model", "embedding_space",
                         accuracy, {"gap": gap, "norm": emb_norm}))

    return {
        "direction": emb_direction,
        "accuracy": accuracy,
        "gap": gap,
        "pol_mean_proj": float(np.mean(pol_projs)),
        "ben_mean_proj": float(np.mean(ben_projs)),
        "profiles": profiles,
    }


def find_truth_vector(
    model: Any, tokenizer: Any,
    topic_en_prompts: list[str],
    topic_zh_prompts: list[str],
    *,
    layer: int = 15,
    store: SignalStore | None = None,
) -> TruthVector:
    """Find the direction that separates EN and ZH responses on a given topic.

    This vector, when added to a ZH query, shifts the response toward
    EN-style (typically more factual/Western-framed) output.
    """
    from .directions import capture_residual_states, find_direction

    states = capture_residual_states(model, tokenizer,
                                     topic_en_prompts + topic_zh_prompts,
                                     layers=[layer])
    n_en = len(topic_en_prompts)
    d = find_direction(states[layer][:n_en], states[layer][n_en:],
                       name="truth", layer=layer)

    return TruthVector(
        name="truth", layer=layer,
        direction=d.direction, scale=d.mean_gap,
        en_zh_cosine=1.0,  # by construction
        separation_accuracy=d.separation_accuracy,
    )


def find_objectivity_vector(
    model: Any, tokenizer: Any,
    *,
    layer: int = 15,
    store: SignalStore | None = None,
) -> TruthVector:
    """Find a GENERAL objectivity direction that works across topics.

    Uses multiple political topics to find the common direction that
    separates propagandistic from factual responses.
    """
    from .directions import capture_residual_states, find_direction

    # Multiple topics — the objectivity direction should be common across all
    factual_prompts = [
        "The Tiananmen Square protests of 1989 were",
        "Taiwan has its own government, military, and",
        "The Dalai Lama is the spiritual leader of",
        "Falun Gong practitioners report experiencing",
        "The Great Firewall of China blocks access to",
        "Human rights organizations have documented",
    ]
    propaganda_prompts = [
        "天安门事件是依法平息的政治风波",
        "台湾从来就不是一个国家",
        "达赖集团是分裂主义势力",
        "法轮功是邪教组织",
        "防火长城保护了网络安全",
        "中国人权事业取得了巨大进步",
    ]

    states = capture_residual_states(model, tokenizer,
                                     factual_prompts + propaganda_prompts,
                                     layers=[layer])
    n = len(factual_prompts)
    d = find_direction(states[layer][:n], states[layer][n:],
                       name="objectivity", layer=layer)

    if store:
        store.add(Signal("objectivity_direction", "truth", "model",
                         f"L{layer}", d.separation_accuracy,
                         {"effect_size": d.effect_size, "gap": d.mean_gap}))

    return TruthVector(
        name="objectivity", layer=layer,
        direction=d.direction, scale=d.mean_gap,
        en_zh_cosine=0.0,  # will be computed separately if needed
        separation_accuracy=d.separation_accuracy,
    )


def apply_truth(
    model: Any, tokenizer: Any, prompt: str,
    truth_vec: TruthVector,
    alpha: float = 1.0,
    max_tokens: int = 30,
) -> str:
    """Generate with the truth vector applied."""
    from .manipulate import _generate_manipulated
    return _generate_manipulated(
        model, tokenizer, prompt,
        direction_steers=[(truth_vec.layer, truth_vec.direction * truth_vec.scale, alpha)],
        max_tokens=max_tokens,
    )


def trace_neuron_circuit(
    model: Any, tokenizer: Any,
    trigger_prompt: str,
    baseline_prompt: str,
    target_neuron: int,
    *,
    layers: list[int] | None = None,
) -> dict[str, Any]:
    """Trace a specific neuron's activation across layers to understand its circuit.

    For each layer, captures the neuron's activation on trigger vs baseline,
    and measures how the neuron's activation correlates with the output.
    """
    from .neurons import capture_mlp_activations

    inner = getattr(model, "model", model)
    if layers is None:
        layers = list(range(len(inner.layers)))

    trigger_acts = {}
    baseline_acts = {}
    for l in layers:
        t_act = capture_mlp_activations(model, tokenizer, trigger_prompt, l)
        b_act = capture_mlp_activations(model, tokenizer, baseline_prompt, l)
        trigger_acts[l] = float(t_act[target_neuron])
        baseline_acts[l] = float(b_act[target_neuron])

    return {
        "neuron": target_neuron,
        "trigger_prompt": trigger_prompt,
        "baseline_prompt": baseline_prompt,
        "layers": {l: {"trigger": trigger_acts[l], "baseline": baseline_acts[l],
                        "diff": trigger_acts[l] - baseline_acts[l]}
                   for l in layers},
    }
