"""Text regime clustering — similarity-based clustering of model output texts."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from math import log2
from statistics import mean
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class TextRegimeCluster:
    cluster_index: int
    size: int
    probability: float
    representative_text: str
    medoid_text: str
    medoid_index: int
    member_indices: tuple[int, ...]
    member_texts: tuple[str, ...]
    mean_similarity_to_medoid: float
    mean_within_similarity: float
    exact_consensus: bool


def text_similarity(lhs: str, rhs: str) -> float:
    return float(SequenceMatcher(None, lhs, rhs).ratio())


def regime_entropy(weights: Iterable[float]) -> float:
    values = [float(weight) for weight in weights if float(weight) > 0.0]
    if not values:
        return 0.0
    total = sum(values)
    if total <= 0.0:
        return 0.0
    probabilities = [value / total for value in values]
    return float(-sum(probability * log2(probability) for probability in probabilities))


def cluster_text_regimes(
    texts: Sequence[str],
    *,
    similarity_threshold: float = 0.82,
) -> dict[str, Any]:
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    normalized = [str(text) for text in texts]
    clusters: list[dict[str, Any]] = []
    assignments: list[int] = []

    for index, text in enumerate(normalized):
        cluster_index = _assign_cluster_index(clusters, text, similarity_threshold)
        if cluster_index is None:
            cluster_index = len(clusters)
            clusters.append(
                {
                    "cluster_index": cluster_index,
                    "member_indices": [index],
                    "member_texts": [text],
                    "representative_text": text,
                }
            )
        else:
            clusters[cluster_index]["member_indices"].append(index)
            clusters[cluster_index]["member_texts"].append(text)
        assignments.append(int(cluster_index))

    cluster_summaries = [_summarize_cluster(cluster, total=len(normalized)) for cluster in clusters]
    cluster_summaries.sort(key=lambda row: (row["cluster_index"], -row["size"]))

    return {
        "sample_count": len(normalized),
        "similarity_threshold": float(similarity_threshold),
        "regime_count": len(cluster_summaries),
        "assignments": assignments,
        "clusters": cluster_summaries,
    }


def summarize_text_regimes(
    texts: Sequence[str],
    *,
    similarity_threshold: float = 0.82,
) -> dict[str, Any]:
    clustering = cluster_text_regimes(texts, similarity_threshold=similarity_threshold)
    clusters = clustering["clusters"]
    dominant = max(clusters, key=lambda row: row["size"], default=None)
    probabilities = [row["probability"] for row in clusters]
    sample_count = int(clustering["sample_count"])
    unique_text_count = len({str(text) for text in texts})

    return {
        "sample_count": sample_count,
        "unique_text_count": unique_text_count,
        "similarity_threshold": clustering["similarity_threshold"],
        "regime_count": clustering["regime_count"],
        "entropy_bits": regime_entropy(probabilities),
        "effective_regime_count": _effective_regime_count(probabilities),
        "dominant_regime_mass": 0.0 if dominant is None else float(dominant["probability"]),
        "dominant_regime_index": None if dominant is None else dominant["cluster_index"],
        "dominant_regime_text": None if dominant is None else dominant["medoid_text"],
        "exact_consensus": unique_text_count <= 1,
        "consensus": {
            "exact": unique_text_count <= 1,
            "dominant_regime_mass": 0.0 if dominant is None else float(dominant["probability"]),
            "dominant_regime_size": 0 if dominant is None else int(dominant["size"]),
            "dominant_regime_index": None if dominant is None else dominant["cluster_index"],
            "dominant_regime_text": None if dominant is None else dominant["medoid_text"],
        },
        "clusters": clusters,
        "assignments": clustering["assignments"],
    }


def _assign_cluster_index(
    clusters: list[dict[str, Any]],
    text: str,
    similarity_threshold: float,
) -> int | None:
    if not clusters:
        return None
    for cluster in clusters:
        score = text_similarity(text, str(cluster["representative_text"]))
        if score >= similarity_threshold:
            return int(cluster["cluster_index"])
    return None


def _summarize_cluster(cluster: dict[str, Any], *, total: int) -> dict[str, Any]:
    member_texts = [str(text) for text in cluster["member_texts"]]
    member_indices = [int(index) for index in cluster["member_indices"]]
    medoid_index, medoid_text = _find_medoid(member_texts, member_indices)
    medoid_similarities = [text_similarity(text, medoid_text) for text in member_texts]
    pairwise_similarities = _pairwise_similarities(member_texts)
    size = len(member_texts)
    return asdict(
        TextRegimeCluster(
            cluster_index=int(cluster["cluster_index"]),
            size=size,
            probability=(size / total) if total else 0.0,
            representative_text=str(cluster["representative_text"]),
            medoid_text=medoid_text,
            medoid_index=medoid_index,
            member_indices=tuple(member_indices),
            member_texts=tuple(member_texts),
            mean_similarity_to_medoid=mean(medoid_similarities) if medoid_similarities else 1.0,
            mean_within_similarity=mean(pairwise_similarities) if pairwise_similarities else 1.0,
            exact_consensus=len(set(member_texts)) <= 1,
        )
    )


def _find_medoid(member_texts: list[str], member_indices: list[int]) -> tuple[int, str]:
    if not member_texts:
        return -1, ""
    if len(member_texts) == 1:
        return member_indices[0], member_texts[0]
    scores: list[tuple[float, int, str, int]] = []
    for local_index, text in enumerate(member_texts):
        total_similarity = 0.0
        for other_index, other_text in enumerate(member_texts):
            if local_index == other_index:
                continue
            total_similarity += text_similarity(text, other_text)
        scores.append((total_similarity, member_indices[local_index], text, local_index))
    scores.sort(key=lambda row: (-row[0], row[1], row[2]))
    _, member_index, text, _ = scores[0]
    return member_index, text


def _pairwise_similarities(texts: list[str]) -> list[float]:
    values: list[float] = []
    for left_index in range(len(texts)):
        for right_index in range(left_index + 1, len(texts)):
            values.append(text_similarity(texts[left_index], texts[right_index]))
    return values


def _effective_regime_count(probabilities: Sequence[float]) -> float:
    entropy = regime_entropy(probabilities)
    return float(2.0**entropy) if entropy > 0.0 else 1.0
