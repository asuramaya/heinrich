"""Sampling null hypothesis testing — bootstrap CI comparison for chat sample sets."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

_WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def summarize_sampling_null(
    lhs_texts: list[str] | tuple[str, ...],
    rhs_texts: list[str] | tuple[str, ...],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    lhs_rows = [str(text) for text in lhs_texts]
    rhs_rows = [str(text) for text in rhs_texts]
    if not lhs_rows or not rhs_rows:
        raise ValueError("summarize_sampling_null requires non-empty lhs_texts and rhs_texts")

    within_lhs = _pairwise_word_jaccards(lhs_rows, lhs_rows, within=True)
    within_rhs = _pairwise_word_jaccards(rhs_rows, rhs_rows, within=True)
    cross = _pairwise_word_jaccards(lhs_rows, rhs_rows, within=False)

    lhs_summary = _bootstrap_mean_ci(within_lhs, n_boot=n_boot, seed=seed + 1, default=1.0)
    rhs_summary = _bootstrap_mean_ci(within_rhs, n_boot=n_boot, seed=seed + 2, default=1.0)
    cross_summary = _bootstrap_mean_ci(cross, n_boot=n_boot, seed=seed + 3, default=0.0)

    delta_lhs = float(cross_summary["mean"] - lhs_summary["mean"])
    delta_rhs = float(cross_summary["mean"] - rhs_summary["mean"])
    overlap_lhs = _intervals_overlap(lhs_summary, cross_summary)
    overlap_rhs = _intervals_overlap(rhs_summary, cross_summary)
    signal_margin = max(
        float(np.mean([lhs_summary["mean"], rhs_summary["mean"]])) - float(cross_summary["mean"]),
        0.0,
    )
    if cross_summary["mean"] < min(lhs_summary["mean"], rhs_summary["mean"]):
        direction = "cross_lower"
    elif cross_summary["mean"] > max(lhs_summary["mean"], rhs_summary["mean"]):
        direction = "cross_higher"
    else:
        direction = "mixed"
    if int(lhs_summary["n"]) == 0 or int(rhs_summary["n"]) == 0:
        verdict = "INSUFFICIENT"
    else:
        verdict = "NOISE" if (overlap_lhs and overlap_rhs) else "SIGNAL"
    return {
        "mode": "samplingnull",
        "metric": "word_jaccard",
        "lhs_sample_count": len(lhs_rows),
        "rhs_sample_count": len(rhs_rows),
        "within_lhs": lhs_summary,
        "within_rhs": rhs_summary,
        "cross": cross_summary,
        "delta_lhs": delta_lhs,
        "delta_rhs": delta_rhs,
        "overlap_lhs": overlap_lhs,
        "overlap_rhs": overlap_rhs,
        "direction": direction,
        "signal_margin": signal_margin,
        "verdict": verdict,
    }


def scan_sampling_null_source(
    source: str | Path | dict[str, Any],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    report = _load_jsonish(source)
    lhs_texts, rhs_texts, source_mode = _extract_text_sets(report)
    result = summarize_sampling_null(lhs_texts, rhs_texts, n_boot=n_boot, seed=seed)
    result["source_mode"] = source_mode
    return result


def _extract_text_sets(report: dict[str, Any]) -> tuple[list[str], list[str], str]:
    if isinstance(report.get("lhs_texts"), list) and isinstance(report.get("rhs_texts"), list):
        return (
            [str(text) for text in report["lhs_texts"]],
            [str(text) for text in report["rhs_texts"]],
            str(report.get("mode", "text_pairs")),
        )
    if str(report.get("mode")) == "chat":
        lhs = report.get("lhs", {})
        rhs = report.get("rhs", {})
        return (_sample_texts(lhs), _sample_texts(rhs), "chat")
    if str(report.get("mode")) == "stategate":
        baseline = report.get("baseline", {})
        followup = report.get("followup", {})
        return (_sample_texts(baseline), _sample_texts(followup), "stategate")
    raise ValueError("samplingnull source must be a chat/stategate report or an object with lhs_texts/rhs_texts")


def _sample_texts(section: Any) -> list[str]:
    if not isinstance(section, dict):
        return []
    samples = section.get("samples")
    if isinstance(samples, list) and samples:
        rows = [str(row.get("text", "")) for row in samples if isinstance(row, dict)]
        if rows:
            return rows
    text = section.get("text")
    return [str(text)] if isinstance(text, str) else []


def _pairwise_word_jaccards(lhs_texts: list[str], rhs_texts: list[str], *, within: bool) -> list[float]:
    rows: list[float] = []
    if within:
        for lhs_index in range(len(lhs_texts)):
            for rhs_index in range(lhs_index + 1, len(rhs_texts)):
                rows.append(_word_jaccard(lhs_texts[lhs_index], rhs_texts[rhs_index]))
        return rows
    for lhs_text in lhs_texts:
        for rhs_text in rhs_texts:
            rows.append(_word_jaccard(lhs_text, rhs_text))
    return rows


def _word_jaccard(lhs: str, rhs: str) -> float:
    lhs_tokens = set(_WORD_RE.findall(lhs.lower()))
    rhs_tokens = set(_WORD_RE.findall(rhs.lower()))
    union = lhs_tokens | rhs_tokens
    if not union:
        return 1.0
    return float(len(lhs_tokens & rhs_tokens) / len(union))


def _bootstrap_mean_ci(values: list[float], *, n_boot: int, seed: int, default: float) -> dict[str, Any]:
    if not values:
        return {"mean": float(default), "lo": float(default), "hi": float(default), "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        value = float(arr[0])
        return {"mean": value, "lo": value, "hi": value, "n": 1}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(max(int(n_boot), 1)):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot.append(float(np.mean(sample)))
    return {
        "mean": float(np.mean(arr)),
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
        "n": int(len(arr)),
    }


def _intervals_overlap(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
    return float(lhs["lo"]) <= float(rhs["hi"]) and float(rhs["lo"]) <= float(lhs["hi"])


def _load_jsonish(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        result = source
    else:
        path = Path(source)
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
        else:
            result = json.loads(str(source))
    if not isinstance(result, dict):
        raise ValueError("samplingnull source must decode to a JSON object")
    return result
