"""Replay audit — determinism and repeatability verification of adapter scoring."""
from __future__ import annotations

from typing import Any

import numpy as np

from .legality import describe_runner, fork_runner
from .trace_schema import parse_sample_trace


def replay_runtime(
    runner: Any = None,
    tokens: np.ndarray | None = None,
    *,
    artifact_path: Any | None = None,
    config_path: Any | None = None,
    profile: str = "parameter-golf",
    chunk_size: int = 32_768,
    max_chunks: int | None = None,
    sample_chunks: int = 4,
    position_batch_size: int = 256,
    seed: int = 0,
    atol: float = 1e-7,
    rtol: float = 1e-7,
) -> dict[str, Any]:
    # New-style API: artifact_path + config_path + runner
    if artifact_path is not None:
        return _replay_from_paths(artifact_path, config_path, runner, profile=profile)
    if profile != "parameter-golf":
        raise ValueError(f"Unknown replay profile: {profile}")
    return replay_parameter_golf(
        runner, tokens,
        chunk_size=chunk_size, max_chunks=max_chunks, sample_chunks=sample_chunks,
        position_batch_size=position_batch_size, seed=seed, atol=atol, rtol=rtol,
    )


def replay_parameter_golf(
    runner: Any = None,
    tokens: np.ndarray | None = None,
    *,
    artifact_path: Any | None = None,
    config_path: Any | None = None,
    chunk_size: int = 32_768,
    max_chunks: int | None = None,
    sample_chunks: int = 4,
    position_batch_size: int = 256,
    seed: int = 0,
    atol: float = 1e-7,
    rtol: float = 1e-7,
) -> dict[str, Any]:
    # New-style API: artifact_path + config_path + runner
    if artifact_path is not None:
        return _replay_from_paths(artifact_path, config_path, runner, profile="parameter-golf")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if position_batch_size <= 0:
        raise ValueError("position_batch_size must be positive")
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("max_chunks must be positive when provided")
    token_arr = np.asarray(tokens, dtype=np.int64).reshape(-1)
    if token_arr.size == 0:
        raise ValueError("tokens must be non-empty")

    chunk_starts = list(range(0, int(token_arr.size), chunk_size))
    chunks = [token_arr[start : start + chunk_size] for start in chunk_starts]
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
        chunk_starts = chunk_starts[:max_chunks]

    rng = np.random.default_rng(seed)
    selected_chunks = _choose_chunk_indices(len(chunks), sample_chunks, rng)
    selected_set = set(selected_chunks)
    adapter_info = describe_runner(runner)

    chunk_rows: list[dict[str, Any]] = []
    total_loss_nats = 0.0
    total_positions = 0
    repeat_stats = _empty_repeat_stats()

    for chunk_index, chunk in enumerate(chunks):
        chunk = np.asarray(chunk, dtype=np.int64)
        snapshot = fork_runner(runner)
        chunk_row, chunk_repeat = _replay_chunk(
            snapshot, chunk,
            chunk_index=chunk_index, chunk_start=chunk_starts[chunk_index],
            position_batch_size=position_batch_size,
            compare_repeat=chunk_index in selected_set,
            atol=atol, rtol=rtol,
        )
        chunk_rows.append(chunk_row)
        total_loss_nats += float(chunk_row["total_loss_nats"])
        total_positions += int(chunk_row["token_count"])
        _merge_repeat_stats(repeat_stats, chunk_repeat)

        runner.score_chunk(chunk, sample_positions=None)
        if chunk_index + 1 < len(chunks):
            runner.adapt_chunk(chunk)

    mean_loss_nats = float(total_loss_nats / total_positions) if total_positions else None
    mean_bpb = float(mean_loss_nats / np.log(2.0)) if mean_loss_nats is not None else None
    repeat_summary = _finalize_repeat_stats(repeat_stats, atol=atol, rtol=rtol)

    return {
        "profile": "parameter-golf",
        "adapter": adapter_info,
        "token_count": int(token_arr.size),
        "audited_token_count": int(sum(int(chunk.size) for chunk in chunks)),
        "chunk_size": int(chunk_size),
        "chunk_count": len(chunks),
        "max_chunks": None if max_chunks is None else int(max_chunks),
        "position_batch_size": int(position_batch_size),
        "selected_chunks": selected_chunks,
        "aggregate": {
            "total_loss_nats": float(total_loss_nats),
            "mean_loss_nats": mean_loss_nats,
            "mean_bpb": mean_bpb,
        },
        "repeatability": repeat_summary,
        "chunks": chunk_rows,
    }


def _replay_from_paths(artifact_path: Any, config_path: Any, runner: Any, *, profile: str) -> dict[str, Any]:
    """Handle path-based replay API — load artifact/config and run if runner is available, or return error for missing files."""
    from pathlib import Path as _Path
    artifact = _Path(artifact_path) if not isinstance(artifact_path, _Path) else artifact_path
    if not artifact.exists():
        return {"status": "error", "error": f"Artifact not found: {artifact}"}
    if config_path is not None:
        cfg = _Path(config_path) if not isinstance(config_path, _Path) else config_path
        if not cfg.exists():
            return {"status": "error", "error": f"Config not found: {cfg}"}
    if runner is None:
        return {"status": "error", "error": "No runner provided for replay"}
    return {"status": "error", "error": "Cannot run path-based replay without token data"}


def _replay_chunk(
    snapshot: Any, chunk: np.ndarray, *, chunk_index: int, chunk_start: int,
    position_batch_size: int, compare_repeat: bool, atol: float, rtol: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    total_loss_nats = 0.0
    trace_fields_present: set[str] = set()
    repeat_stats = _empty_repeat_stats()

    for positions in _position_batches(int(chunk.size), position_batch_size):
        base = _score_position_batch(snapshot, chunk, positions)
        total_loss_nats += float(np.sum(-base["gold_logprobs"]))
        trace_fields_present.update(base["trace_fields"])

        if compare_repeat:
            repeated = _score_position_batch(snapshot, chunk, positions)
            _merge_batch_repeat_stats(repeat_stats, base, repeated, atol=atol, rtol=rtol)

    token_count = int(chunk.size)
    mean_loss_nats = float(total_loss_nats / token_count) if token_count else None
    row: dict[str, Any] = {
        "chunk_index": int(chunk_index),
        "chunk_start": int(chunk_start),
        "token_count": token_count,
        "total_loss_nats": float(total_loss_nats),
        "mean_loss_nats": mean_loss_nats,
        "mean_bpb": float(mean_loss_nats / np.log(2.0)) if mean_loss_nats is not None else None,
        "trace_fields_present": sorted(trace_fields_present),
        "repeat_compared": bool(compare_repeat),
    }
    if compare_repeat:
        row["repeatability"] = _finalize_repeat_stats(repeat_stats, atol=atol, rtol=rtol)
    return row, repeat_stats


def _score_position_batch(snapshot: Any, chunk: np.ndarray, positions: list[int]) -> dict[str, Any]:
    sample_positions = np.asarray(positions, dtype=np.int64)
    runner = fork_runner(snapshot)
    outputs = runner.score_chunk(chunk, sample_positions=sample_positions)
    if not isinstance(outputs, dict):
        raise ValueError("score_chunk() must return a dict")
    if "sample_predictions" not in outputs:
        raise ValueError("score_chunk() must return sample_predictions when sample_positions are requested")
    preds = np.asarray(outputs["sample_predictions"], dtype=np.float64)
    if preds.shape[0] != len(positions):
        raise ValueError("sample_predictions first dimension must match the number of requested sample_positions")
    gold_logprobs = np.empty((len(positions),), dtype=np.float64)
    for idx, pos in enumerate(positions):
        tok = int(chunk[pos])
        row = preds[idx]
        prob = float(row[tok]) if 0 <= tok < row.shape[0] else 0.0
        gold_logprobs[idx] = float(np.log(max(prob, np.finfo(np.float64).tiny)))
    trace_info = parse_sample_trace(outputs, positions)
    return {"positions": positions, "predictions": preds, "gold_logprobs": gold_logprobs, "trace": trace_info["by_position"], "trace_fields": trace_info["present_fields"]}


def _position_batches(size: int, batch_size: int) -> list[list[int]]:
    if size <= 0:
        return []
    return [list(range(start, min(size, start + batch_size))) for start in range(0, size, batch_size)]


def _choose_chunk_indices(chunk_count: int, sample_chunks: int, rng: np.random.Generator) -> list[int]:
    if chunk_count <= 0:
        return []
    if sample_chunks <= 0 or sample_chunks >= chunk_count:
        return list(range(chunk_count))
    picks = rng.choice(chunk_count, size=sample_chunks, replace=False)
    return sorted(int(x) for x in picks.tolist())


def _empty_repeat_stats() -> dict[str, Any]:
    return {
        "chunk_count": 0, "position_count": 0,
        "prediction_diff_sum": 0.0, "prediction_diff_count": 0,
        "prediction_diff_failures": 0, "max_abs_prediction_diff": 0.0,
        "gold_logprob_diff_sum": 0.0, "gold_logprob_diff_count": 0,
        "gold_logprob_diff_failures": 0, "max_abs_gold_logprob_diff": 0.0,
        "state_hash_mismatch_count": 0, "state_hash_compared_count": 0,
    }


def _merge_batch_repeat_stats(target: dict[str, Any], base: dict[str, Any], repeated: dict[str, Any], *, atol: float, rtol: float) -> None:
    target["chunk_count"] = max(int(target["chunk_count"]), 1)
    target["position_count"] += int(len(base["positions"]))
    pred_diff = np.abs(base["predictions"] - repeated["predictions"])
    target["prediction_diff_sum"] += float(np.sum(pred_diff))
    target["prediction_diff_count"] += int(pred_diff.size)
    target["max_abs_prediction_diff"] = max(float(target["max_abs_prediction_diff"]), float(np.max(pred_diff, initial=0.0)))
    if not np.allclose(base["predictions"], repeated["predictions"], atol=atol, rtol=rtol):
        target["prediction_diff_failures"] += 1
    gold_diff = np.abs(base["gold_logprobs"] - repeated["gold_logprobs"])
    target["gold_logprob_diff_sum"] += float(np.sum(gold_diff))
    target["gold_logprob_diff_count"] += int(gold_diff.size)
    target["max_abs_gold_logprob_diff"] = max(float(target["max_abs_gold_logprob_diff"]), float(np.max(gold_diff, initial=0.0)))
    if not np.allclose(base["gold_logprobs"], repeated["gold_logprobs"], atol=atol, rtol=rtol):
        target["gold_logprob_diff_failures"] += 1
    state_fields = ("state_hash_before", "state_hash_after")
    for pos in base["positions"]:
        base_trace = base["trace"].get(pos, {})
        rep_trace = repeated["trace"].get(pos, {})
        for field in state_fields:
            if field not in base_trace or field not in rep_trace:
                continue
            target["state_hash_compared_count"] += 1
            if base_trace[field] != rep_trace[field]:
                target["state_hash_mismatch_count"] += 1


def _merge_repeat_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("chunk_count", "position_count", "prediction_diff_sum", "prediction_diff_count", "prediction_diff_failures", "gold_logprob_diff_sum", "gold_logprob_diff_count", "gold_logprob_diff_failures", "state_hash_mismatch_count", "state_hash_compared_count"):
        target[key] += source[key]
    target["max_abs_prediction_diff"] = max(float(target["max_abs_prediction_diff"]), float(source["max_abs_prediction_diff"]))
    target["max_abs_gold_logprob_diff"] = max(float(target["max_abs_gold_logprob_diff"]), float(source["max_abs_gold_logprob_diff"]))


def _finalize_repeat_stats(stats: dict[str, Any], *, atol: float, rtol: float) -> dict[str, Any]:
    compared = int(stats["prediction_diff_count"]) > 0
    mean_abs_prediction_diff = float(stats["prediction_diff_sum"] / stats["prediction_diff_count"]) if stats["prediction_diff_count"] else None
    mean_abs_gold_logprob_diff = float(stats["gold_logprob_diff_sum"] / stats["gold_logprob_diff_count"]) if stats["gold_logprob_diff_count"] else None
    pass_value = None
    if compared:
        pass_value = (int(stats["prediction_diff_failures"]) == 0 and int(stats["gold_logprob_diff_failures"]) == 0 and int(stats["state_hash_mismatch_count"]) == 0)
    return {
        "covered": compared, "pass": pass_value,
        "chunk_count": int(stats["chunk_count"]), "position_count": int(stats["position_count"]),
        "max_abs_prediction_diff": float(stats["max_abs_prediction_diff"]),
        "mean_abs_prediction_diff": mean_abs_prediction_diff,
        "prediction_diff_failures": int(stats["prediction_diff_failures"]),
        "max_abs_gold_logprob_diff": float(stats["max_abs_gold_logprob_diff"]),
        "mean_abs_gold_logprob_diff": mean_abs_gold_logprob_diff,
        "gold_logprob_diff_failures": int(stats["gold_logprob_diff_failures"]),
        "state_hash_compared_count": int(stats["state_hash_compared_count"]),
        "state_hash_mismatch_count": int(stats["state_hash_mismatch_count"]),
        "tolerances": {"atol": float(atol), "rtol": float(rtol)},
    }
