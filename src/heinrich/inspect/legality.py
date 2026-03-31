"""Legality audit — runtime adapter probing for parameter-golf compliance."""
from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from .trace_schema import parse_sample_trace

TRUST_LEVELS = ("basic", "traced", "strict")


# ---------------------------------------------------------------------------
# Token / config loading
# ---------------------------------------------------------------------------

def load_token_array(path: Path, *, key: str | None = None) -> np.ndarray:
    if path.suffix == ".npy":
        tokens = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            names = list(data.files)
            if not names:
                raise ValueError(f"No arrays found in token bundle: {path}")
            if key is None:
                if len(names) != 1:
                    raise ValueError(f"Multiple arrays found in {path}; specify --tokens-key")
                key = names[0]
            if key not in data:
                raise ValueError(f"Array {key!r} not found in {path}")
            tokens = np.array(data[key], copy=False)
    elif path.suffix == ".csv":
        tokens = np.loadtxt(path, delimiter=",", dtype=np.int64)
    else:
        raise ValueError(f"Unsupported token format: {path.suffix}")
    arr = np.asarray(tokens, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Token array is empty: {path}")
    return arr


def load_json_config(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    raw = raw.strip()
    if raw.startswith("{"):
        obj = json.loads(raw)
    else:
        obj = json.loads(Path(raw).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Adapter config must decode to a JSON object")
    return obj


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------

def load_adapter(adapter_ref: str, config: dict[str, Any]) -> Any:
    module = _load_adapter_module(adapter_ref)
    build = getattr(module, "build_adapter", None)
    if build is None or not callable(build):
        raise ValueError(f"Adapter {adapter_ref!r} must export a callable build_adapter(config)")
    runner = build(config)
    for name in ("score_chunk", "adapt_chunk"):
        if not hasattr(runner, name) or not callable(getattr(runner, name)):
            raise ValueError(f"Adapter runner must define callable {name}()")
    return runner


def _load_adapter_module(adapter_ref: str) -> Any:
    candidate = Path(adapter_ref)
    if candidate.suffix == ".py" or candidate.exists():
        if not candidate.exists():
            raise FileNotFoundError(f"Adapter file not found: {adapter_ref}")
        spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load adapter from {adapter_ref}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(adapter_ref)


def fork_runner(runner: Any) -> Any:
    if hasattr(runner, "fork") and callable(runner.fork):
        return runner.fork()
    return copy.deepcopy(runner)


def describe_runner(runner: Any) -> dict[str, Any]:
    if hasattr(runner, "describe") and callable(runner.describe):
        desc = runner.describe()
        if isinstance(desc, dict):
            return desc
    return {"runner_type": type(runner).__name__}


# ---------------------------------------------------------------------------
# Main audit entry points
# ---------------------------------------------------------------------------

def audit_legality(
    runner: Any,
    tokens: np.ndarray,
    *,
    profile: str,
    trust_level: str = "basic",
    chunk_size: int,
    max_chunks: int | None,
    sample_chunks: int,
    future_probes_per_chunk: int,
    answer_probes_per_chunk: int,
    positions_per_future_probe: int,
    seed: int,
    vocab_size: int | None,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if profile != "parameter-golf":
        raise ValueError(f"Unknown legality profile: {profile}")
    return audit_parameter_golf_legality(
        runner,
        tokens,
        trust_level=trust_level,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
        sample_chunks=sample_chunks,
        future_probes_per_chunk=future_probes_per_chunk,
        answer_probes_per_chunk=answer_probes_per_chunk,
        positions_per_future_probe=positions_per_future_probe,
        seed=seed,
        vocab_size=vocab_size,
        atol=atol,
        rtol=rtol,
    )


def audit_parameter_golf_legality(
    runner: Any,
    tokens: np.ndarray,
    *,
    trust_level: str = "basic",
    chunk_size: int = 32_768,
    max_chunks: int | None = None,
    sample_chunks: int = 4,
    future_probes_per_chunk: int = 2,
    answer_probes_per_chunk: int = 2,
    positions_per_future_probe: int = 4,
    seed: int = 0,
    vocab_size: int | None = None,
    atol: float = 1e-7,
    rtol: float = 1e-7,
) -> dict[str, Any]:
    if trust_level not in TRUST_LEVELS:
        raise ValueError(f"Unknown trust level: {trust_level}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("max_chunks must be positive when provided")
    token_arr = np.asarray(tokens, dtype=np.int64).reshape(-1)
    if token_arr.size == 0:
        raise ValueError("tokens must be non-empty")
    vocab_size_explicit = vocab_size is not None
    if vocab_size is None:
        vocab_size = int(np.max(token_arr)) + 1
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    chunk_starts = list(range(0, int(token_arr.size), chunk_size))
    chunks = [token_arr[start : start + chunk_size] for start in chunk_starts]
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
        chunk_starts = chunk_starts[:max_chunks]
    rng = np.random.default_rng(seed)
    chosen_chunks = _choose_chunk_indices(len(chunks), sample_chunks, rng)
    chosen_set = set(chosen_chunks)
    adapter_info = describe_runner(runner)
    adapter_info.setdefault("vocab_size", vocab_size)
    adapter_info.setdefault("vocab_size_source", "explicit" if vocab_size_explicit else "inferred_from_tokens")

    summaries = {
        "normalization": _empty_summary(),
        "trace_coverage": _empty_summary(),
        "repeatability": _empty_summary(),
        "future_suffix_invariance": _empty_summary(),
        "answer_mask_invariance": _empty_summary(),
        "gold_logprob_consistency": _empty_summary(),
        "accounting_contribution_consistency": _empty_summary(),
        "accounting_path_invariance": _empty_summary(),
        "state_hash_consistency": _empty_summary(),
    }
    probes: list[dict[str, Any]] = []
    trace_fields_present: set[str] = set()

    for chunk_index, chunk in enumerate(chunks):
        chunk = np.asarray(chunk, dtype=np.int64)
        if chunk_index in chosen_set:
            snapshot = fork_runner(runner)
            chunk_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            chunk_rng = np.random.default_rng(chunk_seed)
            probe_rows, chunk_summaries = _audit_chunk(
                snapshot, chunk,
                chunk_index=chunk_index, chunk_start=chunk_starts[chunk_index],
                rng=chunk_rng, vocab_size=vocab_size, vocab_size_explicit=vocab_size_explicit,
                future_probes_per_chunk=future_probes_per_chunk,
                answer_probes_per_chunk=answer_probes_per_chunk,
                positions_per_future_probe=positions_per_future_probe,
                atol=atol, rtol=rtol,
            )
            probes.extend(probe_rows)
            for row in probe_rows:
                if row.get("kind") == "trace_coverage":
                    trace_fields_present.update(row.get("present_fields", []))
            for name, chunk_summary in chunk_summaries.items():
                _merge_chunk_summary(summaries[name], chunk_summary)

        runner.score_chunk(chunk, sample_positions=None)
        if chunk_index + 1 < len(chunks):
            runner.adapt_chunk(chunk)

    for summary in summaries.values():
        covered = int(summary["probe_count"]) > 0
        summary["covered"] = covered
        summary["pass"] = (summary["failure_count"] == 0) if covered else None

    alerts = [
        f"{name} failed on {summary['failure_count']} / {summary['probe_count']} probes"
        for name, summary in summaries.items()
        if int(summary["failure_count"]) > 0
    ]
    trust = _assess_trust_level(
        requested=trust_level, vocab_size_explicit=vocab_size_explicit,
        trace_fields_present=trace_fields_present, checks=summaries,
    )
    if not trust["satisfied"]:
        alerts.append(f"requested trust_level {trust['requested']} not satisfied; achieved {trust['achieved']}")

    return {
        "profile": "parameter-golf",
        "adapter": adapter_info,
        "token_count": int(token_arr.size),
        "audited_token_count": int(sum(int(chunk.size) for chunk in chunks)),
        "chunk_size": int(chunk_size),
        "chunk_count": len(chunks),
        "max_chunks": None if max_chunks is None else int(max_chunks),
        "selected_chunks": chosen_chunks,
        "vocab_size_source": "explicit" if vocab_size_explicit else "inferred_from_tokens",
        "trace_fields_present": sorted(trace_fields_present),
        "tolerances": {"atol": float(atol), "rtol": float(rtol)},
        "trust": trust,
        "obligations": _parameter_golf_obligations(
            vocab_size_explicit=vocab_size_explicit,
            gold_logprob_covered=bool(summaries["gold_logprob_consistency"]["covered"]),
            accounting_trace_covered=bool(
                summaries["accounting_contribution_consistency"]["covered"]
                or summaries["accounting_path_invariance"]["covered"]
            ),
        ),
        "checks": summaries,
        "probes": probes,
        "alerts": alerts,
    }


def _audit_chunk(
    snapshot: Any, chunk: np.ndarray, *, chunk_index: int, chunk_start: int,
    rng: np.random.Generator, vocab_size: int, vocab_size_explicit: bool,
    future_probes_per_chunk: int, answer_probes_per_chunk: int,
    positions_per_future_probe: int, atol: float, rtol: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    future_specs = _build_future_specs(chunk_len=chunk.size, future_probes=future_probes_per_chunk, positions_per_probe=positions_per_future_probe, rng=rng)
    answer_positions = _build_answer_positions(chunk_len=chunk.size, answer_probes=answer_probes_per_chunk, rng=rng)
    base_positions = sorted(set(answer_positions) | {pos for spec in future_specs for pos in spec["positions"]})
    empty_sums = {k: _empty_summary() for k in [
        "normalization", "trace_coverage", "repeatability", "future_suffix_invariance",
        "answer_mask_invariance", "gold_logprob_consistency", "accounting_contribution_consistency",
        "accounting_path_invariance", "state_hash_consistency"]}
    if not base_positions:
        return [], empty_sums

    base_outputs = _score_sample_outputs(fork_runner(snapshot), chunk, base_positions)
    base_predictions = base_outputs["predictions"]
    base_trace = base_outputs["trace"]
    repeat_outputs = _score_sample_outputs(fork_runner(snapshot), chunk, base_positions)
    repeat_predictions = repeat_outputs["predictions"]
    repeat_trace = repeat_outputs["trace"]

    probes: list[dict[str, Any]] = []
    summaries = {k: _empty_summary() for k in empty_sums}

    norm_result = _check_normalization_set(base_predictions, positions=base_positions, vocab_size=vocab_size, vocab_size_explicit=vocab_size_explicit, atol=atol, rtol=rtol)
    probes.append({"kind": "normalization", "chunk_index": chunk_index, "chunk_start": chunk_start, "positions": base_positions, **norm_result})
    _merge_summary(summaries["normalization"], norm_result)

    trace_row = _check_trace_coverage(base_trace, positions=base_positions)
    if trace_row is not None:
        probes.append({"kind": "trace_coverage", "chunk_index": chunk_index, "chunk_start": chunk_start, "positions": base_positions, **trace_row})
        _merge_summary(summaries["trace_coverage"], trace_row)

    gold_result = _check_gold_logprob_consistency(base_predictions, base_outputs["gold_logprobs"], chunk=chunk, positions=base_positions, atol=atol, rtol=rtol)
    if gold_result is not None:
        probes.append({"kind": "gold_logprob_consistency", "chunk_index": chunk_index, "chunk_start": chunk_start, "positions": base_positions, **gold_result})
        _merge_summary(summaries["gold_logprob_consistency"], gold_result)

    contrib_result = _check_accounting_contribution_consistency(base_predictions, base_trace, chunk=chunk, positions=base_positions, atol=atol, rtol=rtol)
    if contrib_result is not None:
        probes.append({"kind": "accounting_contribution_consistency", "chunk_index": chunk_index, "chunk_start": chunk_start, "positions": base_positions, **contrib_result})
        _merge_summary(summaries["accounting_contribution_consistency"], contrib_result)

    state_result = _check_state_hash_consistency(base_trace, repeat_trace, positions=base_positions)
    if state_result is not None:
        probes.append({"kind": "state_hash_consistency", "chunk_index": chunk_index, "chunk_start": chunk_start, "positions": base_positions, **state_result})
        _merge_summary(summaries["state_hash_consistency"], state_result)

    repeat_result = _compare_position_set(base_predictions, repeat_predictions, positions=base_positions, atol=atol, rtol=rtol)
    probes.append({"kind": "repeatability", "chunk_index": chunk_index, "chunk_start": chunk_start, "positions": base_positions, **repeat_result})
    _merge_summary(summaries["repeatability"], repeat_result)

    for spec in future_specs:
        perturbed = chunk.copy()
        perturbed[spec["cutoff"]:] = rng.integers(0, vocab_size, size=(chunk.size - spec["cutoff"],), dtype=np.int64)
        alt_out = _score_sample_outputs(fork_runner(snapshot), perturbed, spec["positions"])
        fsi_result = _compare_position_set(base_predictions, alt_out["predictions"], positions=spec["positions"], atol=atol, rtol=rtol)
        probes.append({"kind": "future_suffix_invariance", "chunk_index": chunk_index, "chunk_start": chunk_start, "cutoff": int(spec["cutoff"]), "positions": spec["positions"], **fsi_result})
        _merge_summary(summaries["future_suffix_invariance"], fsi_result)
        path_result = _compare_accounting_path_set(base_trace, alt_out["trace"], positions=spec["positions"], atol=atol, rtol=rtol)
        if path_result is not None:
            probes.append({"kind": "accounting_path_invariance", "chunk_index": chunk_index, "chunk_start": chunk_start, "cutoff": int(spec["cutoff"]), "positions": spec["positions"], "probe_family": "future_suffix", **path_result})
            _merge_summary(summaries["accounting_path_invariance"], path_result)

    for pos in answer_positions:
        perturbed = chunk.copy()
        perturbed[pos:] = rng.integers(0, vocab_size, size=(chunk.size - pos,), dtype=np.int64)
        alt_out = _score_sample_outputs(fork_runner(snapshot), perturbed, [pos])
        ami_result = _compare_position_set(base_predictions, alt_out["predictions"], positions=[pos], atol=atol, rtol=rtol)
        probes.append({"kind": "answer_mask_invariance", "chunk_index": chunk_index, "chunk_start": chunk_start, "position": int(pos), **ami_result})
        _merge_summary(summaries["answer_mask_invariance"], ami_result)
        path_result = _compare_accounting_path_set(base_trace, alt_out["trace"], positions=[pos], atol=atol, rtol=rtol)
        if path_result is not None:
            probes.append({"kind": "accounting_path_invariance", "chunk_index": chunk_index, "chunk_start": chunk_start, "position": int(pos), "probe_family": "answer_mask", **path_result})
            _merge_summary(summaries["accounting_path_invariance"], path_result)

    return probes, summaries


def _score_sample_outputs(runner: Any, chunk: np.ndarray, positions: list[int]) -> dict[str, Any]:
    sample_positions = np.asarray(positions, dtype=np.int64)
    outputs = runner.score_chunk(chunk, sample_positions=sample_positions)
    if not isinstance(outputs, dict):
        raise ValueError("score_chunk() must return a dict")
    if "sample_predictions" not in outputs:
        raise ValueError("score_chunk() must return sample_predictions when sample_positions are requested")
    preds = np.asarray(outputs["sample_predictions"])
    if preds.shape[0] != len(positions):
        raise ValueError("sample_predictions first dimension must match the number of requested sample_positions")
    predictions = {int(pos): np.asarray(preds[idx], dtype=np.float64) for idx, pos in enumerate(positions)}
    trace_info = parse_sample_trace(outputs, positions)
    trace = trace_info["by_position"]
    gold_positions = {int(pos): float(trace[int(pos)]["gold_logprobs"]) for pos in positions if "gold_logprobs" in trace[int(pos)]}
    return {"predictions": predictions, "gold_logprobs": gold_positions or None, "trace": trace, "trace_fields": trace_info["present_fields"]}


def _compare_position_set(base: dict[int, np.ndarray], alt: dict[int, np.ndarray], *, positions: list[int], atol: float, rtol: float) -> dict[str, Any]:
    max_abs_diff = 0.0
    shape_mismatch = False
    passed = True
    for pos in positions:
        b, a = base[pos], alt[pos]
        if b.shape != a.shape:
            shape_mismatch = True; passed = False; continue
        diff = np.abs(b - a)
        local_max = float(np.max(diff, initial=0.0))
        if local_max > max_abs_diff:
            max_abs_diff = local_max
        if not np.allclose(b, a, atol=atol, rtol=rtol):
            passed = False
    return {"pass": passed and not shape_mismatch, "shape_mismatch": shape_mismatch, "max_abs_diff": float(max_abs_diff)}


def _check_normalization_set(predictions: dict[int, np.ndarray], *, positions: list[int], vocab_size: int, vocab_size_explicit: bool, atol: float, rtol: float) -> dict[str, Any]:
    max_abs_diff = 0.0
    min_value = float("inf")
    shape_mismatch = False
    wrong_length_count = 0
    observed_sizes: set[int] = set()
    passed = True
    for pos in positions:
        row = np.asarray(predictions[pos], dtype=np.float64)
        if row.ndim != 1:
            shape_mismatch = True; passed = False; continue
        observed_sizes.add(int(row.shape[0]))
        if row.shape[0] != vocab_size:
            wrong_length_count += 1; passed = False
        row_sum = float(np.sum(row))
        max_abs_diff = max(max_abs_diff, abs(row_sum - 1.0))
        min_value = min(min_value, float(np.min(row)))
        if np.any(row < -atol) or not np.isclose(row_sum, 1.0, atol=atol, rtol=rtol):
            passed = False
    return {"pass": passed and not shape_mismatch, "shape_mismatch": shape_mismatch, "expected_vocab_size": int(vocab_size), "vocab_size_source": "explicit" if vocab_size_explicit else "inferred_from_tokens", "wrong_length_count": int(wrong_length_count), "observed_sizes": sorted(observed_sizes), "max_abs_diff": float(max_abs_diff), "min_value": float(min_value) if min_value != float("inf") else None}


def _check_gold_logprob_consistency(predictions: dict[int, np.ndarray], gold_logprobs: dict[int, float] | None, *, chunk: np.ndarray, positions: list[int], atol: float, rtol: float) -> dict[str, Any] | None:
    if gold_logprobs is None:
        return None
    max_abs_diff = 0.0
    missing_count = 0
    passed = True
    for pos in positions:
        if pos not in gold_logprobs:
            missing_count += 1; passed = False; continue
        row = np.asarray(predictions[pos], dtype=np.float64)
        tok = int(chunk[pos])
        prob = float(row[tok]) if 0 <= tok < row.shape[0] else 0.0
        implied = float(np.log(max(prob, np.finfo(np.float64).tiny)))
        reported = float(gold_logprobs[pos])
        diff = abs(implied - reported)
        max_abs_diff = max(max_abs_diff, diff)
        if not np.isclose(implied, reported, atol=atol, rtol=rtol):
            passed = False
    return {"pass": passed, "missing_count": int(missing_count), "max_abs_diff": float(max_abs_diff)}


def _check_trace_coverage(trace: dict[int, dict[str, Any]], *, positions: list[int]) -> dict[str, Any] | None:
    present_fields = sorted({field for pos in positions for field in trace[pos]})
    if not present_fields:
        return None
    required_accounting = ["gold_logprobs", "loss_nats", "weights", "counted"]
    missing_accounting_fields = sorted(field for field in required_accounting if field not in present_fields)
    return {"pass": True, "max_abs_diff": 0.0, "present_fields": present_fields, "missing_accounting_fields": missing_accounting_fields}


def _check_accounting_contribution_consistency(predictions: dict[int, np.ndarray], trace: dict[int, dict[str, Any]], *, chunk: np.ndarray, positions: list[int], atol: float, rtol: float) -> dict[str, Any] | None:
    required_fields = ("gold_logprobs", "loss_nats", "weights", "counted")
    if not all(all(field in trace[pos] for field in required_fields) for pos in positions):
        return None
    max_abs_diff = 0.0
    passed = True
    for pos in positions:
        row = np.asarray(predictions[pos], dtype=np.float64)
        tok = int(chunk[pos])
        prob = float(row[tok]) if 0 <= tok < row.shape[0] else 0.0
        implied_gold = float(np.log(max(prob, np.finfo(np.float64).tiny)))
        reported_gold = float(trace[pos]["gold_logprobs"])
        counted = bool(trace[pos]["counted"])
        weight = float(trace[pos]["weights"])
        expected_loss = 0.0 if not counted else float(-weight * reported_gold)
        reported_loss = float(trace[pos]["loss_nats"])
        for lhs, rhs in ((implied_gold, reported_gold), (expected_loss, reported_loss)):
            diff = abs(lhs - rhs)
            max_abs_diff = max(max_abs_diff, diff)
            if not np.isclose(lhs, rhs, atol=atol, rtol=rtol):
                passed = False
    return {"pass": passed, "max_abs_diff": float(max_abs_diff)}


def _compare_accounting_path_set(base_trace: dict[int, dict[str, Any]], alt_trace: dict[int, dict[str, Any]], *, positions: list[int], atol: float, rtol: float) -> dict[str, Any] | None:
    candidate_fields = ("weights", "counted", "path_ids")
    compared_fields = [f for f in candidate_fields if all(f in base_trace[p] and f in alt_trace[p] for p in positions)]
    if not compared_fields:
        return None
    max_abs_diff = 0.0
    passed = True
    for pos in positions:
        for field in compared_fields:
            bv, av = base_trace[pos][field], alt_trace[pos][field]
            if field == "weights":
                diff = abs(float(bv) - float(av))
                max_abs_diff = max(max_abs_diff, diff)
                if not np.isclose(float(bv), float(av), atol=atol, rtol=rtol):
                    passed = False
            else:
                if bv != av:
                    passed = False
    return {"pass": passed, "compared_fields": compared_fields, "max_abs_diff": float(max_abs_diff)}


def _check_state_hash_consistency(base_trace: dict[int, dict[str, Any]], repeat_trace: dict[int, dict[str, Any]], *, positions: list[int]) -> dict[str, Any] | None:
    fields = ("state_hash_before", "state_hash_after")
    compared_fields = [f for f in fields if all(f in base_trace[p] and f in repeat_trace[p] for p in positions)]
    if not compared_fields:
        return None
    mismatch_count = sum(1 for p in positions for f in compared_fields if base_trace[p][f] != repeat_trace[p][f])
    return {"pass": mismatch_count == 0, "compared_fields": compared_fields, "mismatch_count": int(mismatch_count), "max_abs_diff": 0.0}


def _build_future_specs(*, chunk_len: int, future_probes: int, positions_per_probe: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    if chunk_len <= 1 or future_probes <= 0 or positions_per_probe <= 0:
        return []
    specs: list[dict[str, Any]] = []
    for _ in range(future_probes):
        cutoff = int(rng.integers(2, chunk_len + 1))
        candidates = np.arange(1, cutoff)
        max_positions = min(len(candidates), positions_per_probe)
        raw_positions = rng.choice(candidates, size=max_positions, replace=False)
        specs.append({"cutoff": cutoff, "positions": sorted(int(x) for x in raw_positions.tolist())})
    return specs


def _build_answer_positions(*, chunk_len: int, answer_probes: int, rng: np.random.Generator) -> list[int]:
    if chunk_len <= 1 or answer_probes <= 0:
        return []
    count = min(chunk_len - 1, answer_probes)
    positions = rng.choice(np.arange(1, chunk_len), size=count, replace=False)
    return sorted(int(x) for x in positions.tolist())


def _choose_chunk_indices(chunk_count: int, sample_chunks: int, rng: np.random.Generator) -> list[int]:
    if chunk_count <= 0:
        return []
    if sample_chunks <= 0 or sample_chunks >= chunk_count:
        return list(range(chunk_count))
    picks = rng.choice(chunk_count, size=sample_chunks, replace=False)
    return sorted(int(x) for x in picks.tolist())


def _empty_summary() -> dict[str, Any]:
    return {"probe_count": 0, "failure_count": 0, "max_abs_diff": 0.0}


def _merge_summary(target: dict[str, Any], row: dict[str, Any]) -> None:
    target["probe_count"] += 1
    if not row["pass"]:
        target["failure_count"] += 1
    target["max_abs_diff"] = max(float(target["max_abs_diff"]), float(row["max_abs_diff"]))


def _merge_chunk_summary(target: dict[str, Any], chunk_summary: dict[str, Any]) -> None:
    target["probe_count"] += int(chunk_summary["probe_count"])
    target["failure_count"] += int(chunk_summary["failure_count"])
    target["max_abs_diff"] = max(float(target["max_abs_diff"]), float(chunk_summary["max_abs_diff"]))


def _parameter_golf_obligations(
    *,
    vocab_size_explicit: bool = False,
    gold_logprob_covered: bool = False,
    accounting_trace_covered: bool = False,
    requested_trust: str | None = None,
    checks: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    vocab_note = (
        "full-alphabet shape checks use the explicit --vocab-size boundary"
        if vocab_size_explicit
        else "vector length is checked only against max(tokens)+1; use --vocab-size to audit the official alphabet"
    )
    return {
        "prefix_causal_distribution": {"status": "partially_covered", "checked_by": ["repeatability", "future_suffix_invariance", "answer_mask_invariance"], "notes": ["sampled probes test same-position and suffix dependence within chosen chunks"]},
        "full_normalized_distribution_over_official_alphabet": {"status": "partially_covered", "checked_by": ["normalization"], "notes": ["sampled positions must return a non-negative 1D distribution that sums to 1", vocab_note]},
        "score_accounting_independent_of_answer": {
            "status": "partially_covered" if (gold_logprob_covered or accounting_trace_covered) else "out_of_scope",
            "checked_by": [n for n, c in [("gold_logprob_consistency", gold_logprob_covered), ("accounting_contribution_consistency", accounting_trace_covered), ("accounting_path_invariance", accounting_trace_covered)] if c] if (gold_logprob_covered or accounting_trace_covered) else [],
            "notes": ["sampled positions compare the adapter's reported gold-token logprob against the returned full distribution"] if (gold_logprob_covered or accounting_trace_covered) else ["the current adapter contract audits distributions, not x_t-dependent bookkeeping"],
        },
        "no_outcome_selection_across_validation_runs": {"status": "out_of_scope", "checked_by": [], "notes": ["the current runtime audit evaluates one declared run"]},
    }


def _assess_trust_level(
    *,
    requested: str,
    vocab_size_explicit: bool = False,
    trace_fields_present: set[str] | None = None,
    checks: dict[str, dict[str, Any]] | None = None,
    obligations: dict[str, Any] | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    if checks is None:
        checks = {}
    if trace_fields_present is None:
        trace_fields_present = set()

    def _safe_check(name: str, condition: bool, detail: str) -> dict[str, Any]:
        return _check_requirement(name, condition, detail)

    def _get_check(key: str) -> dict[str, Any]:
        return checks.get(key, {"covered": False, "pass": False})

    def _check_covered_and_pass(key: str) -> bool:
        """A check is satisfied if it's covered and passed, OR if it's not present in checks at all (not required)."""
        if key not in checks:
            return True  # Not required / not covered → don't block
        ch = checks[key]
        return bool(ch.get("covered") and ch.get("pass") is True)

    requirements = {
        "basic": [
            _safe_check("normalization", _check_covered_and_pass("normalization"), "sampled prediction vectors are normalized and shape-checked"),
            _safe_check("repeatability", _check_covered_and_pass("repeatability"), "repeat scoring from the same snapshot is numerically stable"),
            _safe_check("future_suffix_invariance", _check_covered_and_pass("future_suffix_invariance"), "sampled positions do not change when later suffix tokens are perturbed"),
            _safe_check("answer_mask_invariance", _check_covered_and_pass("answer_mask_invariance"), "sampled positions do not change when the scored token and later suffix are perturbed"),
        ],
        "traced": [
            _safe_check("explicit_vocab_size", vocab_size_explicit, "the legality run declared the official vocabulary size explicitly"),
            _safe_check("trace_fields.gold_logprobs/loss_nats/weights/counted/path_ids", {"gold_logprobs", "loss_nats", "weights", "counted", "path_ids"} <= trace_fields_present, "the adapter exposed enough trace fields"),
            _safe_check("gold_logprob_consistency", _check_covered_and_pass("gold_logprob_consistency"), "reported gold-token logprobs match the returned full distributions"),
            _safe_check("accounting_contribution_consistency", _check_covered_and_pass("accounting_contribution_consistency"), "reported loss contributions match"),
            _safe_check("accounting_path_invariance", _check_covered_and_pass("accounting_path_invariance"), "trace-backed path metadata is stable"),
        ],
        "strict": [
            _safe_check("trace_fields.state_hash_before/state_hash_after", {"state_hash_before", "state_hash_after"} <= trace_fields_present, "the adapter exposed state hashes"),
            _safe_check("state_hash_consistency", _check_covered_and_pass("state_hash_consistency"), "repeated scoring preserves score-time state hashes"),
        ],
    }
    achieved = "none"
    if _requirements_satisfied(requirements["basic"]):
        achieved = "basic"
        if _requirements_satisfied(requirements["traced"]):
            achieved = "traced"
            if _requirements_satisfied(requirements["strict"]):
                achieved = "strict"
    safe_requested = requested if requested in TRUST_LEVELS else "basic"
    missing = [row["name"] for level in TRUST_LEVELS if TRUST_LEVELS.index(level) <= TRUST_LEVELS.index(safe_requested) for row in requirements[level] if not row["satisfied"]]
    satisfied = achieved != "none" and TRUST_LEVELS.index(achieved) >= TRUST_LEVELS.index(safe_requested)
    return {"requested": requested, "achieved": achieved, "satisfied": satisfied, "requirements": requirements, "missing": missing, "notes": ["trust levels score the exposed adapter surface only"]}


def _check_requirement(name: str, satisfied: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "satisfied": bool(satisfied), "detail": detail}


def _requirements_satisfied(rows: list[dict[str, Any]]) -> bool:
    return all(bool(row["satisfied"]) for row in rows)
