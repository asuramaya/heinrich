# Session 12 Forensics Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse session-12's ad-hoc Python scripts into permanent CLI + MCP tools (T1, T2), add flag-tuning knobs to two existing tools (T6, T7), harden `mri-recapture` for multi-host volume paths (T5), audit silent partial-capture fallthroughs (T8), and anchor the T1 result with a regression test (T10).

**Architecture:** Follow the established `profile-cb-*` pattern. Analysis helpers go in `src/heinrich/profile/compare.py`. CLI parsers and dispatch go in `src/heinrich/cli.py`. MCP tool definitions and subprocess dispatch go in `src/heinrich/mcp.py`. Model-needed tools use subprocess isolation via `_do_subprocess`, matching `profile-cb-causality`. All new tools support `--json` via the existing global `--json` flag + `_json_or(args, result, formatter)` helper.

**Tech Stack:** Python 3.10+, numpy, argparse, pytest, the `decepticons.loader` interface via `heinrich.backend.protocol.load_backend(path, backend="decepticon")`, the `heinrich.backend.decepticon.load_val_sequences` helper for byte-shard reading.

---

## File Structure

**Modified files:**
- `src/heinrich/profile/compare.py` — adds `_cb_effective_context`, `_cb_ablations`, context-manager ablation helpers; patches `_cb_additivity_metrics` to accept `svd_samples`; patches `causal_bank_additivity` to plumb the flag; patches `causal_bank_pc_bands` to accept `n_bootstrap`; audit pass on `.get(...)` call sites.
- `src/heinrich/cli.py` — adds `profile-cb-effective-context`, `profile-cb-ablations` parsers + dispatch; adds `--svd-samples` to `profile-cb-additivity`; adds `--n-bootstrap` to `profile-cb-pc-bands`; patches `_resolve_recapture_source` with path-relative fallback.
- `src/heinrich/mcp.py` — adds `heinrich_cb_effective_context`, `heinrich_cb_ablations` tool definitions + dispatch.
- `tests/test_cb_effective_context.py` — NEW, regression test for T1.
- `tests/test_cb_forensics_tools.py` — NEW, unit tests for T1/T2 helpers that don't need a checkpoint (mock-friendly).

**Created files:**
- Two new test files above.

No files are deleted. No existing files are restructured.

---

## Build Order

1. **Task 1** — T1 helper `_cb_effective_context` in `profile/compare.py` with unit tests (mocked backend).
2. **Task 2** — T1 CLI parser + dispatch in `cli.py`.
3. **Task 3** — T1 MCP tool definition + dispatch in `mcp.py`.
4. **Task 4** — T10 integration test for T1 (gated on `HEINRICH_TEST_CKPT` env var).
5. **Task 5** — T2 context-manager ablation helpers + `_cb_ablations` in `profile/compare.py` with unit tests.
6. **Task 6** — T2 CLI parser + dispatch.
7. **Task 7** — T2 MCP tool definition + dispatch.
8. **Task 8** — T6 `--svd-samples` on `profile-cb-additivity` (parameter + flag + MCP optional arg).
9. **Task 9** — T7 `--n-bootstrap` on `profile-cb-pc-bands` (parameter + flag + MCP optional arg).
10. **Task 10** — T5 `_resolve_recapture_source` path-relative-to-sharts fallback.
11. **Task 11** — T8 silent-capture audit pass on `profile/compare.py`.

---

## Task 1: T1 — `_cb_effective_context` helper

**Files:**
- Modify: `src/heinrich/profile/compare.py` (append new helpers after `causal_bank_reproduce`, approximately after line 4701)
- Test: `tests/test_cb_forensics_tools.py` (create)

- [ ] **Step 1: Write failing unit tests for helper signature and bucket math**

Create `tests/test_cb_forensics_tools.py`:

```python
"""Unit tests for session-12 forensics tools (T1, T2) that don't need a checkpoint.

Integration tests gated on a real checkpoint live in
tests/test_cb_effective_context.py and skip without HEINRICH_TEST_CKPT.
"""
from __future__ import annotations

import numpy as np
import pytest

from heinrich.profile.compare import (
    _bucket_positions,
    _find_knee_bucket,
)


def test_bucket_positions_respects_bounds():
    """Positions 0..seqlen-1 partitioned by [1,2,4,8,16,32,64]; last bucket
    extends to seqlen."""
    buckets = _bucket_positions(seqlen=64, bounds=[1, 2, 4, 8, 16, 32, 64])
    # Expected ranges (inclusive min, exclusive max): [1,2)=[1], [2,4)=[2,3],
    # [4,8)=[4..7], [8,16)=[8..15], [16,32)=[16..31], [32,64)=[32..63].
    # Position 0 is not in any bucket (no prefix to condition on).
    assert buckets[0] == {"min": 1, "max": 2, "indices": [1]}
    assert buckets[2]["indices"] == [4, 5, 6, 7]
    assert buckets[-1]["indices"] == list(range(32, 64))


def test_bucket_positions_handles_shorter_seqlen():
    """Last bucket right-bound clamped to seqlen."""
    buckets = _bucket_positions(seqlen=24, bounds=[1, 2, 4, 8, 16, 32])
    assert buckets[-1]["min"] == 16
    assert buckets[-1]["max"] == 24
    assert buckets[-1]["indices"] == list(range(16, 24))


def test_find_knee_first_delta_below_threshold():
    """Knee = first adjacent-bucket bpb delta below threshold."""
    bucket_bpbs = [3.25, 2.82, 2.41, 2.04, 1.98, 1.975, 1.974]
    result = _find_knee_bucket(bucket_bpbs, threshold=0.01)
    # Deltas: 0.43, 0.41, 0.37, 0.06, 0.005, 0.001.
    # First delta < 0.01 is at index 4 (between bucket 4 and 5).
    assert result == 4


def test_find_knee_returns_none_when_monotone_drop():
    """If every adjacent delta exceeds threshold, there is no knee."""
    bucket_bpbs = [3.25, 2.82, 2.41, 2.04, 1.98]
    assert _find_knee_bucket(bucket_bpbs, threshold=0.01) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cb_forensics_tools.py -v`
Expected: FAIL with `ImportError` for `_bucket_positions` and `_find_knee_bucket`.

- [ ] **Step 3: Implement helpers at the top of compare.py's new section**

Open `src/heinrich/profile/compare.py` and append after `causal_bank_reproduce` (line ~4701):

```python
def _bucket_positions(seqlen: int, bounds: list[int]) -> list[dict]:
    """Partition position indices [1..seqlen-1] into buckets defined by
    ``bounds`` (left-inclusive, right-exclusive). Last bucket right-bound
    is clamped to seqlen. Position 0 is excluded because it has no prefix.

    Returns list of dicts ``{min, max, indices}`` in order.
    """
    bounds = sorted(set(int(b) for b in bounds))
    # Normalize: drop any bound >= seqlen (except we always cap last at seqlen).
    kept = [b for b in bounds if b < seqlen]
    if not kept:
        raise ValueError(f"no bucket bounds fit inside seqlen={seqlen}: {bounds}")
    kept.append(seqlen)
    out = []
    for lo, hi in zip(kept[:-1], kept[1:]):
        indices = list(range(lo, hi))
        out.append({"min": lo, "max": hi, "indices": indices})
    return out


def _find_knee_bucket(bucket_bpbs: list[float], threshold: float) -> int | None:
    """Return the index of the first bucket where bpb[i] - bpb[i+1] < threshold.

    Returns None if no such index exists (the curve is strictly decreasing at
    every step by at least ``threshold`` — i.e., no saturation observed yet).
    """
    for i in range(len(bucket_bpbs) - 1):
        if bucket_bpbs[i] - bucket_bpbs[i + 1] < threshold:
            return i
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cb_forensics_tools.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Write the end-to-end helper test with a fake backend**

Append to `tests/test_cb_forensics_tools.py`:

```python
class _FakeCausalBackend:
    """Minimal fake backend satisfying the _cb_effective_context interface.

    Returns synthetic logits such that cross-entropy at position t is a
    monotone-decreasing function of t — reproducing the expected bpb-curve
    shape so we can verify the full pipeline produces a sensible result.
    """

    def __init__(self, vocab_size: int = 256):
        class _Cfg:
            pass
        self.config = _Cfg()
        self.config.vocab_size = vocab_size

    def forward(self, seq):
        # seq shape: [1, seqlen] int64
        _, seqlen = seq.shape
        logits = np.zeros((1, seqlen, self.config.vocab_size), dtype=np.float32)
        for t in range(seqlen):
            # Correct token at position t is seq[0, t] (trivial case —
            # perfect prediction would make bpb=0 everywhere). To get a
            # realistic curve, add noise inversely proportional to t.
            correct = int(seq[0, t])
            peakedness = 1.0 + np.log1p(t)  # more peaked at deeper positions
            logits[0, t, :] = -peakedness * 0.1  # small negative baseline
            logits[0, t, correct] = peakedness  # correct token gets larger logit
        return logits


def test_cb_effective_context_decreasing_bpb_with_fake_backend(monkeypatch):
    """Helper runs end-to-end on a fake backend and produces a decreasing
    bpb curve + a detected knee (because synthetic curve saturates)."""
    from heinrich.profile import compare as cmp_mod

    fake_backend = _FakeCausalBackend(vocab_size=256)
    monkeypatch.setattr(cmp_mod, "_load_effective_context_backend",
                         lambda path, result_json, tokenizer_path: fake_backend)
    monkeypatch.setattr(cmp_mod, "_load_effective_context_val",
                         lambda val, seqlen, n_trials, vocab_size:
                         np.random.default_rng(0).integers(
                             0, 256, size=(n_trials, seqlen), dtype=np.int64))

    result = cmp_mod._cb_effective_context(
        model_path="IGNORED",
        val=None,
        seqlen=32,
        n_trials=4,
        buckets=[1, 2, 4, 8, 16, 32],
        knee_threshold=0.01,
    )

    assert "buckets" in result
    assert len(result["buckets"]) == 5
    assert all("bpb_mean" in b for b in result["buckets"])
    # Monotone decrease is the synthetic-backend property.
    bpbs = [b["bpb_mean"] for b in result["buckets"]]
    assert bpbs == sorted(bpbs, reverse=True)
    # Fake backend has a saturating synthetic curve; knee is likely detected.
    # (Don't assert a specific index — just assert the field exists.)
    assert "knee_bucket_min" in result
    assert "saturation_bpb" in result
    assert result["saturation_bpb"] == bpbs[-1]
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_cb_forensics_tools.py::test_cb_effective_context_decreasing_bpb_with_fake_backend -v`
Expected: FAIL — `_cb_effective_context`, `_load_effective_context_backend`, and `_load_effective_context_val` do not exist.

- [ ] **Step 7: Implement the end-to-end helper**

Append to `compare.py` after `_find_knee_bucket`:

```python
def _load_effective_context_backend(model_path: str,
                                     result_json: str | None,
                                     tokenizer_path: str | None):
    """Injected by tests; production path calls load_backend + asserts causal-bank."""
    from ..backend.protocol import load_backend
    kwargs = {}
    if result_json:
        kwargs["result_json"] = result_json
    if tokenizer_path:
        kwargs["tokenizer_path"] = tokenizer_path
    backend = load_backend(model_path, backend="decepticon", **kwargs)
    if backend.config.model_type != "causal_bank":
        raise ValueError(
            f"profile-cb-effective-context requires a causal-bank model; "
            f"got model_type={backend.config.model_type!r}")
    return backend


def _load_effective_context_val(val: str | None,
                                 seqlen: int,
                                 n_trials: int,
                                 vocab_size: int) -> np.ndarray:
    """Load val data as [n_trials, seqlen] int64. If val is None, sample
    random tokens from [0, vocab_size) for a null-content baseline (useful
    for shape/instrumentation checks but not for real measurements)."""
    if val is None:
        rng = np.random.default_rng(42)
        return rng.integers(0, vocab_size, size=(n_trials, seqlen),
                             dtype=np.int64)
    from ..backend.decepticon import load_val_sequences
    byte_level = vocab_size == 256
    return load_val_sequences(val, seq_len=seqlen, n_seqs=n_trials,
                              byte_level=byte_level)


def _cb_effective_context(model_path: str,
                          *,
                          val: str | None,
                          seqlen: int,
                          n_trials: int,
                          buckets: list[int],
                          knee_threshold: float,
                          result_json: str | None = None,
                          tokenizer_path: str | None = None) -> dict:
    """Context-knee test: per-position bpb on random-prefix sequences.

    For each of n_trials sequences of length seqlen, compute next-token
    cross-entropy at every position. Partition positions into buckets,
    compute mean bpb per bucket, identify the first adjacent-bucket delta
    below knee_threshold as the context-saturation point.

    The load-bearing session-12 diagnostic. Current finding: all
    substrate-primary causal-bank models plateau at a 16-byte effective
    context regardless of half_life_max or linear_modes.
    """
    backend = _load_effective_context_backend(
        model_path, result_json, tokenizer_path)
    vocab_size = int(backend.config.vocab_size)
    seqs = _load_effective_context_val(val, seqlen, n_trials, vocab_size)

    bucket_defs = _bucket_positions(seqlen=seqs.shape[1], bounds=buckets)

    # Per-position cross-entropy accumulator across trials.
    per_pos_sum = np.zeros(seqs.shape[1], dtype=np.float64)
    per_pos_n = np.zeros(seqs.shape[1], dtype=np.int64)

    for i in range(seqs.shape[0]):
        seq_i = seqs[i:i + 1]  # [1, seqlen]
        logits = backend.forward(seq_i)  # [1, seqlen, vocab]
        logits_2d = np.asarray(logits[0], dtype=np.float32)  # [seqlen, vocab]
        # Cross-entropy at position t predicts token at t+1. We skip t=seqlen-1
        # because there's no target, and we skip t=0 because there's no prefix
        # to condition on (bucket_defs already excludes position 0).
        # Convention: bucket "i..j" reports loss at positions i..j-1, where
        # loss[pos] = -log(p(seq[pos+1] | seq[:pos+1])). Using target at pos
        # aligns with standard per-token conventions.
        # Stable log-softmax + gather.
        log_probs = logits_2d - _logsumexp(logits_2d, axis=-1, keepdims=True)
        targets = np.asarray(seqs[i], dtype=np.int64)
        for pos in range(seqs.shape[1] - 1):
            nll = -float(log_probs[pos, targets[pos + 1]])
            per_pos_sum[pos] += nll
            per_pos_n[pos] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        per_pos_ce = np.where(per_pos_n > 0, per_pos_sum / per_pos_n, np.nan)
    # Convert nats to bpb.
    per_pos_bpb = per_pos_ce / np.log(2.0)

    # Per-bucket mean bpb.
    bucket_reports = []
    for b in bucket_defs:
        ce_vals = per_pos_bpb[b["indices"]]
        ce_vals = ce_vals[~np.isnan(ce_vals)]
        if len(ce_vals) == 0:
            bucket_reports.append({
                "min": b["min"], "max": b["max"],
                "n": 0, "bpb_mean": float("nan"), "bpb_sem": float("nan"),
            })
            continue
        mean = float(np.mean(ce_vals))
        sem = float(np.std(ce_vals, ddof=1) / np.sqrt(len(ce_vals))) \
               if len(ce_vals) > 1 else 0.0
        bucket_reports.append({
            "min": b["min"], "max": b["max"],
            "n": int(len(ce_vals)),
            "bpb_mean": round(mean, 4),
            "bpb_sem": round(sem, 4),
        })

    bucket_bpbs = [b["bpb_mean"] for b in bucket_reports]
    knee_idx = _find_knee_bucket(bucket_bpbs, threshold=knee_threshold)

    return {
        "model": model_path,
        "val_data": val,
        "n_trials": int(n_trials),
        "seqlen": int(seqlen),
        "knee_threshold": float(knee_threshold),
        "buckets": bucket_reports,
        "knee_bucket_min": bucket_reports[knee_idx]["min"] if knee_idx is not None else None,
        "knee_bucket_max": bucket_reports[knee_idx]["max"] if knee_idx is not None else None,
        "saturation_bpb": bucket_reports[-1]["bpb_mean"],
    }


def _logsumexp(x: np.ndarray, *, axis: int, keepdims: bool = False) -> np.ndarray:
    """Numerically-stable log-sum-exp for cross-entropy computation."""
    m = np.max(x, axis=axis, keepdims=True)
    y = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    if not keepdims:
        y = np.squeeze(y, axis=axis)
    return y
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_cb_forensics_tools.py -v`
Expected: 5 tests PASS.

- [ ] **Step 9: Commit**

```bash
git add src/heinrich/profile/compare.py tests/test_cb_forensics_tools.py
git commit -m "feat(cb): _cb_effective_context helper + bucket math unit tests"
```

---

## Task 2: T1 — CLI parser and dispatch

**Files:**
- Modify: `src/heinrich/cli.py` (add parser around line 443, add dispatch around line 849, add `_cmd_cb_effective_context` near line 3517)

- [ ] **Step 1: Add the CLI parser**

Open `src/heinrich/cli.py`. Locate the `p_cb_reproduce` block (around line 440). After its `add_argument` lines (just before the next parser block), insert:

```python
    p_cb_effctx = sub.add_parser("profile-cb-effective-context",
                                  help="Context-knee test: per-position bpb on random-prefix sequences, identifies effective context length")
    p_cb_effctx.add_argument("--model", required=True, help="Checkpoint path (.checkpoint.pt)")
    p_cb_effctx.add_argument("--val", default=None, help="Val bytes file (optional; random tokens used if omitted)")
    p_cb_effctx.add_argument("--seqlen", type=int, default=512, help="Sequence length (default: 512)")
    p_cb_effctx.add_argument("--n-trials", type=int, default=30, help="Number of trial sequences (default: 30)")
    p_cb_effctx.add_argument("--buckets", type=str,
                              default="1,2,4,8,16,32,64,128,256,512",
                              help="Comma-separated bucket bounds (default: 1,2,4,8,16,32,64,128,256,512)")
    p_cb_effctx.add_argument("--knee-threshold", type=float, default=0.01,
                              help="bpb delta below which adjacent buckets count as saturated (default: 0.01)")
    p_cb_effctx.add_argument("--result-json", default=None, help="Path to result.json for model config")
    p_cb_effctx.add_argument("--tokenizer-path", default=None, help="Path to tokenizer model")
```

- [ ] **Step 2: Add the dispatch branch**

Locate the dispatch chain around line 848 (`elif args.command == "profile-cb-reproduce": _cmd_cb_reproduce(args)`). After it, add:

```python
    elif args.command == "profile-cb-effective-context":
        _cmd_cb_effective_context(args)
```

- [ ] **Step 3: Add the command function**

Locate `_cmd_cb_reproduce` (around line 3502). After it, add:

```python
def _cmd_cb_effective_context(args: argparse.Namespace) -> None:
    """Per-position bpb curve; identify the effective-context knee."""
    from .profile.compare import _cb_effective_context

    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]
    result = _cb_effective_context(
        model_path=args.model,
        val=args.val,
        seqlen=args.seqlen,
        n_trials=args.n_trials,
        buckets=buckets,
        knee_threshold=args.knee_threshold,
        result_json=getattr(args, "result_json", None),
        tokenizer_path=getattr(args, "tokenizer_path", None),
    )

    def _fmt(r: dict) -> None:
        print(f"\n=== Effective-context for {r['model']} ===\n")
        print(f"  val_data: {r['val_data']}")
        print(f"  n_trials={r['n_trials']}  seqlen={r['seqlen']}  "
              f"threshold={r['knee_threshold']}\n")
        print(f"  {'bucket':<16} {'n':>6}  {'bpb_mean':>8}  {'bpb_sem':>8}")
        for b in r["buckets"]:
            label = f"[{b['min']},{b['max']})"
            print(f"  {label:<16} {b['n']:>6}  {b['bpb_mean']:>8.4f}  "
                  f"{b['bpb_sem']:>8.4f}")
        print()
        if r["knee_bucket_min"] is None:
            print(f"  knee: NOT DETECTED (curve still decreasing)")
        else:
            print(f"  knee: [{r['knee_bucket_min']},{r['knee_bucket_max']})")
        print(f"  saturation_bpb: {r['saturation_bpb']:.4f}\n")

    _json_or(args, result, _fmt)
```

- [ ] **Step 4: Smoke-test the CLI invocation**

Run: `python -m heinrich.cli profile-cb-effective-context --help`
Expected: argparse prints usage with all flags. No crash.

- [ ] **Step 5: Commit**

```bash
git add src/heinrich/cli.py
git commit -m "feat(cli): profile-cb-effective-context command + dispatch"
```

---

## Task 3: T1 — MCP tool definition and dispatch

**Files:**
- Modify: `src/heinrich/mcp.py` (add to `TOOLS` dict near line 546, add dispatch near line 1049)

- [ ] **Step 1: Add tool definition**

Open `src/heinrich/mcp.py`. Locate the `heinrich_cb_reproduce` entry in the `TOOLS` dict (around line 547). After its closing brace, insert:

```python
    "heinrich_cb_effective_context": {
        "description": "Context-knee test: per-position bpb on random-prefix sequences. Identifies the effective context length. Needs model.",
        "parameters": {
            "model": {"type": "string", "description": "Checkpoint path (.checkpoint.pt)", "required": True},
            "val": {"type": "string", "description": "Val bytes file (optional)"},
            "seqlen": {"type": "integer", "description": "Sequence length (default: 512)"},
            "n_trials": {"type": "integer", "description": "Number of trial sequences (default: 30)"},
            "buckets": {"type": "string", "description": "Comma-separated bucket bounds (default: 1,2,4,8,16,32,64,128,256,512)"},
            "knee_threshold": {"type": "number", "description": "bpb delta threshold for saturation (default: 0.01)"},
        },
    },
```

- [ ] **Step 2: Add dispatch branch**

Locate the `heinrich_cb_reproduce` dispatch block (around line 1046). After its block, insert:

```python
        if name == "heinrich_cb_effective_context":
            return self._do_subprocess(arguments, "profile-cb-effective-context",
                ["--model", arguments["model"]],
                optional={"val": "--val", "seqlen": "--seqlen",
                          "n_trials": "--n-trials", "buckets": "--buckets",
                          "knee_threshold": "--knee-threshold"},
                timeout=1800)
```

Timeout rationale: 30 minutes. A 30-trial × 512-seqlen × small-model run takes a few minutes; 30 minutes covers large models and longer sequences without hanging indefinitely.

- [ ] **Step 3: Smoke-test via direct tool dispatch**

Run: `python -c "from heinrich.mcp import TOOLS; print('heinrich_cb_effective_context' in TOOLS)"`
Expected: `True`

- [ ] **Step 4: Commit**

```bash
git add src/heinrich/mcp.py
git commit -m "feat(mcp): heinrich_cb_effective_context tool definition + dispatch"
```

---

## Task 4: T10 — Regression test for T1

**Files:**
- Test: `tests/test_cb_effective_context.py` (create)

- [ ] **Step 1: Write the integration test**

Create `tests/test_cb_effective_context.py`:

```python
"""Integration regression test for T1 / profile-cb-effective-context.

Anchors the session-12 finding that substrate-primary causal-bank models
plateau at a 16-byte effective context ceiling. Skipped unless a checkpoint
is provided via HEINRICH_TEST_CKPT (and matching val data via
HEINRICH_TEST_VAL).

To run locally:

    export HEINRICH_TEST_CKPT=/Volumes/sharts/heinrich/session11/\\
                               byte-hrr-learnable-s8-50k.checkpoint.pt
    export HEINRICH_TEST_VAL=/path/to/fineweb_val_000000_bytes.bin
    pytest tests/test_cb_effective_context.py -v

"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def substrate_primary_ckpt() -> tuple[str, str]:
    ckpt = os.environ.get("HEINRICH_TEST_CKPT")
    val = os.environ.get("HEINRICH_TEST_VAL")
    if not ckpt or not Path(ckpt).exists():
        pytest.skip("set HEINRICH_TEST_CKPT to a substrate-primary checkpoint")
    if not val or not Path(val).exists():
        pytest.skip("set HEINRICH_TEST_VAL to a matching bytes file")
    return ckpt, val


def test_substrate_primary_knee_at_or_below_32_bytes(substrate_primary_ckpt):
    """Substrate-primary family has a ~16-byte effective context ceiling.
    Assert knee ≤ 32 bytes (one-bucket tolerance around the 16-byte finding).
    """
    from heinrich.profile.compare import _cb_effective_context

    ckpt, val = substrate_primary_ckpt
    result = _cb_effective_context(
        model_path=ckpt,
        val=val,
        seqlen=128,
        n_trials=5,
        buckets=[1, 2, 4, 8, 16, 32, 64, 128],
        knee_threshold=0.01,
    )
    assert result["knee_bucket_max"] is not None, (
        f"no knee detected; bucket curve: "
        f"{[b['bpb_mean'] for b in result['buckets']]}")
    assert result["knee_bucket_max"] <= 32, (
        f"substrate-primary knee expected ≤ 32; got "
        f"{result['knee_bucket_max']}. "
        f"Curve: {[b['bpb_mean'] for b in result['buckets']]}")


def test_effective_context_produces_monotone_nonincreasing_bpb(
        substrate_primary_ckpt):
    """The bpb curve must be nonincreasing (longer context ≤ shorter context).
    If a newer architecture produces a rising curve, something is wrong with
    the measurement or the model.
    """
    from heinrich.profile.compare import _cb_effective_context

    ckpt, val = substrate_primary_ckpt
    result = _cb_effective_context(
        model_path=ckpt,
        val=val,
        seqlen=128,
        n_trials=5,
        buckets=[1, 2, 4, 8, 16, 32, 64, 128],
        knee_threshold=0.01,
    )
    bpbs = [b["bpb_mean"] for b in result["buckets"]]
    # Allow a small rise of up to 0.01 bpb between adjacent buckets —
    # measurement noise at n_trials=5. Larger rises indicate a problem.
    for i in range(len(bpbs) - 1):
        assert bpbs[i + 1] - bpbs[i] <= 0.01, (
            f"bpb rose from {bpbs[i]:.4f} to {bpbs[i+1]:.4f} between "
            f"buckets {i} and {i+1}; expected nonincreasing curve")
```

- [ ] **Step 2: Run the test without env vars**

Run: `pytest tests/test_cb_effective_context.py -v`
Expected: 2 tests SKIPPED (no HEINRICH_TEST_CKPT set). No failures.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cb_effective_context.py
git commit -m "test(cb): regression anchor for profile-cb-effective-context knee at 16 bytes"
```

---

## Task 5: T2 — Ablation context managers and `_cb_ablations` helper

**Files:**
- Modify: `src/heinrich/profile/compare.py` (append after `_cb_effective_context` helpers)
- Test: `tests/test_cb_forensics_tools.py` (append)

- [ ] **Step 1: Write failing unit tests for ablation dispatch and bpb math**

Append to `tests/test_cb_forensics_tools.py`:

```python
def test_parse_ablation_spec_handles_three_modes():
    from heinrich.profile.compare import _parse_ablation_spec
    assert _parse_ablation_spec("substrate") == ("substrate", None)
    assert _parse_ablation_spec("local") == ("local", None)
    assert _parse_ablation_spec("truncate:32") == ("truncate", 32)
    with pytest.raises(ValueError, match="truncate requires"):
        _parse_ablation_spec("truncate")
    with pytest.raises(ValueError, match="unknown ablation"):
        _parse_ablation_spec("silly")


def test_compute_bpb_from_logits_matches_manual_cross_entropy():
    """bpb = -log2(p(correct)) averaged over target positions.

    For a 2-token vocab with uniform logits, bpb = 1.0 exactly.
    """
    from heinrich.profile.compare import _compute_bpb_over_sequences

    # Uniform logits: log-softmax over 2 classes gives -log(2); bpb = 1.0.
    vocab = 2
    seqs = np.array([[0, 1, 0, 1, 0]], dtype=np.int64)

    class _Uniform:
        class config:
            vocab_size = vocab
        def forward(self, seq):
            _, seqlen = seq.shape
            return np.zeros((1, seqlen, vocab), dtype=np.float32)

    bpb = _compute_bpb_over_sequences(_Uniform(), seqs)
    assert abs(bpb - 1.0) < 1e-4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cb_forensics_tools.py::test_parse_ablation_spec_handles_three_modes tests/test_cb_forensics_tools.py::test_compute_bpb_from_logits_matches_manual_cross_entropy -v`
Expected: FAIL — `_parse_ablation_spec` and `_compute_bpb_over_sequences` do not exist.

- [ ] **Step 3: Implement the helpers**

Append to `src/heinrich/profile/compare.py` after `_cb_effective_context` / `_logsumexp`:

```python
def _parse_ablation_spec(spec: str) -> tuple[str, int | None]:
    """Parse ``--ablate`` spec into (mode, arg)."""
    if spec in ("substrate", "local"):
        return spec, None
    if spec.startswith("truncate:"):
        k_str = spec[len("truncate:"):]
        if not k_str:
            raise ValueError("truncate requires a rank: truncate:K")
        try:
            return "truncate", int(k_str)
        except ValueError as e:
            raise ValueError(f"truncate requires integer K; got {k_str!r}") from e
    if spec == "truncate":
        raise ValueError("truncate requires a rank: truncate:K")
    raise ValueError(f"unknown ablation: {spec!r}; expected "
                      f"substrate|local|truncate:K")


def _compute_bpb_over_sequences(backend, seqs: np.ndarray) -> float:
    """Mean bpb over all next-token predictions in seqs.

    seqs: [n_seqs, seqlen] int64. Returns a single scalar bpb.
    """
    total_nll = 0.0
    total_n = 0
    for i in range(seqs.shape[0]):
        seq_i = seqs[i:i + 1]
        logits = np.asarray(backend.forward(seq_i), dtype=np.float32)[0]
        log_probs = logits - _logsumexp(logits, axis=-1, keepdims=True)
        targets = seqs[i].astype(np.int64)
        for t in range(seqs.shape[1] - 1):
            total_nll -= float(log_probs[t, targets[t + 1]])
            total_n += 1
    if total_n == 0:
        raise ValueError("no predictions collected")
    ce_nats = total_nll / total_n
    return ce_nats / float(np.log(2.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cb_forensics_tools.py -v`
Expected: all previous tests + new two PASS.

- [ ] **Step 5: Write the context-manager and top-level helper tests**

Append to `tests/test_cb_forensics_tools.py`:

```python
def test_ablate_local_restores_on_exit():
    """_ablate_local monkey-patches model._local_logits and restores
    even if the body raises."""
    from heinrich.profile.compare import _ablate_local

    class _Model:
        def _local_logits(self, *a, **kw):
            return "ORIGINAL"

    m = _Model()
    orig = m._local_logits
    try:
        with _ablate_local(m):
            assert m._local_logits is not orig
            # Inside ablation, local path returns zeros-like (we check that
            # the patched callable is NOT the original).
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert m._local_logits is orig, "local_logits not restored after exception"


def test_cb_ablations_dispatches_and_computes_delta(monkeypatch):
    """End-to-end with fake backend: baseline and local-ablated runs
    produce a reported delta."""
    from heinrich.profile import compare as cmp_mod

    class _FakeModel:
        def _local_logits(self, *a, **kw):
            return np.zeros((1, 8, 4), dtype=np.float32)

    class _FakeBackend:
        class config:
            vocab_size = 4
            model_type = "causal_bank"
        def __init__(self):
            self.model = _FakeModel()
            self._call = 0
        def forward(self, seq):
            # First set of calls = baseline (bpb=1.5); second set = ablated (bpb=2.0).
            self._call += 1
            _, seqlen = seq.shape
            if self._call <= 2:  # baseline passes
                peak = 2.0  # sharp
            else:  # ablated passes
                peak = 0.5  # less sharp
            logits = np.full((1, seqlen, 4), -peak * 0.1, dtype=np.float32)
            for t in range(seqlen):
                logits[0, t, int(seq[0, t])] = peak
            return logits

    fake_backend = _FakeBackend()
    monkeypatch.setattr(cmp_mod, "_load_effective_context_backend",
                         lambda *a, **kw: fake_backend)
    monkeypatch.setattr(cmp_mod, "_load_effective_context_val",
                         lambda *a, **kw:
                         np.array([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=np.int64))

    result = cmp_mod._cb_ablations(
        model_path="IGNORED",
        ablate="local",
        val=None,
        n_tokens=8,
    )
    assert result["ablation"] == "local"
    assert "baseline_bpb" in result
    assert "ablated_bpb" in result
    assert result["delta_bpb"] == round(
        result["ablated_bpb"] - result["baseline_bpb"], 4)
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_cb_forensics_tools.py::test_ablate_local_restores_on_exit tests/test_cb_forensics_tools.py::test_cb_ablations_dispatches_and_computes_delta -v`
Expected: FAIL — `_ablate_local` and `_cb_ablations` don't exist.

- [ ] **Step 7: Implement context managers and top-level helper**

Append to `compare.py`:

```python
from contextlib import contextmanager


@contextmanager
def _ablate_local(model):
    """Replace model._local_logits with a zero-returning stub; restore on exit."""
    if not hasattr(model, "_local_logits"):
        raise AttributeError(
            "model has no _local_logits; local ablation not supported on this model")
    orig = model._local_logits

    def _zero_local(*args, **kwargs):
        out = orig(*args, **kwargs)
        return np.zeros_like(np.asarray(out))

    model._local_logits = _zero_local
    try:
        yield
    finally:
        model._local_logits = orig


@contextmanager
def _ablate_substrate(model):
    """Zero the substrate contribution to logits.

    Implementation strategy: wrap model._linear_logits (adaptive-substrate
    path) and zero the substrate slice of its _last_features input before
    readout. For non-adaptive models, zero _last_states on the module
    before the linear readout.

    Raises AttributeError on unrecognized model shapes so the caller sees
    a clean error rather than silent no-op.
    """
    patched_any = False
    restoration = []

    if hasattr(model, "_linear_logits"):
        orig_ll = model._linear_logits

        def _zeroed_substrate_ll(*args, **kwargs):
            # Before the readout, zero the substrate slice of _last_features.
            if hasattr(model, "_last_features") and model._last_features is not None:
                f = model._last_features
                n_modes = getattr(model.config, "n_modes", None)
                if n_modes is not None:
                    f[..., :n_modes] = 0
            return orig_ll(*args, **kwargs)

        model._linear_logits = _zeroed_substrate_ll
        restoration.append(("_linear_logits", orig_ll))
        patched_any = True

    if hasattr(model, "_last_states_nonadaptive"):
        # Non-adaptive path zeroing is done at forward time by wrapping the
        # readout method. Use _linear_states if present.
        pass  # handled via _linear_logits above for most cases

    if not patched_any:
        raise AttributeError(
            "model shape not recognized for substrate ablation: "
            "no _linear_logits method")

    try:
        yield
    finally:
        for attr, orig in restoration:
            setattr(model, attr, orig)


@contextmanager
def _ablate_truncate(model, k: int):
    """Zero substrate modes at indices ≥ k before the readout."""
    if not hasattr(model, "_linear_logits"):
        raise AttributeError(
            "model has no _linear_logits; truncate ablation unsupported")
    orig_ll = model._linear_logits
    n_modes = getattr(model.config, "n_modes", None)
    if n_modes is None:
        raise AttributeError("model.config has no n_modes; truncate unsupported")
    if k < 0 or k > n_modes:
        raise ValueError(f"truncate K={k} out of range [0, {n_modes}]")

    def _truncated_ll(*args, **kwargs):
        if hasattr(model, "_last_features") and model._last_features is not None:
            f = model._last_features
            f[..., k:n_modes] = 0
        return orig_ll(*args, **kwargs)

    model._linear_logits = _truncated_ll
    try:
        yield
    finally:
        model._linear_logits = orig_ll


def _cb_ablations(model_path: str,
                  *,
                  ablate: str,
                  val: str | None,
                  n_tokens: int,
                  result_json: str | None = None,
                  tokenizer_path: str | None = None) -> dict:
    """Measure per-path bpb contribution.

    Modes:
      substrate  — zero substrate contribution; local path only
      local      — zero local contribution; substrate path only
      truncate:K — keep only substrate modes [0..K)
    """
    mode, k = _parse_ablation_spec(ablate)
    backend = _load_effective_context_backend(
        model_path, result_json, tokenizer_path)
    vocab_size = int(backend.config.vocab_size)
    # Sequence count and length chosen so n_tokens predictions are produced.
    seqlen = 256
    n_seqs = max(1, int(n_tokens) // (seqlen - 1) + 1)
    seqs = _load_effective_context_val(val, seqlen, n_seqs, vocab_size)

    baseline_bpb = _compute_bpb_over_sequences(backend, seqs)

    model = backend.model
    if mode == "local":
        ctx = _ablate_local(model)
    elif mode == "substrate":
        ctx = _ablate_substrate(model)
    elif mode == "truncate":
        ctx = _ablate_truncate(model, k)
    else:  # pragma: no cover — _parse_ablation_spec raises first
        raise ValueError(f"unknown mode: {mode!r}")

    with ctx:
        ablated_bpb = _compute_bpb_over_sequences(backend, seqs)

    delta = ablated_bpb - baseline_bpb
    multiplier = (ablated_bpb / baseline_bpb) if baseline_bpb > 0 else float("inf")
    return {
        "model": model_path,
        "ablation": ablate,
        "n_tokens": int((seqlen - 1) * n_seqs),
        "baseline_bpb": round(baseline_bpb, 4),
        "ablated_bpb": round(ablated_bpb, 4),
        "delta_bpb": round(delta, 4),
        "multiplier": round(multiplier, 3),
    }
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_cb_forensics_tools.py -v`
Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
git add src/heinrich/profile/compare.py tests/test_cb_forensics_tools.py
git commit -m "feat(cb): _cb_ablations helper + context-manager patches for three ablation modes"
```

---

## Task 6: T2 — CLI parser and dispatch for `profile-cb-ablations`

**Files:**
- Modify: `src/heinrich/cli.py`

- [ ] **Step 1: Add the CLI parser**

In `build_parser`, after the `p_cb_effctx` block from Task 2, insert:

```python
    p_cb_abl = sub.add_parser("profile-cb-ablations",
                              help="Per-path bpb contribution: substrate/local/truncate ablations")
    p_cb_abl.add_argument("--model", required=True, help="Checkpoint path (.checkpoint.pt)")
    p_cb_abl.add_argument("--ablate", required=True,
                          help="Ablation mode: substrate | local | truncate:K")
    p_cb_abl.add_argument("--val", default=None, help="Val bytes file (optional)")
    p_cb_abl.add_argument("--n-tokens", type=int, default=50000,
                          help="Number of token predictions to measure over (default: 50000)")
    p_cb_abl.add_argument("--result-json", default=None, help="Path to result.json")
    p_cb_abl.add_argument("--tokenizer-path", default=None, help="Path to tokenizer model")
```

- [ ] **Step 2: Add the dispatch branch**

After `profile-cb-effective-context` dispatch from Task 2:

```python
    elif args.command == "profile-cb-ablations":
        _cmd_cb_ablations(args)
```

- [ ] **Step 3: Add the command function**

After `_cmd_cb_effective_context` from Task 2:

```python
def _cmd_cb_ablations(args: argparse.Namespace) -> None:
    """Ablation forensics: substrate/local/truncate path contributions to bpb."""
    from .profile.compare import _cb_ablations

    result = _cb_ablations(
        model_path=args.model,
        ablate=args.ablate,
        val=args.val,
        n_tokens=args.n_tokens,
        result_json=getattr(args, "result_json", None),
        tokenizer_path=getattr(args, "tokenizer_path", None),
    )

    def _fmt(r: dict) -> None:
        print(f"\n=== Ablation forensics for {r['model']} ===\n")
        print(f"  ablation: {r['ablation']}")
        print(f"  n_tokens: {r['n_tokens']}\n")
        print(f"  baseline:   {r['baseline_bpb']:.4f} bpb")
        print(f"  ablated:    {r['ablated_bpb']:.4f} bpb")
        sign = "+" if r["delta_bpb"] >= 0 else ""
        print(f"  delta:     {sign}{r['delta_bpb']:.4f} bpb  "
              f"({r['multiplier']:.3f}×)\n")

    _json_or(args, result, _fmt)
```

- [ ] **Step 4: Smoke-test the CLI help**

Run: `python -m heinrich.cli profile-cb-ablations --help`
Expected: argparse prints usage.

- [ ] **Step 5: Commit**

```bash
git add src/heinrich/cli.py
git commit -m "feat(cli): profile-cb-ablations command + dispatch"
```

---

## Task 7: T2 — MCP tool definition and dispatch

**Files:**
- Modify: `src/heinrich/mcp.py`

- [ ] **Step 1: Add tool definition**

After `heinrich_cb_effective_context` in `TOOLS`:

```python
    "heinrich_cb_ablations": {
        "description": "Ablation forensics: substrate/local/truncate path contributions to bpb. Needs model.",
        "parameters": {
            "model": {"type": "string", "description": "Checkpoint path (.checkpoint.pt)", "required": True},
            "ablate": {"type": "string", "description": "substrate | local | truncate:K", "required": True},
            "val": {"type": "string", "description": "Val bytes file (optional)"},
            "n_tokens": {"type": "integer", "description": "Number of token predictions (default: 50000)"},
        },
    },
```

- [ ] **Step 2: Add dispatch branch**

After `heinrich_cb_effective_context` dispatch:

```python
        if name == "heinrich_cb_ablations":
            return self._do_subprocess(arguments, "profile-cb-ablations",
                ["--model", arguments["model"], "--ablate", arguments["ablate"]],
                optional={"val": "--val", "n_tokens": "--n-tokens"},
                timeout=1800)
```

- [ ] **Step 3: Smoke-test tool registry**

Run: `python -c "from heinrich.mcp import TOOLS; print('heinrich_cb_ablations' in TOOLS)"`
Expected: `True`

- [ ] **Step 4: Commit**

```bash
git add src/heinrich/mcp.py
git commit -m "feat(mcp): heinrich_cb_ablations tool definition + dispatch"
```

---

## Task 8: T6 — `--svd-samples N` on `profile-cb-additivity`

**Files:**
- Modify: `src/heinrich/profile/compare.py` (`_cb_additivity_metrics`, `causal_bank_additivity`)
- Modify: `src/heinrich/cli.py` (`p_cb_add`, `_cmd_cb_additivity`)
- Modify: `src/heinrich/mcp.py` (`heinrich_cb_additivity` optional arg)
- Test: `tests/test_cb_forensics_tools.py`

- [ ] **Step 1: Write a failing test for the new parameter**

Append to `tests/test_cb_forensics_tools.py`:

```python
def test_cb_additivity_metrics_accepts_svd_samples(monkeypatch, tmp_path):
    """_cb_additivity_metrics passes svd_samples through to the SVD call."""
    from heinrich.profile import compare as cmp_mod

    observed_sample_sizes = []
    orig_svd = cmp_mod.np.linalg.svd

    def _tracking_svd(x, *a, **kw):
        observed_sample_sizes.append(x.shape[0])
        return orig_svd(x, *a, **kw)

    monkeypatch.setattr(cmp_mod.np.linalg, "svd", _tracking_svd)

    # Fake MRI: make load_mri / causal_bank_loss return a valid substrate.
    def _fake_causal_bank_loss(mri_path):
        return {"overall_bpb": 1.78}

    def _fake_load_mri(mri_path):
        rng = np.random.default_rng(0)
        return {
            "substrate_states": rng.standard_normal(
                (10, 20, 8)).astype(np.float32),
            "loss": None,
        }

    monkeypatch.setattr(cmp_mod, "causal_bank_loss", _fake_causal_bank_loss)
    import heinrich.profile.mri as mri_mod
    monkeypatch.setattr(mri_mod, "load_mri", _fake_load_mri)

    _ = cmp_mod._cb_additivity_metrics("IGNORED", svd_samples=100)
    # SVD called at least once; first call must be on at most 100 rows.
    assert any(n <= 100 for n in observed_sample_sizes)

    observed_sample_sizes.clear()
    _ = cmp_mod._cb_additivity_metrics("IGNORED", svd_samples=5000)
    assert all(n <= 5000 for n in observed_sample_sizes)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_cb_forensics_tools.py::test_cb_additivity_metrics_accepts_svd_samples -v`
Expected: FAIL — TypeError about unexpected keyword argument `svd_samples`.

- [ ] **Step 3: Add `svd_samples` to `_cb_additivity_metrics`**

Open `src/heinrich/profile/compare.py`. Locate `_cb_additivity_metrics` (starts around line 4959). Change its signature from:

```python
def _cb_additivity_metrics(mri_path: str) -> dict:
```

to:

```python
def _cb_additivity_metrics(mri_path: str, *, svd_samples: int = 5000) -> dict:
```

In the body, locate the hardcoded SVD sample line (approximately line 4990):

```python
    _, S, _ = np.linalg.svd(sub_c[:5000], full_matrices=False)
```

Replace with:

```python
    svd_n = max(1, min(int(svd_samples), len(sub_c)))
    _, S, _ = np.linalg.svd(sub_c[:svd_n], full_matrices=False)
```

Add `"svd_samples_used": svd_n` to the returned dict:

```python
    return {
        "bpb": bpb,
        "eff_dim": eff_dim,
        "pos_r2": pos_r2,
        "cont_r2": cont_r2,
        "active_frac": active_frac,
        "svd_samples_used": svd_n,
    }
```

- [ ] **Step 4: Plumb `svd_samples` through `causal_bank_additivity`**

Locate the signature:

```python
def causal_bank_additivity(baseline_mri: str,
                            mutation_mris: list[str],
                            combination_mri: str,
                            *, noise_floor: float = 0.004,
                            metrics: tuple[str, ...] | None = None,
                            noise_floors: dict[str, float] | None = None) -> dict:
```

Change to:

```python
def causal_bank_additivity(baseline_mri: str,
                            mutation_mris: list[str],
                            combination_mri: str,
                            *, noise_floor: float = 0.004,
                            metrics: tuple[str, ...] | None = None,
                            noise_floors: dict[str, float] | None = None,
                            svd_samples: int = 5000) -> dict:
```

Find the `_metrics_for` helper inside the function:

```python
    def _metrics_for(mri: str) -> dict:
        return _cb_additivity_metrics(mri)
```

Change to:

```python
    def _metrics_for(mri: str) -> dict:
        return _cb_additivity_metrics(mri, svd_samples=svd_samples)
```

- [ ] **Step 5: Add CLI flag**

Open `src/heinrich/cli.py`. Locate the `p_cb_add = sub.add_parser("profile-cb-additivity", ...)` block (around line 388). In its `add_argument` calls, add:

```python
    p_cb_add.add_argument("--svd-samples", type=int, default=5000,
                           help="Rows sampled per SVD for tail-PC position R² (default: 5000)")
```

In `_cmd_cb_additivity` (around line 3129), locate the `causal_bank_additivity` call and add `svd_samples=args.svd_samples`:

```python
        result = causal_bank_additivity(
            args.baseline, args.mutations, args.combination,
            noise_floor=args.noise_floor,
            metrics=tuple(args.metrics),
            noise_floors=noise_floors,
            svd_samples=args.svd_samples,
        )
```

- [ ] **Step 6: Add MCP optional arg**

Open `src/heinrich/mcp.py`. Locate `heinrich_cb_additivity` entry in `TOOLS`. In its `parameters` dict, add:

```python
            "svd_samples": {"type": "integer", "description": "SVD sample rows for tail-PC pos R² (default: 5000)"},
```

Locate the dispatch:

```python
        if name == "heinrich_cb_additivity":
            return self._do_subprocess(...)
```

Add `"svd_samples": "--svd-samples"` to the `optional` dict.

- [ ] **Step 7: Run the new and existing tests**

Run: `pytest tests/test_cb_forensics_tools.py -v`
Expected: all tests PASS (including the new svd_samples test).

- [ ] **Step 8: Commit**

```bash
git add src/heinrich/profile/compare.py src/heinrich/cli.py src/heinrich/mcp.py \
        tests/test_cb_forensics_tools.py
git commit -m "feat(cb): --svd-samples flag on profile-cb-additivity (T6)"
```

---

## Task 9: T7 — `--n-bootstrap K` on `profile-cb-pc-bands`

**Files:**
- Modify: `src/heinrich/profile/compare.py` (`causal_bank_pc_bands`)
- Modify: `src/heinrich/cli.py` (`p_cb_pcb`, `_cmd_cb_pc_bands`)
- Modify: `src/heinrich/mcp.py` (`heinrich_cb_pc_bands` optional arg)
- Test: `tests/test_cb_forensics_tools.py`

- [ ] **Step 1: Write a failing test for the bootstrap fields**

Append to `tests/test_cb_forensics_tools.py`:

```python
def test_causal_bank_pc_bands_bootstrap_reports_sem(monkeypatch, tmp_path):
    """n_bootstrap > 0 adds pos_r2_sem and pos_r2_samples to band reports."""
    from heinrich.profile import compare as cmp_mod

    # Minimal synthetic MRI with a clear 2-band structure: 8 content PCs
    # (high var, zero pos correlation) + 4 position PCs.
    rng = np.random.default_rng(0)
    n_seq, seq_len, D = 20, 64, 16
    sub = rng.standard_normal((n_seq, seq_len, D)).astype(np.float32)
    # Inject position-correlated signal on dims 8-11.
    pos = np.tile(np.arange(seq_len, dtype=np.float32), n_seq).reshape(
        n_seq, seq_len)
    sub[:, :, 8:12] += 5.0 * pos[:, :, None]

    mri_dir = tmp_path / "fake.seq.mri"
    mri_dir.mkdir()
    np.savez(str(mri_dir / "tokens.npz"),
              token_ids=rng.integers(0, 256, (n_seq, seq_len)).astype(np.int64))

    def _fake_load_mri(path):
        return {"substrate_states": sub}

    import heinrich.profile.mri as mri_mod
    monkeypatch.setattr(mri_mod, "load_mri", _fake_load_mri)

    result = cmp_mod.causal_bank_pc_bands(str(mri_dir), n_bootstrap=5)
    for band in result["bands"]:
        assert "pos_r2_sem" in band
        assert "pos_r2_samples" in band
        assert band["pos_r2_samples"] == 5
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_cb_forensics_tools.py::test_causal_bank_pc_bands_bootstrap_reports_sem -v`
Expected: FAIL — either `n_bootstrap` TypeError or missing sem field.

- [ ] **Step 3: Extend `causal_bank_pc_bands`**

Open `src/heinrich/profile/compare.py`. Locate `causal_bank_pc_bands` (line ~5259). Update its signature:

```python
def causal_bank_pc_bands(mri_path: str, *,
                          bands: tuple[tuple[int, int], ...] = ((0, 8), (8, 20), (20, 50), (50, 100)),
                          n_fit: int = 6000,
                          n_bootstrap: int = 0) -> dict:
```

Update the docstring to mention bootstrap. After the existing `band_pos_r2` function definition (around line 5317–5326), add a bootstrap variant:

```python
    def band_pos_r2_bootstrap(lo: int, hi: int, K: int) -> tuple[float, float, int]:
        """Run band_pos_r2 K times with different permutation seeds; return
        (mean_pos_r2, sem_pos_r2, K)."""
        samples = []
        hi_c = min(hi, n_pc)
        if lo >= hi_c:
            return 0.0, 0.0, 0
        X = proj[:, lo:hi_c]
        for seed in range(K):
            rng_b = np.random.default_rng(seed + 1)
            idx_b = rng_b.permutation(len(flat_c))
            tr_b = idx_b[:len(flat_c) // 2]
            te_b = idx_b[len(flat_c) // 2:]
            A = np.column_stack([X[tr_b], np.ones(len(tr_b))])
            coef, *_ = np.linalg.lstsq(A, flat_pos[tr_b], rcond=None)
            pred = X[te_b] @ coef[:-1] + coef[-1]
            samples.append(r2(flat_pos[te_b], pred))
        arr = np.asarray(samples, dtype=np.float64)
        mean = float(arr.mean())
        sem = float(arr.std(ddof=1) / np.sqrt(K)) if K > 1 else 0.0
        return mean, sem, K
```

Locate the band-building loop (lines 5343–5354):

```python
    band_results = []
    for lo, hi in bands:
        pos_r2, var_pct = band_pos_r2(lo, hi)
        byte_r2 = band_byte_r2(lo, hi)
        band_results.append({
            "range": f"{lo}-{min(hi, n_pc) - 1}",
            "lo": lo,
            "hi": min(hi, n_pc),
            "var_pct": round(var_pct, 4),
            "pos_r2": round(pos_r2, 6),
            "byte_r2": round(byte_r2, 6),
        })
```

Replace with:

```python
    band_results = []
    for lo, hi in bands:
        pos_r2, var_pct = band_pos_r2(lo, hi)
        byte_r2 = band_byte_r2(lo, hi)
        entry = {
            "range": f"{lo}-{min(hi, n_pc) - 1}",
            "lo": lo,
            "hi": min(hi, n_pc),
            "var_pct": round(var_pct, 4),
            "pos_r2": round(pos_r2, 6),
            "byte_r2": round(byte_r2, 6),
        }
        if n_bootstrap > 0:
            mean_br, sem_br, K = band_pos_r2_bootstrap(lo, hi, n_bootstrap)
            entry["pos_r2_mean"] = round(mean_br, 4)
            entry["pos_r2_sem"] = round(sem_br, 4)
            entry["pos_r2_samples"] = K
        else:
            entry["pos_r2_sem"] = 0.0
            entry["pos_r2_samples"] = 0
        band_results.append(entry)
```

The single-split `pos_r2` stays for back-compat; the bootstrap mean is surfaced as `pos_r2_mean` when bootstrap is on.

- [ ] **Step 4: Add CLI flag**

Open `src/heinrich/cli.py`. Locate `p_cb_pcb = sub.add_parser("profile-cb-pc-bands", ...)` (around line 405). Add:

```python
    p_cb_pcb.add_argument("--n-bootstrap", type=int, default=0,
                           help="Bootstrap K train/test splits for per-band pos_r2 ± SEM (default: 0 = off)")
```

In `_cmd_cb_pc_bands` (around line 3204), locate the call:

```python
        r = causal_bank_pc_bands(mri)
```

Change to:

```python
        r = causal_bank_pc_bands(mri, n_bootstrap=args.n_bootstrap)
```

- [ ] **Step 5: Add MCP optional arg**

Open `src/heinrich/mcp.py`. Locate `heinrich_cb_pc_bands` in `TOOLS`. Add to its `parameters`:

```python
            "n_bootstrap": {"type": "integer", "description": "Bootstrap K train/test splits per band (default: 0)"},
```

Add `"n_bootstrap": "--n-bootstrap"` to its dispatch's `optional` dict.

- [ ] **Step 6: Run all tests**

Run: `pytest tests/test_cb_forensics_tools.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/heinrich/profile/compare.py src/heinrich/cli.py src/heinrich/mcp.py \
        tests/test_cb_forensics_tools.py
git commit -m "feat(cb): --n-bootstrap flag on profile-cb-pc-bands (T7)"
```

---

## Task 10: T5 — `mri-recapture` path-relative fallback

**Files:**
- Modify: `src/heinrich/cli.py` (`_resolve_recapture_source`)
- Test: `tests/test_mri_recapture_paths.py` (create)

- [ ] **Step 1: Write failing tests for the new fallback**

Create `tests/test_mri_recapture_paths.py`:

```python
"""Unit tests for _resolve_recapture_source path-relative-to-sharts fallback (T5)."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_resolves_absolute_path_when_it_exists(tmp_path):
    from heinrich.cli import _resolve_recapture_source

    ckpt = tmp_path / "model.checkpoint.pt"
    ckpt.write_bytes(b"FAKE")
    mri_dir = tmp_path / "out.seq.mri"
    mri_dir.mkdir()
    md = {"provenance": {"model_path": str(ckpt)}}

    model_path, _, why = _resolve_recapture_source(mri_dir, md)
    assert model_path == str(ckpt)
    assert why is None


def test_path_relative_to_sharts_fallback(tmp_path):
    """If the recorded absolute path doesn't exist but the checkpoint exists
    at a new sharts mount, resolve via relative path."""
    from heinrich.cli import _resolve_recapture_source

    # Simulate a remount: new sharts root in tmp_path, but metadata has
    # an old absolute path /Volumes/OldSharts/session11/foo.checkpoint.pt
    # that doesn't exist here.
    new_sharts = tmp_path / "sharts"
    (new_sharts / "heinrich" / "session11").mkdir(parents=True)
    ckpt = new_sharts / "heinrich" / "session11" / "foo.checkpoint.pt"
    ckpt.write_bytes(b"FAKE")

    mri_dir = new_sharts / "heinrich" / "session11" / "foo.seq.mri"
    mri_dir.mkdir()
    md = {"provenance": {
        "model_path": "/Volumes/OldSharts/heinrich/session11/foo.checkpoint.pt",
    }}

    model_path, _, why = _resolve_recapture_source(mri_dir, md)
    assert model_path == str(ckpt), f"expected relative fallback; got {model_path} ({why})"


def test_path_relative_fallback_fails_cleanly(tmp_path):
    """If neither absolute nor relative-to-sharts resolves, fall through
    to legacy naming convention."""
    from heinrich.cli import _resolve_recapture_source

    mri_dir = tmp_path / "nope.seq.mri"
    mri_dir.mkdir()
    md = {"provenance": {
        "model_path": "/Volumes/OldSharts/nope.checkpoint.pt",
    }}

    model_path, _, why = _resolve_recapture_source(mri_dir, md)
    assert model_path is None
    assert why is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mri_recapture_paths.py -v`
Expected: first test PASS; second test FAIL (relative fallback doesn't exist); third test PASS.

- [ ] **Step 3: Implement the fallback**

Open `src/heinrich/cli.py`. Locate `_resolve_recapture_source` (line 2068). After the existing absolute-path check:

```python
    if model_path and _P(model_path).exists():
        return model_path, result_json, None
```

Insert:

```python
    # Path-relative-to-sharts fallback. If the recorded absolute path is
    # gone (different mount point, different host), try to find the same
    # checkpoint under the current MRI's sharts root. Works when both the
    # recorded path and the MRI directory share a "sharts" or "Sharts"
    # component in their path — we peel at that component and re-root at
    # the MRI's sharts.
    if model_path:
        old = _P(model_path)
        cur = mri_dir.resolve()

        def _sharts_split(p: _P) -> tuple[_P, _P] | None:
            """Split path into (root-inclusive-of-sharts, rest-after-sharts).
            Returns None if no sharts component is found."""
            parts = p.parts
            for i in range(len(parts) - 1, -1, -1):
                name = parts[i].lower()
                if name in ("sharts", "volumes") or name.endswith("sharts"):
                    root = _P(*parts[: i + 1])
                    rest = _P(*parts[i + 1:]) if i + 1 < len(parts) else _P()
                    return root, rest
            return None

        old_split = _sharts_split(old)
        cur_split = _sharts_split(cur)
        if old_split and cur_split:
            _, old_rest = old_split
            new_root, _ = cur_split
            candidate = new_root / old_rest
            if candidate.exists():
                import sys as _sys
                print(f"model_path rewritten: {old} → {candidate}",
                      file=_sys.stderr)
                # Also try to relocate result_json the same way.
                new_rj = None
                if result_json:
                    rj_old = _P(result_json)
                    rj_split = _sharts_split(rj_old)
                    if rj_split:
                        _, rj_rest = rj_split
                        rj_cand = new_root / rj_rest
                        if rj_cand.exists():
                            new_rj = str(rj_cand)
                return str(candidate), new_rj or result_json, None
```

Place this block between the existing `if model_path and _P(model_path).exists(): return ...` and the legacy fallback comment (`# Legacy fallback:`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mri_recapture_paths.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/heinrich/cli.py tests/test_mri_recapture_paths.py
git commit -m "feat(mri): _resolve_recapture_source path-relative-to-sharts fallback (T5)"
```

---

## Task 11: T8 — Silent partial-capture audit of `profile/compare.py`

**Files:**
- Modify: `src/heinrich/profile/compare.py` (targeted fixes)
- No new tests (audit is a review pass; any fix should come with a test of its own specific behavior).

- [ ] **Step 1: Inventory all `.get(...)` call sites on MRI-like dicts**

Run: `grep -n "mri\.get\(\|\.get(\"band_logits\|\.get(\"routing\|\.get(\"local\|\.get(\"half_lives\|\.get(\"substrate" src/heinrich/profile/compare.py | tee /tmp/t8_audit.txt`

This produces a plain-text line list. Expected lines (from initial scan):

- `line 3871: half_lives = mri.get('half_lives')`
- `line 4311: if 'routing' not in mri or mri.get('routing') is None:`
- `line 4385: if 'temporal_output' not in mri or mri.get('temporal_output') is None:`
- `line 4389: temporal_weights = mri.get('temporal_weights')`
- `line 4453: half_lives = mri.get('half_lives')`
- `line 4798: substrate = mri.get('substrate_states')`
- `line 4979: loss = mri.get('loss')`
- `line 5448: loss = mri.get('loss')`
- `line 5837–5844: gated_delta_{write,retain,erase}, overwrite_gate`
- `line 5853–5854: loss = mri.get('loss')`
- `line 6019: has_local = 'local_norm' in mri and mri.get('local_norm') is not None`

Some are already guarded (`if ... is None: return early`). Those stay as-is.

- [ ] **Step 2: Classify each call site**

For each line in `/tmp/t8_audit.txt`, read the surrounding 10 lines in `src/heinrich/profile/compare.py` and categorize:

- **R (requires-non-None):** downstream math uses the field unconditionally. Fix: add explicit `if X is None: raise ValueError(f"no <field> in MRI at {mri_path} — capture with fix_level ≥ <N>")` where `<N>` is the fix_level at which the field became standard (the implementer determines this from git history of capture.py / MRI_FIX_LEVEL bumps).

- **S (skip-valid):** the None is handled and skipping is the correct behavior. Leave as-is, or convert to an explicit structured-skip return for clarity.

- **F (already guarded):** explicit `is None` or `not in` check already present. Skip.

Expected distribution: most entries are category F (the codebase has matured). A few category R cases may exist in functions that assume sequence-mode MRIs with loss/routing (these are the silent-failure risk the spec calls out).

- [ ] **Step 3: Apply fixes inline**

For each category-R site identified in Step 2, patch in place. Keep changes minimal — one `raise` per site, no refactoring beyond the check.

Example patch format:

```python
    # Before
    loss = mri.get('loss')
    # ... downstream uses loss.shape, loss.mean() ...

    # After
    loss = mri.get('loss')
    if loss is None:
        raise ValueError(
            f"no 'loss' field in MRI at {mri_path} — requires a "
            f"sequence-mode capture (fix_level ≥ 1)")
    # ... downstream unchanged ...
```

- [ ] **Step 4: Run full test suite to catch regressions**

Run: `pytest tests/ -v --tb=short`
Expected: PASS (no regressions from the audit patches).

- [ ] **Step 5: Write the audit report inline in the commit message**

The commit message should list each site classified and the action taken. Example:

```
audit(compare): T8 — silent partial-capture fallthroughs

Classified 12 .get() sites on MRI dicts:
- F (already guarded, no change): line 3871, 4311, 4385, 4389, 5837-5844, 6019
- R (require-non-None, added raise): line 4798 (substrate), 4979 (loss), 5448 (loss), 5853 (loss)
- S (skip-valid, no change): line 4453 (half_lives — tools tolerate None)

Fix_level for each raised field:
- substrate_states, loss: require fix_level >= 1 (session-11 routing/loader fix)
```

- [ ] **Step 6: Commit**

```bash
git add src/heinrich/profile/compare.py
git commit -m "audit(compare): T8 — silent partial-capture fallthroughs report + fixes

<paste the audit summary from Step 5>"
```

---

## Final verification

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: all tests PASS. Integration test in `test_cb_effective_context.py` will skip unless `HEINRICH_TEST_CKPT` is set.

- [ ] **Step 2: Verify the CLI help for all new/updated commands**

Run:
```bash
python -m heinrich.cli profile-cb-effective-context --help
python -m heinrich.cli profile-cb-ablations --help
python -m heinrich.cli profile-cb-additivity --help | grep svd-samples
python -m heinrich.cli profile-cb-pc-bands --help | grep n-bootstrap
```
Expected: All four commands show expected flags.

- [ ] **Step 3: Verify the MCP tool registry has all entries**

Run:
```bash
python -c "from heinrich.mcp import TOOLS; \
  print('effective_context:', 'heinrich_cb_effective_context' in TOOLS); \
  print('ablations:', 'heinrich_cb_ablations' in TOOLS); \
  print('additivity svd:', 'svd_samples' in TOOLS['heinrich_cb_additivity']['parameters']); \
  print('pc-bands bootstrap:', 'n_bootstrap' in TOOLS['heinrich_cb_pc_bands']['parameters'])"
```
Expected: `True` on every line.

- [ ] **Step 4: Confirm `mri-recapture` still works on an existing MRI directory**

Run:
```bash
python -m heinrich.cli mri-recapture --dir /Volumes/sharts --dry-run 2>&1 | head -20
```
Expected: A dry-run plan lists (or skips) MRIs without crashing. If the volume isn't mounted, the "directory does not exist" message is correct behavior.
