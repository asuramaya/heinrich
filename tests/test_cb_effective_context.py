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
    """Adjacent-bucket rises must be explainable: either within noise
    (2x the combined across-sequence SEM) or part of the documented
    U-shape (session 13: universal in the causal-bank family — bpb falls
    to an optimum then degrades at longer context). A rise that is both
    early (before the optimum) and beyond noise indicates a measurement
    or model problem.
    """
    from heinrich.profile.compare import _cb_effective_context

    ckpt, val = substrate_primary_ckpt
    result = _cb_effective_context(
        model_path=ckpt,
        val=val,
        seqlen=128,
        n_trials=16,
        buckets=[1, 2, 4, 8, 16, 32, 64, 128],
        knee_threshold=0.01,
    )
    buckets = result["buckets"]
    bpbs = [b["bpb_mean"] for b in buckets]
    optimum_idx = min(range(len(bpbs)), key=lambda i: bpbs[i])
    for i in range(len(bpbs) - 1):
        rise = bpbs[i + 1] - bpbs[i]
        if rise <= 0.01:
            continue
        if i >= optimum_idx:
            continue  # U-shape territory: degradation past the optimum
        noise = 2.0 * (buckets[i]["bpb_sem"] ** 2
                       + buckets[i + 1]["bpb_sem"] ** 2) ** 0.5
        assert rise <= noise, (
            f"bpb rose {rise:.4f} from bucket {i} to {i+1} "
            f"({bpbs[i]:.4f} -> {bpbs[i+1]:.4f}), before the optimum "
            f"(bucket {optimum_idx}) and beyond 2x combined SEM "
            f"({noise:.4f}) — measurement or model problem")
