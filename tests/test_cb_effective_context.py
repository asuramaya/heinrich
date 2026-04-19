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
