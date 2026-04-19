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
        self.config.model_type = "causal_bank"

    def forward(self, seq):
        """Peak logit at position t on seq[t+1] (the next-token target).
        Peakedness grows with t → bpb at deeper positions is lower."""
        _, seqlen = seq.shape
        logits = np.zeros((1, seqlen, self.config.vocab_size), dtype=np.float32)
        for t in range(seqlen):
            next_t = min(t + 1, seqlen - 1)
            correct = int(seq[0, next_t])
            peakedness = 1.0 + np.log1p(t)
            logits[0, t, :] = -peakedness * 0.1
            logits[0, t, correct] = peakedness
        return logits


def test_cb_effective_context_decreasing_bpb_with_fake_backend(monkeypatch):
    """Helper runs end-to-end on a fake backend and produces a decreasing
    bpb curve + reports all fields."""
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
    bpbs = [b["bpb_mean"] for b in result["buckets"]]
    assert bpbs == sorted(bpbs, reverse=True)
    assert "knee_bucket_min" in result
    assert "saturation_bpb" in result
    assert result["saturation_bpb"] == bpbs[-1]


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


def test_ablate_local_restores_on_exit():
    """_ablate_local monkey-patches model._local_logits and restores
    even if the body raises. Verified by observing behavior change + revert.
    """
    from heinrich.profile.compare import _ablate_local

    class _Model:
        def _local_logits(self, *a, **kw):
            return np.ones((1, 4, 4), dtype=np.float32)

    m = _Model()
    # Before patch: returns ones.
    assert np.all(m._local_logits() == 1.0)
    try:
        with _ablate_local(m):
            # Inside: returns zeros.
            assert np.all(m._local_logits() == 0.0)
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # After exit (even with exception): returns ones again.
    assert np.all(m._local_logits() == 1.0), \
        "local_logits behavior not restored after exception"


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
            self._call += 1
            _, seqlen = seq.shape
            if self._call <= 2:
                peak = 2.0
            else:
                peak = 0.5
            logits = np.full((1, seqlen, 4), -peak * 0.1, dtype=np.float32)
            for t in range(seqlen):
                next_t = min(t + 1, seqlen - 1)
                logits[0, t, int(seq[0, next_t])] = peak
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
