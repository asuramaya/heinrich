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


def test_cb_additivity_metrics_accepts_svd_samples(monkeypatch):
    """_cb_additivity_metrics passes svd_samples through to the SVD call."""
    from heinrich.profile import compare as cmp_mod

    observed_sample_sizes = []
    orig_svd = cmp_mod.np.linalg.svd

    def _tracking_svd(x, *a, **kw):
        observed_sample_sizes.append(x.shape[0])
        return orig_svd(x, *a, **kw)

    monkeypatch.setattr(cmp_mod.np.linalg, "svd", _tracking_svd)

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
    assert any(n <= 100 for n in observed_sample_sizes), \
        f"expected an SVD call on ≤100 rows; saw {observed_sample_sizes}"

    observed_sample_sizes.clear()
    _ = cmp_mod._cb_additivity_metrics("IGNORED", svd_samples=5000)
    assert all(n <= 5000 for n in observed_sample_sizes), \
        f"SVD calls exceeded 5000 rows: {observed_sample_sizes}"


def test_causal_bank_pc_bands_bootstrap_reports_sem(monkeypatch, tmp_path):
    """n_bootstrap > 0 adds pos_r2_sem and pos_r2_samples to band reports."""
    from heinrich.profile import compare as cmp_mod

    rng = np.random.default_rng(0)
    n_seq, seq_len, D = 20, 64, 16
    sub = rng.standard_normal((n_seq, seq_len, D)).astype(np.float32)
    pos = np.tile(np.arange(seq_len, dtype=np.float32),
                   n_seq).reshape(n_seq, seq_len)
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
    fitted = [b for b in result["bands"] if b["lo"] < b["hi"]]
    assert fitted, "no bands fitted for bootstrap"
    for band in fitted:
        assert "pos_r2_sem" in band
        assert "pos_r2_samples" in band
        assert band["pos_r2_samples"] == 5
    for band in result["bands"]:
        # All bands expose the fields even when they weren't fitted.
        assert "pos_r2_sem" in band
        assert "pos_r2_samples" in band


def test_causal_bank_pc_bands_bootstrap_off_by_default(monkeypatch, tmp_path):
    """With no --n-bootstrap, legacy fields unchanged (no sem reported)."""
    from heinrich.profile import compare as cmp_mod

    rng = np.random.default_rng(1)
    sub = rng.standard_normal((10, 32, 8)).astype(np.float32)

    mri_dir = tmp_path / "fake.seq.mri"
    mri_dir.mkdir()
    np.savez(str(mri_dir / "tokens.npz"),
              token_ids=rng.integers(0, 256, (10, 32)).astype(np.int64))

    def _fake_load_mri(path):
        return {"substrate_states": sub}

    import heinrich.profile.mri as mri_mod
    monkeypatch.setattr(mri_mod, "load_mri", _fake_load_mri)

    result = cmp_mod.causal_bank_pc_bands(str(mri_dir))
    for band in result["bands"]:
        # New fields are always present but flat/zero when bootstrap is off.
        assert band["pos_r2_samples"] == 0
        assert band["pos_r2_sem"] == 0.0
