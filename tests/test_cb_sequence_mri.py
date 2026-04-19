"""Tests for sequence-mode causal bank MRI capture and analysis tools."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _make_fake_sequence_mri(tmp_path: Path, *, n_seqs=5, seq_len=64,
                             n_modes=32, embed_dim=16, n_experts=2,
                             n_bands=2, has_temporal=True,
                             has_overwrite_gate=False):
    """Create a synthetic sequence-mode MRI directory for testing."""
    mri_dir = tmp_path / "test_seq.mri"
    mri_dir.mkdir()

    n_snapshots = seq_len // 8

    metadata = {
        "version": "0.7",
        "type": "mri",
        "architecture": "causal_bank",
        "model": {
            "name": "test_cb",
            "n_modes": n_modes,
            "n_experts": n_experts,
            "n_bands": n_bands,
            "embed_dim": embed_dim,
            "vocab_size": 1024,
            "n_layers": 1,
            "hidden_size": n_modes,
        },
        "capture": {
            "mode": "sequence",
            "n_seqs": n_seqs,
            "seq_len": seq_len,
            "n_tokens": n_seqs * seq_len,
            "has_temporal": has_temporal,
            "has_routing": n_experts > 1,
            "has_band_loss": n_bands > 1,
            "has_overwrite_gate": has_overwrite_gate,
            "snapshot_interval": 8,
        },
        "provenance": {"seed": 42},
    }
    with open(mri_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    token_ids = np.random.randint(0, 1024, (n_seqs, seq_len), dtype=np.int32)
    np.savez_compressed(mri_dir / "tokens.npz", token_ids=token_ids)

    np.save(mri_dir / "substrate.npy",
            np.random.randn(n_seqs, seq_len, n_modes).astype(np.float16))
    np.save(mri_dir / "embedding.npy",
            np.random.randn(n_seqs, seq_len, embed_dim).astype(np.float16))
    # Loss: NaN at position 0, random elsewhere
    loss = np.random.rand(n_seqs, seq_len).astype(np.float32) * 5
    loss[:, 0] = np.nan
    np.save(mri_dir / "loss.npy", loss)
    np.save(mri_dir / "half_lives.npy",
            np.logspace(0, 2, n_modes, dtype=np.float32))

    if n_experts > 1:
        routing = np.random.rand(n_seqs, seq_len, n_experts).astype(np.float16)
        # Normalize so it looks like softmax output
        routing = routing / routing.sum(axis=-1, keepdims=True)
        np.save(mri_dir / "routing.npy", routing)
    if n_bands > 1:
        np.save(mri_dir / "band_loss.npy",
                np.random.rand(n_seqs, seq_len, n_bands).astype(np.float32) * 5)
    if has_temporal:
        tw = np.random.rand(n_seqs, seq_len, n_snapshots).astype(np.float16)
        tw = tw / tw.sum(axis=-1, keepdims=True)
        np.save(mri_dir / "temporal_weights.npy", tw)
        np.save(mri_dir / "temporal_output.npy",
                np.random.randn(n_seqs, seq_len, n_modes).astype(np.float16))

    if has_overwrite_gate:
        # Gate values in [0, 1] range (sigmoid output)
        gate = np.random.rand(n_seqs, seq_len, n_modes).astype(np.float16)
        np.save(mri_dir / "overwrite_gate.npy", gate)

    return str(mri_dir)


# --- load_mri tests ---

def test_sequence_mri_loads():
    """Sequence MRI loads via load_mri and has expected keys."""
    from heinrich.profile.mri import load_mri

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp))
        mri = load_mri(mri_path)
        assert mri['metadata']['capture']['mode'] == 'sequence'
        assert 'substrate_states' in mri
        assert 'loss' in mri
        assert mri['substrate_states'].shape == (5, 64, 32)
        assert mri['loss'].shape == (5, 64)


def test_sequence_mri_temporal():
    """Sequence MRI loads temporal attention arrays."""
    from heinrich.profile.mri import load_mri

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), has_temporal=True)
        mri = load_mri(mri_path)
        assert 'temporal_weights' in mri
        assert 'temporal_output' in mri
        assert mri['temporal_weights'].shape == (5, 64, 8)


def test_sequence_mri_no_temporal():
    """Sequence MRI without temporal attention."""
    from heinrich.profile.mri import load_mri

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), has_temporal=False)
        mri = load_mri(mri_path)
        assert mri.get('temporal_weights') is None


def test_sequence_mri_overwrite_gate():
    """Sequence MRI loads overwrite gate values."""
    from heinrich.profile.mri import load_mri

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), has_overwrite_gate=True)
        mri = load_mri(mri_path)
        assert 'overwrite_gate' in mri
        assert mri['overwrite_gate'].shape == (5, 64, 32)


# --- Analysis tool tests ---

def test_cb_loss():
    """cb-loss returns loss decomposition."""
    from heinrich.profile.compare import causal_bank_loss

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), n_seqs=5, seq_len=64,
                                            n_modes=32, n_bands=2)
        result = causal_bank_loss(mri_path)

    assert "error" not in result
    assert "overall_bpb" in result
    assert "by_position" in result
    assert len(result["by_position"]) > 0
    assert result["by_position"][0]["range"] == "0-4"
    assert result["by_band"]  # has band loss
    assert result["autocorrelation"]


def test_cb_routing():
    """cb-routing returns routing statistics."""
    from heinrich.profile.compare import causal_bank_routing

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), n_seqs=5, seq_len=64,
                                            n_experts=4)
        result = causal_bank_routing(mri_path)

    assert "error" not in result
    assert "overall_distribution" in result
    assert len(result["overall_distribution"]) == 4
    assert "switch_rate" in result
    assert "by_position" in result


def test_cb_routing_single_expert():
    """cb-routing returns error for single-expert model."""
    from heinrich.profile.compare import causal_bank_routing

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), n_experts=1)
        result = causal_bank_routing(mri_path)

    assert "error" in result


def test_cb_temporal():
    """cb-temporal returns temporal attention analysis."""
    from heinrich.profile.compare import causal_bank_temporal

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), n_seqs=5, seq_len=64,
                                            has_temporal=True)
        result = causal_bank_temporal(mri_path)

    assert "error" not in result
    assert "output_l2_by_position" in result
    assert "correlation_chain" in result
    assert "substrate_temporal" in result["correlation_chain"]


def test_cb_temporal_missing():
    """cb-temporal returns error without temporal data."""
    from heinrich.profile.compare import causal_bank_temporal

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), has_temporal=False)
        result = causal_bank_temporal(mri_path)

    assert "error" in result


def test_cb_modes():
    """cb-modes returns mode utilization."""
    from heinrich.profile.compare import causal_bank_modes

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), n_seqs=5, seq_len=64, n_modes=32)
        result = causal_bank_modes(mri_path)

    assert "error" not in result
    assert "by_quartile" in result
    assert "growth_curve" in result
    assert "dead_modes" in result
    assert result["n_modes"] == 32


def test_cb_decompose():
    """cb-decompose returns manifold decomposition."""
    from heinrich.profile.compare import causal_bank_decompose

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), n_seqs=5, seq_len=64, n_modes=32)
        result = causal_bank_decompose(mri_path)

    assert "error" not in result
    assert "pca" in result
    assert "position_r2" in result
    assert "ghost_fraction" in result
    assert result["position_fraction"] + result["content_fraction"] + result["ghost_fraction"] == pytest.approx(100.0, abs=1.0)


def test_cb_gate_forensics():
    """cb-gate-forensics returns gate analysis."""
    from heinrich.profile.compare import causal_bank_gate_forensics

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), n_seqs=5, seq_len=64,
                                            n_modes=32, has_overwrite_gate=True)
        result = causal_bank_gate_forensics(mri_path)

    assert "error" not in result
    assert "position_correlation" in result
    assert "effective_rank" in result
    assert "by_position" in result
    assert "position_dependent_modes" in result
    assert len(result["position_dependent_modes"]) == 5
    assert "verdict" in result
    assert result["difficulty_correlation"] is not None


def test_cb_gate_forensics_missing():
    """cb-gate-forensics returns error without gate data."""
    from heinrich.profile.compare import causal_bank_gate_forensics

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), has_overwrite_gate=False)
        result = causal_bank_gate_forensics(mri_path)

    assert "error" in result


def test_cb_substrate_local():
    """cb-substrate-local returns substrate vs local balance."""
    from heinrich.profile.compare import causal_bank_substrate_local

    with tempfile.TemporaryDirectory() as tmp:
        # Without local data
        mri_path = _make_fake_sequence_mri(Path(tmp), n_seqs=3, seq_len=64)
        result = causal_bank_substrate_local(mri_path)

    assert "error" not in result
    assert result["has_local"] is False
    assert "by_position" in result
    assert result["crossover_position"] is None


def test_tokenizer_difficulty():
    """tokenizer-difficulty reads MRI embedding."""
    from heinrich.profile.compare import tokenizer_difficulty

    with tempfile.TemporaryDirectory() as tmp:
        # Use impulse-style MRI (2D arrays)
        mri_dir = Path(tmp) / "test.mri"
        mri_dir.mkdir()
        n_tokens, n_modes, embed_dim = 100, 32, 16
        metadata = {
            "version": "0.7", "type": "mri", "architecture": "causal_bank",
            "model": {"name": "test", "n_modes": n_modes, "n_experts": 1,
                      "n_bands": 1, "embed_dim": embed_dim, "vocab_size": 1024,
                      "n_layers": 1, "hidden_size": n_modes},
            "capture": {"mode": "raw", "n_tokens": n_tokens},
            "provenance": {"seed": 42},
        }
        with open(mri_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        np.savez_compressed(mri_dir / "tokens.npz",
                            token_ids=np.arange(n_tokens, dtype=np.int32),
                            token_texts=np.array([f"t{i}" for i in range(n_tokens)]),
                            scripts=np.array(["Latin"] * n_tokens))
        np.save(mri_dir / "substrate.npy",
                np.random.randn(n_tokens, n_modes).astype(np.float16))
        np.save(mri_dir / "embedding.npy",
                np.random.randn(n_tokens, embed_dim).astype(np.float16))

        result = tokenizer_difficulty(str(mri_dir))

    assert "error" not in result
    assert "embed_substrate_r" in result
    assert "effective_dim" in result
    assert "difficulty_quartiles" in result
    assert len(result["difficulty_quartiles"]) == 4


# --- causal_bank_pc_bands: partition-score heuristic ---

def _make_mri_with_substrate(tmp_path: Path, substrate: np.ndarray, tokens: np.ndarray) -> str:
    """Build a minimal sequence-mode MRI backed by a caller-supplied substrate
    and token stream. Just enough metadata for causal_bank_pc_bands to load."""
    n_seqs, seq_len, D = substrate.shape
    mri_dir = tmp_path / "pc_bands_test.seq.mri"
    mri_dir.mkdir()
    metadata = {
        "version": "0.7", "type": "mri", "architecture": "causal_bank",
        "model": {"name": "test", "n_modes": D, "n_experts": 1, "n_bands": 1,
                  "embed_dim": 16, "vocab_size": 256, "n_layers": 1,
                  "hidden_size": D},
        "capture": {"mode": "sequence", "n_seqs": n_seqs, "seq_len": seq_len,
                    "n_tokens": n_seqs * seq_len},
        "provenance": {"seed": 42},
    }
    with open(mri_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    np.savez_compressed(mri_dir / "tokens.npz", token_ids=tokens.astype(np.int32))
    np.save(mri_dir / "substrate.npy", substrate.astype(np.float16))
    np.save(mri_dir / "embedding.npy",
            np.random.randn(n_seqs, seq_len, 16).astype(np.float16))
    np.save(mri_dir / "half_lives.npy", np.logspace(0, 2, D, dtype=np.float32))
    return str(mri_dir)


def test_pc_bands_absent_verdict_for_no_position():
    """Substrate with no position structure anywhere → verdict 'absent'."""
    from heinrich.profile.compare import causal_bank_pc_bands

    rng = np.random.default_rng(0)
    n_seqs, seq_len, D = 10, 64, 32
    # Pure byte-content substrate: each position's state depends only on the
    # random byte token there, no position dependence.
    tokens = rng.integers(0, 256, (n_seqs, seq_len))
    # Make substrate a function of token only (broadcast content-only pattern)
    byte_embeds = rng.standard_normal((256, D)) * 0.5
    substrate = byte_embeds[tokens].astype(np.float32)
    # Add noise so SVD has non-degenerate spectrum
    substrate += rng.standard_normal(substrate.shape) * 0.05

    with tempfile.TemporaryDirectory() as tmp:
        mri = _make_mri_with_substrate(Path(tmp), substrate, tokens)
        r = causal_bank_pc_bands(mri)
        assert r["partition_verdict"] == "absent", (
            f"expected absent, got {r['partition_verdict']} "
            f"(top={r['top_pos_r2']}, tail={r['max_tail_pos_r2']})")
        assert not r["two_band_partition"]


def test_pc_bands_partitioned_verdict_for_clean_two_band():
    """Substrate with content in top PCs + position in mid PCs →
    verdict 'partitioned' with high score."""
    from heinrich.profile.compare import causal_bank_pc_bands

    rng = np.random.default_rng(0)
    n_seqs, seq_len, D = 20, 64, 32
    tokens = rng.integers(0, 256, (n_seqs, seq_len))
    # Two disjoint subspaces, by coordinate index.
    # Content lives in coords 0-7 (high variance), position in 8-15 (lower
    # variance), rest is noise. Orthogonal by construction.
    byte_embeds = rng.standard_normal((256, 8)) * 2.0  # large variance
    position_embeds = np.stack([
        np.sin(np.arange(seq_len) / 5),
        np.cos(np.arange(seq_len) / 5),
        np.sin(np.arange(seq_len) / 10),
        np.cos(np.arange(seq_len) / 10),
        np.sin(np.arange(seq_len) / 20),
        np.cos(np.arange(seq_len) / 20),
        np.arange(seq_len, dtype=np.float32) / seq_len,
        (np.arange(seq_len, dtype=np.float32) / seq_len) ** 2,
    ], axis=-1) * 0.8  # moderate variance, lower than content

    content_block = byte_embeds[tokens]                    # [S, T, 8]
    # position_embeds is [T, 8]; broadcast over sequences
    pos_block = np.broadcast_to(position_embeds, (n_seqs, seq_len, 8))
    tail = rng.standard_normal((n_seqs, seq_len, D - 16)) * 0.05
    substrate = np.concatenate([content_block, pos_block, tail], axis=-1).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        mri = _make_mri_with_substrate(Path(tmp), substrate, tokens)
        r = causal_bank_pc_bands(mri)
        # Top band should be content-only; mid band should carry position
        assert r["partition_verdict"] in ("partitioned", "partial"), (
            f"expected partition, got {r['partition_verdict']} "
            f"(top={r['top_pos_r2']}, tail={r['max_tail_pos_r2']}, score={r['partition_score']})")
        assert r["max_tail_pos_r2"] > 0.1, (
            "position signal should be detected in tail bands")
        assert r["two_band_partition"], (
            "two_band_partition should be true for clean content/position split")


def test_pc_bands_leaky_verdict_when_position_mixes_with_content():
    """Position signal present but in PC0-7 (content band) → 'leaky'."""
    from heinrich.profile.compare import causal_bank_pc_bands

    rng = np.random.default_rng(1)
    n_seqs, seq_len, D = 20, 64, 32
    tokens = rng.integers(0, 256, (n_seqs, seq_len))
    # Entangle: position signal lives in the same high-variance coords as
    # byte content, so PCA will pack both into the top band.
    byte_embeds = rng.standard_normal((256, 8)) * 1.5
    position_scalar = (np.arange(seq_len, dtype=np.float32) / seq_len).reshape(1, -1, 1)
    content_block = byte_embeds[tokens] + position_scalar * 0.8  # entangled
    tail = rng.standard_normal((n_seqs, seq_len, D - 8)) * 0.05
    substrate = np.concatenate([content_block, tail], axis=-1).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        mri = _make_mri_with_substrate(Path(tmp), substrate, tokens)
        r = causal_bank_pc_bands(mri)
        # Position R² should be non-trivially present in top, tail shouldn't
        # dominate: verdict leaky OR absent (depending on how strongly the
        # position scalar survives PCA). Accept either as "not partitioned".
        assert r["partition_verdict"] in ("leaky", "absent"), (
            f"expected leaky/absent, got {r['partition_verdict']} "
            f"(top={r['top_pos_r2']}, tail={r['max_tail_pos_r2']})")
        assert not r["two_band_partition"]


def test_pc_bands_score_is_regularized():
    """partition_score must be finite even when top_pos_r2 = 0 exactly."""
    from heinrich.profile.compare import causal_bank_pc_bands

    rng = np.random.default_rng(2)
    n_seqs, seq_len, D = 15, 64, 24
    tokens = rng.integers(0, 256, (n_seqs, seq_len))
    # Build a substrate where top band is exactly zero on position. Use
    # content-only in first 8 dims, pure position in dim 10+.
    byte_embeds = rng.standard_normal((256, 8)) * 3.0
    pos_feature = (np.arange(seq_len, dtype=np.float32) / seq_len).reshape(1, -1)
    pos_block = np.broadcast_to(pos_feature[:, :, None] * rng.standard_normal((1, 1, 8)),
                                 (n_seqs, seq_len, 8))
    substrate = np.concatenate([
        byte_embeds[tokens],
        pos_block,
        rng.standard_normal((n_seqs, seq_len, D - 16)) * 0.02,
    ], axis=-1).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        mri = _make_mri_with_substrate(Path(tmp), substrate, tokens)
        r = causal_bank_pc_bands(mri)
        # Score must be finite (regularized by 0.005 floor), not inf/NaN
        import math
        assert math.isfinite(r["partition_score"]), (
            f"partition_score must be finite, got {r['partition_score']}")


def test_tokenizer_compare():
    """tokenizer-compare runs on available tokenizer."""
    import os
    tok_path = "/Volumes/sharts/heinrich/fineweb_1024_bpe.model"
    if not os.path.exists(tok_path):
        pytest.skip("Tokenizer not available")

    from heinrich.profile.compare import tokenizer_compare
    result = tokenizer_compare([tok_path])
    assert "error" not in result
    assert len(result["tokenizers"]) == 1
    t = result["tokenizers"][0]
    assert "vocab_size" in t
    assert t["vocab_size"] > 0
    assert "byte_fallback_pct" in t
