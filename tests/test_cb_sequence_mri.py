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
