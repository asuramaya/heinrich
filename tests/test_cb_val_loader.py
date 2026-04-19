"""Tests for causal bank validation data loading."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_load_val_sequences_from_bin():
    """Load uint16 .bin file and chunk into sequences."""
    from heinrich.backend.decepticon import load_val_sequences

    tokens = np.arange(2048, dtype=np.uint16)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tokens.tofile(f)
        f.flush()
        seqs = load_val_sequences(f.name, seq_len=512, n_seqs=3, seed=42)

    assert seqs.shape == (3, 512)
    assert seqs.dtype == np.int64
    assert not np.array_equal(seqs[0], seqs[1])


def test_load_val_sequences_short_data():
    """When data is shorter than n_seqs * seq_len, return what fits."""
    from heinrich.backend.decepticon import load_val_sequences

    tokens = np.arange(600, dtype=np.uint16)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tokens.tofile(f)
        f.flush()
        seqs = load_val_sequences(f.name, seq_len=512, n_seqs=10, seed=42)

    assert seqs.shape[0] == 1
    assert seqs.shape[1] == 512


def test_load_val_sequences_deterministic():
    """Same seed produces same sequences."""
    from heinrich.backend.decepticon import load_val_sequences

    tokens = np.arange(4096, dtype=np.uint16)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tokens.tofile(f)
        f.flush()
        a = load_val_sequences(f.name, seq_len=512, n_seqs=3, seed=42)
        b = load_val_sequences(f.name, seq_len=512, n_seqs=3, seed=42)

    np.testing.assert_array_equal(a, b)


def test_load_val_sequences_too_short():
    """Data shorter than one sequence raises ValueError."""
    from heinrich.backend.decepticon import load_val_sequences

    tokens = np.arange(100, dtype=np.uint16)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tokens.tofile(f)
        f.flush()
        with pytest.raises(ValueError, match="too short"):
            load_val_sequences(f.name, seq_len=512, n_seqs=1, seed=42)
