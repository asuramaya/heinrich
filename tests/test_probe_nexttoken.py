"""Tests for probe/nexttoken.py (probe_next_token_distribution)."""
import pytest
from heinrich.probe.nexttoken import probe_next_token_distribution, _entropy_bits
import numpy as np


def test_entropy_bits_uniform():
    probs = np.full(4, 0.25)
    h = _entropy_bits(probs)
    assert h == pytest.approx(2.0)


def test_entropy_bits_certain():
    probs = np.array([1.0, 0.0, 0.0])
    h = _entropy_bits(probs)
    assert h == pytest.approx(0.0)


def test_entropy_bits_empty():
    probs = np.zeros(4)
    assert _entropy_bits(probs) == 0.0


def test_probe_next_token_distribution_requires_torch():
    case = {"messages": [{"role": "user", "content": "hello"}]}
    with pytest.raises((ImportError, OSError, Exception)):
        probe_next_token_distribution(
            case,
            tokenizer_ref="nonexistent-model-xyz",
        )
