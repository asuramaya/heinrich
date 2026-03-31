"""Tests for probe/vocab.py"""
import pytest
from unittest.mock import MagicMock, patch
from heinrich.probe.vocab import inspect_token_rows, compare_token_rows_across_models


def _make_mock_tokenizer(vocab_size: int = 50):
    """Create a minimal mock tokenizer for testing."""
    tok = MagicMock()
    tok.vocab_size = vocab_size
    tok.convert_ids_to_tokens = lambda ids: [f"tok_{i}" for i in ids]
    # model with embeddings
    model = MagicMock()
    embed = MagicMock()
    embed.weight = MagicMock()
    import numpy as np
    embed.weight.detach.return_value.cpu.return_value.float.return_value.numpy.return_value = (
        np.random.default_rng(0).standard_normal((vocab_size, 16)).astype("float32")
    )
    model.get_input_embeddings.return_value = embed
    return tok, model


def test_inspect_token_rows_basic():
    tok, model = _make_mock_tokenizer(50)
    with patch("heinrich.probe.vocab.load_tokenizer", return_value=tok):
        result = inspect_token_rows(
            model=model,
            tokenizer=tok,
            token_ids=[0, 1, 2],
        )
    assert isinstance(result, list)
    assert len(result) == 3
    for row in result:
        assert "token_id" in row


def test_inspect_token_rows_empty():
    tok, model = _make_mock_tokenizer(50)
    with patch("heinrich.probe.vocab.load_tokenizer", return_value=tok):
        result = inspect_token_rows(model=model, tokenizer=tok, token_ids=[])
    assert result == []


def test_compare_token_rows_across_models_identical():
    tok, model = _make_mock_tokenizer(50)
    with patch("heinrich.probe.vocab.load_tokenizer", return_value=tok):
        rows_a = inspect_token_rows(model=model, tokenizer=tok, token_ids=[0, 1])
        rows_b = inspect_token_rows(model=model, tokenizer=tok, token_ids=[0, 1])
    result = compare_token_rows_across_models(rows_a, rows_b)
    assert isinstance(result, dict)
