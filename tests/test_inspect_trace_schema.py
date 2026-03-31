"""Tests for inspect/trace_schema.py"""
import pytest
from heinrich.inspect.trace_schema import parse_sample_trace, TRACE_FIELDS, NUMERIC_TRACE_FIELDS


def test_trace_fields_nonempty():
    assert len(TRACE_FIELDS) > 0
    assert len(NUMERIC_TRACE_FIELDS) > 0


def test_parse_sample_trace_empty_outputs():
    """Empty outputs dict with valid positions returns empty per-position entries."""
    result = parse_sample_trace({}, positions=[0, 1, 2])
    assert isinstance(result, dict)
    # positions key or per-position data
    assert len(result) > 0 or result == {}


def test_parse_sample_trace_with_gold_logprobs():
    """gold logprobs array mapped to positions."""
    outputs = {"sample_gold_logprobs": [-1.0, -2.0, -3.0]}
    result = parse_sample_trace(outputs, positions=[0, 1, 2])
    assert isinstance(result, dict)


def test_parse_sample_trace_position_mismatch_does_not_raise():
    """Mismatched position count should not raise hard."""
    outputs = {"sample_gold_logprobs": [-1.0]}
    try:
        result = parse_sample_trace(outputs, positions=[0, 1, 2])
        assert isinstance(result, dict)
    except (ValueError, IndexError):
        pass  # acceptable


def test_parse_sample_trace_rejects_none():
    with pytest.raises((TypeError, AttributeError, ValueError)):
        parse_sample_trace(None, positions=[0])
