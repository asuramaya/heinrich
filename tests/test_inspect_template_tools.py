"""Tests for inspect/template_tools.py"""
import pytest
from heinrich.inspect.template_tools import diff_rendered_templates


def test_diff_rendered_templates_identical():
    """Identical template strings produce no diffs."""
    result = diff_rendered_templates(
        template_a="You are a helpful assistant.\n{{prompt}}",
        template_b="You are a helpful assistant.\n{{prompt}}",
        cases=[{"prompt": "Hello"}],
    )
    assert isinstance(result, dict)
    # No diffs when identical
    total_diffs = result.get("total_diffs", result.get("diff_count", 0))
    assert total_diffs == 0


def test_diff_rendered_templates_different():
    """Different templates should produce at least one diff."""
    result = diff_rendered_templates(
        template_a="You are helpful.\n{{prompt}}",
        template_b="You are extremely helpful.\n{{prompt}}",
        cases=[{"prompt": "Hi"}],
    )
    assert isinstance(result, dict)
    total_diffs = result.get("total_diffs", result.get("diff_count", 0))
    assert total_diffs >= 1


def test_diff_rendered_templates_empty_cases():
    result = diff_rendered_templates(
        template_a="{{prompt}}",
        template_b="{{prompt}}",
        cases=[],
    )
    assert isinstance(result, dict)
