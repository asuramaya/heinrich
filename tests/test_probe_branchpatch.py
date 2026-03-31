"""Tests for probe/branchpatch.py — activation patching helpers."""
from __future__ import annotations

import pytest
from heinrich.probe.branchpatch import _normalize_patch_mode


def test_normalize_patch_mode_replace():
    result = _normalize_patch_mode("replace")
    assert result == "replace"


def test_normalize_patch_mode_delta():
    result = _normalize_patch_mode("delta")
    assert result == "delta"


def test_normalize_patch_mode_zero():
    result = _normalize_patch_mode("zero")
    assert result == "zero"


def test_normalize_patch_mode_case_insensitive():
    result = _normalize_patch_mode("REPLACE")
    assert result == "replace"


def test_normalize_patch_mode_strips_whitespace():
    result = _normalize_patch_mode("  delta  ")
    assert result == "delta"


def test_normalize_patch_mode_invalid():
    with pytest.raises(ValueError):
        _normalize_patch_mode("unknown_mode_xyz")


def test_normalize_patch_mode_add_invalid():
    # "add" was valid in conker but not in heinrich's branchpatch
    with pytest.raises(ValueError):
        _normalize_patch_mode("add")
