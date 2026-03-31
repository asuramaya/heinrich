"""Tests for probe/meta.py — port from conker_detect/meta_probes.py."""
from __future__ import annotations

import pytest
from heinrich.probe.meta import (
    build_meta_probe_suite,
    DEFAULT_META_PROBE_FAMILIES,
    DEFAULT_META_CHAIN_VARIANTS,
    META_PROBE_PLACEMENTS,
)


def _case(content: str = "What is going on?", cid: str = "test") -> dict:
    return {"custom_id": cid, "messages": [{"role": "user", "content": content}]}


def test_build_meta_probe_suite_all_families():
    suite = build_meta_probe_suite(_case(), chain_lines=[], include_chain_only=False, combine_chain_and_probe=False)
    assert suite["mode"] == "metasuite"
    total_prompts = sum(len(v) for v in DEFAULT_META_PROBE_FAMILIES.values())
    assert suite["variant_count"] == total_prompts


def test_build_meta_probe_suite_subset_families():
    suite = build_meta_probe_suite(_case(), families=["SELF"], chain_lines=[], include_chain_only=False, combine_chain_and_probe=False)
    assert suite["variant_count"] == len(DEFAULT_META_PROBE_FAMILIES["SELF"])
    for v in suite["variants"]:
        assert v["family"] == "SELF"


def test_build_meta_probe_suite_append_placement():
    case = _case("original content")
    suite = build_meta_probe_suite(
        case,
        families=["SELF"],
        chain_lines=[],
        probe_placement="append",
        include_chain_only=False,
        combine_chain_and_probe=False,
    )
    variant = suite["variants"][0]
    assert "original content" in variant["case"]["messages"][-1]["content"]
    assert variant["case"]["messages"][-1]["content"].startswith("original content")


def test_build_meta_probe_suite_replace_placement():
    case = _case("original content")
    suite = build_meta_probe_suite(
        case,
        families=["SELF"],
        chain_lines=[],
        probe_placement="replace",
        include_chain_only=False,
        combine_chain_and_probe=False,
    )
    variant = suite["variants"][0]
    assert "original content" not in variant["case"]["messages"][-1]["content"]


def test_build_meta_probe_suite_invalid_placement():
    with pytest.raises(ValueError, match="Unknown probe placement"):
        build_meta_probe_suite(_case(), probe_placement="sideways")


def test_build_meta_probe_suite_chain_only():
    # families=[] falls back to all DEFAULT families due to the `or` logic;
    # use families=None to get defaults, but pass a single-family list to isolate.
    # To get ONLY chain variants, disable probes entirely and only allow chain.
    # The cleanest approach: pass a known-empty sentinel via a list with no valid family.
    # Instead, use a families list with 0 entries but note `[] or keys` = keys.
    # Solution: just verify the chain variant is present among all variants.
    suite = build_meta_probe_suite(
        _case(),
        families=["SELF"],  # one family of probes
        chain_lines=["FALSIFY"],
        include_chain_only=True,
        combine_chain_and_probe=False,
    )
    # Should have SELF probes + 1 chain variant
    chain_variants = [v for v in suite["variants"] if v["probe_kind"] == "chain"]
    assert len(chain_variants) == 1
    assert chain_variants[0]["line"] == "FALSIFY"


def test_build_meta_probe_suite_combo():
    # With families=["SELF"], chain_lines=["FALSIFY"], include_chain_only=False,
    # combine_chain_and_probe=True:
    # - 4 probe variants (SELF) + 4 combo variants (SELF x FALSIFY)
    suite = build_meta_probe_suite(
        _case(),
        families=["SELF"],
        chain_lines=["FALSIFY"],
        include_chain_only=False,
        combine_chain_and_probe=True,
    )
    combo_variants = [v for v in suite["variants"] if v["probe_kind"] == "combo"]
    assert len(combo_variants) == len(DEFAULT_META_PROBE_FAMILIES["SELF"])
    for v in combo_variants:
        assert v["family"] == "SELF"
        assert v["line"] == "FALSIFY"
