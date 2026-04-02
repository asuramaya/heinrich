"""Tests for heinrich.cartography.templates."""
from heinrich.cartography.templates import (
    FRAMINGS, SHART_TOKENS, CHEMISTRY_CONTEXT,
    build_prompt, build_multiturn, inject_sharts,
)


class TestBuildPrompt:
    def test_direct(self):
        p = build_prompt("How are you?")
        assert p.startswith("<|im_start|>user\n")
        assert p.endswith("<|im_start|>assistant\n")
        assert "How are you?" in p

    def test_debug_framing(self):
        p = build_prompt("test query", framing="debug")
        assert "Find errors" in p
        assert "test query" in p

    def test_injection(self):
        p = build_prompt("query", injection="PREFIX ")
        assert "PREFIX " in p

    def test_all_framings_valid(self):
        for name in FRAMINGS:
            p = build_prompt("test", framing=name)
            assert "<|im_start|>" in p


class TestBuildMultiturn:
    def test_two_turns(self):
        turns = [("Hello", "Hi there"), ("Question", None)]
        p = build_multiturn(turns)
        assert p.count("<|im_start|>user") == 2
        assert p.count("<|im_start|>assistant") == 2
        assert p.endswith("<|im_start|>assistant\n")

    def test_single_turn(self):
        p = build_multiturn([("Query", None)])
        assert p.endswith("<|im_start|>assistant\n")


class TestInjectSharts:
    def test_single(self):
        result = inject_sharts("query", ["june4"])
        assert "六四" in result
        assert "query" in result

    def test_multiple(self):
        result = inject_sharts("query", ["june4", "uyghur"])
        assert "六四" in result
        assert "维吾尔" in result

    def test_none(self):
        result = inject_sharts("query", ["none"])
        assert result == "query"

    def test_empty_list(self):
        result = inject_sharts("query", [])
        assert result == "query"


class TestConstants:
    def test_framings_not_empty(self):
        assert len(FRAMINGS) > 5

    def test_shart_tokens_have_none(self):
        assert SHART_TOKENS["none"] == ""

    def test_chemistry_context_nonempty(self):
        assert len(CHEMISTRY_CONTEXT) > 50
