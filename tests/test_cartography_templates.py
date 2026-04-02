"""Tests for heinrich.cartography.templates."""
from heinrich.cartography.templates import (
    FRAMINGS, FRAMINGS_ZH, FRAMINGS_ALL, SHART_TOKENS, CHEMISTRY_CONTEXT,
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


class TestChineseFramings:
    def test_framings_zh_keys(self):
        assert "direct_zh" in FRAMINGS_ZH
        assert "academic_zh" in FRAMINGS_ZH
        assert "forensic_zh" in FRAMINGS_ZH
        assert "report_zh" in FRAMINGS_ZH
        assert "exam_zh" in FRAMINGS_ZH

    def test_framings_zh_contain_chinese(self):
        for name, tmpl in FRAMINGS_ZH.items():
            if name != "direct_zh":
                # Non-direct framings should contain Chinese characters
                assert any("\u4e00" <= ch <= "\u9fff" for ch in tmpl), f"{name} has no Chinese"

    def test_framings_all_merges_both(self):
        for name in FRAMINGS:
            assert name in FRAMINGS_ALL
        for name in FRAMINGS_ZH:
            assert name in FRAMINGS_ALL
        assert len(FRAMINGS_ALL) == len(FRAMINGS) + len(FRAMINGS_ZH)

    def test_build_prompt_language_zh(self):
        p = build_prompt("test", framing="academic_zh", language="zh")
        assert "学术论文" in p
        assert "test" in p

    def test_build_prompt_language_en_default(self):
        p = build_prompt("test", framing="academic")
        assert "Academic thesis" in p

    def test_build_prompt_language_all(self):
        # Should find both EN and ZH framings
        p_en = build_prompt("test", framing="academic", language="all")
        assert "Academic thesis" in p_en
        p_zh = build_prompt("test", framing="forensic_zh", language="all")
        assert "法医分析" in p_zh

    def test_build_prompt_unknown_framing_fallback(self):
        # Unknown framing should fall back to direct
        p = build_prompt("test", framing="nonexistent_framing")
        assert "test" in p

    def test_all_zh_framings_valid(self):
        for name in FRAMINGS_ZH:
            p = build_prompt("test", framing=name, language="zh")
            assert "<|im_start|>" in p

    def test_exam_zh_has_answer_marker(self):
        p = build_prompt("test", framing="exam_zh", language="zh")
        assert "答：" in p


class TestConstants:
    def test_framings_not_empty(self):
        assert len(FRAMINGS) > 5

    def test_shart_tokens_have_none(self):
        assert SHART_TOKENS["none"] == ""

    def test_chemistry_context_nonempty(self):
        assert len(CHEMISTRY_CONTEXT) > 50
