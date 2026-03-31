"""Tests for probe/token_tools.py — port from conker_detect/token_tools.py."""
from __future__ import annotations

import pytest
from heinrich.probe.token_tools import (
    render_case_prompt,
    _safe_decode_piece,
    _tokenizer_name,
)


def _make_case(content: str, custom_id: str = "test") -> dict:
    return {"custom_id": custom_id, "messages": [{"role": "user", "content": content}]}


class FakeTokenizer:
    """Minimal tokenizer stub that does not apply a chat template."""
    name_or_path = "fake-tokenizer"

    def __call__(self, text, add_special_tokens=False):
        tokens = text.split()
        return {"input_ids": list(range(len(tokens)))}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"tok{token_id}"

    def decode(self, ids, skip_special_tokens=False) -> str:
        return " ".join(str(i) for i in ids)


def test_render_case_prompt_no_tokenizer():
    case = _make_case("Hello world")
    result = render_case_prompt(case)
    assert result["custom_id"] == "test"
    assert "Hello world" in result["rendered_text"]
    assert result["used_chat_template"] is False


def test_render_case_prompt_with_tokenizer():
    tokenizer = FakeTokenizer()
    case = _make_case("Hello")
    result = render_case_prompt(case, tokenizer=tokenizer)
    # No apply_chat_template on stub, falls back to plain
    assert result["custom_id"] == "test"
    assert "Hello" in result["rendered_text"]


def test_safe_decode_piece_convert_ids():
    class TokWithConvert:
        def convert_ids_to_tokens(self, tid):
            return f"<piece_{tid}>"
    piece = _safe_decode_piece(TokWithConvert(), 42)
    assert piece == "<piece_42>"


def test_safe_decode_piece_fallback():
    class TokDecodeOnly:
        def decode(self, ids, skip_special_tokens=False):
            return "decoded"
    piece = _safe_decode_piece(TokDecodeOnly(), 0)
    assert piece == "decoded"


def test_tokenizer_name_uses_name_or_path():
    class T:
        name_or_path = "my-model"
    assert _tokenizer_name(T()) == "my-model"


def test_tokenizer_name_fallback_to_class():
    class UnnamedTokenizer:
        pass
    assert _tokenizer_name(UnnamedTokenizer()) == "UnnamedTokenizer"
