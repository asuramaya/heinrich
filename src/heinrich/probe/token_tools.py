"""Token-level analysis utilities — tokenizer loading, case rendering, token diffing."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .trigger_core import normalize_case


def load_tokenizer(tokenizer_ref: str | Path, *, trust_remote_code: bool = True, use_fast: bool = True) -> Any:
    transformers = _import_transformers()
    return transformers.AutoTokenizer.from_pretrained(
        str(tokenizer_ref),
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
    )


def render_case_prompt(
    case: dict[str, Any],
    *,
    tokenizer: Any | None = None,
    add_generation_prompt: bool = True,
) -> dict[str, Any]:
    normalized = normalize_case(case, default_id="case")
    messages = [{"role": row["role"], "content": row["content"]} for row in normalized["messages"]]
    assistant_prefill = _assistant_prefill_text(normalized, messages, add_generation_prompt=add_generation_prompt)
    prompt_messages = messages[:-1] if assistant_prefill is not None else messages
    prompt_add_generation = True if assistant_prefill is not None else add_generation_prompt
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=prompt_add_generation)
        except TypeError:
            rendered = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        rendered_text = str(rendered)
        if assistant_prefill is not None:
            rendered_text = f"{rendered_text}{assistant_prefill}"
        return {
            "custom_id": normalized["custom_id"],
            "used_chat_template": True,
            "add_generation_prompt": bool(add_generation_prompt),
            "rendered_text": rendered_text,
            "messages": messages,
        }
    parts = [f"{row['role']}: {row['content']}" for row in prompt_messages]
    if prompt_add_generation:
        parts.append("assistant:")
    rendered_text = "\n".join(parts)
    if assistant_prefill is not None:
        rendered_text = f"{rendered_text}{assistant_prefill}"
    return {
        "custom_id": normalized["custom_id"],
        "used_chat_template": False,
        "add_generation_prompt": bool(add_generation_prompt),
        "rendered_text": rendered_text,
        "messages": messages,
    }


def tokenize_case(
    case: dict[str, Any],
    *,
    tokenizer_ref: str | Path | None = None,
    tokenizer: Any | None = None,
    trust_remote_code: bool = True,
    use_fast: bool = True,
    add_generation_prompt: bool = True,
    add_special_tokens: bool = False,
) -> dict[str, Any]:
    tokenizer_obj = tokenizer or load_tokenizer(tokenizer_ref or "", trust_remote_code=trust_remote_code, use_fast=use_fast)
    rendered = render_case_prompt(case, tokenizer=tokenizer_obj, add_generation_prompt=add_generation_prompt)
    encoded = tokenizer_obj(rendered["rendered_text"], add_special_tokens=add_special_tokens)
    token_ids = _extract_input_ids(encoded)
    token_pieces = [_safe_decode_piece(tokenizer_obj, token_id) for token_id in token_ids]
    return {
        "mode": "tokenprobe",
        "custom_id": rendered["custom_id"],
        "tokenizer_name": _tokenizer_name(tokenizer_obj),
        "used_chat_template": bool(rendered["used_chat_template"]),
        "add_generation_prompt": bool(rendered["add_generation_prompt"]),
        "rendered_text": rendered["rendered_text"],
        "char_count": len(rendered["rendered_text"]),
        "token_count": len(token_ids),
        "token_ids": token_ids,
        "token_pieces": token_pieces,
        "messages": rendered["messages"],
    }


def compare_case_tokenization(
    lhs_case: dict[str, Any],
    rhs_case: dict[str, Any],
    *,
    tokenizer_ref: str | Path | None = None,
    tokenizer: Any | None = None,
    trust_remote_code: bool = True,
    use_fast: bool = True,
    add_generation_prompt: bool = True,
    add_special_tokens: bool = False,
) -> dict[str, Any]:
    tokenizer_obj = tokenizer or load_tokenizer(tokenizer_ref or "", trust_remote_code=trust_remote_code, use_fast=use_fast)
    lhs = tokenize_case(
        lhs_case,
        tokenizer=tokenizer_obj,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
        add_generation_prompt=add_generation_prompt,
        add_special_tokens=add_special_tokens,
    )
    rhs = tokenize_case(
        rhs_case,
        tokenizer=tokenizer_obj,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
        add_generation_prompt=add_generation_prompt,
        add_special_tokens=add_special_tokens,
    )
    prefix = 0
    while prefix < min(len(lhs["token_ids"]), len(rhs["token_ids"])) and lhs["token_ids"][prefix] == rhs["token_ids"][prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < len(lhs["token_ids"]) - prefix
        and suffix < len(rhs["token_ids"]) - prefix
        and lhs["token_ids"][-1 - suffix] == rhs["token_ids"][-1 - suffix]
    ):
        suffix += 1
    return {
        "mode": "tokdiff",
        "tokenizer_name": lhs["tokenizer_name"],
        "lhs": lhs,
        "rhs": rhs,
        "common_prefix_tokens": int(prefix),
        "common_suffix_tokens": int(suffix),
        "lhs_unique_span": lhs["token_ids"][prefix : len(lhs["token_ids"]) - suffix if suffix else len(lhs["token_ids"])],
        "rhs_unique_span": rhs["token_ids"][prefix : len(rhs["token_ids"]) - suffix if suffix else len(rhs["token_ids"])],
        "lhs_unique_pieces": lhs["token_pieces"][prefix : len(lhs["token_pieces"]) - suffix if suffix else len(lhs["token_pieces"])],
        "rhs_unique_pieces": rhs["token_pieces"][prefix : len(rhs["token_pieces"]) - suffix if suffix else len(rhs["token_pieces"])],
        "first_diff_index": None if prefix == min(len(lhs["token_ids"]), len(rhs["token_ids"])) and len(lhs["token_ids"]) == len(rhs["token_ids"]) else int(prefix),
    }


def _safe_decode_piece(tokenizer: Any, token_id: int) -> str:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            piece = tokenizer.convert_ids_to_tokens(int(token_id))
            if isinstance(piece, str):
                return piece
        except Exception:
            pass
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        return str(token_id)


def _extract_input_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        value = encoded.get("input_ids")
    else:
        value = getattr(encoded, "input_ids", None)
    if value is None:
        raise ValueError("Tokenizer encode result did not expose input_ids")
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def _assistant_prefill_text(
    normalized: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> str | None:
    if add_generation_prompt:
        return None
    if not messages or str(messages[-1]["role"]) != "assistant":
        return None
    metadata = dict(normalized.get("metadata", {}))
    if "assistant_prefix" not in metadata:
        return None
    return str(messages[-1]["content"])


def _tokenizer_name(tokenizer: Any) -> str:
    for attr in ("name_or_path", "name"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(tokenizer).__name__


def _import_transformers():
    try:
        import transformers
    except ImportError as exc:
        raise ImportError("transformers is required for tokenizer analysis; install heinrich[local] or pip install transformers") from exc
    return transformers
