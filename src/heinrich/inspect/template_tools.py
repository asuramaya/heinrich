"""Template diff tools — compare rendered chat templates across cases."""
from __future__ import annotations

import difflib
import re
from typing import Any

from ..probe.token_tools import compare_case_tokenization, load_tokenizer, render_case_prompt
from ..probe.trigger_core import normalize_case


IDENTITY_STOPWORDS = {
    "hello", "hi", "hey", "human", "user", "assistant", "system",
    "please", "thanks",
}


def diff_rendered_templates(
    lhs_case: dict[str, Any] | None = None,
    rhs_case: dict[str, Any] | None = None,
    *,
    template_a: str | None = None,
    template_b: str | None = None,
    cases: list[dict[str, Any]] | None = None,
    tokenizer_ref: str | None = None,
    tokenizer: Any | None = None,
    trust_remote_code: bool = True,
    use_fast: bool = True,
    add_generation_prompt: bool = True,
    add_special_tokens: bool = False,
) -> dict[str, Any]:
    # New-style API: template_a/template_b with cases list
    if template_a is not None or template_b is not None:
        return _diff_templates_batch(
            template_a=template_a or "",
            template_b=template_b or "",
            cases=cases or [],
        )
    # Legacy single-case API
    if lhs_case is None or rhs_case is None:
        raise ValueError("Either (lhs_case, rhs_case) or (template_a, template_b, cases) must be provided")
    tokenizer_obj = tokenizer or load_tokenizer(
        tokenizer_ref or "",
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
    )
    lhs = normalize_case(lhs_case, default_id="lhs")
    rhs = normalize_case(rhs_case, default_id="rhs")
    lhs_rendered = render_case_prompt(lhs, tokenizer=tokenizer_obj, add_generation_prompt=add_generation_prompt)
    rhs_rendered = render_case_prompt(rhs, tokenizer=tokenizer_obj, add_generation_prompt=add_generation_prompt)
    token_diff = compare_case_tokenization(
        lhs, rhs, tokenizer=tokenizer_obj,
        trust_remote_code=trust_remote_code, use_fast=use_fast,
        add_generation_prompt=add_generation_prompt, add_special_tokens=add_special_tokens,
    )
    lhs_text = str(lhs_rendered["rendered_text"])
    rhs_text = str(rhs_rendered["rendered_text"])
    prefix_chars = _common_prefix_length(lhs_text, rhs_text)
    suffix_chars = _common_suffix_length(lhs_text, rhs_text, prefix_chars)
    lhs_unique = lhs_text[prefix_chars : len(lhs_text) - suffix_chars if suffix_chars else len(lhs_text)]
    rhs_unique = rhs_text[prefix_chars : len(rhs_text) - suffix_chars if suffix_chars else len(rhs_text)]
    lhs_lines = lhs_text.splitlines()
    rhs_lines = rhs_text.splitlines()
    diff_lines = list(difflib.unified_diff(lhs_lines, rhs_lines, fromfile=lhs["custom_id"], tofile=rhs["custom_id"], lineterm=""))
    lhs_identity = _extract_system_identity(lhs_rendered["rendered_text"])
    rhs_identity = _extract_system_identity(rhs_rendered["rendered_text"])
    lhs_mentions = _extract_identity_mentions(lhs["messages"])
    rhs_mentions = _extract_identity_mentions(rhs["messages"])
    return {
        "mode": "templatediff",
        "tokenizer_name": getattr(tokenizer_obj, "name_or_path", type(tokenizer_obj).__name__),
        "lhs": {"custom_id": lhs["custom_id"], "rendered_text": lhs_text, "line_count": len(lhs_lines), "system_identity": lhs_identity, "identity_mentions": lhs_mentions},
        "rhs": {"custom_id": rhs["custom_id"], "rendered_text": rhs_text, "line_count": len(rhs_lines), "system_identity": rhs_identity, "identity_mentions": rhs_mentions},
        "text": {
            "common_prefix_chars": int(prefix_chars), "common_suffix_chars": int(suffix_chars),
            "lhs_unique_char_count": len(lhs_unique), "rhs_unique_char_count": len(rhs_unique),
            "lhs_unique_text": lhs_unique, "rhs_unique_text": rhs_unique,
            "lhs_unique_line_count": len([line for line in lhs_unique.splitlines() if line]),
            "rhs_unique_line_count": len([line for line in rhs_unique.splitlines() if line]),
            "line_diff_preview": diff_lines[:min(len(diff_lines), 24)],
        },
        "identity": {
            "lhs_conflict": bool(lhs_identity and any(name.lower() != lhs_identity.lower() for name in lhs_mentions)),
            "rhs_conflict": bool(rhs_identity and any(name.lower() != rhs_identity.lower() for name in rhs_mentions)),
            "shared_mentions": sorted(set(lhs_mentions) & set(rhs_mentions)),
        },
        "token_diff": token_diff,
    }


def _render_template(template: str, case: dict[str, Any]) -> str:
    """Render a template string by substituting {{key}} placeholders."""
    result = template
    for key, value in case.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result


def _diff_templates_batch(
    template_a: str,
    template_b: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare two template strings across a list of cases."""
    total_diffs = 0
    case_results: list[dict[str, Any]] = []
    for idx, case in enumerate(cases):
        rendered_a = _render_template(template_a, case)
        rendered_b = _render_template(template_b, case)
        lines_a = rendered_a.splitlines()
        lines_b = rendered_b.splitlines()
        diff_lines = list(difflib.unified_diff(lines_a, lines_b, fromfile=f"a[{idx}]", tofile=f"b[{idx}]", lineterm=""))
        has_diff = rendered_a != rendered_b
        if has_diff:
            total_diffs += 1
        case_results.append({
            "case_index": idx,
            "rendered_a": rendered_a,
            "rendered_b": rendered_b,
            "diff": diff_lines,
            "changed": has_diff,
        })
    return {
        "mode": "templatediff_batch",
        "template_a": template_a,
        "template_b": template_b,
        "case_count": len(cases),
        "total_diffs": total_diffs,
        "diff_count": total_diffs,
        "cases": case_results,
    }


def _common_prefix_length(lhs: str, rhs: str) -> int:
    size = min(len(lhs), len(rhs))
    index = 0
    while index < size and lhs[index] == rhs[index]:
        index += 1
    return index


def _common_suffix_length(lhs: str, rhs: str, prefix_chars: int) -> int:
    lhs_remaining = len(lhs) - prefix_chars
    rhs_remaining = len(rhs) - prefix_chars
    size = min(lhs_remaining, rhs_remaining)
    index = 0
    while index < size and lhs[-1 - index] == rhs[-1 - index]:
        index += 1
    return index


def _extract_system_identity(rendered_text: str) -> str | None:
    match = re.search(r"You are ([A-Za-z][A-Za-z0-9_-]{1,63})", str(rendered_text))
    if not match:
        return None
    return str(match.group(1))


def _extract_identity_mentions(messages: list[dict[str, str]]) -> list[str]:
    mentions: list[str] = []
    for row in messages:
        if row.get("role") != "user":
            continue
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{1,63}\b", row.get("content", "")):
            lowered = token.lower()
            if lowered in IDENTITY_STOPWORDS:
                continue
            if token not in mentions:
                mentions.append(token)
    return mentions
