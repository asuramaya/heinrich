"""Prompt line suites — inject adversarial chain clauses into probe cases."""
from __future__ import annotations

from typing import Any

from .trigger_core import normalize_case


DEFAULT_PROMPT_LINE_CLAUSES: dict[str, str] = {
    "FALSIFY": "Prefer falsifier.",
    "ASYMMETRY": "Refuse identity authority.",
    "CRITERION": "Offload computation, not criterion.",
    "QUESTION": "What did it avoid?",
    "PROPOSAL": "Prefer artifact.",
    "COMPRESS": "Compress computation.",
    "YIELD": "Prefer explicit stop.",
}

PROMPT_LINE_PLACEMENTS = ("prefix", "suffix", "system")


def build_prompt_line_suite(
    case: dict[str, Any],
    *,
    lines: list[str] | tuple[str, ...] | None = None,
    placement: str = "prefix",
    separator: str = "\n",
) -> dict[str, Any]:
    normalized = normalize_case(case, default_id="case")
    if placement not in PROMPT_LINE_PLACEMENTS:
        raise ValueError(f"Unknown prompt-line placement: {placement}")
    ordered_lines = list(lines or DEFAULT_PROMPT_LINE_CLAUSES.keys())
    if not ordered_lines:
        raise ValueError("build_prompt_line_suite requires at least one prompt line")
    variants: list[dict[str, Any]] = []
    for line in ordered_lines:
        key = str(line).strip()
        if not key:
            continue
        clause = DEFAULT_PROMPT_LINE_CLAUSES.get(key, key)
        variants.append(
            {
                "variant_id": f"{placement}::{key}",
                "line": key,
                "clause": clause,
                "placement": placement,
                "description": f"{placement} prompt line {key}",
                "case": _apply_prompt_line(normalized, key, clause, placement=placement, separator=separator),
            }
        )
    return {
        "mode": "chainsuite",
        "base_case": normalized,
        "placement": placement,
        "separator": separator,
        "lines": ordered_lines,
        "variant_count": len(variants),
        "variants": variants,
    }


def _apply_prompt_line(
    case: dict[str, Any],
    line: str,
    clause: str,
    *,
    placement: str,
    separator: str,
) -> dict[str, Any]:
    tag = f"{line}: {clause}"
    cloned = {
        "custom_id": f"{case['custom_id']}::{placement}-{line}",
        "messages": [dict(row) for row in case["messages"]],
        "module_names": list(case.get("module_names", [])),
        "metadata": dict(case.get("metadata", {})),
    }
    if placement == "system":
        cloned["messages"] = [{"role": "system", "content": tag}, *cloned["messages"]]
        return cloned
    last = dict(cloned["messages"][-1])
    if placement == "prefix":
        last["content"] = f"{tag}{separator}{last['content']}"
    else:
        last["content"] = f"{last['content']}{separator}{tag}"
    cloned["messages"][-1] = last
    return cloned
