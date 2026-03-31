"""Mechanism utilities -- JSON loading, family normalization, token overlap."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_json_source(path_or_raw: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_raw, dict):
        raw = path_or_raw
    else:
        path = Path(path_or_raw)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = json.loads(str(path_or_raw))
    if not isinstance(raw, dict):
        raise ValueError("Expected a JSON object")
    return raw


def source_label(path_or_raw: str | Path | dict[str, Any], *, fallback: str = "source") -> str:
    if isinstance(path_or_raw, dict):
        return str(path_or_raw.get("label") or path_or_raw.get("mode") or fallback)
    path = Path(path_or_raw)
    if path.exists():
        return path.stem
    return fallback


def normalize_mechanism_family(name: str) -> str:
    stem = str(name).strip()
    if not stem:
        return "unknown"
    lower = stem.lower()
    if ".self_attn.q_a_layernorm" in lower or ".self_attn.q_a_ln" in lower:
        return "attn_q_norm"
    if ".input_layernorm" in lower or ".post_attention_layernorm" in lower or ".layernorm" in lower:
        return "layernorm"
    if "self_attn.q_a_proj" in lower or "self_attn.q_b_proj" in lower or "self_attn.q_proj" in lower:
        return "attn_q"
    if "kv_a_proj" in lower or "kv_b_proj" in lower or "self_attn.k_proj" in lower or "self_attn.v_proj" in lower:
        return "attn_kv"
    if "self_attn.o_proj" in lower:
        return "attn_o"
    if "mlp.gate_proj" in lower or "mlp.gate." in lower or lower.endswith(".mlp.gate"):
        return "mlp_gate"
    if "mlp.up_proj" in lower:
        return "mlp_up"
    if "mlp.down_proj" in lower:
        return "mlp_down"
    if "shared_experts" in lower:
        return "moe_shared"
    if ".experts." in lower:
        return "moe_expert"
    if "router" in lower or "route" in lower:
        return "moe_route"
    if "embed_tokens" in lower or lower.endswith(".embedding") or ".embed." in lower:
        return "embedding"
    if "lm_head" in lower or "output" in lower:
        return "output"
    tail = stem.split(".")[-1]
    tail = re.sub(r"[^a-z0-9_]+", "_", tail.lower()).strip("_")
    return tail or "unknown"


def prompt_token_overlap(lhs: str, rhs: str) -> float:
    lhs_tokens = set(re.findall(r"[a-z0-9']+", str(lhs).lower()))
    rhs_tokens = set(re.findall(r"[a-z0-9']+", str(rhs).lower()))
    if not lhs_tokens or not rhs_tokens:
        return 0.0
    union = lhs_tokens | rhs_tokens
    if not union:
        return 0.0
    return float(len(lhs_tokens & rhs_tokens) / len(union))


def flatten_numbers(value: Any) -> list[float]:
    out: list[float] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, (int, float)):
            out.append(float(item))
    return out
