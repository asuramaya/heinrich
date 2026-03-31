"""Tensor family classification and summarization."""
from __future__ import annotations
from typing import Any
import numpy as np

def classify_tensor_family(name: str) -> str:
    if "embed_tokens" in name: return "embed"
    if name.startswith("lm_head"): return "lm_head"
    if ".mlp.experts." in name:
        if ".down_proj." in name: return "mlp_expert_down"
        if ".gate_proj." in name: return "mlp_expert_gate"
        if ".up_proj." in name: return "mlp_expert_up"
    if ".mlp.shared_experts." in name:
        if ".down_proj." in name: return "mlp_shared_down"
        if ".gate_proj." in name: return "mlp_shared_gate"
        if ".up_proj." in name: return "mlp_shared_up"
    if ".mlp.gate.weight" in name: return "mlp_gate"
    if ".mlp.gate.e_score_correction_bias" in name: return "mlp_gate_bias"
    if ".mlp.gate_proj." in name: return "mlp_gate_proj"
    if ".mlp.up_proj." in name: return "mlp_up_proj"
    if ".mlp.down_proj." in name: return "mlp_down_proj"
    if ".input_layernorm." in name: return "input_layernorm"
    if ".post_attention_layernorm." in name: return "post_attention_layernorm"
    if ".self_attn.kv_a_proj" in name: return "attn_kv_a"
    if ".self_attn.kv_b_proj." in name: return "attn_kv_b"
    if ".self_attn.o_proj." in name: return "attn_o"
    if ".self_attn.q_a_proj." in name: return "attn_q_a"
    if ".self_attn.q_b_proj." in name: return "attn_q_b"
    if ".self_attn.q_proj." in name: return "attn_q"
    if ".self_attn.k_proj." in name: return "attn_k"
    if ".self_attn.v_proj." in name: return "attn_v"
    if "norm" in name.lower(): return "norm"
    return "other"

def summarize_families(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        family = classify_tensor_family(str(item["name"]))
        grouped.setdefault(family, []).append(item)
    out: list[dict[str, Any]] = []
    for family, members in sorted(grouped.items()):
        sigma1_values = [float(m["spectral"]["sigma1"]) for m in members]
        fro_values = [float(m["spectral"]["fro_norm"]) for m in members]
        top = sorted(members, key=lambda m: float(m["spectral"]["sigma1"]), reverse=True)[:3]
        out.append({
            "family": family, "count": len(members),
            "mean_sigma1": float(np.mean(sigma1_values)),
            "max_sigma1": float(np.max(sigma1_values)),
            "mean_fro_norm": float(np.mean(fro_values)),
            "top_sigma1": [{"name": m["name"], "sigma1": float(m["spectral"]["sigma1"])} for m in top],
        })
    return out
