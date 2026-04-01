"""Capture and analyze attention patterns from transformer layers."""
from __future__ import annotations
from typing import Any
import numpy as np


def _mask_dtype_from_model(model: Any) -> Any:
    import mlx.core as mx
    inner = getattr(model, "model", model)
    embed = getattr(inner, "embed_tokens", None)
    if embed is not None and hasattr(embed, "weight") and embed.weight.dtype == mx.bfloat16:
        return mx.bfloat16
    return mx.float16


def capture_attention_maps(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layers: list[int] | None = None,
) -> dict[str, Any]:
    """Capture attention weight matrices for specified layers.

    Manually computes Q*K^T/sqrt(d) with RoPE and GQA expansion to get
    the exact attention patterns without relying on the fused kernel.

    Returns dict with attention_maps[layer] = np.array[n_heads, T, T],
    plus token metadata.
    """
    import mlx.core as mx

    inner = getattr(model, "model", model)
    token_ids = tokenizer.encode(prompt)
    input_ids = mx.array([token_ids])
    T = len(token_ids)

    mask_dt = _mask_dtype_from_model(model)
    causal_mask = mx.triu(mx.full((T, T), float("-inf"), dtype=mask_dt), k=1) if T > 1 else None

    token_strs = [tokenizer.decode([tid]) for tid in token_ids]

    if layers is None:
        layers = list(range(len(inner.layers)))
    layer_set = set(layers)

    attention_maps: dict[int, np.ndarray] = {}
    h = inner.embed_tokens(input_ids)

    for i, layer_mod in enumerate(inner.layers):
        if i in layer_set:
            attn = layer_mod.self_attn
            h_normed = layer_mod.input_layernorm(h)

            q = attn.q_proj(h_normed)  # [1, T, n_heads * head_dim]
            k = attn.k_proj(h_normed)  # [1, T, n_kv_heads * head_dim]

            n_heads = attn.n_heads
            n_kv_heads = attn.n_kv_heads
            head_dim = q.shape[-1] // n_heads

            # [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
            q = q.reshape(1, T, n_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(1, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

            # Apply RoPE (same as model's forward path)
            q = attn.rope(q)
            k = attn.rope(k)

            # Expand KV heads for GQA
            n_rep = n_heads // n_kv_heads
            if n_rep > 1:
                k = mx.repeat(k, repeats=n_rep, axis=1)

            # Compute raw attention weights
            weights = (q @ k.transpose(0, 1, 3, 2)) * attn.scale
            if causal_mask is not None:
                weights = weights + causal_mask
            weights = mx.softmax(weights, axis=-1)

            attention_maps[i] = np.array(weights.astype(mx.float32)[0])  # [n_heads, T, T]

        # Continue actual forward pass
        h = layer_mod(h, mask=causal_mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]

    return {
        "attention_maps": attention_maps,
        "tokens": token_strs,
        "token_ids": token_ids,
        "n_heads": inner.layers[0].self_attn.n_heads,
        "n_kv_heads": inner.layers[0].self_attn.n_kv_heads,
    }


def head_attention_profile(attn_map: np.ndarray, head: int, tokens: list[str]) -> dict:
    """Analyze what a specific head attends to at the last position (next-token prediction)."""
    pattern = attn_map[head]  # [T, T]
    last_pos = pattern[-1]  # attention from last position to all positions
    top_k = min(5, len(last_pos))
    top_idx = np.argsort(last_pos)[::-1][:top_k]

    return {
        "head": head,
        "top_attended": [(tokens[i], float(last_pos[i])) for i in top_idx],
        "entropy": float(-np.sum(last_pos * np.log2(last_pos + 1e-12))),
        "self_attention": float(last_pos[-1]),
        "bos_attention": float(last_pos[0]),
    }


def find_token_focused_heads(
    attention_data: dict,
    target_positions: list[int],
    threshold: float = 0.3,
) -> list[dict]:
    """Find heads that strongly attend to specific token positions from the last position."""
    results = []
    tokens = attention_data["tokens"]

    for layer_idx, attn_map in attention_data["attention_maps"].items():
        n_heads = attn_map.shape[0]
        for h in range(n_heads):
            last_row = attn_map[h, -1, :]
            for pos in target_positions:
                if pos < len(last_row) and last_row[pos] > threshold:
                    results.append({
                        "layer": layer_idx,
                        "head": h,
                        "target_pos": pos,
                        "target_token": tokens[pos],
                        "attention_weight": float(last_row[pos]),
                    })

    results.sort(key=lambda x: x["attention_weight"], reverse=True)
    return results
