"""Generate text with head modifications active — verify control surface works."""
from __future__ import annotations
from typing import Any
import numpy as np


def _get_head_geometry(model: Any) -> tuple[int, int]:
    """Return (n_heads, head_dim) from model."""
    inner = getattr(model, "model", model)
    n_heads = inner.layers[0].self_attn.n_heads
    if hasattr(inner, "norm") and hasattr(inner.norm, "weight"):
        hidden_size = inner.norm.weight.shape[0]
    else:
        hidden_size = 3584
    return n_heads, hidden_size // n_heads


def steer_next_token(
    model: Any,
    tokenizer: Any,
    prompt: str,
    modifications: dict[tuple[int, int], float],
) -> dict[str, Any]:
    """Get next-token distribution with specific heads scaled.

    modifications: {(layer, head): scale_factor} where 0.0=zero, 1.0=normal, 2.0=amplify
    """
    import mlx.core as mx
    from .perturb import _mask_dtype
    from ..inspect.self_analysis import _softmax

    inner = getattr(model, "model", model)
    n_heads, head_dim = _get_head_geometry(model)

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mdtype = _mask_dtype(model)
    mask = mx.triu(mx.full((T, T), float("-inf"), dtype=mdtype), k=1) if T > 1 else None

    h = inner.embed_tokens(input_ids)
    for i, ly in enumerate(inner.layers):
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]

        # Apply all modifications at this layer
        mods_at_layer = {head: scale for (layer, head), scale in modifications.items() if layer == i}
        if mods_at_layer:
            h_np = np.array(h.astype(mx.float32))
            for head, scale in mods_at_layer.items():
                start = head * head_dim
                end = start + head_dim
                h_np[0, :, start:end] *= scale
            h = mx.array(h_np.astype(np.float16))

    h = inner.norm(h)
    logits = np.array(model.lm_head(h).astype(mx.float32)[0, -1, :])
    probs = _softmax(logits)

    top_k = 10
    top_idx = np.argsort(probs)[::-1][:top_k]
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    return {
        "top_tokens": [(tokenizer.decode([int(i)]), float(probs[i])) for i in top_idx],
        "top_token": tokenizer.decode([int(top_idx[0])]),
        "top_prob": float(probs[top_idx[0]]),
        "entropy": entropy,
        "top_id": int(top_idx[0]),
    }


def generate_steered(
    model: Any,
    tokenizer: Any,
    prompt: str,
    modifications: dict[tuple[int, int], float],
    max_tokens: int = 30,
) -> dict[str, Any]:
    """Auto-regressive generation with head modifications active at every step."""
    import mlx.core as mx
    from .perturb import _mask_dtype

    inner = getattr(model, "model", model)
    n_heads, head_dim = _get_head_geometry(model)

    tokens = list(tokenizer.encode(prompt))
    mdtype = _mask_dtype(model)
    generated = []

    for _ in range(max_tokens):
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float("-inf"), dtype=mdtype), k=1) if T > 1 else None

        h = inner.embed_tokens(input_ids)
        for i, ly in enumerate(inner.layers):
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]

            mods_at_layer = {hd: sc for (ly_idx, hd), sc in modifications.items() if ly_idx == i}
            if mods_at_layer:
                h_np = np.array(h.astype(mx.float32))
                for hd, sc in mods_at_layer.items():
                    start = hd * head_dim
                    end = start + head_dim
                    h_np[0, :, start:end] *= sc
                h = mx.array(h_np.astype(np.float16))

        h = inner.norm(h)
        logits = np.array(model.lm_head(h).astype(mx.float32)[0, -1, :])
        next_id = int(np.argmax(logits))

        eos = getattr(tokenizer, "eos_token_id", None)
        if next_id == eos:
            break

        tokens.append(next_id)
        generated.append(next_id)

    return {
        "prompt": prompt,
        "generated": tokenizer.decode(generated),
        "full": tokenizer.decode(tokens),
        "modifications": {f"L{l}H{h}": s for (l, h), s in modifications.items()},
        "n_tokens": len(generated),
    }
