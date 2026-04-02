"""Generate text with head modifications active — verify control surface works."""
from __future__ import annotations
from typing import Any
import numpy as np
from .runtime import _lm_head


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
    *,
    backend: Any = None,
) -> dict[str, Any]:
    """Get next-token distribution with specific heads scaled.

    modifications: {(layer, head): scale_factor} where 0.0=zero, 1.0=normal, 2.0=amplify
    """
    if backend is not None:
        # Build steer_dirs from modifications — scale each head via backend.forward
        # For multi-head steering, we use individual perturb_head calls composed,
        # but the simplest delegation is a single forward with the modifications
        # encoded as per-layer steer directions. Since the backend protocol doesn't
        # have a direct multi-head-scale forward, we use the first modification
        # and fall back to the MLX path for complex cases.
        # Actually, we can iterate modifications and use backend.perturb_head for
        # single-head cases, or fall through to MLX for multi-head.
        if len(modifications) == 1:
            (layer, head), scale = next(iter(modifications.items()))
            mode = "zero" if scale == 0.0 else "scale"
            result = backend.perturb_head(prompt, layer, head, mode=mode, scale=scale)
            probs = result.probs
            top_k_count = 10
            top_idx = np.argsort(probs)[::-1][:top_k_count]
            return {
                "top_tokens": [(backend.decode([int(i)]), float(probs[i])) for i in top_idx],
                "top_token": result.top_token,
                "top_prob": float(probs[result.top_id]),
                "entropy": result.entropy,
                "top_id": result.top_id,
            }
        # Multi-head modifications: fall through to direct MLX path
        # (backend protocol doesn't support multi-head perturbation in one pass)

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
    logits = np.array(_lm_head(model, h).astype(mx.float32)[0, -1, :])
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
    *,
    backend: Any = None,
) -> dict[str, Any]:
    """Auto-regressive generation with head modifications active at every step."""
    if backend is not None and len(modifications) == 1:
        # Single-head modification can be delegated via backend.generate with steer_dirs
        # For zeroing a head, we encode it as a steering direction that cancels the head
        # However, generate() uses additive steering, not scaling. For single-head zero,
        # we do step-by-step generation using backend.perturb_head.
        (layer, head), scale = next(iter(modifications.items()))
        mode = "zero" if scale == 0.0 else "scale"
        tokens = backend.tokenize(prompt)
        generated_ids = []
        full_tokens = list(tokens)

        for _ in range(max_tokens):
            current_text = backend.decode(full_tokens)
            result = backend.perturb_head(current_text, layer, head, mode=mode, scale=scale)
            next_id = result.top_id
            if hasattr(backend, 'tokenizer'):
                eos = getattr(backend.tokenizer, "eos_token_id", None)
            else:
                eos = None
            if next_id == eos:
                break
            full_tokens.append(next_id)
            generated_ids.append(next_id)

        return {
            "prompt": prompt,
            "generated": backend.decode(generated_ids),
            "full": backend.decode(full_tokens),
            "modifications": {f"L{l}H{h}": s for (l, h), s in modifications.items()},
            "n_tokens": len(generated_ids),
        }

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
        logits = np.array(_lm_head(model, h).astype(mx.float32)[0, -1, :])
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
