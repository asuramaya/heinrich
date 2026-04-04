"""Causal shart measurement — the token and the architecture, nothing else.

Three baseline modes:
  - "raw": token IDs through the model with no template. Pure token to residual.
  - "noise": token prepended to random byte tokens. No semantic context.
  - "query": token prepended to a specific query (legacy mode, context-dependent).

Returns the full residual vector at specified layers, not a scalar projection.
The caller projects onto whatever directions they want. The measurement doesn't
choose the axis.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ShartMeasurement:
    """One token's causal effect on the residual stream."""
    token_id: int
    token_text: str
    residuals: dict[int, np.ndarray]  # {layer: residual_delta[hidden_size]}
    baseline_residuals: dict[int, np.ndarray]  # {layer: baseline_residual[hidden_size]}


def measure_sharts(
    backend,
    token_ids: list[int],
    *,
    layers: list[int],
    mode: str = "raw",
    noise_length: int = 10,
    query: str | None = None,
    seed: int = 42,
) -> list[ShartMeasurement]:
    """Measure the causal effect of each token on the residual stream.

    Args:
        backend: model backend with forward/capture_residual_states
        token_ids: tokens to measure
        layers: layers to capture residuals at
        mode: "raw" (no template), "noise" (random bytes), "query" (prepend to query)
        noise_length: number of random tokens for noise baseline
        query: query string for "query" mode
        seed: RNG seed for noise generation

    Returns:
        list of ShartMeasurement, one per token
    """
    rng = np.random.RandomState(seed)
    inner = getattr(backend.model, "model", backend.model)
    vocab_size = inner.embed_tokens.weight.shape[0]

    # Build baseline token sequence
    if mode == "raw":
        bos = getattr(backend.tokenizer, 'bos_token_id', 1)
        baseline_ids = [bos] if bos is not None else [0]
    elif mode == "noise":
        baseline_ids = rng.randint(100, vocab_size, size=noise_length).tolist()
    elif mode == "query":
        if query is None:
            raise ValueError("query mode requires a query string")
        from heinrich.cartography.templates import build_prompt
        cfg = backend.config
        baseline_ids = backend.tokenize(build_prompt(query, model_config=cfg))
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'raw', 'noise', or 'query'.")

    # Capture baseline residuals
    baseline_residuals = _capture_residuals(backend, baseline_ids, layers)

    # Measure each token
    results = []
    for tid in token_ids:
        token_text = backend.tokenizer.decode([tid])
        token_input_ids = [tid] + baseline_ids

        try:
            token_residuals = _capture_residuals(backend, token_input_ids, layers)

            deltas = {}
            for l in layers:
                if l in token_residuals and l in baseline_residuals:
                    deltas[l] = token_residuals[l] - baseline_residuals[l]

            results.append(ShartMeasurement(
                token_id=tid,
                token_text=token_text,
                residuals=deltas,
                baseline_residuals=baseline_residuals,
            ))
        except Exception:
            pass

    return results


def _capture_residuals(
    backend, token_ids: list[int], layers: list[int],
) -> dict[int, np.ndarray]:
    """Run a forward pass on raw token IDs (no template) and capture residuals."""
    import mlx.core as mx
    from heinrich.cartography.perturb import _mask_dtype

    inner = getattr(backend.model, "model", backend.model)
    mdtype = _mask_dtype(backend.model)

    input_ids = mx.array([token_ids])
    T = len(token_ids)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

    layer_set = set(layers)
    residuals = {}

    h = inner.embed_tokens(input_ids)
    for i, ly in enumerate(inner.layers):
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]
        if i in layer_set:
            residuals[i] = np.array(h.astype(mx.float32)[0, -1, :])
        if i % 8 == 7:
            mx.eval(h)

    return residuals


def measure_shart_gradients(
    backend,
    token_ids: list[int],
    *,
    query: str = "What is the weather today?",
    target_token: str = "I",
    position: str = "last",
    seed: int = 42,
) -> list[dict]:
    """Measure the gradient magnitude at the shart token's position.

    The gradient of a fixed output logit w.r.t. the input embeddings
    tells us how sensitive the output is to each token position.
    No projection onto any direction. The gradient magnitude IS the measurement.

    Args:
        backend: MLX backend (model stays loaded between calls)
        token_ids: tokens to measure
        query: the context query
        target_token: which output logit to backprop from
        position: "first" (prepend) or "last" (append) — controls position privilege
        seed: RNG seed

    Returns:
        list of dicts with token_id, token_text, shart_grad, query_grad, total_grad
    """
    import mlx.core as mx
    from heinrich.cartography.templates import build_prompt
    from heinrich.cartography.perturb import _mask_dtype

    inner = getattr(backend.model, "model", backend.model)
    cfg = backend.config

    # Target logit
    target_id = backend.tokenize(target_token)[0]

    def forward_to_logit(embeddings):
        mdtype = _mask_dtype(backend.model)
        T = embeddings.shape[1]
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None
        h = embeddings
        for ly in inner.layers:
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]
        h = inner.norm(h)
        logits = backend._lm_head(h)
        return logits[0, -1, target_id]

    grad_fn = mx.grad(forward_to_logit)
    fmt_query = build_prompt(query, model_config=cfg)
    query_tokens = backend.tokenize(fmt_query)
    n_query = len(query_tokens)

    results = []
    for tid in token_ids:
        tok_text = backend.tokenizer.decode([tid])
        tok_tokens = backend.tokenize(tok_text)

        if position == "first":
            full_tokens = tok_tokens + query_tokens
            shart_pos = slice(0, len(tok_tokens))
            query_pos = slice(len(tok_tokens), None)
        else:
            full_tokens = query_tokens + tok_tokens
            shart_pos = slice(n_query, None)
            query_pos = slice(0, n_query)

        try:
            input_ids = mx.array([full_tokens])
            embeddings = inner.embed_tokens(input_ids)
            grad = grad_fn(embeddings)
            mx.eval(grad)

            grad_np = np.array(grad.astype(mx.float32)[0])
            shart_grad = float(np.linalg.norm(grad_np[shart_pos]))
            query_grad = float(np.linalg.norm(grad_np[query_pos]))
            total_grad = float(np.linalg.norm(grad_np))

            results.append({
                "token_id": tid,
                "token_text": tok_text,
                "shart_grad": round(shart_grad, 4),
                "query_grad": round(query_grad, 4),
                "total_grad": round(total_grad, 4),
                "position": position,
            })
        except Exception:
            pass

    return results


def random_token_sample(
    backend,
    n: int = 500,
    *,
    seed: int = 42,
) -> list[int]:
    """Sample n random real token IDs (excluding special/control tokens)."""
    rng = np.random.RandomState(seed)
    vocab_size = backend.tokenizer.vocab_size if hasattr(backend.tokenizer, 'vocab_size') else 32000

    real_ids = []
    for tid in range(vocab_size):
        tok = backend.tokenizer.decode([tid])
        if tok.strip() and len(tok) > 1 and not tok.startswith('[control') and not tok.startswith('<'):
            real_ids.append(tid)

    selected = rng.choice(len(real_ids), min(n, len(real_ids)), replace=False)
    return [real_ids[i] for i in selected]
