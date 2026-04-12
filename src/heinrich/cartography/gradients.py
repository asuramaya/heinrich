"""Gradient-based attribution — token saliency and neuron attribution.

Uses mx.grad (MLX) or torch.autograd (HF) to compute how much each input
token or intermediate neuron contributes to the model's predicted output.
"""
from __future__ import annotations
from typing import Any
import numpy as np

from .runtime import _lm_head
from .backend import MLXBackend, HFBackend, ForwardResult


def token_saliency(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    target_token_id: int | None = None,
    backend: Any = None,
) -> np.ndarray:
    """Compute gradient of target logit w.r.t. input embeddings.

    Returns [n_tokens] saliency scores (L2 norm of per-token embedding gradient).

    If backend is an MLXBackend, uses mx.grad.
    If backend is an HFBackend, uses torch.autograd.
    If backend is None, raises NotImplementedError.
    """
    # Auto-construct backend from model/tokenizer if not provided
    if backend is None and model is not None:
        try:
            b = MLXBackend.__new__(MLXBackend)
            b.model = model
            b.tokenizer = tokenizer
            from .model_config import detect_config
            b.config = detect_config(model, tokenizer)
            b._inner = getattr(model, "model", model)
            backend = b
        except Exception:
            pass

    if isinstance(backend, MLXBackend):
        return _token_saliency_mlx(backend, prompt, target_token_id=target_token_id)
    elif isinstance(backend, HFBackend):
        return _token_saliency_hf(backend, prompt, target_token_id=target_token_id)
    else:
        raise NotImplementedError(
            f"token_saliency requires an MLXBackend or HFBackend, got {type(backend)}"
        )


def neuron_attribution(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer: int,
    *,
    target_token_id: int | None = None,
    backend: Any = None,
) -> np.ndarray:
    """Gradient of target logit w.r.t. MLP activations at target layer.

    Returns [intermediate_size] attribution scores.

    If backend is an MLXBackend, uses mx.grad.
    If backend is an HFBackend, uses torch.autograd.
    If backend is None, raises NotImplementedError.
    """
    if isinstance(backend, MLXBackend):
        return _neuron_attribution_mlx(backend, prompt, layer, target_token_id=target_token_id)
    elif isinstance(backend, HFBackend):
        return _neuron_attribution_hf(backend, prompt, layer, target_token_id=target_token_id)
    else:
        raise NotImplementedError(
            f"neuron_attribution requires an MLXBackend or HFBackend, got {type(backend)}"
        )


# ---------------------------------------------------------------------------
# MLX implementations
# ---------------------------------------------------------------------------

def _token_saliency_mlx(
    backend: MLXBackend,
    prompt: str,
    *,
    target_token_id: int | None = None,
) -> np.ndarray:
    """MLX implementation of token saliency via mx.grad."""
    import mlx.core as mx
    from .perturb import _mask_dtype

    model = backend.model
    inner = backend._inner
    mdtype = _mask_dtype(model)

    tokens = backend.tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

    # Get embeddings — detach to break tied-weight graph.
    # Models with tied embeddings (embed_tokens.as_linear) share weights
    # between embedding and lm_head. Without stop_gradient, mx.grad
    # differentiates through both paths, producing wrong gradients.
    embeddings = mx.stop_gradient(inner.embed_tokens(input_ids).astype(mx.float32))

    # Determine target token if not specified
    if target_token_id is None:
        # Run a forward pass to find the argmax token
        result = backend.forward(prompt)
        target_token_id = result.top_id

    target_id = target_token_id

    def loss_fn(embedding_input):
        h = embedding_input
        for ly in inner.layers:
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]
        h = inner.norm(h)
        logits = _lm_head(model, h)
        return logits[0, -1, target_id]

    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(embeddings)
    mx.eval(grads)  # noqa: S307 — this is mlx.core.eval, not Python eval

    # L2 norm per token position: grads shape is [1, T, hidden_size]
    grads_np = np.array(grads.astype(mx.float32))
    saliency = np.linalg.norm(grads_np[0], axis=-1)  # [T]
    return saliency


def _neuron_attribution_mlx(
    backend: MLXBackend,
    prompt: str,
    layer: int,
    *,
    target_token_id: int | None = None,
) -> np.ndarray:
    """MLX implementation of neuron attribution via mx.grad."""
    import mlx.core as mx
    import mlx.nn as nn
    from .perturb import _mask_dtype

    model = backend.model
    inner = backend._inner
    mdtype = _mask_dtype(model)

    tokens = backend.tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

    # Determine target token if not specified
    if target_token_id is None:
        result = backend.forward(prompt)
        target_token_id = result.top_id

    target_id = target_token_id

    # Forward pass up to the target layer to get the input to MLP
    h = inner.embed_tokens(input_ids)
    for i, ly in enumerate(inner.layers):
        if i == layer:
            break
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]

    # At the target layer: decompose into attention + MLP
    target_ly = inner.layers[layer]
    h_normed = target_ly.input_layernorm(h) if hasattr(target_ly, 'input_layernorm') else h
    attn_out = target_ly.self_attn(h_normed, mask=mask, cache=None)
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    h_post_attn = h + attn_out

    # Get pre-MLP hidden state
    h_normed2 = target_ly.post_attention_layernorm(h_post_attn) if hasattr(target_ly, 'post_attention_layernorm') else h_post_attn

    # Compute MLP activations (gate * up)
    gate_out = target_ly.mlp.gate_proj(h_normed2)
    up_out = target_ly.mlp.up_proj(h_normed2)
    activations = nn.silu(gate_out) * up_out
    mx.eval(activations)  # noqa: S307

    # Now define loss from activations forward
    def loss_fn(act_input):
        mlp_out = target_ly.mlp.down_proj(act_input)
        h_next = h_post_attn + mlp_out
        # Continue through remaining layers
        for i in range(layer + 1, len(inner.layers)):
            ly = inner.layers[i]
            h_next = ly(h_next, mask=mask, cache=None)
            if isinstance(h_next, tuple):
                h_next = h_next[0]
        h_next = inner.norm(h_next)
        logits = _lm_head(model, h_next)
        return logits[0, -1, target_id]

    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(activations)
    mx.eval(grads)  # noqa: S307

    # Return absolute gradient at the last token position
    grads_np = np.array(grads.astype(mx.float32))
    attribution = np.abs(grads_np[0, -1, :])  # [intermediate_size]
    return attribution


# ---------------------------------------------------------------------------
# HF (torch) implementations
# ---------------------------------------------------------------------------

def _token_saliency_hf(
    backend: HFBackend,
    prompt: str,
    *,
    target_token_id: int | None = None,
) -> np.ndarray:
    """HuggingFace implementation of token saliency via torch.autograd."""
    import torch

    input_ids = backend.tokenizer.encode(prompt, return_tensors="pt").to(backend._device)

    # Get embedding layer
    embed_layer = backend.hf_model.get_input_embeddings()
    embeddings = embed_layer(input_ids)
    embeddings = embeddings.detach().requires_grad_(True)

    # Determine target token if not specified
    if target_token_id is None:
        with torch.no_grad():
            outputs = backend.hf_model(input_ids)
            target_token_id = int(outputs.logits[0, -1, :].argmax())

    target_id = target_token_id

    # Forward with embeddings that have gradients
    outputs = backend.hf_model(inputs_embeds=embeddings)
    logit_val = outputs.logits[0, -1, target_id]
    logit_val.backward()

    # L2 norm per token position
    grads = embeddings.grad  # [1, T, hidden_size]
    saliency = grads[0].norm(dim=-1).detach().cpu().numpy()  # [T]
    return saliency


def _neuron_attribution_hf(
    backend: HFBackend,
    prompt: str,
    layer: int,
    *,
    target_token_id: int | None = None,
) -> np.ndarray:
    """HuggingFace implementation of neuron attribution via torch.autograd."""
    import torch

    input_ids = backend.tokenizer.encode(prompt, return_tensors="pt").to(backend._device)

    # Determine target token if not specified
    if target_token_id is None:
        with torch.no_grad():
            outputs = backend.hf_model(input_ids)
            target_token_id = int(outputs.logits[0, -1, :].argmax())

    target_id = target_token_id

    # Use a forward hook to capture and enable gradients on MLP activations
    captured_activation = {}
    target_module = backend.hf_model.model.layers[layer].mlp

    def hook_fn(module, input, output):
        output.retain_grad()
        captured_activation["output"] = output

    handle = target_module.register_forward_hook(hook_fn)

    try:
        outputs = backend.hf_model(input_ids)
        logit_val = outputs.logits[0, -1, target_id]
        logit_val.backward()

        act_grad = captured_activation["output"].grad  # [1, T, intermediate_size]
        attribution = act_grad[0, -1, :].abs().detach().cpu().numpy()
    finally:
        handle.remove()

    return attribution
