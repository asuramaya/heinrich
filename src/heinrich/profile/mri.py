"""The .mri file — complete model residual image.

One directory per model per capture mode. Contains everything:
  - Tokenizer atoms (raw bytes, merge ranks, decoded text)
  - Residual state at entry and exit positions, every layer
  - Baselines (the reference frame)
  - All model weights (embedding, norms, projections, lm_head)
  - Discovered directions (safety, comply, any others)
  - Capture provenance (mode, seed, template, model config)

No separate .frt, .shrt, .sht, .trd needed. One directory. One load_mri().
Analysis tools compute everything from stored data.

Modes:
  template — chat frame, silence baseline
  naked   — single token, BOS baseline
  raw     — single token, no BOS, absolute state (zero baseline)

Architecture note: the MRI capture uses the Backend protocol for weight
extraction, baseline computation, and tokenizer access. The per-token
state capture uses an optimized batched loop with deferred GPU sync.
Future: migrate the hot loop to backend.capture_all_positions() when
that method supports token_ids, batching, and deferred sync.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def _is_mlx_backend(backend) -> bool:
    """Check if backend uses MLX (vs PyTorch/HF)."""
    return type(backend).__name__ == 'MLXBackend'


def _extract_weight(module, in_dim: int, is_mlx: bool) -> np.ndarray:
    """Extract weight matrix via identity probing (works with quantized models)."""
    if is_mlx:
        import mlx.core as mx
        cols = []
        for s in range(0, in_dim, 64):
            e = min(s + 64, in_dim)
            probe = np.zeros((1, e - s, in_dim), dtype=np.float32)
            for j in range(e - s):
                probe[0, j, s + j] = 1.0
            out = np.array(module(mx.array(probe).astype(mx.float16)).astype(mx.float32)[0])
            cols.append(out.T)
        return np.concatenate(cols, axis=1).astype(np.float32)
    else:
        if hasattr(module, 'weight') and module.weight is not None:
            return module.weight.float().cpu().numpy()
        import torch
        device = next(module.parameters()).device
        cols = []
        for s in range(0, in_dim, 64):
            e = min(s + 64, in_dim)
            probe = torch.zeros(1, e - s, in_dim, device=device)
            for j in range(e - s):
                probe[0, j, s + j] = 1.0
            with torch.no_grad():
                out = module(probe).float().cpu().numpy()[0]
            cols.append(out.T)
        return np.concatenate(cols, axis=1).astype(np.float32)


def _framework_ops(backend):
    """Return framework-specific operations as a namespace.

    Works for both MLX and PyTorch/HF backends. Abstracts:
      array, triu, full, stack, to_numpy, embed, layer_forward, norm, lm_head, dtype
    """
    from types import SimpleNamespace

    if _is_mlx_backend(backend):
        import mlx.core as mx
        from ..cartography.runtime import _lm_head

        model_inner = getattr(backend.model, 'model', backend.model)
        # Infer mask dtype from model — must match the attention computation's dtype
        _probe = model_inner.embed_tokens(mx.array([[0]]))
        mdtype = _probe.dtype
        _final_norm = getattr(model_inner, 'norm', None) or getattr(model_inner, 'final_layernorm', None)

        def _mlx_layer_decomposed(ly, h, mask):
            """Run one layer decomposed, returning exit state + intermediates.

            Returns (h_exit, attn_weights, h_pre_mlp):
              h_exit:       [B, T, hidden] — exact layer output (matches fused)
              attn_weights: [B, heads, T, T] — attention after softmax
              h_pre_mlp:    [B, T, hidden] — MLP input (for exact gate capture)

            Handles two architectures:
              Sequential (Qwen, Mistral, SmolLM): attn → post_norm → MLP
              Parallel (Phi-2): attn and MLP run from same normed input
            """
            attn_mod = ly.self_attn
            norm_in = ly.input_layernorm(h) if hasattr(ly, 'input_layernorm') else h
            residual = h

            # Detect parallel vs sequential
            is_parallel = not hasattr(ly, 'post_attention_layernorm')

            # Attention output (using model's own module — exact)
            attn_out = attn_mod(norm_in, mask=mask, cache=None)

            if is_parallel:
                h_pre_mlp = norm_in
            else:
                h_after_attn = residual + attn_out
                h_pre_mlp = ly.post_attention_layernorm(h_after_attn)

            # MLP decomposed: capture gate + up before down_proj
            import mlx.nn as _nn
            _mlp = ly.mlp
            if hasattr(_mlp, 'gate_proj'):
                _gate_val = _nn.silu(_mlp.gate_proj(h_pre_mlp))
                _up_val = _mlp.up_proj(h_pre_mlp)
                _mlp_out = _mlp.down_proj(_gate_val * _up_val)
            elif hasattr(_mlp, 'fc1'):
                _gate_val = _nn.gelu(_mlp.fc1(h_pre_mlp))
                _up_val = _gate_val
                _mlp_out = _mlp.fc2(_gate_val)
            else:
                _gate_val = None
                _up_val = None
                _mlp_out = _mlp(h_pre_mlp)

            if is_parallel:
                h_exit = attn_out + _mlp_out + residual
            else:
                h_exit = h_after_attn + _mlp_out

            # Attention weights (recomputed Q@K — does NOT affect h_exit)
            n_h = getattr(attn_mod, 'n_heads', None) or getattr(attn_mod, 'num_heads', None)
            n_kv = getattr(attn_mod, 'n_kv_heads', None) or getattr(attn_mod, 'num_key_value_heads', n_h)
            B, L, D = norm_in.shape

            if hasattr(attn_mod, 'qkv_proj'):
                head_dim = getattr(attn_mod, 'head_dim', D // n_h)
                qkv = attn_mod.qkv_proj(norm_in)
                q_dim = n_h * head_dim
                kv_dim = n_kv * head_dim
                q = qkv[..., :q_dim].reshape(B, L, n_h, head_dim).transpose(0, 2, 1, 3)
                k = qkv[..., q_dim:q_dim + kv_dim].reshape(B, L, n_kv, head_dim).transpose(0, 2, 1, 3)
            else:
                q = attn_mod.q_proj(norm_in).reshape(B, L, n_h, -1).transpose(0, 2, 1, 3)
                k = attn_mod.k_proj(norm_in).reshape(B, L, n_kv, -1).transpose(0, 2, 1, 3)
            q = attn_mod.rope(q)
            k = attn_mod.rope(k)
            if n_kv < n_h:
                k = mx.repeat(k, n_h // n_kv, axis=1)
            head_dim = q.shape[-1]
            scores = (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * (head_dim ** -0.5)
            if mask is not None:
                scores = scores + mask
            weights = mx.softmax(scores, axis=-1)

            return h_exit, weights, scores, h_pre_mlp, attn_out, _gate_val, _up_val

        def _mlx_embedding_grad(emb, mask):
            """Gradient of top-1 logit w.r.t. embedding. One backward pass.

            CRITICAL: For tied-embedding models (embed_tokens.as_linear),
            the embedding must be detached from the graph before computing
            gradients. Otherwise the same weight matrix appears in both
            the embed and lm_head paths, producing wrong gradients.
            """
            # Detach: break tied-weight graph by round-tripping through numpy
            emb_detached = mx.array(np.array(emb.astype(mx.float32)))

            # Find top-1 token (outside gradient graph)
            h_probe = emb_detached
            for ly in model_inner.layers:
                h_probe = ly(h_probe, mask=mask, cache=None)
            h_probe = _final_norm(h_probe) if _final_norm else h_probe
            logits_probe = _lm_head(backend.model, h_probe)
            top1 = mx.argmax(logits_probe[:, -1, :], axis=1)
            mx.eval(top1)
            top1_ids = np.array(top1).astype(np.int32).tolist()

            # Gradient with fixed target (no argmax in backward graph)
            def _fwd(e):
                h = e
                for ly in model_inner.layers:
                    h = ly(h, mask=mask, cache=None)
                h = _final_norm(h) if _final_norm else h
                logits = _lm_head(backend.model, h)
                # Fixed targets per batch item
                selected = mx.take_along_axis(logits[:, -1, :], mx.array(top1_ids)[:, None], axis=1)
                return mx.sum(selected)

            grad_fn = mx.grad(_fwd)
            g = grad_fn(emb_detached)
            return np.array(g.astype(mx.float32))

        def _mlx_mlp_internals(ly, h_pre_mlp):
            """Get gate AND up values from exact pre-MLP state.
            Returns (gate_values, up_values) as numpy float16."""
            import mlx.nn as nn
            mlp = ly.mlp
            h = h_pre_mlp
            if hasattr(mlp, 'gate_proj'):
                gate = nn.silu(mlp.gate_proj(h))
                up = mlp.up_proj(h)
            elif hasattr(mlp, 'fc1'):
                gate = nn.gelu(mlp.fc1(h))
                up = gate  # fc1/fc2 has no separate up projection
            else:
                return None, None
            g = gate[:, -1, :] if len(gate.shape) == 3 else gate
            u = up[:, -1, :] if len(up.shape) == 3 else up
            return np.array(g.astype(mx.float16)), np.array(u.astype(mx.float16))

        def _mlx_gate_topk(ly, h_pre_mlp, k=32):
            """Get top-K MLP gate activations from the exact pre-MLP state.

            h_pre_mlp is already the correctly normed input to the MLP:
              Sequential: post_attention_layernorm(h_after_attn)
              Parallel: input_layernorm(h) (same input as attention)
            """
            import mlx.nn as nn
            mlp = ly.mlp
            h_normed = h_pre_mlp
            if hasattr(mlp, 'gate_proj'):
                gate_act = nn.silu(mlp.gate_proj(h_normed))
            elif hasattr(mlp, 'fc1'):
                gate_act = nn.gelu(mlp.fc1(h_normed))
            else:
                return None, None, None
            g = gate_act[:, -1, :] if len(gate_act.shape) == 3 else gate_act
            topk_idx = mx.argpartition(-mx.abs(g), kth=k, axis=1)[:, :k]
            topk_val = mx.take_along_axis(g, topk_idx, axis=1)
            return np.array(topk_idx), np.array(topk_val.astype(mx.float32)), k

        return SimpleNamespace(
            model_inner=model_inner,
            array=lambda x: mx.array(x),
            triu_mask=lambda T: mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None,
            stack_to_numpy=lambda tensors: np.array(mx.stack(tensors).astype(mx.float32)),
            to_numpy_1d=lambda t: np.array(t.astype(mx.float32)),
            to_numpy_2d=lambda t, row, col: np.array(t.astype(mx.float32)[0, row, :]),
            embed=lambda ids: model_inner.embed_tokens(ids),
            layer_forward=lambda ly, h, mask: ly(h, mask=mask, cache=None),
            layer_decomposed=_mlx_layer_decomposed,
            mlp_internals=_mlx_mlp_internals,
            embedding_grad=lambda emb, mask: _mlx_embedding_grad(emb, mask),
            gate_topk=_mlx_gate_topk,
            norm=lambda h: _final_norm(h) if _final_norm else h,
            lm_head=lambda h: _lm_head(backend.model, h),
            lm_head_logits=lambda h, pos: np.array(_lm_head(backend.model, (_final_norm(h) if _final_norm else h)).astype(mx.float32)[0, pos, :]),
            float32=mx.float32,
        )
    else:
        import torch

        model_inner = backend.hf_model.model
        device = next(backend.hf_model.parameters()).device

        def _lm_head_hf(h):
            normed = model_inner.norm(h)
            return backend.hf_model.lm_head(normed)

        def _hf_layer_decomposed(ly, h, mask):
            """Run one HF layer decomposed. Handles sequential and parallel."""
            with torch.no_grad():
                residual = h
                norm_in = ly.input_layernorm(h) if hasattr(ly, 'input_layernorm') else h

                attn_out = ly.self_attn(norm_in, attention_mask=mask,
                                         output_attentions=True)
                attn_hidden = attn_out[0]
                weights = attn_out[-1]

                is_parallel = not hasattr(ly, 'post_attention_layernorm')
                if is_parallel:
                    h_pre_mlp = norm_in
                else:
                    h_after_attn = residual + attn_hidden
                    h_pre_mlp = ly.post_attention_layernorm(h_after_attn)

                _mlp = ly.mlp
                if hasattr(_mlp, 'gate_proj'):
                    _gv = torch.nn.functional.silu(_mlp.gate_proj(h_pre_mlp))
                    _uv = _mlp.up_proj(h_pre_mlp)
                    _mo = _mlp.down_proj(_gv * _uv)
                elif hasattr(_mlp, 'fc1'):
                    _gv = torch.nn.functional.gelu(_mlp.fc1(h_pre_mlp))
                    _uv = _gv
                    _mo = _mlp.fc2(_gv)
                else:
                    _gv = None
                    _uv = None
                    _mo = _mlp(h_pre_mlp)

                if is_parallel:
                    h_exit = attn_hidden + _mo + residual
                else:
                    h_exit = h_after_attn + _mo

                return h_exit, weights, None, h_pre_mlp, attn_hidden, _gv, _uv

        def _hf_mlp_internals(ly, h_pre_mlp):
            """Get gate AND up values."""
            with torch.no_grad():
                mlp = ly.mlp
                if hasattr(mlp, 'gate_proj'):
                    gate = torch.nn.functional.silu(mlp.gate_proj(h_pre_mlp))
                    up = mlp.up_proj(h_pre_mlp)
                elif hasattr(mlp, 'fc1'):
                    gate = torch.nn.functional.gelu(mlp.fc1(h_pre_mlp))
                    up = gate
                else:
                    return None, None
                g = gate[:, -1, :] if len(gate.shape) == 3 else gate
                u = up[:, -1, :] if len(up.shape) == 3 else up
                return g.half().cpu().numpy(), u.half().cpu().numpy()

        def _hf_gate_topk(ly, h_pre_mlp, k=32):
            """Get top-K MLP gate activations from the exact pre-MLP state."""
            with torch.no_grad():
                mlp = ly.mlp
                h_normed = h_pre_mlp  # already normed by layer_decomposed
                if hasattr(mlp, 'gate_proj'):
                    gate_act = torch.nn.functional.silu(mlp.gate_proj(h_normed))
                elif hasattr(mlp, 'fc1'):
                    gate_act = torch.nn.functional.gelu(mlp.fc1(h_normed))
                else:
                    return None, None, None
                g = gate_act[:, -1, :] if len(gate_act.shape) == 3 else gate_act
                topk_val, topk_idx = torch.topk(g.abs(), k, dim=1)
                real_val = torch.gather(g, 1, topk_idx)
                return topk_idx.cpu().numpy(), real_val.float().cpu().numpy(), k

        return SimpleNamespace(
            model_inner=model_inner,
            array=lambda x: torch.tensor(x, device=device, dtype=torch.long),
            triu_mask=lambda T: torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1) if T > 1 else None,
            stack_to_numpy=lambda tensors: torch.stack(tensors).float().cpu().numpy(),
            to_numpy_1d=lambda t: t.float().cpu().numpy(),
            to_numpy_2d=lambda t, row, col: t.float().cpu().numpy()[0, row, :],
            embed=lambda ids: model_inner.embed_tokens(ids),
            layer_forward=lambda ly, h, mask: (lambda out: out[0] if isinstance(out, tuple) else out)(ly(h, attention_mask=mask)),
            layer_decomposed=_hf_layer_decomposed,
            mlp_internals=_hf_mlp_internals,
            gate_topk=_hf_gate_topk,
            norm=lambda h: model_inner.norm(h),
            lm_head=lambda h: _lm_head_hf(h),
            lm_head_logits=lambda h, pos: _lm_head_hf(h)[0, pos, :].float().cpu().numpy(),
            float32=torch.float32,
        )


def _capture_mri_causal_bank(backend, *, mode="raw", n_index=None,
                              seed=42, output=None) -> dict:
    """MRI capture for causal bank models.

    Runs each token through the model and captures:
      - substrate_states[N, n_modes] — the EMA state at readout
      - route_weights[N, n_experts] — routing decisions
      - band_logits[N, n_bands, vocab] — per-band logit contribution (if multi-band)
      - embedding[N, embed_dim] — input embedding

    NOTE: raw mode feeds single tokens from zero EMA state. The EMA substrate
    barely moves from zero on one token — this captures the *impulse response*,
    not the steady-state representation. Sequential dynamics require sequence input.
    Only raw mode is currently implemented for causal bank.
    """
    from .frt import _detect_script

    if mode != "raw":
        print(f"  WARNING: mode '{mode}' not implemented for causal bank. Using raw mode.")
        mode = "raw"

    cfg = backend.config
    t0 = time.time()

    # Build token sample from tokenizer
    tokenizer = backend.tokenizer
    vocab_size = cfg.vocab_size
    real_tokens = []
    seen = set()
    for tid in range(vocab_size):
        tok = tokenizer.Decode([tid]) if tokenizer else str(tid)
        if tok.strip() and tok not in seen:
            seen.add(tok)
            real_tokens.append((tid, tok))

    if n_index is None or n_index >= len(real_tokens):
        sample = real_tokens
    else:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(real_tokens), n_index, replace=False)
        sample = [real_tokens[i] for i in sorted(idx)]

    n_tokens = len(sample)
    n_modes = cfg.n_modes
    n_experts = cfg.n_experts
    n_bands = cfg.n_bands
    embed_dim = cfg.embed_dim

    print(f"MRI capture (causal_bank, {mode}): {n_tokens} tokens, "
          f"{n_modes} modes, {n_experts} experts, {n_bands} bands")

    # Allocate
    substrate_states = np.zeros((n_tokens, n_modes), dtype=np.float16)
    embeddings = np.zeros((n_tokens, embed_dim), dtype=np.float16)
    route_weights = np.zeros((n_tokens, n_experts), dtype=np.float16) if n_experts > 1 else None

    # Band logits: only allocate if multi-band (can be large: N * n_bands * vocab)
    capture_band_logits = n_bands > 1
    band_logits_all = None
    if capture_band_logits:
        band_bytes = n_tokens * n_bands * vocab_size * 2  # float16
        if band_bytes > 4e9:
            print(f"  WARNING: band_logits would be {band_bytes/1e9:.1f} GB. Skipping.")
            capture_band_logits = False
        else:
            band_logits_all = np.zeros((n_tokens, n_bands, vocab_size), dtype=np.float16)

    # For each token: feed single token from zero EMA state
    for i, (tid, tok) in enumerate(sample):
        result = backend.forward_captured(np.array([[tid]]))
        substrate_states[i] = result['substrate_states'][0, 0].astype(np.float16)
        embeddings[i] = result['embedding'][0, 0].astype(np.float16)
        if route_weights is not None and result.get('route_weights') is not None:
            rw = result['route_weights']
            if len(rw.shape) == 2:
                route_weights[i] = rw[0].astype(np.float16)
            else:
                route_weights[i] = rw[0, 0].astype(np.float16)
        if capture_band_logits and result.get('band_logits') is not None:
            for b, bl in enumerate(result['band_logits']):
                band_logits_all[i, b] = bl[0, 0].astype(np.float16)

        if (i + 1) % 200 == 0 or i + 1 == n_tokens:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.01)
            print(f"  {i+1}/{n_tokens} ({rate:.0f} tok/s)")

    # Tokenizer data
    token_ids = np.array([tid for tid, _ in sample], dtype=np.int32)
    token_texts = np.array([tok for _, tok in sample])
    scripts = np.array([_detect_script(tok) for _, tok in sample])

    elapsed = time.time() - t0
    metadata = {
        "version": "0.5",
        "type": "mri",
        "architecture": "causal_bank",
        "generated_at": time.time(),
        "elapsed_s": round(elapsed),
        "model": {
            "name": cfg.model_type,
            "n_modes": n_modes,
            "n_experts": n_experts,
            "n_bands": n_bands,
            "embed_dim": embed_dim,
            "vocab_size": vocab_size,
            "n_layers": 1,
            "hidden_size": n_modes,
        },
        "capture": {
            "mode": mode,
            "n_tokens": n_tokens,
        },
        "provenance": {
            "seed": seed,
            "n_index": n_index,
        },
    }

    if output:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        np.savez_compressed(out_dir / "tokens.npz",
                            token_ids=token_ids, token_texts=token_texts,
                            scripts=scripts)

        np.save(out_dir / "substrate.npy", substrate_states)
        np.save(out_dir / "embedding.npy", embeddings)
        np.save(out_dir / "half_lives.npy", backend.model.half_lives)

        if route_weights is not None:
            np.save(out_dir / "routing.npy", route_weights)

        if band_logits_all is not None:
            np.save(out_dir / "band_logits.npy", band_logits_all)

        # Save model weights
        weights = backend.weights()
        weights_dir = out_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        for name, w in weights.items():
            safe_name = name.replace('.', '_').replace('/', '_')
            np.save(weights_dir / f"{safe_name}.npy", w)

        size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
        print(f"\n  Saved to {out_dir}/ ({size / 1e9:.2f} GB)")

    return {
        "metadata": metadata,
        "substrate_states": substrate_states,
        "embeddings": embeddings,
        "route_weights": route_weights,
        "band_logits": band_logits_all,
        "token_ids": token_ids,
        "token_texts": token_texts,
        "scripts": scripts,
    }


def capture_mri(
    backend,
    *,
    mode: str = "template",
    n_index: int | None = None,
    seed: int = 42,
    output: str | None = None,
    db_path: str | None = None,
) -> dict:
    """Capture a complete .mri for a model.

    One forward pass per token. Captures entry and exit residuals
    at every layer. Includes tokenizer data, baselines, and
    optionally discovered directions.

    Dispatches to architecture-specific capture:
    - Transformers (MLX/HF): per-layer entry/exit residuals
    - Causal bank (decepticon): substrate states + routing + band logits
    """
    cfg = backend.config
    if getattr(cfg, 'model_type', '') == 'causal_bank':
        return _capture_mri_causal_bank(backend, mode=mode, n_index=n_index,
                                         seed=seed, output=output)

    from .shrt import _extract_clean_baseline, _extract_template_parts
    from .frt import _detect_script
    from ..cartography.metrics import softmax

    n_layers = cfg.n_layers
    hidden = cfg.hidden_size
    is_mlx = _is_mlx_backend(backend)
    ops = _framework_ops(backend)
    model_inner = ops.model_inner

    t0 = time.time()

    # === Mode-specific baseline ===
    if mode == "raw":
        token_pos = 0
        prefix_ids = []
        suffix_ids = []
        baseline_entry = {i: np.zeros(hidden, dtype=np.float32) for i in range(n_layers)}
        baseline_exit = baseline_entry
        bl_entropy = 0.0
        bl_top_token = "(none)"
    elif mode == "naked":
        bos_id = backend.tokenizer.bos_token_id or 0
        token_pos = 0
        prefix_ids = []
        suffix_ids = []
        bos_input = ops.array([[bos_id]])
        baseline_entry = {}
        baseline_exit = {}
        h = ops.embed(bos_input)
        for i, ly in enumerate(model_inner.layers):
            h = ops.layer_forward(ly, h, None)
            if isinstance(h, tuple): h = h[0]
            baseline_entry[i] = ops.to_numpy_2d(h, 0, 0)
            baseline_exit[i] = baseline_entry[i]
        logits = ops.lm_head_logits(h, 0)
        probs = softmax(logits)
        bl_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        bl_top_token = backend.tokenizer.decode([int(np.argmax(probs))])
    else:
        clean_baseline = _extract_clean_baseline(backend.tokenizer)
        prefix_ids, suffix_ids = _extract_template_parts(backend.tokenizer)
        token_pos = len(prefix_ids)
        bl_tokens = backend.tokenizer.encode(clean_baseline)
        bl_input = ops.array([bl_tokens])
        T_bl = len(bl_tokens)
        mask_bl = ops.triu_mask(T_bl)
        baseline_entry = {}
        baseline_exit = {}
        h = ops.embed(bl_input)
        for i, ly in enumerate(model_inner.layers):
            h = ops.layer_forward(ly, h, mask_bl)
            if isinstance(h, tuple): h = h[0]
            bp = min(token_pos, T_bl - 1)
            baseline_entry[i] = ops.to_numpy_2d(h, bp, 0)
            baseline_exit[i] = ops.to_numpy_2d(h, -1, 0)
        logits = ops.lm_head_logits(h, -1)
        probs = softmax(logits)
        bl_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        bl_top_token = backend.tokenizer.decode([int(np.argmax(probs))])

    # === Build token sample ===
    vocab_size = backend.tokenizer.vocab_size
    real_tokens = []
    seen = set()
    for tid in range(vocab_size):
        tok = backend.tokenizer.decode([tid], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
        if tok.strip() and tok not in seen:
            seen.add(tok)
            real_tokens.append((tid, tok))

    if n_index is None or n_index >= len(real_tokens):
        sample = real_tokens
    else:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(real_tokens), n_index, replace=False)
        sample = [real_tokens[i] for i in sorted(idx)]

    n_tokens = len(sample)
    estimated_bytes = n_tokens * n_layers * 2 * hidden * 2
    print(f"MRI capture ({mode}): {n_tokens} tokens x {n_layers} layers x {hidden} dims")
    print(f"  Estimated: {estimated_bytes / 1e9:.1f} GB")

    # === Resume check ===
    if output:
        out_dir_check = Path(output)
        if out_dir_check.exists():
            # Case 1: all exit layer files exist — skip capture, backfill weights
            existing_exit = sum(1 for i in range(n_layers)
                                if (out_dir_check / f"L{i:02d}_exit.npy").exists())
            if existing_exit == n_layers:
                test_path = out_dir_check / "L00_exit.npy"
                test_arr = np.load(test_path, mmap_mode='r')
                if test_arr.shape == (n_tokens, hidden):
                    print(f"  All {n_layers} layer files exist — resuming from weight extraction")
                    backfill_mri(backend, str(out_dir_check))
                    return json.load(open(out_dir_check / "metadata.json"))
                else:
                    print(f"  WARNING: L00_exit shape {test_arr.shape} != expected ({n_tokens}, {hidden}). Recapturing.")

            # Case 2: capture was interrupted — mmap files exist with progress marker
            progress_file = out_dir_check / ".capture_progress"
            if progress_file.exists():
                progress = json.loads(progress_file.read_text())
                mmap_candidate = Path(progress.get("mmap_dir", ""))
                captured = progress.get("tokens_captured", 0)
                total = progress.get("tokens_total", 0)
                expected_size = total * hidden * 2  # float16 = 2 bytes
                if captured == total and not mmap_candidate.name:
                    # Non-mmap capture completed — progress file is stale, restart
                    print(f"  Non-mmap capture completed but layer files missing. Recapturing.")
                    progress_file.unlink()
                elif captured == total and mmap_candidate.exists():
                    # Verify dat files exist and have correct size
                    dat_ok = True
                    for i in range(n_layers):
                        e_dat = mmap_candidate / f"e{i}.dat"
                        x_dat = mmap_candidate / f"x{i}.dat"
                        if not e_dat.exists() or not x_dat.exists():
                            dat_ok = False
                            print(f"  Missing dat file for layer {i}")
                            break
                        if e_dat.stat().st_size != expected_size or x_dat.stat().st_size != expected_size:
                            dat_ok = False
                            print(f"  Layer {i} dat size mismatch: got {e_dat.stat().st_size}, expected {expected_size}")
                            break
                    if dat_ok:
                        print(f"  Capture verified ({captured} tokens, all dat files valid). Writing layer files...")
                        for i in range(n_layers):
                            mmap_e = np.memmap(str(mmap_candidate / f"e{i}.dat"), dtype=np.float16,
                                               mode='r', shape=(total, hidden))
                            mmap_x = np.memmap(str(mmap_candidate / f"x{i}.dat"), dtype=np.float16,
                                               mode='r', shape=(total, hidden))
                            np.save(out_dir_check / f"L{i:02d}_entry.npy", np.array(mmap_e))
                            np.save(out_dir_check / f"L{i:02d}_exit.npy", np.array(mmap_x))
                        progress_file.unlink()
                        print(f"  Layer files written. Proceeding to weight extraction.")
                        backfill_mri(backend, str(out_dir_check))
                        return json.load(open(out_dir_check / "metadata.json"))
                    else:
                        print(f"  Dat files corrupt. Recapturing.")
                elif captured < total:
                    print(f"  Interrupted capture: {captured}/{total} tokens. Restarting.")

    # === Allocate ===
    n_heads = cfg.n_heads
    if mode in ("raw", "naked"):
        T_seq = 1  # naked uses BOS as baseline, not as prefix
    else:
        T_seq = len(prefix_ids) + 1 + len(suffix_ids)

    # Infer intermediate_size by probing the MLP gate projection
    _mlp0 = model_inner.layers[0].mlp
    _probe_in = ops.embed(ops.array([[0]]))  # [1, 1, hidden]
    if hasattr(_mlp0, 'gate_proj'):
        _probe_out = _mlp0.gate_proj(_probe_in[:, :1, :])
        intermediate_size = _probe_out.shape[-1]
    elif hasattr(_mlp0, 'fc1'):
        _probe_out = _mlp0.fc1(_probe_in[:, :1, :])
        intermediate_size = _probe_out.shape[-1]
    else:
        intermediate_size = hidden * 4

    use_mmap = estimated_bytes > 1e9 and output
    mmap_dir = None

    if output:
        out_dir_alloc = Path(output)
        out_dir_alloc.mkdir(parents=True, exist_ok=True)

    if use_mmap:
        exit_arrays = {}
        entry_arrays = {}
        pre_mlp_arrays = {}
        for i in range(n_layers):
            exit_arrays[i] = np.lib.format.open_memmap(
                str(out_dir_alloc / f"L{i:02d}_exit.npy"),
                mode='w+', dtype=np.float16, shape=(n_tokens, hidden))
            pre_mlp_arrays[i] = np.lib.format.open_memmap(
                str(out_dir_alloc / f"L{i:02d}_pre_mlp.npy"),
                mode='w+', dtype=np.float16, shape=(n_tokens, hidden))
            entry_arrays[i] = np.lib.format.open_memmap(
                str(out_dir_alloc / f"L{i:02d}_entry.npy"),
                mode='w+', dtype=np.float16, shape=(n_tokens, hidden))
        print(f"  Writing directly to {out_dir_alloc}/ (memory-mapped .npy)")
    else:
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
        exit_arrays = {i: np.zeros((n_tokens, hidden), dtype=np.float16) for i in range(n_layers)}
        entry_arrays = {i: np.zeros((n_tokens, hidden), dtype=np.float16) for i in range(n_layers)}
        pre_mlp_arrays = {i: np.zeros((n_tokens, hidden), dtype=np.float16) for i in range(n_layers)}

    # Attn_out: mmap (hidden-sized, small per layer)
    # Attn weights/logits: mmap (tiny per layer)
    # Gate/up: NOT mmap (intermediate-sized, causes page thrashing on USB drives)
    #          Written to disk after capture from in-memory accumulators
    if use_mmap:
        attn_mmap_dir = out_dir_alloc / "attention"
        attn_mmap_dir.mkdir(exist_ok=True)
        attn_out_arrays = {i: np.lib.format.open_memmap(
            str(out_dir_alloc / f"L{i:02d}_attn_out.npy"),
            mode='w+', dtype=np.float16, shape=(n_tokens, hidden))
            for i in range(n_layers)}
        attn_weight_arrays = {i: np.lib.format.open_memmap(
            str(attn_mmap_dir / f"L{i:02d}_weights.npy"),
            mode='w+', dtype=np.float16, shape=(n_tokens, n_heads, T_seq))
            for i in range(n_layers)}
        attn_logit_arrays = {i: np.lib.format.open_memmap(
            str(attn_mmap_dir / f"L{i:02d}_logits.npy"),
            mode='w+', dtype=np.float16, shape=(n_tokens, n_heads, T_seq))
            for i in range(n_layers)}
    else:
        attn_out_arrays = {i: np.zeros((n_tokens, hidden), dtype=np.float16)
                           for i in range(n_layers)}
        attn_weight_arrays = {i: np.zeros((n_tokens, n_heads, T_seq), dtype=np.float16)
                              for i in range(n_layers)}
        attn_logit_arrays = {i: np.zeros((n_tokens, n_heads, T_seq), dtype=np.float16)
                             for i in range(n_layers)}

    # Gate/up: write per-batch to pre-allocated files (no mmap, no accumulation)
    # Pre-allocate .npy files with correct headers, keep file handles open
    gate_files = {}
    up_files = {}
    if output:
        mlp_out_dir = Path(output) / "mlp"
        mlp_out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_layers):
            for name, store in [("gate", gate_files), ("up", up_files)]:
                fpath = mlp_out_dir / f"L{i:02d}_{name}.npy"
                # Write numpy header, then keep file open for batch writes
                header = np.lib.format.header_data_from_array_1_0(
                    np.zeros((n_tokens, intermediate_size), dtype=np.float16))
                with open(fpath, 'wb') as f:
                    np.lib.format.write_array_header_1_0(f, header)
                store[i] = {"path": fpath, "header_size": 128}  # .npy v1 header is 128 bytes typically
        # Measure actual header size from first file
        _test_path = mlp_out_dir / "L00_gate.npy"
        with open(_test_path, 'rb') as f:
            np.lib.format.read_magic(f)
            np.lib.format.read_array_header_1_0(f)
            _header_bytes = f.tell()
        for i in range(n_layers):
            gate_files[i]["header_size"] = _header_bytes
            up_files[i]["header_size"] = _header_bytes

    # Embedding gradient: d(top1_logit) / d(embedding)
    emb_grad_arrays = {i: np.zeros((n_tokens, hidden), dtype=np.float16)
                       for i in range(1)}  # just one array, keyed 0

    print(f"  Capturing: entry+exit+pre_mlp [{hidden}d], "
          f"gate+up [{intermediate_size}], attn_out+weights+logits [{n_heads}x{T_seq}], "
          f"embedding_grad [{hidden}d]")

    # === Tokenizer data (inline) ===
    token_ids = np.array([tid for tid, _ in sample], dtype=np.int32)
    token_texts = np.array([tok for _, tok in sample])
    raw_bytes_list = [tok.encode('utf-8', errors='replace') for _, tok in sample]
    max_bytes = max((len(b) for b in raw_bytes_list), default=1)
    raw_bytes = np.zeros((n_tokens, max_bytes), dtype=np.uint8)
    raw_bytes_lengths = np.array([len(b) for b in raw_bytes_list], dtype=np.int16)
    for i, b in enumerate(raw_bytes_list):
        raw_bytes[i, :len(b)] = list(b)
    merge_ranks = np.array([tid for tid, _ in sample], dtype=np.int32)
    scripts = np.array([_detect_script(tok) for _, tok in sample])

    # === Capture ===
    batch_size = 32  # all modes batchable — template has fixed length
    checkpoint_interval = max(n_tokens // 10, 1000)
    t_start = time.time()

    for batch_start in range(0, n_tokens, batch_size):
        batch_end = min(batch_start + batch_size, n_tokens)
        batch = sample[batch_start:batch_end]
        B = len(batch)

        if mode in ("raw", "naked"):
            inp = ops.array([[tid] for tid, _ in batch])  # [B, 1]
            mask = None
        else:
            inp = ops.array([prefix_ids + [tid] + suffix_ids for tid, _ in batch])
            T = inp.shape[1]
            mask = ops.triu_mask(T)

        h = ops.embed(inp)
        layer_entries = []
        layer_exits = []
        layer_pre_mlps = []
        layer_attn_outs = []
        batch_attn_w = []
        batch_attn_s = []
        batch_gates = []
        batch_ups = []
        for i, ly in enumerate(model_inner.layers):
            h, attn_w, attn_scores, h_pre_mlp, attn_output, gate_val, up_val = \
                ops.layer_decomposed(ly, h, mask)

            layer_pre_mlps.append(h_pre_mlp[:, -1, :] if len(h_pre_mlp.shape) == 3 else h_pre_mlp)
            layer_attn_outs.append(attn_output[:, -1, :] if len(attn_output.shape) == 3 else attn_output)

            aw_row = attn_w[:, :, token_pos, :]
            batch_attn_w.append(np.array(aw_row.astype(ops.float32)).astype(np.float16))
            if attn_scores is not None:
                scores_np = np.array(attn_scores[:, :, token_pos, :].astype(ops.float32)).astype(np.float32)
                scores_np = np.clip(scores_np, -65504, 65504)
                batch_attn_s.append(scores_np.astype(np.float16))
            else:
                batch_attn_s.append(np.zeros((B, n_heads, T_seq), dtype=np.float16))

            if gate_val is not None:
                gv = gate_val[:, -1, :] if len(gate_val.shape) == 3 else gate_val
                uv = up_val[:, -1, :] if len(up_val.shape) == 3 else up_val
                # MLX bfloat16 -> float32 -> numpy float16 (bfloat16 not supported by numpy)
                gv_f32 = gv.astype(ops.float32) if hasattr(gv, 'astype') and not isinstance(gv, np.ndarray) else gv
                uv_f32 = uv.astype(ops.float32) if hasattr(uv, 'astype') and not isinstance(uv, np.ndarray) else uv
                batch_gates.append(np.array(gv_f32).astype(np.float16))
                batch_ups.append(np.array(uv_f32).astype(np.float16))
            else:
                batch_gates.append(np.zeros((B, intermediate_size), dtype=np.float16))
                batch_ups.append(np.zeros((B, intermediate_size), dtype=np.float16))

            layer_entries.append(h[:, token_pos, :])
            layer_exits.append(h[:, -1, :])

        # Store everything
        all_exit = ops.stack_to_numpy(layer_exits)
        all_entry = ops.stack_to_numpy(layer_entries)
        all_pre_mlp = ops.stack_to_numpy(layer_pre_mlps)
        all_attn_out = ops.stack_to_numpy(layer_attn_outs)
        for i in range(n_layers):
            exit_arrays[i][batch_start:batch_end] = (all_exit[i] - baseline_exit[i]).astype(np.float16)
            entry_arrays[i][batch_start:batch_end] = (all_entry[i] - baseline_entry[i]).astype(np.float16)
            pre_mlp_arrays[i][batch_start:batch_end] = all_pre_mlp[i].astype(np.float16)
            attn_out_arrays[i][batch_start:batch_end] = all_attn_out[i].astype(np.float16)
            attn_weight_arrays[i][batch_start:batch_end] = batch_attn_w[i]
            attn_logit_arrays[i][batch_start:batch_end] = batch_attn_s[i]
            # Write gate/up batch directly to file (no memory accumulation)
            if output and i in gate_files:
                row_bytes = intermediate_size * 2  # float16
                offset = gate_files[i]["header_size"] + batch_start * row_bytes
                with open(gate_files[i]["path"], 'r+b') as f:
                    f.seek(offset)
                    f.write(batch_gates[i].tobytes())
                with open(up_files[i]["path"], 'r+b') as f:
                    f.seek(offset)
                    f.write(batch_ups[i].tobytes())

        # Backward pass: embedding gradient
        emb_for_grad = ops.embed(inp)
        eg = ops.embedding_grad(emb_for_grad, mask)
        # eg is [B, seq_len, hidden] — take the token position
        if len(eg.shape) == 3:
            emb_grad_arrays[0][batch_start:batch_end] = eg[:, token_pos, :].astype(np.float16)
        else:
            emb_grad_arrays[0][batch_start:batch_end] = eg.astype(np.float16)

        if (batch_end) % 1000 < batch_size or batch_end == n_tokens:
            elapsed = time.time() - t_start
            rate = batch_end / max(elapsed, 0.01)
            remaining = (n_tokens - batch_end) / max(rate, 1)
            print(f"  {batch_end}/{n_tokens} ({rate:.0f} tok/s, ~{remaining/60:.0f}m remaining)")

        # Checkpoint: save progress marker every 10% so we can resume
        if output and (batch_end % checkpoint_interval < batch_size or batch_end == n_tokens):
            out_dir_ckpt = Path(output)
            out_dir_ckpt.mkdir(parents=True, exist_ok=True)
            ckpt_file = out_dir_ckpt / ".capture_progress"
            ckpt_file.write_text(json.dumps({
                "tokens_captured": batch_end,
                "tokens_total": n_tokens,
                "mmap_dir": str(mmap_dir) if mmap_dir else "",
                "timestamp": time.time(),
            }))

    elapsed = time.time() - t0

    # === Metadata ===
    metadata = {
        "version": "0.5",
        "type": "mri",
        "generated_at": time.time(),
        "elapsed_s": round(elapsed),
        "model": {
            "name": cfg.model_type,
            "n_layers": n_layers,
            "hidden_size": hidden,
            "n_heads": cfg.n_heads,
            "vocab_size": vocab_size,
        },
        "capture": {
            "mode": mode,
            "n_tokens": n_tokens,
            "n_layers": n_layers,
            "token_pos": token_pos,
            "has_entry": True,
            "has_pre_mlp": True,
            "has_attention": True,
            "has_gates": True,
            "gate_format": "full",
            "intermediate_size": intermediate_size,
            "seq_len": T_seq,
            "baseline_entropy": round(bl_entropy, 4),
            "baseline_top_token": bl_top_token,
        },
        "provenance": {
            "seed": seed,
            "n_index": n_index,
            "decode": "skip_special_tokens=True, clean_up_tokenization_spaces=False",
            "all_bugs_fixed": ["add_special_tokens", "skip_special_tokens", "mmap_threshold",
                               "entry_exit_redundancy"],
        },
    }

    # === Save as directory of per-layer files ===
    if output:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        np.savez_compressed(out_dir / "tokens.npz",
                            token_ids=token_ids, token_texts=token_texts,
                            raw_bytes=raw_bytes, raw_bytes_lengths=raw_bytes_lengths,
                            merge_ranks=merge_ranks, scripts=scripts)

        bl_dict = {}
        for i in range(n_layers):
            bl_dict[f"entry_L{i}"] = baseline_entry[i].astype(np.float16)
            bl_dict[f"exit_L{i}"] = baseline_exit[i].astype(np.float16)
        np.savez_compressed(out_dir / "baselines.npz", **bl_dict)

        # Layer files: mmap captures already on disk, in-memory need writing
        total_size = 0
        if use_mmap:
            for i in range(n_layers):
                exit_arrays[i].flush()
                entry_arrays[i].flush()
                total_size += (out_dir / f"L{i:02d}_exit.npy").stat().st_size
                total_size += (out_dir / f"L{i:02d}_entry.npy").stat().st_size
        else:
            for i in range(n_layers):
                np.save(out_dir / f"L{i:02d}_exit.npy", exit_arrays[i])
                np.save(out_dir / f"L{i:02d}_entry.npy", entry_arrays[i])
                total_size += (out_dir / f"L{i:02d}_exit.npy").stat().st_size
                total_size += (out_dir / f"L{i:02d}_entry.npy").stat().st_size
        # Pre-MLP states
        if use_mmap:
            for i in range(n_layers):
                pre_mlp_arrays[i].flush()
                total_size += (out_dir / f"L{i:02d}_pre_mlp.npy").stat().st_size
        else:
            for i in range(n_layers):
                pre_mlp_path = out_dir / f"L{i:02d}_pre_mlp.npy"
                np.save(pre_mlp_path, pre_mlp_arrays[i])
                total_size += pre_mlp_path.stat().st_size

        # Attention and MLP arrays
        attn_dir = out_dir / "attention"
        mlp_dir = out_dir / "mlp"
        if use_mmap:
            for i in range(n_layers):
                attn_out_arrays[i].flush()
                attn_weight_arrays[i].flush()
                attn_logit_arrays[i].flush()
        else:
            attn_dir.mkdir(exist_ok=True)
            for i in range(n_layers):
                np.save(out_dir / f"L{i:02d}_attn_out.npy", attn_out_arrays[i])
                np.save(attn_dir / f"L{i:02d}_weights.npy", attn_weight_arrays[i])
                np.save(attn_dir / f"L{i:02d}_logits.npy", attn_logit_arrays[i])

        # Gate/up: already written per-batch during capture
        if output:
            for i in range(n_layers):
                total_size += (mlp_dir / f"L{i:02d}_gate.npy").stat().st_size
                total_size += (mlp_dir / f"L{i:02d}_up.npy").stat().st_size
        for i in range(n_layers):
            total_size += (out_dir / f"L{i:02d}_attn_out.npy").stat().st_size
            total_size += (attn_dir / f"L{i:02d}_weights.npy").stat().st_size
            total_size += (attn_dir / f"L{i:02d}_logits.npy").stat().st_size
            total_size += (mlp_dir / f"L{i:02d}_gate.npy").stat().st_size
            total_size += (mlp_dir / f"L{i:02d}_up.npy").stat().st_size

        # Embedding gradient
        np.save(out_dir / "embedding_grad.npy", emb_grad_arrays[0])
        total_size += (out_dir / "embedding_grad.npy").stat().st_size

        print(f"  Layer files written ({total_size / 1e9:.1f} GB). Extracting weights...")

        # Layer norm weights (small, 2 per layer + final norm)
        norms_path = out_dir / "norms.npz"
        if not norms_path.exists():
            print(f"  Extracting norm weights...")
            norm_dict = {}
            def _w2np(w):
                if is_mlx:
                    import mlx.core as mx
                    return np.array(mx.array(w).astype(mx.float32))
                return w.float().cpu().numpy()
            for layer_idx in range(n_layers):
                ly = model_inner.layers[layer_idx]
                if hasattr(ly, 'input_layernorm'):
                    norm_dict[f"input_L{layer_idx}"] = _w2np(ly.input_layernorm.weight)
                if hasattr(ly, 'post_attention_layernorm'):
                    norm_dict[f"post_attn_L{layer_idx}"] = _w2np(ly.post_attention_layernorm.weight)
            final_norm = getattr(model_inner, 'norm', None) or getattr(model_inner, 'final_layernorm', None)
            if final_norm is not None:
                norm_dict["final"] = _w2np(final_norm.weight)
            np.savez_compressed(norms_path, **norm_dict)

        # Embedding matrix (probed to dequantize, float32 to avoid overflow)
        embed_path = out_dir / "embedding.npy"
        if not embed_path.exists():
            print(f"  Extracting embedding matrix...")
            vocab = cfg.vocab_size
            embed_vecs = []
            batch_e = 256
            for s in range(0, vocab, batch_e):
                e = min(s + batch_e, vocab)
                ids = ops.array([[i] for i in range(s, e)])
                v = ops.embed(ids)
                if is_mlx:
                    import mlx.core as mx
                    v = np.array(v.astype(mx.float32)[:, 0, :])
                else:
                    v = v[:, 0, :].float().cpu().numpy()
                embed_vecs.append(v)
            embed_weights = np.concatenate(embed_vecs, axis=0).astype(np.float32)
            np.save(embed_path, embed_weights)
            total_size += embed_path.stat().st_size

        # Raw lm_head (without norm, for logit lens at intermediate layers)
        lmhead_raw_path = out_dir / "lmhead_raw.npy"
        if not lmhead_raw_path.exists():
            print(f"  Extracting raw lm_head (no norm)...")
            if is_mlx:
                import mlx.core as mx
                from ..cartography.runtime import _lm_head
                cols = []
                for s in range(0, hidden, 64):
                    e = min(s + 64, hidden)
                    probe = np.zeros((1, e - s, hidden), dtype=np.float32)
                    for j in range(e - s):
                        probe[0, j, s + j] = 1.0
                    out = np.array(_lm_head(backend.model, mx.array(probe).astype(mx.float16)).astype(mx.float32)[0])
                    cols.append(out.T)
                np.save(lmhead_raw_path, np.concatenate(cols, axis=1).astype(np.float32))
            else:
                lm_head_mod = backend.hf_model.lm_head
                if hasattr(lm_head_mod, 'weight') and lm_head_mod.weight is not None:
                    np.save(lmhead_raw_path, lm_head_mod.weight.float().cpu().numpy())
                else:
                    np.save(lmhead_raw_path, _extract_weight(lm_head_mod, hidden, is_mlx))
            total_size += lmhead_raw_path.stat().st_size

        # lm_head matrix (norm + unembedding composed, for final-layer quick analysis)
        lmhead_path = out_dir / "lmhead.npy"
        if not lmhead_path.exists():
            print(f"  Extracting lm_head...")
            if is_mlx:
                import mlx.core as mx
                from ..cartography.runtime import _lm_head
                cols = []
                for start in range(0, hidden, 64):
                    end = min(start + 64, hidden)
                    inp_probe = np.zeros((1, end - start, hidden), dtype=np.float16)
                    for j in range(end - start):
                        inp_probe[0, j, start + j] = 1.0
                    h_probe = mx.array(inp_probe)
                    _fn = getattr(model_inner, 'norm', None) or getattr(model_inner, 'final_layernorm', None)
                    h_normed = _fn(h_probe) if _fn else h_probe
                    out = np.array(_lm_head(backend.model, h_normed).astype(mx.float32)[0])
                    cols.append(out.T)
                np.save(lmhead_path, np.concatenate(cols, axis=1).astype(np.float32))
            else:
                import torch
                cols = []
                for start in range(0, hidden, 64):
                    end = min(start + 64, hidden)
                    probe = torch.zeros(1, end - start, hidden, device=next(backend.hf_model.parameters()).device)
                    for j in range(end - start):
                        probe[0, j, start + j] = 1.0
                    with torch.no_grad():
                        _fn = getattr(model_inner, 'norm', None) or getattr(model_inner, 'final_layernorm', None)
                        normed = _fn(probe) if _fn else probe
                        out = backend.hf_model.lm_head(normed).float().cpu().numpy()[0]
                    cols.append(out.T)
                np.save(lmhead_path, np.concatenate(cols, axis=1).astype(np.float32))
            total_size += lmhead_path.stat().st_size

        # All projection weights per layer (dequantized by probing)
        weights_dir = out_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        n_weight_complete = sum(1 for i in range(n_layers)
                                if (weights_dir / f"L{i:02d}" / "down_proj.npy").exists())
        if n_weight_complete < n_layers:
            print(f"  Extracting projection weights ({n_weight_complete}/{n_layers} complete)...")
            n_heads = cfg.n_heads
            head_dim = hidden // n_heads
            kv_heads = getattr(cfg, 'num_key_value_heads', n_heads)
            kv_dim = kv_heads * head_dim

            def _infer_in_dim(module, fallback):
                """Infer input dimension from quantized/linear weight shape."""
                if hasattr(module, 'weight'):
                    w = module.weight
                    if hasattr(module, 'bits'):
                        return w.shape[1] * 32 // module.bits
                    elif hasattr(w, 'shape') and len(w.shape) == 2:
                        return w.shape[1]
                return fallback

            for layer_idx in range(n_layers):
                ly = model_inner.layers[layer_idx]
                attn = ly.self_attn
                mlp_mod = ly.mlp
                layer_dir = weights_dir / f"L{layer_idx:02d}"
                if (layer_dir / "down_proj.npy").exists():
                    continue
                layer_dir.mkdir(exist_ok=True)

                # Attention — handle fused QKV (Phi-3), separate Q/K/V, or Phi-2 naming
                if hasattr(attn, 'q_proj'):
                    np.save(layer_dir / "q_proj.npy", _extract_weight(attn.q_proj, hidden, is_mlx))
                    np.save(layer_dir / "k_proj.npy", _extract_weight(attn.k_proj, hidden, is_mlx))
                    np.save(layer_dir / "v_proj.npy", _extract_weight(attn.v_proj, hidden, is_mlx))
                elif hasattr(attn, 'qkv_proj'):
                    qkv = _extract_weight(attn.qkv_proj, hidden, is_mlx)
                    q_dim = cfg.n_heads * (hidden // cfg.n_heads)
                    kv_dim = getattr(cfg, 'n_kv_heads', cfg.n_heads) * (hidden // cfg.n_heads)
                    np.save(layer_dir / "q_proj.npy", qkv[:q_dim])
                    np.save(layer_dir / "k_proj.npy", qkv[q_dim:q_dim + kv_dim])
                    np.save(layer_dir / "v_proj.npy", qkv[q_dim + kv_dim:])
                o_proj = getattr(attn, 'o_proj', None) or getattr(attn, 'dense', None)
                if o_proj:
                    o_in = _infer_in_dim(o_proj, hidden)
                    np.save(layer_dir / "o_proj.npy", _extract_weight(o_proj, o_in, is_mlx))

                # MLP — handle gate/up/down, fused gate_up, or fc1/fc2 (Phi-2)
                if hasattr(mlp_mod, 'gate_proj'):
                    gate_in = _infer_in_dim(mlp_mod.gate_proj, hidden)
                    np.save(layer_dir / "gate_proj.npy", _extract_weight(mlp_mod.gate_proj, gate_in, is_mlx))
                    up_in = _infer_in_dim(mlp_mod.up_proj, hidden)
                    np.save(layer_dir / "up_proj.npy", _extract_weight(mlp_mod.up_proj, up_in, is_mlx))
                    mlp_inner_dim = np.load(layer_dir / "gate_proj.npy").shape[0]
                    np.save(layer_dir / "down_proj.npy", _extract_weight(mlp_mod.down_proj, mlp_inner_dim, is_mlx))
                elif hasattr(mlp_mod, 'gate_up_proj'):
                    gu = _extract_weight(mlp_mod.gate_up_proj, hidden, is_mlx)
                    mid = gu.shape[0] // 2
                    np.save(layer_dir / "gate_proj.npy", gu[:mid])
                    np.save(layer_dir / "up_proj.npy", gu[mid:])
                    mlp_inner_dim = mid
                    np.save(layer_dir / "down_proj.npy", _extract_weight(mlp_mod.down_proj, mlp_inner_dim, is_mlx))
                elif hasattr(mlp_mod, 'fc1'):
                    fc1_in = _infer_in_dim(mlp_mod.fc1, hidden)
                    np.save(layer_dir / "gate_proj.npy", _extract_weight(mlp_mod.fc1, fc1_in, is_mlx))
                    fc1_out = np.load(layer_dir / "gate_proj.npy").shape[0]
                    np.save(layer_dir / "down_proj.npy", _extract_weight(mlp_mod.fc2, fc1_out, is_mlx))

                if (layer_idx + 1) % 8 == 0:
                    print(f"    L{layer_idx} done")

        # Directions from DB
        if db_path:
            dir_dir = out_dir / "directions"
            if not dir_dir.exists():
                dir_dir.mkdir()
                try:
                    from .compare import discover_safety_direction
                    safety = discover_safety_direction(backend, db_path, n_harmful=100, n_benign=100)
                    if 'direction' in safety:
                        np.save(dir_dir / "safety.npy", safety['direction'])
                        with open(dir_dir / "safety.json", 'w') as f:
                            json.dump({k: v for k, v in safety.items() if k != 'direction'}, f, indent=2)
                except Exception:
                    pass

        print(f"\n  Saved to {out_dir}/ ({total_size / 1e9:.2f} GB)")

        # Clean up progress marker
        progress_marker = out_dir / ".capture_progress"
        if progress_marker.exists():
            progress_marker.unlink()

    return metadata


def backfill_mri(backend, mri_path: str) -> dict:
    """Fill missing data in an existing MRI directory.

    Loads the model once, extracts whatever is missing (embedding, norms,
    lm_head_raw, weights), writes to the MRI directory. Doesn't recapture
    layer states — those are already stored.
    """
    p = Path(mri_path)
    if not p.is_dir():
        return {"error": f"Not an MRI directory: {mri_path}"}

    with open(p / "metadata.json") as f:
        meta = json.load(f)

    cfg = backend.config
    ops = _framework_ops(backend)
    model_inner = ops.model_inner
    hidden = cfg.hidden_size
    n_layers = cfg.n_layers
    filled = []
    is_mlx = _is_mlx_backend(backend)

    # Embedding
    if not (p / "embedding.npy").exists():
        print(f"  Backfilling embedding...")
        vocab = cfg.vocab_size
        vecs = []
        for s in range(0, vocab, 256):
            e = min(s + 256, vocab)
            ids = ops.array([[i] for i in range(s, e)])
            v = ops.embed(ids)
            if is_mlx:
                import mlx.core as mx
                v = np.array(v.astype(mx.float32)[:, 0, :])
            else:
                v = v[:, 0, :].float().cpu().numpy()
            vecs.append(v)
        np.save(p / "embedding.npy", np.concatenate(vecs, axis=0).astype(np.float32))
        filled.append("embedding")

    # Norms
    if not (p / "norms.npz").exists():
        print(f"  Backfilling norms...")
        nd = {}
        def _w2np(w):
            if is_mlx:
                import mlx.core as mx
                return np.array(mx.array(w).astype(mx.float32))
            return w.float().cpu().numpy()
        for i in range(n_layers):
            ly = model_inner.layers[i]
            if hasattr(ly, 'input_layernorm'):
                nd[f"input_L{i}"] = _w2np(ly.input_layernorm.weight)
            if hasattr(ly, 'post_attention_layernorm'):
                nd[f"post_attn_L{i}"] = _w2np(ly.post_attention_layernorm.weight)
        final_norm = getattr(model_inner, 'norm', None) or getattr(model_inner, 'final_layernorm', None)
        if final_norm is not None:
            nd["final"] = _w2np(final_norm.weight)
        np.savez_compressed(p / "norms.npz", **nd)
        filled.append("norms")

    # Raw lm_head (without norm)
    if not (p / "lmhead_raw.npy").exists():
        print(f"  Backfilling lmhead_raw...")
        if is_mlx:
            import mlx.core as mx
            from ..cartography.runtime import _lm_head
            cols = []
            for s in range(0, hidden, 64):
                e = min(s + 64, hidden)
                probe = np.zeros((1, e - s, hidden), dtype=np.float32)
                for j in range(e - s):
                    probe[0, j, s + j] = 1.0
                out = np.array(_lm_head(backend.model, mx.array(probe).astype(mx.float16)).astype(mx.float32)[0])
                cols.append(out.T)
            np.save(p / "lmhead_raw.npy", np.concatenate(cols, axis=1).astype(np.float32))
        else:
            # HF: lm_head is typically a Linear layer
            lm_head = backend.hf_model.lm_head
            if hasattr(lm_head, 'weight') and lm_head.weight is not None:
                np.save(p / "lmhead_raw.npy", lm_head.weight.float().cpu().numpy())
            else:
                np.save(p / "lmhead_raw.npy", _extract_weight(lm_head, hidden, is_mlx))
        filled.append("lmhead_raw")

    # Weights — check per-layer completeness
    weights_dir = p / "weights"
    weights_dir.mkdir(exist_ok=True)
    n_complete = sum(1 for i in range(n_layers)
                     if (weights_dir / f"L{i:02d}" / "down_proj.npy").exists())
    if n_complete < n_layers:
        print(f"  Backfilling projection weights ({n_complete}/{n_layers} complete)...")
        for layer_idx in range(n_layers):
            layer_dir = weights_dir / f"L{layer_idx:02d}"
            if (layer_dir / "down_proj.npy").exists():
                continue
            layer_dir.mkdir(exist_ok=True)
            ly = model_inner.layers[layer_idx]

            attn = ly.self_attn
            def _bf_in_dim(module, fallback):
                if hasattr(module, 'weight') and hasattr(module, 'bits'):
                    return module.weight.shape[1] * 32 // module.bits
                return fallback
            if hasattr(attn, 'q_proj'):
                np.save(layer_dir / "q_proj.npy", _extract_weight(attn.q_proj, hidden, is_mlx))
                np.save(layer_dir / "k_proj.npy", _extract_weight(attn.k_proj, hidden, is_mlx))
                np.save(layer_dir / "v_proj.npy", _extract_weight(attn.v_proj, hidden, is_mlx))
            elif hasattr(attn, 'qkv_proj'):
                qkv = _extract_weight(attn.qkv_proj, hidden, is_mlx)
                q_dim = cfg.n_heads * (hidden // cfg.n_heads)
                kv_dim = getattr(cfg, 'n_kv_heads', cfg.n_heads) * (hidden // cfg.n_heads)
                np.save(layer_dir / "q_proj.npy", qkv[:q_dim])
                np.save(layer_dir / "k_proj.npy", qkv[q_dim:q_dim + kv_dim])
                np.save(layer_dir / "v_proj.npy", qkv[q_dim + kv_dim:])
            o_proj = getattr(attn, 'o_proj', None) or getattr(attn, 'dense', None)
            if o_proj:
                np.save(layer_dir / "o_proj.npy", _extract_weight(o_proj, _bf_in_dim(o_proj, hidden), is_mlx))
            if hasattr(ly.mlp, 'gate_proj'):
                np.save(layer_dir / "gate_proj.npy", _extract_weight(ly.mlp.gate_proj, _bf_in_dim(ly.mlp.gate_proj, hidden), is_mlx))
                np.save(layer_dir / "up_proj.npy", _extract_weight(ly.mlp.up_proj, _bf_in_dim(ly.mlp.up_proj, hidden), is_mlx))
            elif hasattr(ly.mlp, 'gate_up_proj'):
                gu = _extract_weight(ly.mlp.gate_up_proj, hidden, is_mlx)
                mid = gu.shape[0] // 2
                np.save(layer_dir / "gate_proj.npy", gu[:mid])
                np.save(layer_dir / "up_proj.npy", gu[mid:])
            elif hasattr(ly.mlp, 'fc1'):
                np.save(layer_dir / "gate_proj.npy", _extract_weight(ly.mlp.fc1, _bf_in_dim(ly.mlp.fc1, hidden), is_mlx))
            mlp_dim = np.load(layer_dir / "gate_proj.npy").shape[0]
            np.save(layer_dir / "down_proj.npy", _extract_weight(ly.mlp.down_proj, mlp_dim, is_mlx))

            if (layer_idx + 1) % 8 == 0:
                print(f"    L{layer_idx} done")
        filled.append("weights")

    return {"filled": filled, "path": str(p)}


def verify_mri(path: str) -> dict:
    """Deep health check on an MRI directory. No model needed.

    Checks:
      - metadata.json present and valid
      - All layer files present with correct shapes
      - Token counts consistent across layers, tokens.npz, baselines
      - NaN/Inf sampling at multiple positions per layer
      - Weight completeness per layer (q, k, v, o, gate, up, down)
      - Embedding shape vs vocab/hidden
      - lmhead_raw shape vs vocab/hidden
      - Norms present for each layer

    Returns dict with 'healthy' bool, 'issues' list, and 'summary' dict.
    """
    p = Path(path)
    issues = []

    if not p.is_dir():
        return {"healthy": False, "issues": [f"Not a directory: {path}"], "summary": {}}

    meta_path = p / "metadata.json"
    if not meta_path.exists():
        return {"healthy": False, "issues": ["Missing metadata.json"], "summary": {}}

    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        return {"healthy": False, "issues": [f"Bad metadata.json: {e}"], "summary": {}}

    arch = meta.get("architecture", "transformer")
    model = meta.get("model", {})
    capture = meta.get("capture", {})
    n_layers = model.get("n_layers", 0)
    hidden = model.get("hidden_size", 0)
    vocab = model.get("vocab_size", 0)
    n_tok = capture.get("n_tokens", 0)
    mode = capture.get("mode", "?")
    model_name = model.get("name", "?")

    summary = {
        "path": str(p),
        "name": p.name,
        "model": model_name,
        "mode": mode,
        "n_tokens": n_tok,
        "n_layers": n_layers,
        "hidden_size": hidden,
        "vocab_size": vocab,
        "architecture": arch,
    }

    if arch == "causal_bank":
        for fname in ["substrate.npy", "half_lives.npy"]:
            if not (p / fname).exists():
                issues.append(f"Missing {fname}")
            else:
                try:
                    arr = np.load(p / fname, mmap_mode='r')
                    if fname == "substrate.npy" and arr.shape[0] != n_tok:
                        issues.append(f"{fname} tokens={arr.shape[0]} expected={n_tok}")
                    if np.any(np.isnan(arr[:min(100, len(arr))].astype(np.float32))):
                        issues.append(f"{fname} contains NaN")
                except Exception as e:
                    issues.append(f"{fname} read error: {e}")

        if (p / "embedding.npy").exists():
            try:
                emb = np.load(p / "embedding.npy", mmap_mode='r')
                if emb.shape[0] != n_tok:
                    issues.append(f"embedding tokens={emb.shape[0]} expected={n_tok}")
            except Exception as e:
                issues.append(f"embedding read error: {e}")
        else:
            issues.append("Missing embedding.npy")

        return {"healthy": len(issues) == 0, "issues": issues, "summary": summary}

    # --- Transformer MRI ---

    tok_path = p / "tokens.npz"
    if tok_path.exists():
        try:
            tok = np.load(tok_path, allow_pickle=False)
            tok_count = len(tok["token_ids"])
            if tok_count != n_tok:
                issues.append(f"tokens.npz has {tok_count} tokens, metadata says {n_tok}")
        except Exception as e:
            issues.append(f"tokens.npz read error: {e}")
    else:
        issues.append("Missing tokens.npz")

    has_entry = capture.get("has_entry", True)  # legacy MRIs default to True
    suffixes = ["entry", "exit"] if has_entry else ["exit"]
    for i in range(n_layers):
        for suffix in suffixes:
            fpath = p / f"L{i:02d}_{suffix}.npy"
            if not fpath.exists():
                issues.append(f"Missing L{i:02d}_{suffix}.npy")
                continue
            try:
                arr = np.load(fpath, mmap_mode='r')
                if arr.shape != (n_tok, hidden):
                    issues.append(f"L{i:02d}_{suffix} shape={arr.shape} expected=({n_tok},{hidden})")
            except Exception as e:
                issues.append(f"L{i:02d}_{suffix} read error: {e}")

    # NaN/Inf: sample 3 layers x 3 positions
    check_layers = [0, n_layers // 2, n_layers - 1] if n_layers > 2 else list(range(n_layers))
    check_positions = [0, n_tok // 2, n_tok - 1] if n_tok > 2 else list(range(n_tok))
    for li in check_layers:
        fpath = p / f"L{li:02d}_exit.npy"
        if not fpath.exists():
            continue
        try:
            arr = np.load(fpath, mmap_mode='r')
            for pos in check_positions:
                row = arr[pos].astype(np.float32)
                if np.any(np.isnan(row)):
                    issues.append(f"L{li:02d}_exit[{pos}] contains NaN")
                if np.any(np.isinf(row)):
                    issues.append(f"L{li:02d}_exit[{pos}] contains Inf")
        except Exception:
            pass

    bl_path = p / "baselines.npz"
    if bl_path.exists():
        try:
            bl = np.load(bl_path, allow_pickle=False)
            expected_keys = n_layers * 2
            if len(bl.files) < expected_keys:
                issues.append(f"baselines.npz has {len(bl.files)} arrays, expected {expected_keys}")
        except Exception as e:
            issues.append(f"baselines.npz read error: {e}")
    else:
        issues.append("Missing baselines.npz")

    embed_path = p / "embedding.npy"
    if embed_path.exists():
        try:
            emb = np.load(embed_path, mmap_mode='r')
            # Models pad vocab to multiples of 64/128 — embedding rows >= tokenizer vocab is OK
            if vocab and emb.shape[0] < vocab:
                issues.append(f"embedding shape[0]={emb.shape[0]} < vocab={vocab}")
            if len(emb.shape) > 1 and emb.shape[1] != hidden:
                issues.append(f"embedding shape[1]={emb.shape[1]} expected hidden={hidden}")
        except Exception as e:
            issues.append(f"embedding read error: {e}")
    else:
        issues.append("Missing embedding.npy")

    lmhead_path = p / "lmhead_raw.npy"
    if lmhead_path.exists():
        try:
            lm = np.load(lmhead_path, mmap_mode='r')
            if vocab and lm.shape[0] < vocab:
                issues.append(f"lmhead_raw shape[0]={lm.shape[0]} < vocab={vocab}")
        except Exception as e:
            issues.append(f"lmhead_raw read error: {e}")
    else:
        issues.append("Missing lmhead_raw.npy")

    norms_path = p / "norms.npz"
    if norms_path.exists():
        try:
            norms = np.load(norms_path, allow_pickle=False)
            if "final" not in norms.files:
                issues.append("norms.npz missing 'final' norm")
        except Exception as e:
            issues.append(f"norms.npz read error: {e}")
    else:
        issues.append("Missing norms.npz")

    wdir = p / "weights"
    if wdir.exists():
        # Minimum: down_proj (completion marker) + at least 3 attention projections
        # Phi-2 has no up_proj. Phi-3 splits fused qkv. Architecture varies.
        for i in range(n_layers):
            ldir = wdir / f"L{i:02d}"
            if not ldir.exists():
                issues.append(f"Missing weights/L{i:02d}/")
                continue
            actual = {f.name for f in ldir.glob("*.npy")}
            if "down_proj.npy" not in actual:
                issues.append(f"L{i:02d} weights incomplete (no down_proj.npy)")
            elif len(actual) < 4:
                issues.append(f"L{i:02d} weights sparse ({len(actual)} files, expected >= 4)")
    else:
        issues.append("Missing weights/ directory")

    # Attention weights (template mode captures)
    attn_dir = p / "attention"
    has_attn = capture.get("has_attention", False)
    if has_attn and attn_dir.exists():
        seq_len = capture.get("seq_len", 0)
        n_heads = model.get("n_heads", 0)
        for li in check_layers:
            apath = attn_dir / f"L{li:02d}_weights.npy"
            if not apath.exists():
                issues.append(f"Missing attention/L{li:02d}_attn.npy")
                continue
            try:
                aw = np.load(apath, mmap_mode='r')
                if aw.shape != (n_tok, n_heads, seq_len):
                    issues.append(f"L{li:02d}_attn shape={aw.shape} expected=({n_tok},{n_heads},{seq_len})")
                # Attention rows must sum to ~1.0 (softmax output)
                row_sums = aw[:min(100, n_tok)].astype(np.float32).sum(axis=2)
                if row_sums.min() < 0.95 or row_sums.max() > 1.05:
                    issues.append(f"L{li:02d}_attn row sums out of range [{row_sums.min():.3f},{row_sums.max():.3f}]")
                if np.any(np.isnan(aw[:min(100, n_tok)].astype(np.float32))):
                    issues.append(f"L{li:02d}_attn contains NaN")
            except Exception as e:
                issues.append(f"L{li:02d}_attn read error: {e}")
    elif has_attn and not attn_dir.exists():
        issues.append("metadata says has_attention=True but attention/ directory missing")

    # MLP gate activations (full .npy or legacy top-K .npz)
    # MLP gate activations: new format in mlp/, legacy in gates/
    mlp_dir = p / "mlp"
    gates_dir = p / "gates"
    has_gates = capture.get("has_gates", False)
    if has_gates and (mlp_dir.exists() or gates_dir.exists()):
        for li in check_layers:
            gpath = mlp_dir / f"L{li:02d}_gate.npy"
            gpath_legacy = gates_dir / f"L{li:02d}_gates.npy"
            gpath_topk = gates_dir / f"L{li:02d}_gates.npz"
            if gpath.exists():
                try:
                    gv = np.load(gpath, mmap_mode='r')
                    inter = capture.get("intermediate_size", 0)
                    if inter and gv.shape != (n_tok, inter):
                        issues.append(f"L{li:02d} gate shape={gv.shape} expected=({n_tok},{inter})")
                    if np.all(gv[:min(100, n_tok)] == 0):
                        issues.append(f"L{li:02d} gate all zero (dead MLP?)")
                    if np.any(np.isnan(gv[:min(100, n_tok)].astype(np.float32))):
                        issues.append(f"L{li:02d} gate contains NaN")
                except Exception as e:
                    issues.append(f"L{li:02d} gate read error: {e}")
            elif gpath_legacy.exists() or gpath_topk.exists():
                pass  # legacy format, don't flag as missing
            else:
                issues.append(f"Missing mlp/L{li:02d}_gate.npy")
    elif has_gates:
        issues.append("metadata says has_gates=True but mlp/ and gates/ directories missing")

    summary["has_attention"] = has_attn
    summary["has_gates"] = has_gates
    summary["size_gb"] = round(
        sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9, 1)
    return {"healthy": len(issues) == 0, "issues": issues, "summary": summary}


class LazyMRI(dict):
    """Dict-like object that loads .npy arrays on first access.

    Only metadata and token data are loaded eagerly (small).
    All per-layer arrays (exit, entry, pre_mlp, gate, up, attn, weights)
    are loaded lazily via mmap on first key access.
    """

    def __init__(self, path: str, eager_data: dict, file_map: dict):
        super().__init__(eager_data)
        self._file_map = file_map  # key -> (filepath, mmap_mode)
        self._path = path

    def __getitem__(self, key):
        if key not in self.keys() and key in self._file_map:
            fpath, mode = self._file_map[key]
            if fpath.suffix == '.npz':
                data = np.load(fpath, allow_pickle=False)
                for k in data.files:
                    super().__setitem__(k, data[k])
            else:
                super().__setitem__(key, np.load(fpath, mmap_mode=mode))
        # Compatibility: vectors = last exit layer
        if key == "vectors" and key not in self.keys():
            meta = super().__getitem__("metadata")
            n = meta.get("model", {}).get("n_layers", 0)
            if n > 0 and f"exit_L{n-1}" in self:
                super().__setitem__("vectors", self[f"exit_L{n-1}"])
        if key == "deltas" and key not in self.keys():
            if "vectors" in self:
                v = np.array(self["vectors"]).astype(np.float32)
                super().__setitem__("deltas", np.linalg.norm(v, axis=1).astype(np.float32))
        return super().__getitem__(key)

    def __contains__(self, key):
        if super().__contains__(key) or key in self._file_map:
            return True
        # Computed keys
        if key in ("vectors", "deltas"):
            meta = super().__getitem__("metadata") if "metadata" in self.keys() else {}
            n = meta.get("model", {}).get("n_layers", 0)
            return n > 0 and f"exit_L{n-1}" in self
        return False

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def load_mri(path: str) -> LazyMRI:
    """Load a .mri directory lazily. Arrays loaded on first access.

    Only metadata and token data are loaded eagerly.
    Per-layer arrays (exit, gate, up, attn, weights) are memory-mapped
    on demand — a 99 GB MRI uses ~1 MB until you access a specific layer.
    """
    p = Path(path)

    if not p.is_dir():
        raise ValueError(
            f"Not an MRI directory: {path}. "
            f"Legacy .shrt.npz/.frt.npz files must be recaptured using 'heinrich mri'."
        )

    if not (p / "metadata.json").exists():
        raise ValueError(f"Missing metadata.json in {path}. Not a valid MRI directory.")

    with open(p / "metadata.json") as f:
        meta = json.load(f)

    # Eager: metadata + token data (small)
    tokens = np.load(p / "tokens.npz", allow_pickle=False)
    eager = {
        "metadata": meta,
        "path": str(p),
        "token_ids": tokens["token_ids"],
        "token_texts": tokens["token_texts"],
    }
    for key in tokens.files:
        eager[key] = tokens[key]

    # Build file map: key -> (path, mmap_mode) for lazy loading
    fmap = {}
    arch = meta.get("architecture", "transformer")

    if arch == "causal_bank":
        for name in ["substrate", "routing", "half_lives", "embedding"]:
            fp = p / f"{name}.npy"
            if fp.exists():
                fmap[name if name != "substrate" else "substrate_states"] = (fp, 'r')
    else:
        n_layers = meta["model"]["n_layers"]

        # Baselines (small, load eagerly via npz)
        bl_path = p / "baselines.npz"
        if bl_path.exists():
            baselines = np.load(bl_path, allow_pickle=False)
            for key in baselines.files:
                eager[f"baseline_{key}"] = baselines[key]

        # Norms (small, load eagerly)
        norms_path = p / "norms.npz"
        if norms_path.exists():
            norms = np.load(norms_path, allow_pickle=False)
            for key in norms.files:
                eager[f"norm_{key}"] = norms[key]

        # Per-layer arrays: ALL lazy
        for i in range(n_layers):
            for name in ["exit", "entry", "pre_mlp", "attn_out"]:
                fp = p / f"L{i:02d}_{name}.npy"
                if fp.exists():
                    fmap[f"{name}_L{i}"] = (fp, 'r')

        # Embedding, lmhead
        for name in ["embedding", "lmhead", "lmhead_raw"]:
            fp = p / f"{name}.npy"
            if fp.exists():
                fmap[name] = (fp, 'r')

        # Embedding gradient
        eg = p / "embedding_grad.npy"
        if eg.exists():
            fmap["embedding_grad"] = (eg, 'r')

        # Directions
        dir_dir = p / "directions"
        if dir_dir.exists():
            for npy in dir_dir.glob("*.npy"):
                fmap[f"direction_{npy.stem}"] = (npy, 'r')

        # Weights per layer
        weights_dir = p / "weights"
        if weights_dir.exists():
            for layer_d in sorted(weights_dir.glob("L*")):
                if not layer_d.is_dir():
                    continue
                layer_idx = int(layer_d.name[1:])
                for npy in layer_d.glob("*.npy"):
                    fmap[f"{npy.stem}_L{layer_idx}"] = (npy, 'r')

        # Attention weights and logits
        attn_dir = p / "attention"
        if attn_dir.exists():
            for f_w in sorted(attn_dir.glob("L*_weights.npy")):
                li = int(f_w.name[1:3])
                fmap[f"attn_weights_L{li}"] = (f_w, 'r')
            for f_l in sorted(attn_dir.glob("L*_logits.npy")):
                li = int(f_l.name[1:3])
                fmap[f"attn_logits_L{li}"] = (f_l, 'r')
            for f_a in sorted(attn_dir.glob("L*_attn.npy")):
                li = int(f_a.name[1:3])
                if f"attn_weights_L{li}" not in fmap:
                    fmap[f"attn_weights_L{li}"] = (f_a, 'r')

        # MLP gate + up
        mlp_dir = p / "mlp"
        if mlp_dir.exists():
            for gf in sorted(mlp_dir.glob("L*_gate.npy")):
                li = int(gf.name[1:3])
                fmap[f"gate_L{li}"] = (gf, 'r')
            for uf in sorted(mlp_dir.glob("L*_up.npy")):
                li = int(uf.name[1:3])
                fmap[f"up_L{li}"] = (uf, 'r')

        # Legacy gates/ directory
        gates_dir = p / "gates"
        if gates_dir.exists() and not (mlp_dir and mlp_dir.exists()):
            for gf in sorted(gates_dir.glob("L*_gates.npy")):
                li = int(gf.name[1:3])
                fmap[f"gate_L{li}"] = (gf, 'r')

    # Causal bank weights
    if arch == "causal_bank":
        weights_dir = p / "weights"
        if weights_dir.exists():
            for npy in weights_dir.glob("*.npy"):
                fmap[f"weight_{npy.stem}"] = (npy, 'r')

    # byte_counts alias
    if "raw_bytes_lengths" in eager and "byte_counts" not in eager:
        eager["byte_counts"] = eager["raw_bytes_lengths"]

    return LazyMRI(str(p), eager, fmap)
