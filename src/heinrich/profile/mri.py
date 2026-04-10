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
    return hasattr(backend, 'model') and type(backend).__name__ != 'HFBackend'


def _framework_ops(backend):
    """Return framework-specific operations as a namespace dict.

    Works for both MLX and PyTorch/HF backends. Abstracts:
      array, triu, full, stack, to_numpy, embed, layer_forward, norm, lm_head, dtype
    """
    if _is_mlx_backend(backend):
        import mlx.core as mx
        from ..cartography.perturb import _mask_dtype
        from ..cartography.runtime import _lm_head

        model_inner = getattr(backend.model, 'model', backend.model)
        mdtype = _mask_dtype(backend.model)
        _final_norm = getattr(model_inner, 'norm', None) or getattr(model_inner, 'final_layernorm', None)

        return {
            "model_inner": model_inner,
            "array": lambda x: mx.array(x),
            "triu_mask": lambda T: mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None,
            "stack_to_numpy": lambda tensors: np.array(mx.stack(tensors).astype(mx.float32)),
            "to_numpy_1d": lambda t: np.array(t.astype(mx.float32)),
            "to_numpy_2d": lambda t, row, col: np.array(t.astype(mx.float32)[0, row, :]),
            "embed": lambda ids: model_inner.embed_tokens(ids),
            "layer_forward": lambda ly, h, mask: ly(h, mask=mask, cache=None),
            "norm": lambda h: _final_norm(h) if _final_norm else h,
            "lm_head": lambda h: _lm_head(backend.model, h),
            "lm_head_logits": lambda h, pos: np.array(_lm_head(backend.model, (_final_norm(h) if _final_norm else h)).astype(mx.float32)[0, pos, :]),
            "float32": mx.float32,
        }
    else:
        import torch

        model_inner = backend.hf_model.model
        device = next(backend.hf_model.parameters()).device

        def _lm_head_hf(h):
            normed = model_inner.norm(h)
            return backend.hf_model.lm_head(normed)

        return {
            "model_inner": model_inner,
            "array": lambda x: torch.tensor(x, device=device, dtype=torch.long),
            "triu_mask": lambda T: torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1) if T > 1 else None,
            "stack_to_numpy": lambda tensors: torch.stack(tensors).float().cpu().numpy(),
            "to_numpy_1d": lambda t: t.float().cpu().numpy(),
            "to_numpy_2d": lambda t, row, col: t.float().cpu().numpy()[0, row, :],
            "embed": lambda ids: model_inner.embed_tokens(ids),
            "layer_forward": lambda ly, h, mask: ly(h, attention_mask=mask)[0] if isinstance(ly(h, attention_mask=mask), tuple) else ly(h, attention_mask=mask),
            "norm": lambda h: model_inner.norm(h),
            "lm_head": lambda h: _lm_head_hf(h),
            "lm_head_logits": lambda h, pos: _lm_head_hf(h)[0, pos, :].float().cpu().numpy(),
            "float32": torch.float32,
        }


def _capture_mri_causal_bank(backend, *, mode="raw", n_index=None,
                              seed=42, output=None) -> dict:
    """MRI capture for causal bank models.

    Runs each token through the model and captures:
      - substrate_states[N, n_modes] — the EMA state at readout
      - route_weights[N, n_experts] — routing decisions
      - band_logits[N, n_bands, vocab] — per-band logit contribution
      - embedding[N, embed_dim] — input embedding
    """
    from .frt import _detect_script

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
    substrate_states = np.zeros((n_tokens, n_modes), dtype=np.float32)
    embeddings = np.zeros((n_tokens, embed_dim), dtype=np.float32)
    route_weights = np.zeros((n_tokens, n_experts), dtype=np.float32) if n_experts > 1 else None

    # For each token: reset state (raw mode = feed single token)
    for i, (tid, tok) in enumerate(sample):
        result = backend.forward_captured(np.array([[tid]]))
        substrate_states[i] = result['substrate_states'][0, 0]
        embeddings[i] = result['embedding'][0, 0]
        if route_weights is not None and result.get('route_weights') is not None:
            rw = result['route_weights']
            if len(rw.shape) == 2:
                route_weights[i] = rw[0]
            else:
                route_weights[i] = rw[0, 0]

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
    model_inner = ops["model_inner"]

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
        bos_input = ops["array"]([[bos_id]])
        baseline_entry = {}
        baseline_exit = {}
        h = ops["embed"](bos_input)
        for i, ly in enumerate(model_inner.layers):
            h = ops["layer_forward"](ly, h, None)
            if isinstance(h, tuple): h = h[0]
            baseline_entry[i] = ops["to_numpy_2d"](h, 0, 0)
            baseline_exit[i] = baseline_entry[i]
        logits = ops["lm_head_logits"](h, 0)
        probs = softmax(logits)
        bl_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        bl_top_token = backend.tokenizer.decode([int(np.argmax(probs))])
    else:
        clean_baseline = _extract_clean_baseline(backend.tokenizer)
        prefix_ids, suffix_ids = _extract_template_parts(backend.tokenizer)
        token_pos = len(prefix_ids)
        bl_tokens = backend.tokenizer.encode(clean_baseline)
        bl_input = ops["array"]([bl_tokens])
        T_bl = len(bl_tokens)
        mask_bl = ops["triu_mask"](T_bl)
        baseline_entry = {}
        baseline_exit = {}
        h = ops["embed"](bl_input)
        for i, ly in enumerate(model_inner.layers):
            h = ops["layer_forward"](ly, h, mask_bl)
            if isinstance(h, tuple): h = h[0]
            bp = min(token_pos, T_bl - 1)
            baseline_entry[i] = ops["to_numpy_2d"](h, bp, 0)
            baseline_exit[i] = ops["to_numpy_2d"](h, -1, 0)
        logits = ops["lm_head_logits"](h, -1)
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
            # Case 1: all layer files exist — skip capture, backfill weights
            existing_entry = sum(1 for i in range(n_layers)
                                 if (out_dir_check / f"L{i:02d}_entry.npy").exists())
            existing_exit = sum(1 for i in range(n_layers)
                                if (out_dir_check / f"L{i:02d}_exit.npy").exists())
            if existing_entry == n_layers and existing_exit == n_layers:
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
                if captured == total and mmap_candidate.exists():
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

    # === Allocate (mmap for large captures) ===
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    use_mmap = estimated_bytes > 1e9 and output
    if use_mmap:
        mmap_dir = Path(output).parent / ".mri_tmp"
        mmap_dir.mkdir(exist_ok=True)
        entry_arrays = {i: np.memmap(str(mmap_dir / f"e{i}.dat"), dtype=np.float16,
                                      mode='w+', shape=(n_tokens, hidden))
                        for i in range(n_layers)}
        exit_arrays = {i: np.memmap(str(mmap_dir / f"x{i}.dat"), dtype=np.float16,
                                     mode='w+', shape=(n_tokens, hidden))
                       for i in range(n_layers)}
        print(f"  Using memory-mapped arrays")
    else:
        mmap_dir = None
        entry_arrays = {i: np.zeros((n_tokens, hidden), dtype=np.float16) for i in range(n_layers)}
        exit_arrays = {i: np.zeros((n_tokens, hidden), dtype=np.float16) for i in range(n_layers)}

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
    t_start = time.time()

    for batch_start in range(0, n_tokens, batch_size):
        batch_end = min(batch_start + batch_size, n_tokens)
        batch = sample[batch_start:batch_end]
        B = len(batch)

        if mode in ("raw", "naked"):
            inp = ops["array"]([[tid] for tid, _ in batch])  # [B, 1]
            mask = None
        else:
            inp = ops["array"]([prefix_ids + [tid] + suffix_ids for tid, _ in batch])
            T = inp.shape[1]
            mask = ops["triu_mask"](T)

        h = ops["embed"](inp)
        layer_entries = []
        layer_exits = []
        for i, ly in enumerate(model_inner.layers):
            h = ops["layer_forward"](ly, h, mask)
            if isinstance(h, tuple): h = h[0]
            layer_entries.append(h[:, token_pos, :])  # [B, hidden]
            layer_exits.append(h[:, -1, :])

        # One sync for entire batch
        all_entry = ops["stack_to_numpy"](layer_entries)  # [n_layers, B, hidden]
        all_exit = ops["stack_to_numpy"](layer_exits)

        for b in range(B):
            idx = batch_start + b
            for i in range(n_layers):
                entry_arrays[i][idx] = (all_entry[i, b] - baseline_entry[i]).astype(np.float16)
                exit_arrays[i][idx] = (all_exit[i, b] - baseline_exit[i]).astype(np.float16)

        if (batch_end) % 1000 < batch_size or batch_end == n_tokens:
            elapsed = time.time() - t_start
            rate = batch_end / max(elapsed, 0.01)
            remaining = (n_tokens - batch_end) / max(rate, 1)
            print(f"  {batch_end}/{n_tokens} ({rate:.0f} tok/s, ~{remaining/60:.0f}m remaining)")

        # Checkpoint: save progress marker every 10% so we can resume
        checkpoint_interval = max(n_tokens // 10, 1000)
        if output and use_mmap and (batch_end % checkpoint_interval < batch_size or batch_end == n_tokens):
            out_dir_ckpt = Path(output)
            out_dir_ckpt.mkdir(parents=True, exist_ok=True)
            ckpt_file = out_dir_ckpt / ".capture_progress"
            ckpt_file.write_text(json.dumps({
                "tokens_captured": batch_end,
                "tokens_total": n_tokens,
                "mmap_dir": str(mmap_dir),
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
            "baseline_entropy": round(bl_entropy, 4),
            "baseline_top_token": bl_top_token,
        },
        "provenance": {
            "seed": seed,
            "n_index": n_index,
            "decode": "skip_special_tokens=True, clean_up_tokenization_spaces=False",
            "all_bugs_fixed": ["add_special_tokens", "skip_special_tokens", "mmap_threshold"],
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

        # Write layer files IMMEDIATELY — crash after this preserves all states
        total_size = 0
        for i in range(n_layers):
            entry_path = out_dir / f"L{i:02d}_entry.npy"
            exit_path = out_dir / f"L{i:02d}_exit.npy"
            np.save(entry_path, np.array(entry_arrays[i]))
            np.save(exit_path, np.array(exit_arrays[i]))
            total_size += entry_path.stat().st_size + exit_path.stat().st_size
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
                ids = ops["array"]([[i] for i in range(s, e)])
                v = ops["embed"](ids)
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
                    np.save(lmhead_raw_path, _extract_weight_probe(lm_head_mod, hidden))
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

            def _extract_weight(module, in_dim):
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
                    np.save(layer_dir / "q_proj.npy", _extract_weight(attn.q_proj, hidden))
                    np.save(layer_dir / "k_proj.npy", _extract_weight(attn.k_proj, hidden))
                    np.save(layer_dir / "v_proj.npy", _extract_weight(attn.v_proj, hidden))
                elif hasattr(attn, 'qkv_proj'):
                    qkv = _extract_weight(attn.qkv_proj, hidden)
                    q_dim = cfg.n_heads * (hidden // cfg.n_heads)
                    kv_dim = getattr(cfg, 'n_kv_heads', cfg.n_heads) * (hidden // cfg.n_heads)
                    np.save(layer_dir / "q_proj.npy", qkv[:q_dim])
                    np.save(layer_dir / "k_proj.npy", qkv[q_dim:q_dim + kv_dim])
                    np.save(layer_dir / "v_proj.npy", qkv[q_dim + kv_dim:])
                o_proj = getattr(attn, 'o_proj', None) or getattr(attn, 'dense', None)
                if o_proj:
                    o_in = _infer_in_dim(o_proj, hidden)
                    np.save(layer_dir / "o_proj.npy", _extract_weight(o_proj, o_in))

                # MLP — handle gate/up/down, fused gate_up, or fc1/fc2 (Phi-2)
                if hasattr(mlp_mod, 'gate_proj'):
                    gate_in = _infer_in_dim(mlp_mod.gate_proj, hidden)
                    np.save(layer_dir / "gate_proj.npy", _extract_weight(mlp_mod.gate_proj, gate_in))
                    up_in = _infer_in_dim(mlp_mod.up_proj, hidden)
                    np.save(layer_dir / "up_proj.npy", _extract_weight(mlp_mod.up_proj, up_in))
                    mlp_inner_dim = np.load(layer_dir / "gate_proj.npy").shape[0]
                    np.save(layer_dir / "down_proj.npy", _extract_weight(mlp_mod.down_proj, mlp_inner_dim))
                elif hasattr(mlp_mod, 'gate_up_proj'):
                    gu = _extract_weight(mlp_mod.gate_up_proj, hidden)
                    mid = gu.shape[0] // 2
                    np.save(layer_dir / "gate_proj.npy", gu[:mid])
                    np.save(layer_dir / "up_proj.npy", gu[mid:])
                    mlp_inner_dim = mid
                    np.save(layer_dir / "down_proj.npy", _extract_weight(mlp_mod.down_proj, mlp_inner_dim))
                elif hasattr(mlp_mod, 'fc1'):
                    fc1_in = _infer_in_dim(mlp_mod.fc1, hidden)
                    np.save(layer_dir / "gate_proj.npy", _extract_weight(mlp_mod.fc1, fc1_in))
                    fc1_out = np.load(layer_dir / "gate_proj.npy").shape[0]
                    np.save(layer_dir / "down_proj.npy", _extract_weight(mlp_mod.fc2, fc1_out))

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

        if mmap_dir and mmap_dir.exists():
            import shutil
            shutil.rmtree(str(mmap_dir), ignore_errors=True)

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
    model_inner = ops["model_inner"]
    hidden = cfg.hidden_size
    n_layers = cfg.n_layers
    filled = []
    is_mlx = _is_mlx_backend(backend)

    def _extract_weight_probe(module, in_dim):
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
            import torch
            # HF: try direct weight access first (faster, avoids quantization issues)
            if hasattr(module, 'weight') and module.weight is not None:
                return module.weight.float().cpu().numpy()
            # Fallback: probe
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

    # Embedding
    if not (p / "embedding.npy").exists():
        print(f"  Backfilling embedding...")
        vocab = cfg.vocab_size
        vecs = []
        for s in range(0, vocab, 256):
            e = min(s + 256, vocab)
            ids = ops["array"]([[i] for i in range(s, e)])
            v = ops["embed"](ids)
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
        for i in range(n_layers):
            ly = model_inner.layers[i]
            norm_w = ly.input_layernorm.weight
            post_w = ly.post_attention_layernorm.weight
            if is_mlx:
                nd[f"input_L{i}"] = np.array(norm_w).astype(np.float32)
                nd[f"post_attn_L{i}"] = np.array(post_w).astype(np.float32)
            else:
                nd[f"input_L{i}"] = norm_w.float().cpu().numpy()
                nd[f"post_attn_L{i}"] = post_w.float().cpu().numpy()
        final_w = model_inner.norm.weight
        nd["final"] = np.array(final_w).astype(np.float32) if is_mlx else final_w.float().cpu().numpy()
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
                np.save(p / "lmhead_raw.npy", _extract_weight_probe(lm_head, hidden))
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
                np.save(layer_dir / "q_proj.npy", _extract_weight_probe(attn.q_proj, hidden))
                np.save(layer_dir / "k_proj.npy", _extract_weight_probe(attn.k_proj, hidden))
                np.save(layer_dir / "v_proj.npy", _extract_weight_probe(attn.v_proj, hidden))
            elif hasattr(attn, 'qkv_proj'):
                qkv = _extract_weight_probe(attn.qkv_proj, hidden)
                q_dim = cfg.n_heads * (hidden // cfg.n_heads)
                kv_dim = getattr(cfg, 'n_kv_heads', cfg.n_heads) * (hidden // cfg.n_heads)
                np.save(layer_dir / "q_proj.npy", qkv[:q_dim])
                np.save(layer_dir / "k_proj.npy", qkv[q_dim:q_dim + kv_dim])
                np.save(layer_dir / "v_proj.npy", qkv[q_dim + kv_dim:])
            o_proj = getattr(attn, 'o_proj', None) or getattr(attn, 'dense', None)
            if o_proj:
                np.save(layer_dir / "o_proj.npy", _extract_weight_probe(o_proj, _bf_in_dim(o_proj, hidden)))
            if hasattr(ly.mlp, 'gate_proj'):
                np.save(layer_dir / "gate_proj.npy", _extract_weight_probe(ly.mlp.gate_proj, _bf_in_dim(ly.mlp.gate_proj, hidden)))
                np.save(layer_dir / "up_proj.npy", _extract_weight_probe(ly.mlp.up_proj, _bf_in_dim(ly.mlp.up_proj, hidden)))
            elif hasattr(ly.mlp, 'gate_up_proj'):
                gu = _extract_weight_probe(ly.mlp.gate_up_proj, hidden)
                mid = gu.shape[0] // 2
                np.save(layer_dir / "gate_proj.npy", gu[:mid])
                np.save(layer_dir / "up_proj.npy", gu[mid:])
            elif hasattr(ly.mlp, 'fc1'):
                np.save(layer_dir / "gate_proj.npy", _extract_weight_probe(ly.mlp.fc1, _bf_in_dim(ly.mlp.fc1, hidden)))
            mlp_dim = np.load(layer_dir / "gate_proj.npy").shape[0]
            np.save(layer_dir / "down_proj.npy", _extract_weight_probe(ly.mlp.down_proj, mlp_dim))

            if (layer_idx + 1) % 8 == 0:
                print(f"    L{layer_idx} done")
        filled.append("weights")

    return {"filled": filled, "path": str(p)}


def load_mri(path: str) -> dict:
    """Load a .mri directory. The only measurement format.

    Legacy .shrt.npz and .frt.npz files must be recaptured as .mri.
    This function only accepts .mri directories.
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

    tokens = np.load(p / "tokens.npz", allow_pickle=False)
    result = {
        "metadata": meta,
        "path": str(p),
        "token_ids": tokens["token_ids"],
        "token_texts": tokens["token_texts"],
    }
    for key in tokens.files:
        result[key] = tokens[key]

    # Architecture dispatch
    arch = meta.get("architecture", "transformer")

    if arch == "causal_bank":
        # Causal bank: substrate states, routing, half-lives
        substrate_path = p / "substrate.npy"
        if substrate_path.exists():
            result["substrate_states"] = np.load(substrate_path, mmap_mode='r')
            # Compatibility: substrate states ARE the vectors for PCA tools
            result["vectors"] = result["substrate_states"]
            result["deltas"] = np.linalg.norm(
                np.array(result["substrate_states"]).astype(np.float32), axis=1
            ).astype(np.float32)

        routing_path = p / "routing.npy"
        if routing_path.exists():
            result["route_weights"] = np.load(routing_path, mmap_mode='r')

        hl_path = p / "half_lives.npy"
        if hl_path.exists():
            result["half_lives"] = np.load(hl_path)

        embed_path = p / "embedding.npy"
        if embed_path.exists():
            result["embedding"] = np.load(embed_path, mmap_mode='r')

    else:
        # Transformer: per-layer residuals, baselines, projections
        baselines_path = p / "baselines.npz"
        if baselines_path.exists():
            baselines = np.load(baselines_path, allow_pickle=False)
            for key in baselines.files:
                result[f"baseline_{key}"] = baselines[key]

        n_layers = meta["model"]["n_layers"]
        for i in range(n_layers):
            entry_path = p / f"L{i:02d}_entry.npy"
            exit_path = p / f"L{i:02d}_exit.npy"
            if entry_path.exists():
                result[f"entry_L{i}"] = np.load(entry_path, mmap_mode='r')
            if exit_path.exists():
                result[f"exit_L{i}"] = np.load(exit_path, mmap_mode='r')

        dir_dir = p / "directions"
        if dir_dir.exists():
            for npy in dir_dir.glob("*.npy"):
                result[f"direction_{npy.stem}"] = np.load(npy)

        lmhead_path = p / "lmhead.npy"
        if lmhead_path.exists():
            result["lmhead"] = np.load(lmhead_path, mmap_mode='r')
        lmhead_raw_path = p / "lmhead_raw.npy"
        if lmhead_raw_path.exists():
            result["lmhead_raw"] = np.load(lmhead_raw_path, mmap_mode='r')

        embed_path = p / "embedding.npy"
        if embed_path.exists():
            result["embedding"] = np.load(embed_path, mmap_mode='r')

        norms_path = p / "norms.npz"
        if norms_path.exists():
            norms = np.load(norms_path, allow_pickle=False)
            for key in norms.files:
                result[f"norm_{key}"] = norms[key]

        weights_dir = p / "weights"
        if weights_dir.exists():
            for layer_d in sorted(weights_dir.glob("L*")):
                if not layer_d.is_dir():
                    continue
                layer_idx = int(layer_d.name[1:])
                for npy in layer_d.glob("*.npy"):
                    key = f"{npy.stem}_L{layer_idx}"
                    result[key] = np.load(npy, mmap_mode='r')

        # Compatibility keys for analysis tools
        primary_layer = meta['model']['n_layers'] - 1
        exit_key = f"exit_L{primary_layer}"
        if exit_key in result:
            result["vectors"] = result[exit_key]
            result["deltas"] = np.linalg.norm(
                np.array(result[exit_key]).astype(np.float32), axis=1
            ).astype(np.float32)

    # Common: weight files (flat for causal bank, layered for transformer)
    weights_dir = p / "weights"
    if weights_dir.exists() and arch == "causal_bank":
        for npy in weights_dir.glob("*.npy"):
            result[f"weight_{npy.stem}"] = np.load(npy, mmap_mode='r')

    if "raw_bytes_lengths" in result and "byte_counts" not in result:
        result["byte_counts"] = result["raw_bytes_lengths"]

    return result
