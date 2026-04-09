"""The .mri file — complete model residual image.

One file per model per capture mode. Contains everything:
  - Tokenizer atoms (raw bytes, merge ranks, decoded text)
  - Residual state at entry and exit positions, every layer
  - Baselines (the reference frame)
  - Discovered directions (safety, comply, any others)
  - Capture provenance (mode, seed, template, model config)

No separate .frt, .shrt, .sht, .trd needed. One file. One load.
Analysis tools compute everything from stored data.

Modes:
  template — chat frame, silence baseline
  naked   — single token, BOS baseline
  raw     — single token, no BOS, absolute state (zero baseline)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


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
    """
    from .shrt import _extract_clean_baseline, _extract_template_parts
    from .frt import _detect_script
    import mlx.core as mx
    from ..cartography.perturb import _mask_dtype
    from ..cartography.runtime import _lm_head
    from ..cartography.metrics import softmax

    cfg = backend.config
    n_layers = cfg.n_layers
    hidden = cfg.hidden_size
    model_inner = getattr(backend.model, 'model', backend.model)
    mdtype = _mask_dtype(backend.model)

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
        bos_input = mx.array([[bos_id]])
        baseline_entry = {}
        baseline_exit = {}
        h = model_inner.embed_tokens(bos_input)
        for i, ly in enumerate(model_inner.layers):
            h = ly(h, mask=None, cache=None)
            if isinstance(h, tuple): h = h[0]
            baseline_entry[i] = np.array(h.astype(mx.float32)[0, 0, :])
            baseline_exit[i] = baseline_entry[i]
        h_normed = model_inner.norm(h)
        logits = np.array(_lm_head(backend.model, h_normed).astype(mx.float32)[0, 0, :])
        probs = softmax(logits)
        bl_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        bl_top_token = backend.tokenizer.decode([int(np.argmax(probs))])
    else:
        clean_baseline = _extract_clean_baseline(backend.tokenizer)
        prefix_ids, suffix_ids = _extract_template_parts(backend.tokenizer)
        token_pos = len(prefix_ids)
        bl_tokens = backend.tokenizer.encode(clean_baseline)
        bl_input = mx.array([bl_tokens])
        T_bl = len(bl_tokens)
        mask_bl = mx.triu(mx.full((T_bl, T_bl), float('-inf'), dtype=mdtype), k=1) if T_bl > 1 else None
        baseline_entry = {}
        baseline_exit = {}
        h = model_inner.embed_tokens(bl_input)
        for i, ly in enumerate(model_inner.layers):
            h = ly(h, mask=mask_bl, cache=None)
            if isinstance(h, tuple): h = h[0]
            bp = min(token_pos, T_bl - 1)
            baseline_entry[i] = np.array(h.astype(mx.float32)[0, bp, :])
            baseline_exit[i] = np.array(h.astype(mx.float32)[0, -1, :])
        h_normed = model_inner.norm(h)
        logits = np.array(_lm_head(backend.model, h_normed).astype(mx.float32)[0, -1, :])
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
    t_start = time.time()
    for idx, (tid, tok) in enumerate(sample):
        if mode in ("raw", "naked"):
            input_ids = [tid]
        else:
            input_ids = prefix_ids + [tid] + suffix_ids

        inp = mx.array([input_ids])
        T = len(input_ids)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        h = model_inner.embed_tokens(inp)
        entry_mlx = []
        exit_mlx = []
        for i, ly in enumerate(model_inner.layers):
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple): h = h[0]
            entry_mlx.append(h[0, token_pos, :])  # stays in MLX, no sync
            exit_mlx.append(h[0, -1, :])
        # One sync: stack all layers, convert to numpy once
        all_entry = np.array(mx.stack(entry_mlx).astype(mx.float32))  # [n_layers, hidden]
        all_exit = np.array(mx.stack(exit_mlx).astype(mx.float32))
        for i in range(n_layers):
            entry_arrays[i][idx] = (all_entry[i] - baseline_entry[i]).astype(np.float16)
            exit_arrays[i][idx] = (all_exit[i] - baseline_exit[i]).astype(np.float16)

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            remaining = (n_tokens - idx - 1) / rate
            print(f"  {idx+1}/{n_tokens} ({rate:.0f} tok/s, ~{remaining/60:.0f}m remaining)")

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

    # === Save ===
    if output:
        save_dict = {
            "metadata": np.array([json.dumps(metadata, ensure_ascii=False)]),
            "token_ids": token_ids,
            "token_texts": token_texts,
            "raw_bytes": raw_bytes,
            "raw_bytes_lengths": raw_bytes_lengths,
            "merge_ranks": merge_ranks,
            "scripts": scripts,
        }
        for i in range(n_layers):
            save_dict[f"entry_L{i}"] = np.array(entry_arrays[i])
            save_dict[f"exit_L{i}"] = np.array(exit_arrays[i])
            save_dict[f"baseline_entry_L{i}"] = baseline_entry[i].astype(np.float16)
            save_dict[f"baseline_exit_L{i}"] = baseline_exit[i].astype(np.float16)

        np.savez_compressed(output, **save_dict)
        file_size = Path(output).stat().st_size / 1e9
        print(f"\n  Saved to {output} ({file_size:.2f} GB)")

        if mmap_dir and mmap_dir.exists():
            import shutil
            shutil.rmtree(str(mmap_dir), ignore_errors=True)

    return metadata


def load_mri(path: str) -> dict:
    """Load a .mri file. Single load function for all analysis tools."""
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d['metadata'][0]))
    result = {"metadata": meta}
    for key in d.files:
        if key != "metadata":
            result[key] = d[key]
    return result
