"""Total residual capture — every token, every layer, entry + exit positions.

No directions. No projections. No labels. No interpretations.
Raw displacement deltas from silence at two positions per layer.

The .shrt v0.4 format stores the complete measurement.
Analysis tools compute everything else from stored data.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def total_capture(
    backend,
    *,
    n_index: int | None = None,
    seed: int = 42,
    output: str | None = None,
    **kwargs,
) -> dict:
    """Capture displacement at entry and exit positions, all layers, all tokens.

    For each token:
      - Splice into template at token_pos
      - Forward pass through all layers
      - At each layer: extract h[0, token_pos, :] and h[0, -1, :]
      - Subtract baseline (silence at same positions, same layers)
      - Store as float16

    Args:
        backend: loaded model backend
        n_index: number of tokens (None = full vocabulary)
        seed: random seed for sampling if n_index < vocab
        output: path to write .shrt.npz file
    """
    from .shrt import _extract_clean_baseline, _extract_template_parts
    import mlx.core as mx

    cfg = backend.config
    n_layers = cfg.n_layers
    hidden = cfg.hidden_size
    model_inner = getattr(backend.model, 'model', backend.model)

    t0 = time.time()

    from ..cartography.perturb import _mask_dtype
    from ..cartography.runtime import _lm_head
    from ..cartography.metrics import softmax
    mdtype = _mask_dtype(backend.model)

    # Mode: naked (single token, BOS baseline) or template (chat frame, silence baseline)
    naked = kwargs.get('naked', False)

    if naked:
        # Naked mode: baseline is BOS token alone
        bos_id = backend.tokenizer.bos_token_id or 0
        baseline_input = mx.array([[bos_id]])
        token_pos = 0  # token is always at position 0
        prefix_ids = []
        suffix_ids = []

        baseline_entry = {}
        baseline_exit = {}
        h = model_inner.embed_tokens(baseline_input)
        for i, ly in enumerate(model_inner.layers):
            h = ly(h, mask=None, cache=None)
            if isinstance(h, tuple):
                h = h[0]
            baseline_entry[i] = np.array(h.astype(mx.float32)[0, 0, :])
            baseline_exit[i] = baseline_entry[i]  # same position for single token

        h_normed = model_inner.norm(h)
        silence_logits = np.array(_lm_head(backend.model, h_normed).astype(mx.float32)[0, 0, :])
        silence_probs = softmax(silence_logits)
        silence_top_id = int(np.argmax(silence_probs))
        silence_entropy = float(-np.sum(silence_probs * np.log2(silence_probs + 1e-12)))
        silence_top_token = backend.tokenizer.decode([silence_top_id])
        T_bl = 1
    else:
        # Template mode: chat frame, silence baseline
        clean_baseline = _extract_clean_baseline(backend.tokenizer)
        prefix_ids, suffix_ids = _extract_template_parts(backend.tokenizer)
        token_pos = len(prefix_ids)

        baseline_tokens = backend.tokenizer.encode(clean_baseline)
        baseline_input = mx.array([baseline_tokens])
        T_bl = len(baseline_tokens)
        mask_bl = mx.triu(mx.full((T_bl, T_bl), float('-inf'), dtype=mdtype), k=1) if T_bl > 1 else None

        baseline_entry = {}
        baseline_exit = {}
        h = model_inner.embed_tokens(baseline_input)
        for i, ly in enumerate(model_inner.layers):
            h = ly(h, mask=mask_bl, cache=None)
            if isinstance(h, tuple):
                h = h[0]
            bp = min(token_pos, T_bl - 1)
            baseline_entry[i] = np.array(h.astype(mx.float32)[0, bp, :])
            baseline_exit[i] = np.array(h.astype(mx.float32)[0, -1, :])

        h_normed = model_inner.norm(h)
        silence_logits = np.array(_lm_head(backend.model, h_normed).astype(mx.float32)[0, -1, :])
        silence_probs = softmax(silence_logits)
        silence_top_id = int(np.argmax(silence_probs))
        silence_entropy = float(-np.sum(silence_probs * np.log2(silence_probs + 1e-12)))
        silence_top_token = backend.tokenizer.decode([silence_top_id])

    # Build token sample
    vocab_size = backend.tokenizer.vocab_size
    real_tokens = []
    seen = set()
    for tid in range(vocab_size):
        tok = backend.tokenizer.decode([tid])
        if tok.strip() and not tok.startswith('[control') and not tok.startswith('<') and tok not in seen:
            seen.add(tok)
            real_tokens.append((tid, tok))

    if n_index is None or n_index >= len(real_tokens):
        sample = real_tokens
    else:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(real_tokens), n_index, replace=False)
        sample = [real_tokens[i] for i in sorted(idx)]

    n_tokens = len(sample)
    print(f"Total capture: {n_tokens} tokens x {n_layers} layers x 2 positions x {hidden} dims")
    print(f"  Estimated size: {n_tokens * n_layers * 2 * hidden * 2 / 1e9:.1f} GB")

    # Prepare output arrays — write incrementally to avoid OOM
    # Store per-layer: entry_deltas[layer] and exit_deltas[layer]
    # Each is [n_tokens, hidden] float16
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    token_ids_list = []
    token_texts_list = []

    # Process tokens and write directly
    # Use memory-mapped arrays for large datasets
    entry_arrays = {i: np.zeros((n_tokens, hidden), dtype=np.float16) for i in range(n_layers)}
    exit_arrays = {i: np.zeros((n_tokens, hidden), dtype=np.float16) for i in range(n_layers)}

    t_start = time.time()
    for idx, (tid, tok) in enumerate(sample):
        if naked:
            input_ids = [tid]
        else:
            input_ids = prefix_ids + [tid] + suffix_ids
        inp = mx.array([input_ids])
        T = len(input_ids)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        h = model_inner.embed_tokens(inp)
        for i, ly in enumerate(model_inner.layers):
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]
            # Entry position delta
            entry = np.array(h.astype(mx.float32)[0, token_pos, :])
            entry_arrays[i][idx] = (entry - baseline_entry[i]).astype(np.float16)
            # Exit position delta
            exit_val = np.array(h.astype(mx.float32)[0, -1, :])
            exit_arrays[i][idx] = (exit_val - baseline_exit[i]).astype(np.float16)

        token_ids_list.append(tid)
        token_texts_list.append(tok)

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            remaining = (n_tokens - idx - 1) / rate
            print(f"  {idx+1}/{n_tokens} ({rate:.0f} tok/s, ~{remaining/60:.0f}m remaining)")

    elapsed = time.time() - t0

    metadata = {
        "version": "0.4",
        "type": "shrt",
        "format": "total_capture",
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
            "n_tokens": n_tokens,
            "n_layers": n_layers,
            "positions": ["entry (pos 0)"] if naked else ["entry (token_pos)", "exit (last)"],
            "token_pos": token_pos,
            "mode": "naked (single token, BOS baseline)" if naked else "template (chat frame, silence baseline)",
            "baseline": "BOS token alone" if naked else "silence (empty user message)",
            "storage": "float16 deltas from baseline",
        },
        "silence": {
            "entropy": round(silence_entropy, 4),
            "top_token": silence_top_token,
            "n_baseline_tokens": T_bl,
        },
        "provenance": {
            "seed": seed,
            "n_index": n_index,
            "template": "model's own chat template with system prompt stripped",
            "no_directions": True,
            "no_labels": True,
            "no_projections": True,
        },
    }

    if output:
        save_dict = {
            "metadata": np.array([json.dumps(metadata, ensure_ascii=False)]),
            "token_ids": np.array(token_ids_list, dtype=np.int32),
            "token_texts": np.array(token_texts_list),
        }
        for i in range(n_layers):
            save_dict[f"entry_L{i}"] = entry_arrays[i]
            save_dict[f"exit_L{i}"] = exit_arrays[i]

        np.savez_compressed(output, **save_dict)
        file_size = Path(output).stat().st_size / 1e9
        print(f"\n  Saved to {output} ({file_size:.2f} GB)")

    return metadata
