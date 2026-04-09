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
    batch_size = 32  # all modes batchable — template has fixed length
    t_start = time.time()

    for batch_start in range(0, n_tokens, batch_size):
        batch_end = min(batch_start + batch_size, n_tokens)
        batch = sample[batch_start:batch_end]
        B = len(batch)

        if mode in ("raw", "naked"):
            inp = mx.array([[tid] for tid, _ in batch])  # [B, 1]
            mask = None
        else:
            # Template: [prefix + tid + suffix] — same length for every token
            inp = mx.array([prefix_ids + [tid] + suffix_ids for tid, _ in batch])  # [B, seq_len]
            T = inp.shape[1]
            mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        h = model_inner.embed_tokens(inp)
        entry_mlx = []
        exit_mlx = []
        for i, ly in enumerate(model_inner.layers):
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple): h = h[0]
            entry_mlx.append(h[:, token_pos, :])  # [B, hidden]
            exit_mlx.append(h[:, -1, :])

        # One sync for entire batch
        all_entry = np.array(mx.stack(entry_mlx).astype(mx.float32))  # [n_layers, B, hidden]
        all_exit = np.array(mx.stack(exit_mlx).astype(mx.float32))

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

        total_size = 0
        for i in range(n_layers):
            entry_path = out_dir / f"L{i:02d}_entry.npy"
            exit_path = out_dir / f"L{i:02d}_exit.npy"
            np.save(entry_path, np.array(entry_arrays[i]))
            np.save(exit_path, np.array(exit_arrays[i]))
            total_size += entry_path.stat().st_size + exit_path.stat().st_size

        # Layer norm weights (small, 2 per layer + final norm)
        norms_path = out_dir / "norms.npz"
        if not norms_path.exists():
            print(f"  Extracting norm weights...")
            norm_dict = {}
            for layer_idx in range(n_layers):
                ly = model_inner.layers[layer_idx]
                norm_dict[f"input_L{layer_idx}"] = np.array(ly.input_layernorm.weight).astype(np.float32)
                norm_dict[f"post_attn_L{layer_idx}"] = np.array(ly.post_attention_layernorm.weight).astype(np.float32)
            norm_dict["final"] = np.array(model_inner.norm.weight).astype(np.float32)
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
                ids = mx.array([[i] for i in range(s, e)])  # [batch, 1]
                vecs = np.array(model_inner.embed_tokens(ids).astype(mx.float32)[:, 0, :])  # [batch, hidden]
                embed_vecs.append(vecs)
            embed_weights = np.concatenate(embed_vecs, axis=0).astype(np.float32)  # float32, not float16
            np.save(embed_path, embed_weights)
            total_size += embed_path.stat().st_size

        # lm_head matrix (norm + unembedding composed)
        lmhead_path = out_dir / "lmhead.npy"
        if not lmhead_path.exists():
            print(f"  Extracting lm_head...")
            cols = []
            batch_sz = 64
            for start in range(0, hidden, batch_sz):
                end = min(start + batch_sz, hidden)
                inp_probe = np.zeros((1, end - start, hidden), dtype=np.float16)
                for j in range(end - start):
                    inp_probe[0, j, start + j] = 1.0
                h_probe = mx.array(inp_probe)
                h_normed = model_inner.norm(h_probe)
                out = np.array(_lm_head(backend.model, h_normed).astype(mx.float32)[0])
                cols.append(out.T)
            W = np.concatenate(cols, axis=1).astype(np.float32)
            np.save(lmhead_path, W)
            total_size += lmhead_path.stat().st_size

        # All projection weights per layer (dequantized by probing)
        weights_dir = out_dir / "weights"
        if not weights_dir.exists():
            weights_dir.mkdir()
            print(f"  Extracting projection weights (all layers)...")
            n_heads = cfg.n_heads
            head_dim = hidden // n_heads
            # GQA: some models have fewer KV heads
            kv_heads = getattr(cfg, 'num_key_value_heads', n_heads)
            kv_dim = kv_heads * head_dim

            for layer_idx in range(n_layers):
                ly = model_inner.layers[layer_idx]
                attn = ly.self_attn
                mlp_mod = ly.mlp
                layer_dir = weights_dir / f"L{layer_idx:02d}"
                layer_dir.mkdir(exist_ok=True)

                # Probe each projection by sending identity-like vectors through
                def extract_weight(module, in_dim):
                    cols = []
                    b = 64
                    for s in range(0, in_dim, b):
                        e = min(s + b, in_dim)
                        probe = np.zeros((1, e - s, in_dim), dtype=np.float32)
                        for j in range(e - s):
                            probe[0, j, s + j] = 1.0
                        out = np.array(module(mx.array(probe).astype(mx.float16)).astype(mx.float32)[0])
                        cols.append(out.T)
                    return np.concatenate(cols, axis=1).astype(np.float32)

                # Attention projections
                np.save(layer_dir / "q_proj.npy", extract_weight(attn.q_proj, hidden))
                np.save(layer_dir / "k_proj.npy", extract_weight(attn.k_proj, hidden))
                np.save(layer_dir / "v_proj.npy", extract_weight(attn.v_proj, hidden))
                np.save(layer_dir / "o_proj.npy", extract_weight(attn.o_proj, hidden))

                # MLP projections
                np.save(layer_dir / "gate_proj.npy", extract_weight(mlp_mod.gate_proj, hidden))
                np.save(layer_dir / "up_proj.npy", extract_weight(mlp_mod.up_proj, hidden))

                mlp_inner_dim = np.load(layer_dir / "gate_proj.npy").shape[0]
                np.save(layer_dir / "down_proj.npy", extract_weight(mlp_mod.down_proj, mlp_inner_dim))

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


def load_mri(path: str) -> dict:
    """Load measurement data. Handles both .mri directories and legacy .shrt.npz files.

    This is the ONLY load function analysis tools should call.
    Returns a consistent dict regardless of source format.
    """
    p = Path(path)

    # === .mri directory format ===
    if p.is_dir():
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

        # Embedding matrix
        embed_path = p / "embedding.npy"
        if embed_path.exists():
            result["embedding"] = np.load(embed_path, mmap_mode='r')

        # Norm weights
        norms_path = p / "norms.npz"
        if norms_path.exists():
            norms = np.load(norms_path, allow_pickle=False)
            for key in norms.files:
                result[f"norm_{key}"] = norms[key]

        # Legacy o_proj directory
        oproj_dir = p / "oproj"
        if oproj_dir.exists():
            for npy in sorted(oproj_dir.glob("L*.npy")):
                layer_idx = int(npy.stem[1:])
                result[f"oproj_L{layer_idx}"] = np.load(npy, mmap_mode='r')

        # Full projection weights per layer
        weights_dir = p / "weights"
        if weights_dir.exists():
            for layer_d in sorted(weights_dir.glob("L*")):
                if not layer_d.is_dir():
                    continue
                layer_idx = int(layer_d.name[1:])
                for npy in layer_d.glob("*.npy"):
                    key = f"{npy.stem}_L{layer_idx}"
                    result[key] = np.load(npy, mmap_mode='r')

        return result

    # === Legacy .shrt.npz format (v0.2, v0.3, v0.4) ===
    d = np.load(str(p), allow_pickle=False)
    meta = json.loads(str(d['metadata'][0]))

    import warnings
    baseline_ent = meta.get('baseline', {}).get('entropy', 0)
    if baseline_ent > 5.0:
        warnings.warn(f"{path}: baseline entropy {baseline_ent:.2f} is high.", stacklevel=2)

    result = {
        "metadata": meta,
        "path": str(p),
        "token_ids": d["token_ids"],
        "token_texts": d["token_texts"],
    }

    # Map legacy arrays to consistent keys
    if "vectors" in d.files:
        # v0.2/v0.3: vectors at primary layer = exit vectors
        layer = meta.get('baseline', {}).get('layer', meta.get('layers', [0])[0] if meta.get('layers') else 0)
        result[f"exit_L{layer}"] = d["vectors"]
        result["vectors"] = d["vectors"]  # backwards compat

    if "deltas" in d.files:
        result["deltas"] = d["deltas"]

    # v0.3 arrays
    for key in ["kl_divs", "output_entropies", "byte_counts", "scripts",
                 "raw_bytes", "raw_bytes_lengths", "merge_ranks"]:
        if key in d.files:
            result[key] = d[key]

    # Per-layer deltas (v0.2 all-layers)
    for key in d.files:
        if key.startswith("deltas_L"):
            result[key] = d[key]
        elif key.startswith("entry_L") or key.startswith("exit_L"):
            result[key] = d[key]
        elif key.startswith("baseline_"):
            result[key] = d[key]

    # Layer array
    if "layer" in d.files:
        result["layer"] = d["layer"]

    return result
