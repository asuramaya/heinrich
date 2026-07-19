"""Full-vocabulary frozen-frame projection — the .mri vocab extension.

Captures per-layer exit states for EVERY vocabulary token and projects them
through an existing decomposition's frozen PCA frame. The frame (components +
means) is never re-derived here — the sample decomposition stays the
coordinate system, and the full vocabulary becomes addressable inside it.
Supports raw, naked, and template capture modes (template splices each token
into the chat frame exactly like capture_mri).

Because the stored components are orthonormal and K = hidden_size by default,
full-K score-space distances equal hidden-space distances exactly (up to f16
rounding). One artifact gives exact geometry for every token:

  decomp/vocab_scores.bin   VSCR header + [n_rows, total_layers, K] float16
  decomp/vocab_ids.npy      [n_rows] int32 token ids (row -> token id)
  decomp/vocab_tokens.json  [n_rows] decoded texts (row -> text)
  decomp/vocab_meta.json    provenance + built-in sample agreement check

Row layout matches token_scores.bin: real layers first, then emb, then lmh
virtual layers. Per-row access is an O(1) seek: 16 + row * total_layers*K*2.
"""
from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

from .mri import _ensure_chat_template, _framework_ops, _is_mlx_backend


def _vocab_token_list(backend, mode: str) -> tuple[list[tuple[int, str]], list[int], list[int]]:
    """The full-vocabulary token list, dedup'd exactly like capture_mri (by
    decoded text, first token id wins) — shared by every vocab-derived
    artifact so `vocab_row` means the same row everywhere (vocab_scores.bin,
    vocab_pc16.bin, vocab_gate_heatmap.npy, vocab_token_neurons.bin, ...).
    Returns (real_tokens, prefix_ids, suffix_ids) — prefix/suffix are only
    non-empty in template mode (same splice capture_mri's template mode uses).
    """
    vocab_size = backend.tokenizer.vocab_size
    real_tokens = []
    seen = set()
    for tid in range(vocab_size):
        tok = backend.tokenizer.decode([tid], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
        if tok.strip() and tok not in seen:
            seen.add(tok)
            real_tokens.append((tid, tok))
    prefix_ids: list[int] = []
    suffix_ids: list[int] = []
    if mode == 'template':
        from .shrt import _extract_template_parts
        # Match capture_mri: a base tokenizer with no chat_template gets the
        # same ChatML fallback, so the splice reproduces the capture exactly.
        _ensure_chat_template(backend.tokenizer)
        prefix_ids, suffix_ids = _extract_template_parts(backend.tokenizer)
    return real_tokens, prefix_ids, suffix_ids


def build_vocab_scripts(mri_path: str) -> dict:
    """Full-vocabulary script/language classification — a pure text-classification
    backfill over decomp/vocab_tokens.json (row -> text, already row-ordered by
    _vocab_token_list), no backend/tokenizer/forward-pass needed. Writes
    decomp/vocab_scripts.json (row -> script), same shape/order as vocab_tokens.json,
    using the same _detect_script classifier the sample capture already uses — so
    the two data-scopes color-classify tokens identically.
    """
    from .frt import _detect_script

    mri_dir = Path(mri_path)
    decomp = mri_dir / 'decomp'
    tokens_path = decomp / 'vocab_tokens.json'
    if not tokens_path.exists():
        return {"error": f"No vocab_tokens.json — run: heinrich mri-vocab --model <hf_id> --mri {mri_path}"}
    t0 = time.time()
    texts = json.loads(tokens_path.read_text())
    scripts = [_detect_script(t) for t in texts]
    out_path = decomp / 'vocab_scripts.json'
    out_path.write_text(json.dumps(scripts))
    return {
        "mri_path": str(mri_dir),
        "n_rows": len(scripts),
        "size_mb": round(out_path.stat().st_size / (1024 * 1024), 2),
        "elapsed_s": round(time.time() - t0, 2),
    }


def capture_vocab_projection(backend, mri_path: str, *, batch_size: int = 0,
                             verify: bool = True) -> dict:
    """Project the full vocabulary through an existing frozen decomposition.

    The capture replicates the MRI's exit semantics exactly: same forward
    (single-token for raw/naked; prefix+[tid]+suffix splice with causal mask
    for template, exit read at the LAST position), same stored per-layer
    baseline subtraction (zeros for raw, BOS states for naked, clean-template
    exits for template), same mean-centering, same components. A built-in
    check compares the rows of the original sample tokens against their
    stored scores and reports the agreement in vocab_meta.json — if the
    forward path drifted from the original capture, the artifact says so
    itself.
    """
    mri_dir = Path(mri_path)
    meta = json.loads((mri_dir / 'metadata.json').read_text())
    mode = meta.get('capture', {}).get('mode', 'raw')
    if mode not in ('raw', 'naked', 'template'):
        return {"error": f"vocab projection supports raw/naked/template captures; this MRI is '{mode}'"}

    decomp = mri_dir / 'decomp'
    dmeta_path = decomp / 'meta.json'
    if not dmeta_path.exists():
        return {"error": f"No decomposition at {decomp} — run mri-decompose first"}
    dmeta = json.loads(dmeta_path.read_text())
    n_real = int(dmeta['n_real_layers'])
    total_layers = int(dmeta['n_layers'])
    K = int(dmeta['n_components'])
    hidden = int(meta['model']['hidden_size'])

    # === Frozen frame: means + components per real layer ===
    means, comps = [], []
    for li in range(n_real):
        mp = decomp / f'L{li:02d}_mean.npy'
        cp = decomp / f'L{li:02d}_components.npy'
        if not mp.exists() or not cp.exists():
            return {"error": f"missing {mp.name if not mp.exists() else cp.name} — "
                             f"run: heinrich mri-decompose --mri {mri_dir} --backfill-means"}
        means.append(np.load(str(mp)).astype(np.float32))
        comps.append(np.load(str(cp)).astype(np.float32))

    # === Stored baselines (raw mode: zeros; naked: BOS states; template:
    # clean-template exit states — REUSE, never recompute) ===
    baselines = []
    bl_path = mri_dir / 'baselines.npz'
    bl = np.load(str(bl_path)) if bl_path.exists() else {}
    for li in range(n_real):
        key = f'exit_L{li}'
        baselines.append(bl[key].astype(np.float32) if key in bl
                         else np.zeros(hidden, dtype=np.float32))

    real_tokens, prefix_ids, suffix_ids = _vocab_token_list(backend, mode)
    n_rows = len(real_tokens)
    ids = np.array([tid for tid, _ in real_tokens], dtype=np.int32)

    est_mb = (16 + n_rows * total_layers * K * 2) / (1024 * 1024)
    print(f"vocab projection ({mode}): {n_rows} tokens x {total_layers} layers x {K} PCs "
          f"= {est_mb:.0f}MB", file=sys.stderr)

    ops = _framework_ops(backend)
    is_mlx = _is_mlx_backend(backend)
    if is_mlx:
        import mlx.core as mx

        def _to_np(t):
            return np.array(t.astype(mx.float32))
        import contextlib
        _grad_ctx = contextlib.nullcontext
    else:
        import torch

        def _to_np(t):
            return t.float().cpu().numpy()
        _grad_ctx = torch.no_grad

    if batch_size <= 0:
        if mode == 'template':
            # spliced sequences are ~20-60 tokens each; the intermediates are
            # B x T x hidden x layers. Stay conservative — a model may already
            # be resident in a companion process on the same 12GB GPU.
            batch_size = 16
        else:
            # single-token rows are cheap; scale down for deep/wide models
            batch_size = max(16, min(512, int(6e8 // max(1, n_real * hidden * 4))))

    # === Output memmap ===
    out_path = decomp / 'vocab_scores.bin'
    hdr = struct.pack('<4sIII', b'VSCR', n_rows, total_layers, K)
    with open(out_path, 'wb') as f:
        f.write(hdr)
        f.write(b'\x00' * (n_rows * total_layers * K * 2))
    out = np.memmap(str(out_path), dtype=np.float16, mode='r+', offset=16,
                    shape=(n_rows, total_layers, K))

    # === Real layers: batched single-token forwards, frozen-frame projection ===
    t0 = time.time()
    n_batches = (n_rows + batch_size - 1) // batch_size
    with _grad_ctx():
        for bi, b0 in enumerate(range(0, n_rows, batch_size)):
            b1 = min(b0 + batch_size, n_rows)
            if mode == 'template':
                # prefix + [tid] + suffix — single token spliced, so every
                # sequence is the same length: one clean batch, one triu mask.
                inp = ops.array([prefix_ids + [tid] + suffix_ids
                                 for tid, _ in real_tokens[b0:b1]])
                mask = ops.triu_mask(inp.shape[1])
            else:
                inp = ops.array([[tid] for tid, _ in real_tokens[b0:b1]])  # [B, 1]
                mask = None
            h = ops.embed(inp)
            for li, ly in enumerate(ops.model_inner.layers):
                # layer_decomposed, not layer_forward: the MRI captured its
                # exits through the decomposed path, and bf16/f16 reduction
                # order differs between the two — coordinates must come from
                # the identical computation to sit in the frozen frame.
                h = ops.layer_decomposed(ly, h, mask)[0]
                ex = _to_np(h[:, -1, :] if h.ndim == 3 else h)  # [B, hidden]
                sc = (ex - baselines[li] - means[li]) @ comps[li].T
                out[b0:b1, li, :sc.shape[1]] = sc.astype(np.float16)
            if (bi + 1) % 20 == 0 or b1 == n_rows:
                rate = b1 / max(time.time() - t0, 1e-6)
                print(f"  batch {bi + 1}/{n_batches} ({b1}/{n_rows} tokens, "
                      f"{rate:.0f} tok/s)", file=sys.stderr)

    # === Virtual layers: weight-derived, full vocab already on disk ===
    for vi, (vname, vpath) in enumerate([('emb', 'embedding.npy'), ('lmh', 'lmhead.npy')]):
        li_out = n_real + vi
        if li_out >= total_layers:
            break
        vf = mri_dir / vpath
        vmp = decomp / f'{vname}_mean.npy'
        vcp = decomp / f'{vname}_components.npy'
        if not (vf.exists() and vmp.exists() and vcp.exists()):
            print(f"  {vname}: missing file(s), rows left zero", file=sys.stderr)
            continue
        raw = np.load(str(vf), mmap_mode='r')
        v_mu = np.load(str(vmp)).astype(np.float32)
        v_C = np.load(str(vcp)).astype(np.float32)
        for c0 in range(0, n_rows, 8192):
            c1 = min(c0 + 8192, n_rows)
            vecs = raw[ids[c0:c1]].astype(np.float32)
            sc = (vecs - v_mu) @ v_C.T
            out[c0:c1, li_out, :sc.shape[1]] = sc.astype(np.float16)
        print(f"  {vname}: projected {n_rows} rows", file=sys.stderr)

    out.flush()

    # === Built-in falsification: sample tokens vs their stored scores ===
    # Exact agreement is unreachable: the f16 forward has a reproducibility
    # floor (~sqrt(D) f16 ulps at the layer's common-mode magnitude), and the
    # stored cloud carries the same floor from its own capture. So instead of
    # a pass/fail on bit-parity, MEASURE the floor per layer and store it —
    # consumers can then show "distances below X are capture noise".
    agreement = None
    if verify:
        tokens = dict(np.load(mri_dir / 'tokens.npz', allow_pickle=True))
        sample_ids = tokens['token_ids']
        si = dmeta.get('sample_indices', 'all')
        idx = np.arange(len(sample_ids)) if si == 'all' else np.asarray(si)
        row_of = {int(t): r for r, t in enumerate(ids)}
        check_layers = sorted({0, n_real // 4, n_real // 2, (3 * n_real) // 4, n_real - 1})
        per_layer = []
        worst_med_ratio = 0.0
        for li in check_layers:
            sp = decomp / f'L{li:02d}_scores.npy'
            if not sp.exists():
                continue
            stored = np.load(str(sp)).astype(np.float32)
            rows = np.array([row_of.get(int(sample_ids[j]), -1) for j in idx])
            ok = rows >= 0
            if not ok.any():
                continue
            got = out[rows[ok], li, :].astype(np.float32)
            want = stored[np.where(ok)[0]]
            kk = min(got.shape[1], want.shape[1])
            delta = np.linalg.norm(got[:, :kk] - want[:, :kk], axis=1)
            row_norm = np.linalg.norm(want[:, :kk], axis=1)
            med_delta = float(np.median(delta))
            med_norm = float(np.median(row_norm))
            per_layer.append({
                "layer": li,
                "noise_floor_abs": round(med_delta, 3),
                "noise_floor_p99": round(float(np.percentile(delta, 99)), 3),
                "median_row_norm": round(med_norm, 3),
                "median_rel": round(med_delta / (med_norm + 1e-9), 5),
            })
            worst_med_ratio = max(worst_med_ratio, med_delta / (med_norm + 1e-9))
        agreement = {
            "n_rows_checked": int(len(idx)),
            "layers": per_layer,
            "worst_median_rel": round(worst_med_ratio, 5),
            "note": "noise_floor_abs = median |fresh - stored| over sample rows; "
                    "the f16-forward reproducibility floor shared with the frozen cloud",
        }
        status = ("OK" if worst_med_ratio < 0.10 else
                  "WARNING: capture noise exceeds 10% of typical row norm — "
                  "forward path may have drifted from the original capture")
        for pl in per_layer:
            print(f"  L{pl['layer']:02d}: noise floor |Δ|={pl['noise_floor_abs']:.1f} "
                  f"(median row |s|={pl['median_row_norm']:.1f}, rel {pl['median_rel']:.3f})",
                  file=sys.stderr)
        print(f"  sample agreement: worst median rel {worst_med_ratio:.3f} — {status}",
              file=sys.stderr)

    np.save(decomp / 'vocab_ids.npy', ids)
    (decomp / 'vocab_tokens.json').write_text(
        json.dumps([tok for _, tok in real_tokens], ensure_ascii=False))

    vocab_meta = {
        "n_rows": n_rows,
        "n_layers": total_layers,
        "n_real_layers": n_real,
        "n_components": K,
        "dtype": "float16",
        "layout": "VSCR: 16-byte header <4sIII (magic, n_rows, n_layers, K), row-major [n_rows, n_layers, K]",
        "distance_exact": bool(K >= hidden),
        "frame": {
            "source": "sample decomposition (frozen)",
            "n_sample": int(dmeta.get('n_sample', 0)),
            "mode": mode,
            "mri_fix_level": meta.get('fix_level'),
        },
        "sample_agreement": agreement,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (decomp / 'vocab_meta.json').write_text(json.dumps(vocab_meta, indent=2))

    return {
        "mri_path": str(mri_dir),
        "n_rows": n_rows,
        "n_layers": total_layers,
        "n_components": K,
        "size_mb": round(est_mb, 1),
        "distance_exact": bool(K >= hidden),
        "sample_agreement": agreement,
        "elapsed_s": vocab_meta["elapsed_s"],
    }


def capture_vocab_gate_summary(backend, mri_path: str, *, batch_size: int = 0,
                                top_n: int = 50) -> dict:
    """Full-vocabulary MLP gate/up signal — the one thing capture_vocab_projection
    deliberately never touches (it only re-projects hidden-state/PC geometry).

    Two artifacts, from two passes over the SAME per-layer forward structure
    capture_vocab_projection uses (ops.layer_decomposed; same token list via
    _vocab_token_list, so vocab_row means the same row everywhere):

      Pass 1 (accumulate) — gate/up ARE ALREADY computed inside
      ops.layer_decomposed's return tuple (indices 5, 6); capture_vocab_projection
      just discards them. Reading them instead costs nothing extra per batch:
        - vocab_gate_heatmap.npy: max|gate*up| per token per layer — the
          full-vocab depth-curve summary (tiny, a few MB: this closes the exact
          gap the whole full-vocab investigation started from).
        - a running per-layer sum|gate*up| accumulator (kept in memory only) —
          ranks neuron importance over the FULL vocabulary, fixing the bias in
          the sample's neuron_importance.json (ranked from just 2,000 tokens).

      Pass 2 (targeted extract, skipped entirely if top_n<=0) — now knowing each
      layer's true important neurons, re-run the same forward structure and
      keep ONLY those columns' signed gate*up per token per layer:
        - vocab_token_neurons.bin (VTKN): same semantics as the sample's
          token_neurons.bin (signed gate*up, not absolute) — a per-token *data*
          reduction (top_n of intermediate_size columns), not a token-count
          reduction; every vocab row is covered.
        - vocab_neuron_importance.json: the corrected top-N indices/contributions.

    Two passes is NOT a "one forward pass, not two" violation — that invariant
    (generate_with_geometry) is about text generation and geometry capture never
    diverging from the same event. This is inherently sequential: which neurons
    matter can only be known after seeing the whole vocabulary, so a single pass
    cannot both rank importance and extract the ranked columns. Holding the full
    [n_rows, n_layers, intermediate] tensor in memory to dodge a second pass
    would run tens of GB at this scale — not viable.
    """
    mri_dir = Path(mri_path)
    meta = json.loads((mri_dir / 'metadata.json').read_text())
    mode = meta.get('capture', {}).get('mode', 'raw')
    if mode not in ('raw', 'naked', 'template'):
        return {"error": f"vocab gate summary supports raw/naked/template captures; this MRI is '{mode}'"}

    decomp = mri_dir / 'decomp'
    dmeta_path = decomp / 'meta.json'
    if not dmeta_path.exists():
        return {"error": f"No decomposition at {decomp} — run mri-decompose first"}
    dmeta = json.loads(dmeta_path.read_text())
    n_real = int(dmeta['n_real_layers'])
    hidden = int(meta['model']['hidden_size'])

    real_tokens, prefix_ids, suffix_ids = _vocab_token_list(backend, mode)
    n_rows = len(real_tokens)

    # Row parity with vocab_scores.bin — vocab_row must mean the same row
    # everywhere, or Phase 1's predict-chip pinning would silently point a
    # depth-curve/neuron-field lookup at the wrong token.
    vpath = decomp / 'vocab_scores.bin'
    if vpath.exists():
        with open(vpath, 'rb') as f:
            vmagic, vn_rows, _, _ = struct.unpack('<4sIII', f.read(16))
        if vmagic == b'VSCR' and vn_rows != n_rows:
            return {"error": f"Row count mismatch vs vocab_scores.bin ({vn_rows} vs {n_rows}) — "
                             "tokenizer/dedup drifted since mri-vocab ran; re-run mri-vocab first"}

    ops = _framework_ops(backend)
    is_mlx = _is_mlx_backend(backend)
    if is_mlx:
        import mlx.core as mx

        def _to_np(t):
            return np.array(t.astype(mx.float32))
        import contextlib
        _grad_ctx = contextlib.nullcontext
    else:
        import torch

        def _to_np(t):
            return t.float().cpu().numpy()
        _grad_ctx = torch.no_grad

    if batch_size <= 0:
        if mode == 'template':
            batch_size = 16
        else:
            batch_size = max(16, min(512, int(6e8 // max(1, n_real * hidden * 4))))

    def _run_layers(inp, mask):
        """One forward: embed + every real layer's layer_decomposed, yielding
        (li, gate_val, up_val) at the LAST position — None,None if this
        architecture has no separate gate/up MLP (fc1/fc2-only, nothing to
        capture here)."""
        h = ops.embed(inp)
        for li, ly in enumerate(ops.model_inner.layers):
            res = ops.layer_decomposed(ly, h, mask)
            h = res[0]
            gate_val, up_val = res[5], res[6]
            if gate_val is None or up_val is None:
                yield li, None, None
                continue
            g = _to_np(gate_val[:, -1, :] if gate_val.ndim == 3 else gate_val)
            u = _to_np(up_val[:, -1, :] if up_val.ndim == 3 else up_val)
            yield li, g, u

    def _batch_input(b0, b1):
        if mode == 'template':
            inp = ops.array([prefix_ids + [tid] + suffix_ids for tid, _ in real_tokens[b0:b1]])
            return inp, ops.triu_mask(inp.shape[1])
        return ops.array([[tid] for tid, _ in real_tokens[b0:b1]]), None

    t0 = time.time()
    n_batches = (n_rows + batch_size - 1) // batch_size
    gate_heat = np.zeros((n_rows, n_real), dtype=np.float16)
    importance_sum = None  # [n_real, intermediate] float64, sized on first sight

    with _grad_ctx():
        for bi, b0 in enumerate(range(0, n_rows, batch_size)):
            b1 = min(b0 + batch_size, n_rows)
            inp, mask = _batch_input(b0, b1)
            for li, g, u in _run_layers(inp, mask):
                if g is None:
                    continue
                gu = g * u
                if importance_sum is None:
                    importance_sum = np.zeros((n_real, gu.shape[1]), dtype=np.float64)
                gate_heat[b0:b1, li] = np.abs(gu).max(axis=1).astype(np.float16)
                importance_sum[li] += np.abs(gu).sum(axis=0)
            if (bi + 1) % 20 == 0 or b1 == n_rows:
                rate = b1 / max(time.time() - t0, 1e-6)
                print(f"  pass 1 (depth curve + importance): batch {bi + 1}/{n_batches} "
                      f"({b1}/{n_rows} tokens, {rate:.0f} tok/s)", file=sys.stderr)

    if importance_sum is None:
        return {"error": "Model has no gate/up MLP (fc1/fc2-only architecture) — nothing to capture"}

    np.save(decomp / 'vocab_gate_heatmap.npy', gate_heat)
    pass1_s = time.time() - t0

    result = {
        "mri_path": str(mri_dir),
        "n_rows": n_rows,
        "n_real_layers": n_real,
        "gate_heatmap_size_mb": round((128 + n_rows * n_real * 2) / (1024 * 1024), 2),
        "pass1_s": round(pass1_s, 1),
    }

    if top_n <= 0:
        result["top_n"] = 0
        result["elapsed_s"] = round(time.time() - t0, 1)
        return result

    # === Derive top-N neuron indices per layer from the FULL-VOCAB accumulator ===
    importance_mean = importance_sum / n_rows
    top_n = min(top_n, importance_mean.shape[1])
    top_idx = np.argsort(importance_mean, axis=1)[:, ::-1][:, :top_n]  # [n_real, top_n]

    # === Pass 2: signed gate*up at just those columns, every token, every layer ===
    t2 = time.time()
    tok_neurons = np.zeros((n_rows, n_real, top_n), dtype=np.float16)
    with _grad_ctx():
        for bi, b0 in enumerate(range(0, n_rows, batch_size)):
            b1 = min(b0 + batch_size, n_rows)
            inp, mask = _batch_input(b0, b1)
            for li, g, u in _run_layers(inp, mask):
                if g is None:
                    continue
                gu = (g * u)[:, top_idx[li]]
                tok_neurons[b0:b1, li, :] = gu.astype(np.float16)
            if (bi + 1) % 20 == 0 or b1 == n_rows:
                rate = b1 / max(time.time() - t2, 1e-6)
                print(f"  pass 2 (top-{top_n} neuron field): batch {bi + 1}/{n_batches} "
                      f"({b1}/{n_rows} tokens, {rate:.0f} tok/s)", file=sys.stderr)

    hdr = struct.pack('<4sIII', b'VTKN', n_rows, n_real, top_n)
    with open(decomp / 'vocab_token_neurons.bin', 'wb') as f:
        f.write(hdr)
        f.write(tok_neurons.tobytes())

    neuron_importance = [{
        "layer": li,
        "top_neurons": top_idx[li].tolist(),
        "top_contrib": importance_mean[li, top_idx[li]].tolist(),
    } for li in range(n_real)]
    (decomp / 'vocab_neuron_importance.json').write_text(json.dumps(neuron_importance))

    result["top_n"] = int(top_n)
    result["token_neurons_size_mb"] = round((16 + n_rows * n_real * top_n * 2) / (1024 * 1024), 1)
    result["pass2_s"] = round(time.time() - t2, 1)
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def frame_falsification(mri_path: str, *, k: int = 50, z_cut: float = 6.0) -> dict:
    """Does the sample frame hold at full vocabulary? Falsify, don't assume.

    Recomputes PCA per layer over the FULL vocabulary (crystal-suppressed:
    rows with row-norm |z| > z_cut excluded, per Session 11) and measures the
    principal-angle overlap between its top-k subspace and the frozen sample
    frame's top-k subspace.

    Runs entirely in score space: vocab_scores.bin at full K is a lossless
    rotation of the centered residuals, and principal angles are rotation-
    invariant. In this basis the frozen top-k subspace is the first k
    coordinate axes, so overlap = mean squared mass of the full-vocab PCs
    inside the first k coordinates. 1.0 = identical subspace, 1/K = random.

    Writes results into decomp/vocab_meta.json under "frame_falsification".
    """
    import struct as _struct

    mri_dir = Path(mri_path)
    decomp = mri_dir / 'decomp'
    vpath = decomp / 'vocab_scores.bin'
    if not vpath.exists():
        return {"error": f"No vocab_scores.bin at {decomp} — run mri-vocab first"}
    with open(vpath, 'rb') as f:
        magic, n_rows, n_layers, K = _struct.unpack('<4sIII', f.read(16))
    if magic != b'VSCR':
        return {"error": f"Bad magic {magic!r} in vocab_scores.bin"}
    dmeta = json.loads((decomp / 'meta.json').read_text())
    n_real = int(dmeta.get('n_real_layers', n_layers))
    k = min(k, K)

    blob = np.memmap(str(vpath), dtype=np.float16, mode='r', offset=16,
                     shape=(n_rows, n_layers, K))
    from sklearn.utils.extmath import randomized_svd

    layers_out = []
    worst = 1.0
    for li in range(n_real):
        X = blob[:, li, :].astype(np.float32)
        norms = np.linalg.norm(X, axis=1)
        z = (norms - norms.mean()) / (norms.std() + 1e-9)
        keep = np.abs(z) <= z_cut
        Xk = X[keep] - X[keep].mean(axis=0)
        _, S, Vt = randomized_svd(Xk, n_components=k, random_state=42)
        # Vt rows are full-vocab PCs expressed in frozen-PC coordinates.
        mass_in_frozen_topk = (Vt[:, :k] ** 2).sum(axis=1)   # per new PC
        w = (S ** 2) / (S ** 2).sum()
        overlap_topk = float(mass_in_frozen_topk[:k].mean())
        overlap_weighted = float((mass_in_frozen_topk * w).sum())
        pc1_mass = float(mass_in_frozen_topk[0])
        layers_out.append({
            "layer": li,
            "overlap_topk": round(overlap_topk, 4),
            "overlap_var_weighted": round(overlap_weighted, 4),
            "pc1_mass_in_frozen": round(pc1_mass, 4),
            "n_excluded": int((~keep).sum()),
        })
        worst = min(worst, overlap_weighted)
        print(f"  L{li:02d}: overlap top-{k}={overlap_topk:.3f} "
              f"var-weighted={overlap_weighted:.3f} pc1={pc1_mass:.3f} "
              f"excluded={int((~keep).sum())}", file=sys.stderr)

    verdict = ("frame holds" if worst >= 0.8 else
               "frame partially holds — interpret tail PCs with care" if worst >= 0.5 else
               "frame does not represent the full vocabulary — re-decompose deliberately")
    result = {
        "k": k,
        "z_cut": z_cut,
        "n_rows": int(n_rows),
        "worst_var_weighted_overlap": round(worst, 4),
        "verdict": verdict,
        "layers": layers_out,
    }
    vm_path = decomp / 'vocab_meta.json'
    if vm_path.exists():
        vm = json.loads(vm_path.read_text())
        vm["frame_falsification"] = result
        vm_path.write_text(json.dumps(vm, indent=2))
    return result


def _sample_display_k(mri_path: str) -> int | None:
    """The PC count the SAMPLE decomposition actually exposes (pc_scores.bin's
    own `full_k` header field) — the true ceiling on any PC index the UI can
    ever select, whatever that happens to be for this MRI (commonly the full
    hidden_size — n_components defaults to 0 = hidden_size in mri-decompose).
    Returns None if no sample decomposition exists (vocab_pc16 falls back to 16).
    """
    p = Path(mri_path) / 'decomp' / 'pc_scores.bin'
    if not p.exists():
        return None
    with open(p, 'rb') as f:
        magic, _n_layers, _n_tok, full_k = struct.unpack('<4sIII', f.read(16))
    return int(full_k) if magic == b'PCSC' else None


def build_vocab_pc16(mri_path: str, *, n_pcs: int | None = None) -> dict:
    """Layer-major display companion to vocab_scores.bin: vocab_pc16.bin.

    vocab_scores.bin is row-major ([token, layer, K]) — perfect for "one
    token, all layers" (targeting, homing), useless for "all tokens, one
    view" (a cloud needs per-layer PC columns; at the edge that would be
    48K strided range reads). This writes the transpose truncated to the
    top n_pcs DISPLAY components:

        vocab_pc16.bin  VP16 header <4sIII (magic, n_layers, n_pcs, n_rows)
                        + [n_layers, n_pcs, n_rows] float16

    One (layer, pc) column = one O(1) range read (~2 bytes * n_rows).
    Display truncation only — measurement stays exact in the row-major blob.

    n_pcs defaults to the SAMPLE decomposition's own PC ceiling (pc_scores.bin's
    `full_k` header) rather than a fixed 16: the UI never lets a user select a
    PC index beyond that ceiling anyway (vpPairs/_applyPCSelection are bounded
    by the client's own `nK`, which IS that same header field), so this keeps
    every selectable PC pair working against the full vocabulary. In practice
    `full_k` is commonly the model's full hidden_size (mri-decompose defaults
    n_components to 0 = hidden_size), so this is frequently the SAME size as
    vocab_scores.bin, just transposed — not the much smaller ~50-PC file the
    old fixed default implied.
    """
    import struct as _struct

    decomp = Path(mri_path) / 'decomp'
    vpath = decomp / 'vocab_scores.bin'
    if not vpath.exists():
        return {"error": f"No vocab_scores.bin at {decomp} — run mri-vocab first"}
    with open(vpath, 'rb') as f:
        magic, n_rows, n_layers, K = _struct.unpack('<4sIII', f.read(16))
    if magic != b'VSCR':
        return {"error": f"Bad magic {magic!r} in vocab_scores.bin"}
    if n_pcs is None:
        n_pcs = _sample_display_k(mri_path) or 16
    n_pcs = min(n_pcs, K)

    blob = np.memmap(str(vpath), dtype=np.float16, mode='r', offset=16,
                     shape=(n_rows, n_layers, K))
    out_path = decomp / 'vocab_pc16.bin'
    hdr = _struct.pack('<4sIII', b'VP16', n_layers, n_pcs, n_rows)
    with open(out_path, 'wb') as f:
        f.write(hdr)
        f.write(b'\x00' * (n_layers * n_pcs * n_rows * 2))
    out = np.memmap(str(out_path), dtype=np.float16, mode='r+', offset=16,
                    shape=(n_layers, n_pcs, n_rows))
    # Chunk over rows: one sequential pass through the row-major blob.
    for c0 in range(0, n_rows, 8192):
        c1 = min(c0 + 8192, n_rows)
        chunk = np.asarray(blob[c0:c1, :, :n_pcs])          # [rows, layers, pcs]
        out[:, :, c0:c1] = np.transpose(chunk, (1, 2, 0))   # [layers, pcs, rows]
    out.flush()
    size_mb = (16 + n_layers * n_pcs * n_rows * 2) / (1024 * 1024)
    print(f"  vocab_pc16: {n_layers} layers x {n_pcs} PCs x {n_rows} rows "
          f"= {size_mb:.0f}MB", file=sys.stderr)
    return {"mri_path": str(mri_path), "n_layers": int(n_layers),
            "n_pcs": int(n_pcs), "n_rows": int(n_rows),
            "size_mb": round(size_mb, 1)}
