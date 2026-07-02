"""Full-vocabulary frozen-frame projection — the .mri vocab extension.

Captures per-layer exit states for EVERY vocabulary token and projects them
through an existing decomposition's frozen PCA frame. The frame (components +
means) is never re-derived here — the sample decomposition stays the
coordinate system, and the full vocabulary becomes addressable inside it.

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

from .mri import _framework_ops, _is_mlx_backend


def capture_vocab_projection(backend, mri_path: str, *, batch_size: int = 0,
                             verify: bool = True) -> dict:
    """Project the full vocabulary through an existing frozen decomposition.

    The capture replicates the MRI's exit semantics exactly: same single-token
    forward (raw/naked), same stored per-layer baseline subtraction, same
    mean-centering, same components. A built-in check compares the rows of
    the original sample tokens against their stored scores and reports the
    agreement in vocab_meta.json — if the forward path drifted from the
    original capture, the artifact says so itself.
    """
    mri_dir = Path(mri_path)
    meta = json.loads((mri_dir / 'metadata.json').read_text())
    mode = meta.get('capture', {}).get('mode', 'raw')
    if mode not in ('raw', 'naked'):
        return {"error": f"vocab projection supports raw/naked captures; this MRI is '{mode}' "
                         "(template mode needs prefix/suffix splice — not implemented)"}

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

    # === Stored baselines (raw mode: zeros; naked: BOS forward states) ===
    baselines = []
    bl_path = mri_dir / 'baselines.npz'
    bl = np.load(str(bl_path)) if bl_path.exists() else {}
    for li in range(n_real):
        key = f'exit_L{li}'
        baselines.append(bl[key].astype(np.float32) if key in bl
                         else np.zeros(hidden, dtype=np.float32))

    # === Full token list — same filter as capture_mri (dedup by decoded text,
    # first token id wins), so sample rows land on identical ids ===
    vocab_size = backend.tokenizer.vocab_size
    real_tokens = []
    seen = set()
    for tid in range(vocab_size):
        tok = backend.tokenizer.decode([tid], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
        if tok.strip() and tok not in seen:
            seen.add(tok)
            real_tokens.append((tid, tok))
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
            inp = ops.array([[tid] for tid, _ in real_tokens[b0:b1]])  # [B, 1]
            h = ops.embed(inp)
            for li, ly in enumerate(ops.model_inner.layers):
                # layer_decomposed, not layer_forward: the MRI captured its
                # exits through the decomposed path, and bf16/f16 reduction
                # order differs between the two — coordinates must come from
                # the identical computation to sit in the frozen frame.
                h = ops.layer_decomposed(ly, h, None)[0]
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
