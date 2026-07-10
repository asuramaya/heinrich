"""Information-per-PC vs variance-per-PC for TRANSFORMERS — arm C's
transformer half, and the arrow of memory on residual streams.

The causal-bank triptych (rulings e45eb1c8, bafe1c44, b7b915b7) found:
variance spectra are kernel x marginal; the trained organ's information is
variance-blind (superposition proper); and training moves the state's
information mass from a deep-past archive to the present edge. This
instrument asks the same questions of a transformer's residual stream,
per layer.

For each layer and PC band: variance share, position R2, and LAGGED-TOKEN
information — how much of the band's variance is explained by (equiv.,
how well the band linearly recovers) the input embedding of the token k
positions away. k > 0 is retrospective, k < 0 prospective; k = -1 is
next-token embedding recovery, the probe target staked in the arm-C queue.
Embedding-recovery R2 replaces the byte class-mean R2 of the causal-bank
ledger because a 50k vocabulary starves per-class means; a linear map onto
the embedding vector is the vocab-size-free generalization.

Prompts come from the DB via require_prompts (constitution). First 2
positions per prompt are dropped from the STATE samples (BOS/onset
transients) but remain available as lag targets.
"""
from __future__ import annotations

import numpy as np

INFO_BANDS = ((0, 8), (8, 20), (20, 50), (50, 100), (100, 256), (256, 2048))
INFO_LAGS = (-8, -1, 0, 1, 8, 64)


def _r2_lstsq(X_tr, Y_tr, X_te, Y_te) -> float:
    """Total-variance R2 of a linear map (with intercept) X -> Y on test."""
    A = np.column_stack([X_tr, np.ones(len(X_tr), dtype=np.float32)])
    coef, *_ = np.linalg.lstsq(A, Y_tr, rcond=None)
    pred = X_te @ coef[:-1] + coef[-1]
    ss_r = float(((Y_te - pred) ** 2).sum())
    ss_t = float(((Y_te - Y_te.mean(0)) ** 2).sum())
    return float(max(0.0, 1 - ss_r / max(ss_t, 1e-12)))


def transformer_pc_information(model_name: str,
                               *,
                               bands: tuple = INFO_BANDS,
                               lags: tuple = INFO_LAGS,
                               n_prompts: int = 256,
                               max_len: int = 256,
                               batches: int = 64,
                               layers: list[int] | None = None,
                               seed: int = 42,
                               device: str | None = None) -> dict:
    """Per-layer information ledger for an HF transformer.

    Returns, for each reported layer (0 = embedding output) and PC band:
    var_pct, pos_r2, and emb_r2_lag{k} for each lag. The divergence
    between the variance column and the information columns is
    superposition proper; the lag profile's tilt is the arrow of memory.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..core.db import SignalDB

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    db = SignalDB()
    texts = [p["text"] for p in
             db.require_prompts(is_benign=True, limit=n_prompts)]

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32).eval().to(dev)
    emb_w = model.get_input_embeddings().weight.detach().cpu().numpy() \
        .astype(np.float32)

    enc = [tok(t, return_tensors="pt", truncation=True,
               max_length=max_len).input_ids for t in texts]
    enc = [e for e in enc if e.shape[1] >= 16][:batches]
    if len(enc) < 8:
        raise RuntimeError(f"only {len(enc)} usable prompts >= 16 tokens")

    # collect per-layer states + aligned (prompt token stream, position)
    per_layer: list[list[np.ndarray]] = []
    tok_streams: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    sample_prompt: list[np.ndarray] = []
    with torch.no_grad():
        for pi, ids in enumerate(enc):
            out = model(ids.to(dev), output_hidden_states=True)
            hs = out.hidden_states
            if not per_layer:
                per_layer = [[] for _ in hs]
            seq = ids.shape[1]
            for li, h in enumerate(hs):
                per_layer[li].append(h[0, 2:, :].float().cpu().numpy())
            tok_streams.append(ids[0].numpy())
            positions.append(np.arange(2, seq))
            sample_prompt.append(np.full(seq - 2, pi))
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    pos_all = np.concatenate(positions).astype(np.float32)
    prm_all = np.concatenate(sample_prompt)
    n_samples = len(pos_all)

    # lag targets: embedding of the token at position p - lag within the
    # same prompt; invalid (masked) where p - lag falls outside the prompt.
    lag_targets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for lag in lags:
        tgt = np.zeros((n_samples, emb_w.shape[1]), dtype=np.float32)
        ok = np.zeros(n_samples, dtype=bool)
        i = 0
        for pi, (stream, pos) in enumerate(zip(tok_streams, positions)):
            src = pos - lag
            valid = (src >= 0) & (src < len(stream))
            tgt[i:i + len(pos)][valid] = emb_w[stream[src[valid]]]
            ok[i:i + len(pos)] = valid
            i += len(pos)
        lag_targets[lag] = (tgt, ok)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    tr, te = idx[: n_samples // 2], idx[n_samples // 2:]

    n_layers_total = len(per_layer)
    report_layers = layers if layers is not None else list(
        range(n_layers_total))

    rows = []
    for li in report_layers:
        vecs = np.concatenate(per_layer[li], axis=0)
        d = vecs.shape[1]
        vecs = vecs - vecs.mean(0)
        ev, U = np.linalg.eigh((vecs.T @ vecs).astype(np.float64)
                               / len(vecs))
        ev, U = ev[::-1], U[:, ::-1]
        var_frac = ev / ev.sum()
        proj = (vecs @ U).astype(np.float32)

        layer_bands = []
        for lo, hi in bands:
            hi = min(hi, d)
            if lo >= hi:
                continue
            X = proj[:, lo:hi]
            row = {
                "band": f"{lo}-{hi - 1}",
                "var_pct": round(float(var_frac[lo:hi].sum()) * 100, 4),
                "pos_r2": round(_r2_lstsq(
                    X[tr], pos_all[tr, None], X[te], pos_all[te, None]), 4),
            }
            for lag in lags:
                tgt, ok = lag_targets[lag]
                vtr = tr[ok[tr]]
                vte = te[ok[te]]
                row[f"emb_r2_lag{lag}"] = round(
                    _r2_lstsq(X[vtr], tgt[vtr], X[vte], tgt[vte]), 4)
            layer_bands.append(row)
        rows.append({"layer": int(li), "dim": int(d),
                     "eigs_head": [float(e) for e in ev[:32]],
                     "bands": layer_bands})

    return {
        "model": model_name,
        "n_layers_total": n_layers_total,  # includes embedding at index 0
        "n_samples": int(n_samples),
        "n_prompts_used": len(enc),
        "lags": list(lags),
        "note": ("emb_r2_lag{k}: linear recovery of the input embedding of "
                 "the token k positions BACK (k<0 = future; -1 = next "
                 "token). Layer 0 = embedding output."),
        "layers": rows,
    }
