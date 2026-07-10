#!/usr/bin/env python3
"""Arm C of the anisotropy program (thread 87/113): information-per-PC vs
variance-per-PC — superposition proper.

Arm A closed the variance ledger: spectra = kernel x marginal, learning's
only signature there is regulation/whitening. If anything trained lives in
the geometry, it must show as INFORMATION sitting where VARIANCE doesn't —
the divergence of the two ledgers is the whole remaining question.

This reads sequence-mode .seq.mri captures and reports, per PC band of the
substrate (deep bands, down to the variance floor):
  var_pct        — the variance ledger (arm A says: kernel physics)
  pos_r2         — position/clock information
  byte_r2 lag=0  — current byte identity
  byte_r2 lag=k  — byte k positions BACK (deep-history information; the
                   binding channel's raw material, k in 8/64/512)
R2 here = class-mean explanation: fraction of the band's variance explained
by conditioning on the (lagged) byte class — information the band carries
ABOUT that byte, normalized by the band's own scale so a tiny-variance band
can still score high.

Covariance eigendecomposition uses ALL captured positions (48x2048 rows),
so tail PCs are estimated from 98k samples, not a 6k subsample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from heinrich.profile.mri import load_mri

BANDS = [(0, 8), (8, 20), (20, 50), (50, 100), (100, 256),
         (256, 512), (512, 1024), (1024, 2048)]
LAGS = [0, 8, 64, 512]
# The arrow of memory: negative lags score PROSPECTIVE information — the
# byte |lag| positions AHEAD, which the state can only know through learned
# structure. A recorder points backward (past >> future); a prediction
# organ should point forward.
LAGS_ARROW = [-64, -8, -1, 0, 1, 8, 64, 512]
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "data"


def class_mean_r2(X_tr: np.ndarray, c_tr: np.ndarray,
                  X_te: np.ndarray, c_te: np.ndarray) -> float:
    """Fraction of X_te variance explained by per-class means of c."""
    means = np.zeros((256, X_tr.shape[1]), dtype=np.float64)
    for b in range(256):
        m = c_tr == b
        if m.sum() > 3:
            means[b] = X_tr[m].mean(0)
    pred = means[c_te]
    ss_r = ((X_te - pred) ** 2).sum()
    ss_t = ((X_te - X_te.mean(0)) ** 2).sum()
    return float(max(0.0, 1 - ss_r / max(ss_t, 1e-12)))


def analyze(mri_path: str, dims: int | None = None,
            lags: list[int] = LAGS) -> dict:
    mri = load_mri(mri_path)
    sub = mri["substrate_states"].astype(np.float32)
    if dims is not None:
        # Adaptive captures concat [substrate modes, x_embed]; slicing to the
        # first n_modes dims removes the trivial current-byte carrier so lag0
        # information must come from the recurrence itself.
        sub = sub[:, :, :dims]
    tok = np.load(Path(mri_path) / "tokens.npz")["token_ids"]
    n_seqs, seq_len, D = sub.shape

    flat = sub.reshape(-1, D).astype(np.float64)
    mu = flat.mean(0)
    flat -= mu
    ev, U = np.linalg.eigh((flat.T @ flat) / len(flat))
    ev, U = ev[::-1], U[:, ::-1]
    var_frac = ev / ev.sum()
    proj = (flat @ U).astype(np.float32)

    pos = np.tile(np.arange(seq_len, dtype=np.float32), n_seqs)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(flat))
    tr, te = idx[: len(flat) // 2], idx[len(flat) // 2:]

    bands_out = []
    for lo, hi in BANDS:
        hi = min(hi, D)
        if lo >= hi:
            continue
        X = proj[:, lo:hi]
        # position R2 (predict position from band; ridge-free lstsq)
        A = np.column_stack([X[tr], np.ones(len(tr))])
        coef, *_ = np.linalg.lstsq(A, pos[tr], rcond=None)
        pred = X[te] @ coef[:-1] + coef[-1]
        ss_r = ((pos[te] - pred) ** 2).sum()
        ss_t = ((pos[te] - pos[te].mean()) ** 2).sum()
        row = {
            "band": f"{lo}-{hi - 1}",
            "var_pct": round(float(var_frac[lo:hi].sum()) * 100, 4),
            "pos_r2": round(float(max(0.0, 1 - ss_r / ss_t)), 4),
        }
        for lag in lags:
            # byte at position p-lag: lag>0 = past, lag<0 = FUTURE (valid
            # only where p+|lag| < seq_len)
            lagged = np.full((n_seqs, seq_len), -1, dtype=np.int64)
            if lag == 0:
                lagged[:] = tok[:, :seq_len]
            elif lag > 0:
                lagged[:, lag:] = tok[:, : seq_len - lag]
            else:
                k = -lag
                lagged[:, : seq_len - k] = tok[:, k:seq_len]
            lf = lagged.reshape(-1)
            vtr = tr[lf[tr] >= 0]
            vte = te[lf[te] >= 0]
            row[f"byte_r2_lag{lag}"] = round(
                class_mean_r2(X[vtr], lf[vtr], X[vte], lf[vte]), 4)
        bands_out.append(row)

    return {
        "mri": mri_path,
        "shape": [int(n_seqs), int(seq_len), int(D)],
        "eigs_head": [float(e) for e in ev[:64]],
        "bands": bands_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mris", nargs="+", required=True)
    ap.add_argument("--dims", type=int, default=None,
                    help="keep only the first N state dims (e.g. n_modes to "
                         "drop the concatenated x_embed columns)")
    ap.add_argument("--arrow", action="store_true",
                    help="use the arrow-of-memory lag set (prospective "
                         "negative lags alongside retrospective)")
    args = ap.parse_args()
    lags = LAGS_ARROW if args.arrow else LAGS

    reports = []
    for m in args.mris:
        r = analyze(m, dims=args.dims, lags=lags)
        reports.append(r)
        name = Path(m).name.replace(".seq.mri", "")
        print(f"\n=== {name}  [{r['shape'][0]}x{r['shape'][1]}x{r['shape'][2]}] ===")
        print(f"  {'band':<11} {'var%':>8} {'pos_r2':>7} "
              + " ".join(f"{'lag' + str(l):>7}" for l in lags))
        for b in r["bands"]:
            print(f"  {b['band']:<11} {b['var_pct']:>8.4f} {b['pos_r2']:>7.4f} "
                  + " ".join(f"{b['byte_r2_lag' + str(l)]:>7.4f}" for l in lags))

    suffix = f"-dims{args.dims}" if args.dims else ""
    if args.arrow:
        suffix += "-arrow"
    out = OUT_DIR / f"anisotropy-c-pc-information{suffix}.json"
    out.write_text(json.dumps({"lags": lags, "dims": args.dims,
                               "reports": reports}, indent=1))
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
