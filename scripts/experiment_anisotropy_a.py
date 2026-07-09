#!/usr/bin/env python3
"""Experiment A of the anisotropy program (thread 87): does the steep bank
spectrum come from the DATA (input law) or the KERNEL (multiscale EMA family)?

Protocol (pre-registered, msgs 88/91):
  - UNTRAINED bank: linear_in_proj + linear_decays lifted from the OFF-body
    checkpoint (the frozen innate substrate — untrained by definition), driven
    through an UNTRAINED byte embedding (seeded randn; scale is irrelevant to
    a log-log slope). States streamed with the m3_knn FFT template (the same
    math as chronohorn/experiments/harnesses/knn_stream.py).
  - Six streams from decepticons (20MB each): enwik_real, enwik_shuffled
    (full byte permutation — marginal preserved, ALL sequential structure
    destroyed), world, code, math, wikiml.
  - Eigenspectrum of the state second moment; exponent fit on PCs
    8..min(512, M/4) = 8..512 at M=2048; full curve saved.
  - The discriminating quantity is the GAP real-minus-shuffled (DC's hedge:
    shuffled collapses to the kernel-only exponent, not to white).
  - Bonus row: enwik_real through the TRAINED embedding (the deployed body's
    drive) — how much the one trained tensor moves the spectrum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from decepticons.loader import load_checkpoint

STREAM_DIR = Path("/home/asuramaya/code/REPOS/chronohorn/data/spectrum_streams")
BODY = "/home/asuramaya/code/REPOS/chronohorn/out/results/fo-noadapt-enwik90-50k.checkpoint.pt"
OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "anisotropy-a.json"
STREAMS = ["enwik_real", "enwik_shuffled", "world", "code", "math", "wikiml"]
FIT_LO = 8
CH = 32768
SEED = 42


def fit_exponent(eigs: np.ndarray, dim: int) -> tuple[float, int, int]:
    hi = min(512, dim // 4)
    window = eigs[FIT_LO - 1:hi]
    ranks = np.arange(FIT_LO, FIT_LO + len(window))
    mask = window > 0
    coef = np.polyfit(np.log(ranks[mask]), np.log(window[mask]), 1)
    return float(-coef[0]), FIT_LO, hi


def spectrum_for_stream(data: np.ndarray, drive: torch.Tensor,
                        decays: torch.Tensor, dev: str) -> dict:
    """Eigenspectrum of the streamed bank state (m3_knn FFT template)."""
    m = decays.shape[0]
    nf = 2 * CH
    jj = torch.arange(CH, device=dev, dtype=torch.float64)
    la = torch.log(decays)[:, None]
    kf = torch.fft.rfft(((1 - decays)[:, None] * torch.exp(la * jj)).float(), n=nf, dim=1)
    apow = torch.exp(la * (jj + 1.0)).float()

    gram = torch.zeros(m, m, device=dev, dtype=torch.float64)
    n = 0
    h = torch.zeros(m, device=dev)
    for s in range(0, len(data), CH):
        chunk = torch.as_tensor(data[s:s + CH].astype(np.int64), device=dev)
        d = drive[:, chunk]
        df = torch.fft.rfft(d, n=nf, dim=1)
        st = torch.fft.irfft(df * kf, n=nf, dim=1)[:, :len(chunk)] \
            + h[:, None] * apow[:, :len(chunk)]
        h = st[:, -1].clone()
        # fp64 matmul is 1/32-rate on this card; fp32 chunk-grams into an
        # fp64 accumulator + 4x position subsampling keep the slope exact
        # (5M samples per stream still >> 2048 dims) and ~100x faster.
        sm = st.T[::4].float()
        gram += (sm.T @ sm).to(torch.float64)
        n += sm.shape[0]

    ev = torch.linalg.eigvalsh(gram / n).flip(0).cpu().numpy()
    total = float(ev.sum())
    cum = np.cumsum(ev) / total
    alpha, lo, hi = fit_exponent(ev, m)
    return {
        "alpha": round(alpha, 3),
        "fit_window": [lo, hi],
        "pc1_pct": round(float(ev[0] / total) * 100, 2),
        "top128_pct": round(float(cum[127]) * 100, 2),
        "pcs_for_90pct": int(np.searchsorted(cum, 0.90)) + 1,
        "n_positions": n,
        "eigs_head": [float(e) for e in ev[:512]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    inf = load_checkpoint(BODY, device=dev)
    w = inf.weights()
    in_proj = torch.tensor(w["linear_in_proj"], device=dev)          # [256, M]
    decays = torch.tensor(w["linear_decays"], device=dev, dtype=torch.float64)
    emb_trained = torch.tensor(w["linear_embedding.weight"], device=dev)

    torch.manual_seed(SEED)
    emb_untrained = torch.randn(256, 256, device=dev)

    rows: dict[str, dict] = {}
    for name in STREAMS:
        data = np.fromfile(STREAM_DIR / f"{name}.bin", dtype=np.uint8)
        drive = (emb_untrained @ in_proj).T.contiguous()
        rows[name] = spectrum_for_stream(data, drive, decays, dev)
        print(f"  {name:<16} alpha {rows[name]['alpha']:>6.3f}   "
              f"pc1 {rows[name]['pc1_pct']:>6.2f}%   "
              f"top128 {rows[name]['top128_pct']:>6.2f}%   "
              f"pcs90 {rows[name]['pcs_for_90pct']:>5}", flush=True)

    data = np.fromfile(STREAM_DIR / "enwik_real.bin", dtype=np.uint8)
    drive = (emb_trained @ in_proj).T.contiguous()
    rows["enwik_real_TRAINED_emb"] = spectrum_for_stream(data, drive, decays, dev)
    r = rows["enwik_real_TRAINED_emb"]
    print(f"  {'real+TRAINED emb':<16} alpha {r['alpha']:>6.3f}   "
          f"pc1 {r['pc1_pct']:>6.2f}%   top128 {r['top128_pct']:>6.2f}%   "
          f"pcs90 {r['pcs_for_90pct']:>5}")

    gap = rows["enwik_real"]["alpha"] - rows["enwik_shuffled"]["alpha"]
    print(f"\n  KERNEL-ONLY exponent (shuffled): {rows['enwik_shuffled']['alpha']:.3f}")
    print(f"  INPUT-LAW share (real - shuffled gap): {gap:+.3f}")
    print(f"  TRAINED-emb shift on real: "
          f"{rows['enwik_real_TRAINED_emb']['alpha'] - rows['enwik_real']['alpha']:+.3f}")

    OUT.write_text(json.dumps(
        {"body": BODY, "seed": SEED, "chunk": CH,
         "fit_window_rule": "PCs 8..min(512, M/4)", "rows": rows}, indent=1))
    print(f"\n  saved -> {OUT}")


if __name__ == "__main__":
    main()
