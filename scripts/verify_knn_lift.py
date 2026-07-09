#!/usr/bin/env python3
"""Two-witness verification of the state-kNN lift (thread 90).

CLAIM UNDER TEST (Soundwave's round 3, ruling d9d723ba): kNN-LM mix at
lam=0.05 lifts fo-noadapt-enwik90-50k by -0.0144 +/- 0.0122 bpb on held-out
enwik8[95M:96.5M]; 8M-key store from enwik8[0:8M]; 18/32 windows improved.

Protocol (staked in msgs 93/98 BEFORE this run):
  - Spec from their harness, scoring from THIS instrument: their key
    construction (CONTINUOUS states streamed from stream-start via the body's
    own bank tensors — m3 FFT form; NOT window-truncated forward_captured
    states) against MY independent CE accounting and CI.
  - Base term stays windowed forward_captured logits, L=1024 burn 256 —
    deliberately: the model as deployed. The asymmetry is the claim.
  - Their exact 32 test windows (rng(0) over [95M:96.5M], first 12 = cal)
    reported as the DIRECT two-witness row; 32 extra windows (rng(1) over
    [96.5M:100M], clean of both store and their queries) pool to 64 for the
    tightened CI. Pinned hyperparams (k=64 tau=20 eps=0.1 lam=0.05) scored
    alongside my own cal-grid choice.
  - Kill: pooled delta CI crossing zero AND direct row sign-flipped.
    Bank: direct row within their CI, same direction. Either outcome pays.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from decepticons.loader import load_checkpoint

R = Path("/home/asuramaya/code/REPOS/chronohorn/out/results")
BODY = R / "fo-noadapt-enwik90-50k.checkpoint.pt"
ENWIK = Path("/home/asuramaya/code/REPOS/chronohorn/data/roots/enwik/enwik8")
OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "knn-two-witness.json"

STORE_BYTES = 8_000_000
L, BURN = 1024, 256
KEY_DIM = 128
CH = 32768
PINNED = {"k": 64, "tau": 20.0, "eps": 0.1, "lam": 0.05}
DEV = "cuda"


def stream_states(data: np.ndarray, drive, decays):
    m = decays.shape[0]
    nf = 2 * CH
    jj = torch.arange(CH, device=DEV, dtype=torch.float64)
    la = torch.log(decays)[:, None]
    kf = torch.fft.rfft(((1 - decays)[:, None] * torch.exp(la * jj)).float(), n=nf, dim=1)
    apow = torch.exp(la * (jj + 1.0)).float()
    h = torch.zeros(m, device=DEV)
    for s in range(0, len(data), CH):
        chunk = torch.as_tensor(data[s:s + CH], device=DEV)
        d = drive[:, chunk]
        df = torch.fft.rfft(d, n=nf, dim=1)
        st = torch.fft.irfft(df * kf, n=nf, dim=1)[:, :len(chunk)] + h[:, None] * apow[:, :len(chunk)]
        h = st[:, -1].clone()
        yield st.T, s


def main() -> None:
    enwik = np.fromfile(ENWIK, dtype=np.uint8)
    train = enwik[:STORE_BYTES].astype(np.int64)
    marg = torch.tensor(np.bincount(train, minlength=256) / len(train),
                        device=DEV, dtype=torch.float32)

    inf = load_checkpoint(str(BODY), device=DEV)
    w = inf.weights()
    emb = torch.tensor(w["linear_embedding.weight"], device=DEV)
    in_proj = torch.tensor(w["linear_in_proj"], device=DEV)
    decays = torch.tensor(w["linear_decays"], device=DEV, dtype=torch.float64)
    drive = (emb @ in_proj).T.contiguous()

    # whitening basis from the store stream (truncate-then-whiten, top-128)
    m = decays.shape[0]
    gram = torch.zeros(m, m, device=DEV, dtype=torch.float64)
    n = 0
    for st, _ in stream_states(train, drive, decays):
        # fp64 matmul is 1/32-rate on this card (arm-A lesson): fp32
        # chunk-products into an fp64 accumulator — same samples, and the
        # whitening basis is insensitive to product precision at n=8M.
        gram += (st.T @ st).to(torch.float64)
        n += len(st)
    ev, u = torch.linalg.eigh(gram / n)
    ev, u = ev.flip(0), u.flip(1)
    proj = (u[:, :KEY_DIM] / torch.sqrt(ev[:KEY_DIM].clamp_min(1e-10))).float()
    print(f"whitening ready (n={n}); top-{KEY_DIM} share "
          f"{float(ev[:KEY_DIM].sum() / ev.sum()):.3f}", flush=True)

    keys = torch.empty(STORE_BYTES - 1, KEY_DIM, device=DEV, dtype=torch.float16)
    for st, s in stream_states(train, drive, decays):
        hi = min(s + len(st), STORE_BYTES - 1)
        keys[s:hi] = F.normalize(st[:hi - s] @ proj, dim=1).half()
    vals = torch.as_tensor(train[1:], device=DEV)
    torch.cuda.empty_cache()

    # query regions: A = theirs (direct row), B = mine (extension)
    regions = {
        "A": (enwik[95_000_000:96_500_000].astype(np.int64), np.random.default_rng(0), 44),
        "B": (enwik[96_500_000:100_000_000].astype(np.int64), np.random.default_rng(1), 32),
    }

    def gather(region, rng, count):
        starts = rng.integers(0, len(region) - L - 1, size=count)
        states = torch.empty(len(region), KEY_DIM, device=DEV, dtype=torch.float16)
        for st, s in stream_states(region, drive, decays):
            states[s:s + len(st)] = F.normalize(st @ proj, dim=1).half()
        Q, Y, B = [], [], []
        for s in starts:
            seq = region[s:s + L]
            cap = inf.forward_captured(seq[None, :])
            lg = torch.as_tensor(cap["logits"][0], device=DEV)[BURN:-1]
            pos = torch.arange(s + BURN, s + L - 1, device=DEV)
            Q.append(states[pos])
            Y.append(torch.as_tensor(seq[BURN + 1:], device=DEV))
            B.append(lg)
        del states
        torch.cuda.empty_cache()
        return torch.cat(Q), torch.cat(Y), torch.cat(B)

    qa, ya, ba = gather(*regions["A"])
    qb, yb, bb = gather(*regions["B"])
    npos = L - BURN - 1
    cal = slice(0, 12 * npos)
    tstA = slice(12 * npos, 44 * npos)

    def search(Q, k=64, qtile=1024, ktile=1_000_000):
        ov, oi = [], []
        for qs in range(0, len(Q), qtile):
            q = Q[qs:qs + qtile]
            bv = torch.full((len(q), k), -1e4, device=DEV, dtype=torch.float16)
            bi = torch.zeros((len(q), k), device=DEV, dtype=torch.long)
            for ks in range(0, len(keys), ktile):
                sc = q @ keys[ks:ks + ktile].T
                v, i = sc.topk(k, dim=1)
                bv, sel = torch.cat([bv, v], 1).topk(k, dim=1)
                bi = torch.cat([bi, i + ks], 1).gather(1, sel)
            ov.append(bv.float())
            oi.append(bi)
        return torch.cat(ov), torch.cat(oi)

    va, ia = search(qa)
    vb, ib = search(qb)
    print("searched", flush=True)

    def vote(v, i, k, tau, eps):
        wts = torch.softmax(v[:, :k] * tau, dim=1)
        p = torch.zeros(len(v), 256, device=DEV)
        p.scatter_add_(1, vals[i[:, :k]], wts)
        return (1 - eps) * p + eps * marg[None, :]

    def bits(logp, y):
        return -logp[torch.arange(len(y), device=DEV), y] / np.log(2)

    def mix_bits(v, i, base, y, k, tau, eps, lam):
        p_knn = vote(v, i, k, tau, eps)
        p_base = torch.softmax(base.float(), -1)
        return bits(torch.log((lam * p_knn + (1 - lam) * p_base).clamp_min(1e-12)), y)

    # independent cal grid on their 12 cal windows
    best = None
    for k in (8, 16, 32, 64):
        for tau in (5.0, 10.0, 20.0, 40.0):
            for eps in (0.05, 0.1, 0.25):
                for lam in (0.02, 0.05, 0.1, 0.2):
                    b = float(mix_bits(va[cal], ia[cal], ba[cal], ya[cal],
                                       k, tau, eps, lam).mean())
                    if best is None or b < best[0]:
                        best = (b, {"k": k, "tau": tau, "eps": eps, "lam": lam})
    cal_params = best[1]
    print(f"cal grid chose {cal_params} (cal bpb {best[0]:.4f}); pinned {PINNED}", flush=True)

    def report(tag, v, i, base, y, n_windows, params):
        mb = mix_bits(v, i, base, y, **params).reshape(n_windows, -1).mean(1)
        bb_ = bits(torch.log_softmax(base.float(), -1), y).reshape(n_windows, -1).mean(1)
        d = (mb - bb_).cpu().numpy()
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        row = {"delta": round(float(d.mean()), 4), "ci95": round(float(ci), 4),
               "improved": f"{int((d < 0).sum())}/{len(d)}",
               "base_bpb": round(float(bb_.mean()), 4), "mix_bpb": round(float(mb.mean()), 4)}
        print(f"  {tag:<28} delta {row['delta']:+.4f} +/-{row['ci95']:.4f}  "
              f"improved {row['improved']}  base {row['base_bpb']:.4f}", flush=True)
        return row

    print("\n=== two-witness rows ===")
    out = {"claim": "-0.0144 +/- 0.0122, 18/32 improved", "cal_params": cal_params,
           "pinned": PINNED, "rows": {}}
    out["rows"]["direct_pinned"] = report(
        "DIRECT (their 32, pinned)", va[tstA], ia[tstA], ba[tstA], ya[tstA], 32, PINNED)
    out["rows"]["direct_calgrid"] = report(
        "direct (their 32, my cal)", va[tstA], ia[tstA], ba[tstA], ya[tstA], 32, cal_params)
    pooled = (torch.cat([va[tstA], vb]), torch.cat([ia[tstA], ib]),
              torch.cat([ba[tstA], bb]), torch.cat([ya[tstA], yb]))
    out["rows"]["pooled64_pinned"] = report(
        "POOLED (64 windows, pinned)", *pooled, 64, PINNED)
    out["rows"]["pooled64_calgrid"] = report(
        "pooled (64 windows, my cal)", *pooled, 64, cal_params)

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
