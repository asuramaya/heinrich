#!/usr/bin/env python3
"""Experiment B of the anisotropy program (thread 87): is the steep residual
spectrum LEARNED or SUBSTRATE PHYSICS?

Decepticons' frozen bank shows eigenvalues ~ i^-2.48 with NOTHING learned.
This script separates their contribution (2) [kernel law: untrained
architecture on real data is already steep] from (3) [superposition proper:
training steepens/reshapes] on the transformer side of the boundary:

  For each model: run the SAME text through the TRAINED weights and a
  RANDOM-INIT twin (same config), capture residual states at every layer,
  fit the eigenspectrum power-law exponent per layer.

Pre-registered fit window (msg 88, guards against ruler disagreements):
  fit log(eig) ~ -alpha*log(i) on PCs 8..min(512, d/4); full curve saved.

Verdicts this can return:
  - untrained exponent ~ trained exponent  -> steepness is substrate+input
    physics; superposition must live elsewhere (the divergence, experiment C).
  - trained >> untrained                   -> training steepens: learned
    concentration is real on top of the kernel law.
  - steep already at the embedding layer   -> input law visible pre-mixing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from heinrich.core.db import SignalDB

FIT_LO = 8                       # pre-registered window (msg 88)
SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "data"


def fit_exponent(eigs: np.ndarray, dim: int) -> tuple[float, int, int]:
    """Slope of log(eig) vs log(rank) on the pre-registered window."""
    hi = min(512, dim // 4)
    lo = FIT_LO
    window = eigs[lo - 1:hi]
    ranks = np.arange(lo, lo + len(window))
    mask = window > 0
    coef = np.polyfit(np.log(ranks[mask]), np.log(window[mask]), 1)
    return float(-coef[0]), lo, hi


def spectrum_by_layer(model, ids_batches: list[torch.Tensor],
                      n_sample: int) -> list[dict]:
    """Per-layer eigenspectrum stats from hidden states over all batches."""
    per_layer: list[list[np.ndarray]] = []
    with torch.no_grad():
        for ids in ids_batches:
            out = model(ids, output_hidden_states=True)
            hs = out.hidden_states          # embedding + one per layer
            if not per_layer:
                per_layer = [[] for _ in hs]
            for li, h in enumerate(hs):
                # all positions except the first 2 (BOS/onset transients)
                per_layer[li].append(h[0, 2:, :].float().cpu().numpy())

    rng = np.random.default_rng(SEED)
    rows = []
    for li, chunks in enumerate(per_layer):
        vecs = np.concatenate(chunks, axis=0)
        if len(vecs) > n_sample:
            vecs = vecs[rng.choice(len(vecs), n_sample, replace=False)]
        centered = vecs - vecs.mean(axis=0)
        s = np.linalg.svd(centered, compute_uv=False)
        eigs = s ** 2
        total = float(eigs.sum())
        dim = vecs.shape[1]
        alpha, lo, hi = fit_exponent(eigs, dim)
        cum = np.cumsum(eigs) / total
        rows.append({
            "layer": li,                       # 0 = embedding output
            "alpha": round(alpha, 3),
            "fit_window": [lo, hi],
            "pc1_pct": round(float(eigs[0] / total) * 100, 2),
            "top128_pct": round(float(cum[min(127, len(cum) - 1)]) * 100, 2),
            "pcs_for_90pct": int(np.searchsorted(cum, 0.90)) + 1,
            "eigs_head": [round(float(e), 6) for e in eigs[:64]],
        })
    return rows


def run_model(name: str, texts: list[str], n_sample: int,
              max_len: int, batches: int) -> dict:
    tok = AutoTokenizer.from_pretrained(name)
    enc = [tok(t, return_tensors="pt", truncation=True,
               max_length=max_len).input_ids for t in texts]
    enc = [e for e in enc if e.shape[1] >= 16][:batches]
    if len(enc) < 8:
        raise RuntimeError(f"only {len(enc)} usable prompts >= 16 tokens")

    result: dict = {"model": name, "n_texts": len(enc),
                    "fit_window_rule": "PCs 8..min(512, d/4)"}

    trained = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float32).eval()
    result["trained"] = spectrum_by_layer(trained, enc, n_sample)
    del trained

    torch.manual_seed(SEED)
    cfg = AutoConfig.from_pretrained(name)
    untrained = AutoModelForCausalLM.from_config(cfg).float().eval()
    result["untrained"] = spectrum_by_layer(untrained, enc, n_sample)
    del untrained
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="HuggingFaceTB/SmolLM2-135M",
                    help="comma-separated HF model names")
    ap.add_argument("--n-sample", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batches", type=int, default=64)
    args = ap.parse_args()

    db = SignalDB()
    texts = [p["text"] for p in db.require_prompts(is_benign=True, limit=256)]

    for name in args.models.split(","):
        name = name.strip()
        res = run_model(name, texts, args.n_sample, args.max_len, args.batches)
        slug = name.split("/")[-1].lower()
        out = OUT_DIR / f"anisotropy-b-{slug}.json"
        out.write_text(json.dumps(res, indent=1))

        print(f"\n=== {name} — spectrum exponent by depth "
              f"(window {res['trained'][0]['fit_window']}) ===")
        print(f"  {'layer':>5}  {'alpha_tr':>8}  {'alpha_un':>8}  "
              f"{'pc1%_tr':>7}  {'pc1%_un':>7}  {'top128%_tr':>10}  {'top128%_un':>10}")
        for tr, un in zip(res["trained"], res["untrained"]):
            tag = "emb" if tr["layer"] == 0 else f"L{tr['layer'] - 1:02d}"
            print(f"  {tag:>5}  {tr['alpha']:>8.3f}  {un['alpha']:>8.3f}  "
                  f"{tr['pc1_pct']:>7.2f}  {un['pc1_pct']:>7.2f}  "
                  f"{tr['top128_pct']:>10.2f}  {un['top128_pct']:>10.2f}")
        mean_tr = np.mean([r["alpha"] for r in res["trained"][1:]])
        mean_un = np.mean([r["alpha"] for r in res["untrained"][1:]])
        print(f"\n  mean alpha (residual layers): trained {mean_tr:.3f}  "
              f"untrained {mean_un:.3f}  ratio {mean_tr / mean_un:.2f}")
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
