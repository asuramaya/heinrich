#!/usr/bin/env python3
"""Trough channel decompose (thread 104): is the stationary [1280,1792)
deficit a BINDING-channel crater or a LOCAL-channel artifact?

The 100k 256-bin re-read (Soundwave V, msg 117) showed the trough at
2.5-3.5x linear_half_life_max does not close with training — fixed
structure where no mode's half-life reaches. If that reading is right,
the deficit must live in the binding (substrate) channel: the local head
has an 8-byte window and is depth-blind by construction.

Protocol: EXACTLY the definitive effective-context protocol (same val
file, same loader, n_trials=48, seqlen=2048, 256-wide deep bins), but
scoring three logit sets from ONE forward per sequence (the model
stashes both heads on eval forwards — no ablation, no second pass):
  joined        — logits_linear + gate * logits_local  (the deployed model)
  binding solo  — logits_linear alone
  local solo    — logits_local alone

Per-bucket mean bpb for each, plus the binding-minus-joined gap (what
the local head is buying at that depth).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from heinrich.profile.compare import (
    _load_effective_context_backend,
    _load_effective_context_val,
)

EDGES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1280, 1536, 1792, 2048]
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "data"


def per_position_bits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """CE in bits at each position: logits[t] predicts targets[t]."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    nll = -logp.gather(-1, targets[:, :, None]).squeeze(-1)
    return nll / np.log(2.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help=".checkpoint.pt path")
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--val", required=True, help="validation .bin (byte shard)")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--n-trials", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42,
                    help="val slice seed. The definitive effctx runs pin 42; a "
                         "different seed resamples WHICH bytes sit at each depth "
                         "— the discriminator between a kernel hole (trough "
                         "stays) and fixed-content difficulty (trough moves)")
    args = ap.parse_args()

    backend = _load_effective_context_backend(args.model, args.result_json, None)
    vocab = int(backend.config.vocab_size)
    if args.seed == 42:
        seqs = _load_effective_context_val(args.val, args.seqlen, args.n_trials, vocab)
    else:
        from heinrich.backend.decepticon import load_val_sequences
        seqs = load_val_sequences(args.val, seq_len=args.seqlen,
                                  n_seqs=args.n_trials, seed=args.seed,
                                  byte_level=vocab == 256)
    inner = backend.model._model  # torch CausalBankModel (stash lives here)
    dev = next(inner.parameters()).device
    inner.eval()

    n_seqs, seqlen = seqs.shape
    bits = {k: np.full((n_seqs, seqlen - 1), np.nan, dtype=np.float64)
            for k in ("joined", "binding", "local")}
    gate_val = None

    for i in range(n_seqs):
        x = torch.as_tensor(seqs[i:i + 1], dtype=torch.long, device=dev)
        with torch.inference_mode():
            joined = inner(x)
        lin = getattr(inner, "_last_logits_linear", None)
        loc = getattr(inner, "_last_logits_local", None)
        gate = getattr(inner, "_last_gate", None)
        if lin is None or loc is None:
            raise RuntimeError(
                "model did not stash _last_logits_linear/_last_logits_local — "
                "local path inactive on this body; nothing to decompose")
        if loc.dim() == 4:  # patch-at-readout broadcast form [B,T,1,V]
            loc = loc.squeeze(2)
        if lin.dim() == 4:
            lin = lin.squeeze(2)
        if joined.shape != lin.shape or joined.shape != loc.shape:
            raise RuntimeError(f"head shape mismatch: joined {tuple(joined.shape)} "
                               f"lin {tuple(lin.shape)} loc {tuple(loc.shape)}")
        gate_val = float(gate) if gate is not None and gate.numel() == 1 else None

        tgt = x[:, 1:]
        bits["joined"][i] = per_position_bits(joined[:, :-1], tgt)[0].cpu().numpy()
        bits["binding"][i] = per_position_bits(lin[:, :-1], tgt)[0].cpu().numpy()
        bits["local"][i] = per_position_bits(loc[:, :-1], tgt)[0].cpu().numpy()

    # Position t (0-based) predicts byte t+1 with t+1 bytes of context;
    # bucket by context length t+1 — matches the effctx bucket convention.
    ctx = np.arange(1, seqlen)
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        mask = (ctx >= lo) & (ctx < hi)
        if not mask.any():
            continue
        row = {"bucket": f"[{lo},{hi})", "n": int(mask.sum())}
        for k in bits:
            m = bits[k][:, mask]
            row[f"bpb_{k}"] = round(float(m.mean()), 4)
            row[f"sem_{k}"] = round(float(m.mean(axis=1).std(ddof=1)
                                          / np.sqrt(n_seqs)), 4)
        row["local_buys"] = round(row["bpb_binding"] - row["bpb_joined"], 4)
        rows.append(row)

    name = Path(args.model).stem.replace(".checkpoint", "")
    if args.seed != 42:
        name += f"-seed{args.seed}"
    print(f"\n=== Trough channel decompose: {name} ===")
    print(f"  gate (local_scale): {gate_val}   n_trials={n_seqs} seqlen={seqlen}")
    print(f"\n  {'bucket':<14} {'n':>5}  {'joined':>8} {'binding':>8} "
          f"{'local':>8}  {'local_buys':>10}")
    for r in rows:
        print(f"  {r['bucket']:<14} {r['n']:>5}  {r['bpb_joined']:>8.4f} "
              f"{r['bpb_binding']:>8.4f} {r['bpb_local']:>8.4f}  "
              f"{r['local_buys']:>10.4f}")

    out = OUT_DIR / f"trough-channels-{name}.json"
    out.write_text(json.dumps(
        {"model": args.model, "val": args.val, "n_trials": n_seqs,
         "seqlen": seqlen, "seed": args.seed, "gate": gate_val,
         "buckets": rows}, indent=1))
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
