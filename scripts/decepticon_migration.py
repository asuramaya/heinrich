#!/usr/bin/env python3
"""Head-ablation migration forensic for the causal-bank daemon.

Does associative memory EMERGE from gradient (bits migrate off the local n-gram floor
onto the binding organ) or must it be INSTALLED? The daemon's logits are a two-head join
    logits = logits_linear + gate * logits_local
  LOCAL head  = local_embedding + local_readout (order-8 conv; the ~2.37 bpb rote floor)
  BINDING head= frozen 2048-bank + adaptive complex recurrence + linear_readout (associative)
Both heads are captured pre-join in ONE forward, so we score three EXACT bpb per checkpoint:
  binding-only = CE(logits_linear)   local-only = CE(logits_local)   full = CE(join)
plus the local gate share ||gate*local|| / ||linear|| (= ||full-linear||/||linear||).

Migration = binding-only bpb FALLING across training in the learnable trail while the
matched FROZEN twin (adaptive learning off) stays flat -> the organ is being populated by
SGD (Adam bit the planted tree). Bits riding the local floor in both = installation needed.

Reproduce: python scripts/decepticon_migration.py [--smoke]
"""
from __future__ import annotations
import math
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from heinrich.backend.decepticon import DecepticonBackend, load_val_sequences

R = os.path.expanduser("~/code/REPOS/chronohorn/out/results")
# enwik8[95M:95.5M] held-out test region (0806072e: diet_text/enwik_val IS this slice, in-distribution)
DATA = os.path.expanduser("~/code/REPOS/chronohorn/data/roots/diet_text/enwik_val_000000.bin")
N_SEQS, SEQ_LEN, SEED = 128, 1024, 42
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_FLOOR = 2.37  # order-5/8 count-table floor (0806072e)

# (label, step, path, arm)
CKPTS = [
    ("genome",    0,     f"{R}/fo-learnable-innate-seed42.checkpoint.pt", "learnable"),
    ("learn-s8",  2500,  f"{R}/fo-learnable-s8_step2500.checkpoint.pt",   "learnable"),
    ("learn-s8",  5000,  f"{R}/fo-learnable-s8_step5000.checkpoint.pt",   "learnable"),
    ("learn-s8",  7500,  f"{R}/fo-learnable-s8_step7500.checkpoint.pt",   "learnable"),
    ("learn-s8",  10000, f"{R}/fo-learnable-s8_step10000.checkpoint.pt",  "learnable"),
    ("frozen-s8", 2500,  f"{R}/fo-frozen-s8_step2500.checkpoint.pt",      "frozen"),
    ("frozen-s8", 5000,  f"{R}/fo-frozen-s8_step5000.checkpoint.pt",      "frozen"),
    ("frozen-s8", 7500,  f"{R}/fo-frozen-s8_step7500.checkpoint.pt",      "frozen"),
    ("frozen-s8", 10000, f"{R}/fo-frozen-s8_step10000.checkpoint.pt",     "frozen"),
    ("learn-50k", 10000, f"{R}/fo-learnable-50k_step10000.checkpoint.pt", "learnable"),
    ("learn-50k", 20000, f"{R}/fo-learnable-50k_step20000.checkpoint.pt", "learnable"),
    ("learn-50k", 30000, f"{R}/fo-learnable-50k_step30000.checkpoint.pt", "learnable"),
    ("learn-50k", 40000, f"{R}/fo-learnable-50k_step40000.checkpoint.pt", "learnable"),
    ("learn-50k", 50000, f"{R}/fo-learnable-50k_step50000.checkpoint.pt", "learnable"),
]

# 0806072e's matched organ ablation — same data(diet_text)/seed42/schedule, ONLY the
# adaptive organ differs. Image ON-vs-OFF WITHIN this pair (do NOT compare to fo-learnable-50k).
PAIR = [
    ("adapt-s8",   2500,  f"{R}/fo-adapt-s8_step2500.checkpoint.pt",    "adapt"),
    ("adapt-s8",   5000,  f"{R}/fo-adapt-s8_step5000.checkpoint.pt",    "adapt"),
    ("adapt-s8",   7500,  f"{R}/fo-adapt-s8_step7500.checkpoint.pt",    "adapt"),
    ("adapt-s8",   10000, f"{R}/fo-adapt-s8_step10000.checkpoint.pt",   "adapt"),
    ("noadapt-s8", 2500,  f"{R}/fo-noadapt-s8_step2500.checkpoint.pt",  "noadapt"),
    ("noadapt-s8", 5000,  f"{R}/fo-noadapt-s8_step5000.checkpoint.pt",  "noadapt"),
    ("noadapt-s8", 7500,  f"{R}/fo-noadapt-s8_step7500.checkpoint.pt",  "noadapt"),
    ("noadapt-s8", 10000, f"{R}/fo-noadapt-s8_step10000.checkpoint.pt", "noadapt"),
]


def make_sample():
    # Reference loader: chronohorn shards keep a 1024-byte header and store bytes
    # (0-255) in uint16 slots — raw uint8 fromfile is WRONG (scrambles + header junk).
    return load_val_sequences(DATA, seq_len=SEQ_LEN + 1, n_seqs=N_SEQS, seed=SEED, byte_level=True)


def measure(path, sample):
    """Score binding-only / local-only / full bpb using the CANONICAL head tensors
    0806072e exposed via forward_captured (linear_logits/local_logits) — the identical
    tensors the kernel-native witness reads, not a hook guess."""
    be = DecepticonBackend(path, device=DEV)
    ce_full = ce_bind = ce_loc = 0.0
    ntok = 0
    have_lin = have_loc = False
    ln2 = math.log(2.0)
    BATCH = 32
    for i in range(0, sample.shape[0], BATCH):
        chunk = sample[i:i + BATCH]
        inp = chunk[:, :SEQ_LEN].astype(np.int64)                       # [b, L]
        tgt = torch.from_numpy(chunk[:, 1:SEQ_LEN + 1].astype(np.int64)).to(DEV).reshape(-1)
        cap = be.model.forward_captured(inp)
        V = np.asarray(cap["logits"]).shape[-1]
        full = torch.as_tensor(np.asarray(cap["logits"]), device=DEV).reshape(-1, V)
        ce_full += F.cross_entropy(full, tgt, reduction="sum").item()
        if cap.get("linear_logits") is not None:
            have_lin = True
            lin = torch.as_tensor(np.asarray(cap["linear_logits"]), device=DEV).reshape(-1, V)
            ce_bind += F.cross_entropy(lin, tgt, reduction="sum").item()
        if cap.get("local_logits") is not None:
            have_loc = True
            loc = torch.as_tensor(np.asarray(cap["local_logits"]), device=DEV).reshape(-1, V)
            ce_loc += F.cross_entropy(loc, tgt, reduction="sum").item()
        ntok += int(tgt.numel())
    out = {
        "full_bpb": ce_full / ntok / ln2,
        "binding_bpb": (ce_bind / ntok / ln2) if have_lin else None,
        "local_bpb": (ce_loc / ntok / ln2) if have_loc else None,
        "gate_share": None,
        "ntok": ntok,
    }
    del be
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return out


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s8 = sorted([r for r in rows if r["label"] == "learn-s8"] + [r for r in rows if r["label"] == "genome"], key=lambda r: r["step"])
    fz = sorted([r for r in rows if r["label"] == "frozen-s8"], key=lambda r: r["step"])
    k50 = sorted([r for r in rows if r["label"] == "learn-50k"], key=lambda r: r["step"])
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.axhline(8.0, ls=":", color="#bbb", lw=1, label="random (8 bpb)")
    ax.axhline(LOCAL_FLOOR, ls=":", color="#888", lw=1.2, label=f"order-5 local floor ({LOCAL_FLOOR})")
    ax.plot([r["step"] for r in s8], [r["binding_bpb"] for r in s8], "o-", color="#4a90d9", lw=2, label="binding-only — learnable (organ ON)")
    ax.plot([r["step"] for r in k50], [r["binding_bpb"] for r in k50], "o-", color="#4a90d9", lw=2)
    ax.plot([r["step"] for r in fz], [r["binding_bpb"] for r in fz], "s--", color="#d9534f", lw=2, label="binding-only — FROZEN twin (organ OFF)")
    ax.plot([r["step"] for r in s8 + k50], [r["full_bpb"] for r in s8 + k50], "^-", color="#2ca02c", lw=1.4, alpha=0.8, label="full (both heads, learnable)")
    ax.annotate("Adam bites: organ populated by SGD", (10000, 2.13), (16000, 3.2),
                arrowprops=dict(arrowstyle="->", color="#4a90d9"), fontsize=9, color="#4a90d9")
    ax.annotate("frozen substrate stalls", (10000, 2.49), (16000, 2.7),
                arrowprops=dict(arrowstyle="->", color="#d9534f"), fontsize=9, color="#d9534f")
    ax.set_xlabel("training step"); ax.set_ylabel("bits / byte"); ax.set_ylim(1.4, 8.3)
    ax.set_title("Did associative memory emerge, or must it be installed?\nbinding-only bpb: the organ migrates off random; its frozen twin can't")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("docs/data/decepticon-migration.png", dpi=135)
    print("wrote docs/data/decepticon-migration.png")


def main():
    if "--plot" in sys.argv:
        plot(json.load(open("docs/data/decepticon-migration.json")))
        return
    if "--pair" in sys.argv:
        sample = make_sample()
        print(f"sample {sample.shape}; second-witnessing 0806072e's matched organ pair (ON vs OFF), dev={DEV}")
        rows = []
        for label, step, path, arm in PAIR:
            if not os.path.exists(path):
                print(f"  SKIP (missing): {os.path.basename(path)}")
                continue
            m = measure(path, sample)
            m.update(label=label, step=step, arm=arm)
            rows.append(m)
            bb = f"{m['binding_bpb']:.3f}" if m["binding_bpb"] is not None else "  -  "
            lb = f"{m['local_bpb']:.3f}" if m["local_bpb"] is not None else "  -  "
            print(f"  {arm:8} step {step:6}  full={m['full_bpb']:.3f}  binding={bb}  local={lb}")
        # ON-minus-OFF full-bpb delta per step (the clean organ contribution)
        for st in sorted({r["step"] for r in rows}):
            on = next((r for r in rows if r["arm"] == "adapt" and r["step"] == st), None)
            off = next((r for r in rows if r["arm"] == "noadapt" and r["step"] == st), None)
            if on and off:
                print(f"  step {st:6}: organ delta (OFF - ON) full = {off['full_bpb'] - on['full_bpb']:+.3f} bpb")
        json.dump(rows, open("docs/data/decepticon-organ-pair.json", "w"), indent=0)
        return
    smoke = "--smoke" in sys.argv
    sample = make_sample()
    print(f"sample: {sample.shape} bytes from {os.path.basename(DATA)} (enwik8[95M:95.5M]); dev={DEV}")

    # SELF-CHECK: step-50k full bpb must reproduce the known 1.6587
    print("\n[self-check] fo-learnable-50k_step50000 full bpb (expect ~1.659)...")
    chk = measure(f"{R}/fo-learnable-50k_step50000.checkpoint.pt", sample)
    print(f"  full={chk['full_bpb']:.4f}  binding={chk['binding_bpb']}  local={chk['local_bpb']}  gate_share={chk['gate_share']}")
    ok = 1.45 < chk["full_bpb"] < 1.90
    print(f"  self-check {'PASS' if ok else 'FAIL — pipeline wrong, aborting'}")
    if not ok or smoke:
        return

    rows = []
    for label, step, path, arm in CKPTS:
        if not os.path.exists(path):
            print(f"  SKIP (missing): {os.path.basename(path)}")
            continue
        m = measure(path, sample)
        m.update(label=label, step=step, arm=arm)
        rows.append(m)
        bb = f"{m['binding_bpb']:.3f}" if m["binding_bpb"] is not None else "  -  "
        lb = f"{m['local_bpb']:.3f}" if m["local_bpb"] is not None else "  -  "
        gs = f"{m['gate_share']:.3f}" if m["gate_share"] is not None else "  -  "
        print(f"  {label:10} step {step:6}  {arm:9}  full={m['full_bpb']:.3f}  binding={bb}  local={lb}  gate_share(local)={gs}")

    json.dump(rows, open("docs/data/decepticon-migration.json", "w"), indent=0)
    print(f"\nwrote docs/data/decepticon-migration.json ({len(rows)} checkpoints). local floor = {LOCAL_FLOOR} bpb")


if __name__ == "__main__":
    main()
