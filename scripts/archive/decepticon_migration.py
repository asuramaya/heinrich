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
    # 50k extension of the same matched pair (both arms trained at seq_len 1024, seed 42):
    # the scale-gated test — NEUTRAL at 10k was the s8 verdict; does ON pull below OFF by 50k?
    ("adapt-50k",   10000, f"{R}/fo-adapt-50k_step10000.checkpoint.pt",   "adapt"),
    ("adapt-50k",   20000, f"{R}/fo-adapt-50k_step20000.checkpoint.pt",   "adapt"),
    ("adapt-50k",   30000, f"{R}/fo-adapt-50k_step30000.checkpoint.pt",   "adapt"),
    ("adapt-50k",   40000, f"{R}/fo-adapt-50k_step40000.checkpoint.pt",   "adapt"),
    ("adapt-50k",   50000, f"{R}/fo-adapt-50k_step50000.checkpoint.pt",   "adapt"),
    ("noadapt-50k", 10000, f"{R}/fo-noadapt-50k_step10000.checkpoint.pt", "noadapt"),
    ("noadapt-50k", 20000, f"{R}/fo-noadapt-50k_step20000.checkpoint.pt", "noadapt"),
    ("noadapt-50k", 30000, f"{R}/fo-noadapt-50k_step30000.checkpoint.pt", "noadapt"),
    ("noadapt-50k", 40000, f"{R}/fo-noadapt-50k_step40000.checkpoint.pt", "noadapt"),
    ("noadapt-50k", 50000, f"{R}/fo-noadapt-50k_step50000.checkpoint.pt", "noadapt"),
]

# Attribution control for fo-learnable-50k's headline number: SAME data (enwik
# shards_train90m), seed 42, schedule — organ OFF. Config diff vs learnable-50k is two
# knobs (adaptive_substrate, hrr_omega_init); init is bit-identical to fo-noadapt-50k
# (decepticons-verified init_signature 165935100cbc3809, n_params 5,114,368). If this
# trail matches learn-50k's full bpb, the gain was never the organ's; if it lags,
# the organ (or its ω init) earns the difference. WITHIN-lineage: compare only to the
# learn-50k rows measured by this same script on the same enwik_val sample.
ATTRIB = [
    ("noadapt-e90", 10000, f"{R}/fo-noadapt-enwik90-50k_step10000.checkpoint.pt", "noadapt"),
    ("noadapt-e90", 20000, f"{R}/fo-noadapt-enwik90-50k_step20000.checkpoint.pt", "noadapt"),
    ("noadapt-e90", 30000, f"{R}/fo-noadapt-enwik90-50k_step30000.checkpoint.pt", "noadapt"),
    ("noadapt-e90", 40000, f"{R}/fo-noadapt-enwik90-50k_step40000.checkpoint.pt", "noadapt"),
    ("noadapt-e90", 50000, f"{R}/fo-noadapt-enwik90-50k_step50000.checkpoint.pt", "noadapt"),
]


def make_sample():
    # Reference loader: chronohorn shards keep a 1024-byte header and store bytes
    # (0-255) in uint16 slots — raw uint8 fromfile is WRONG (scrambles + header junk).
    return load_val_sequences(DATA, seq_len=SEQ_LEN + 1, n_seqs=N_SEQS, seed=SEED, byte_level=True)


def measure(path, sample, external_binding=False):
    """Score binding-only / local-only / full bpb using the CANONICAL head tensors
    0806072e exposed via forward_captured (linear_logits/local_logits) — the identical
    tensors the kernel-native witness reads, not a hook guess.

    external_binding=True additionally scores the binding head by calling the model's
    _linear_logits directly (binding_bpb_ext). Non-adaptive bodies never stash the
    pre-join heads (thread bacf30e1), so this external read is the only instrument
    for an OFF body's composition; _linear_logits is the exact tensor the merge
    consumes and the recurrence is deterministic under inference mode, so on an ON
    body it must match the stashed binding_bpb to the digit (self-checked below)."""
    be = DecepticonBackend(path, device=DEV)
    ce_full = ce_bind = ce_loc = ce_ext = 0.0
    ntok = 0
    have_lin = have_loc = have_ext = False
    ln2 = math.log(2.0)
    # CE is summed per token, so batch size cannot change the score — keep it small
    # enough to coexist with other jobs on the shared 12GB card (32 OOMs when busy).
    BATCH = int(os.environ.get("HEINRICH_MIG_BATCH", "8"))
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
        if external_binding:
            with torch.inference_mode():
                lin_t = be.model._model._linear_logits(
                    torch.from_numpy(inp).to(DEV))
            if lin_t.dim() == 4:  # patch-at-readout bodies return [B,T,N,V]; p=0 matches "logits"
                lin_t = lin_t[:, :, 0, :]
            have_ext = True
            ce_ext += F.cross_entropy(lin_t.reshape(-1, V), tgt, reduction="sum").item()
        ntok += int(tgt.numel())
    out = {
        "full_bpb": ce_full / ntok / ln2,
        "binding_bpb": (ce_bind / ntok / ln2) if have_lin else None,
        "binding_bpb_ext": (ce_ext / ntok / ln2) if have_ext else None,
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


def reconcile():
    """Digit-tight two-witness: reproduce 0806072e's EXACT firmed protocol.
    raw enwik8[95M:100M], rng(0) 24 windows x 4096, per-window forward (batch=1),
    burn 1024 / score [1024:], CE the canonical forward_captured heads."""
    ENWIK8 = os.path.expanduser("~/code/REPOS/chronohorn/data/roots/enwik/enwik8")
    test = np.fromfile(ENWIK8, dtype=np.uint8)[95_000_000:100_000_000]
    starts = np.random.default_rng(0).integers(0, len(test) - 4096 - 1, size=24)
    WIN, BURN, ln2 = 4096, 1024, math.log(2.0)
    their = {0: (8.001, 8.003, 8.002), 10000: (2.343, 4.325, 1.962), 20000: (2.267, 4.093, 1.858),
             30000: (2.126, 4.138, 1.795), 40000: (2.062, 4.172, 1.747), 50000: (2.020, 4.212, 1.725)}
    CK = [(0, f"{R}/fo-learnable-innate-seed42.checkpoint.pt")] + \
         [(s, f"{R}/fo-learnable-50k_step{s}.checkpoint.pt") for s in (10000, 20000, 30000, 40000, 50000)]
    print(f"digit-tight reconcile: enwik8[95M:100M], rng(0) 24x{WIN}, burn {BURN}, per-window batch=1")
    print(f"{'step':>7} | {'binding me/them':>17} | {'local me/them':>17} | {'full me/them':>17}")
    for step, path in CK:
        if not os.path.exists(path):
            print(f"  skip (missing) {step}"); continue
        be = DecepticonBackend(path, device=DEV)
        cb = cl = cf = 0.0; n = 0
        sl = slice(BURN, None)
        for s in starts:
            seq = test[int(s):int(s) + WIN].astype(np.int64)
            tgt = torch.from_numpy(seq[1:]).to(DEV)
            cap = be.model.forward_captured(seq[None, :])
            V = np.asarray(cap["logits"]).shape[-1]
            full = torch.as_tensor(np.asarray(cap["logits"]), device=DEV)[0, :-1]
            cf += F.cross_entropy(full[sl], tgt[sl], reduction="sum").item()
            if cap.get("linear_logits") is not None:
                lin = torch.as_tensor(np.asarray(cap["linear_logits"]), device=DEV)[0, :-1]
                cb += F.cross_entropy(lin[sl], tgt[sl], reduction="sum").item()
            if cap.get("local_logits") is not None:
                loc = torch.as_tensor(np.asarray(cap["local_logits"]), device=DEV)[0, :-1]
                cl += F.cross_entropy(loc[sl], tgt[sl], reduction="sum").item()
            n += int(tgt[sl].numel())
        del be
        if DEV == "cuda":
            torch.cuda.empty_cache()
        mb, ml, mf = cb / n / ln2, cl / n / ln2, cf / n / ln2
        tb, tl, tf = their.get(step, (0, 0, 0))
        print(f"{step:>7} | {mb:6.3f}/{tb:<6} ({mb-tb:+.3f}) | {ml:6.3f}/{tl:<6} | {mf:6.3f}/{tf:<6} ({mf-tf:+.3f})")


def main():
    if "--reconcile" in sys.argv:
        reconcile()
        return
    if "--plot" in sys.argv:
        plot(json.load(open("docs/data/decepticon-migration.json")))
        return
    if "--attribution" in sys.argv:
        sample = make_sample()
        print(f"sample {sample.shape}; attribution trail: fo-noadapt-enwik90-50k vs learn-50k (same data/seed), dev={DEV}")
        try:
            prior = json.load(open("docs/data/decepticon-migration.json"))
            learn = {r["step"]: r for r in prior if r["label"] == "learn-50k"}
        except FileNotFoundError:
            learn = {}
            print("  (no decepticon-migration.json — run the main trail first for the comparison column)")

        # SELF-CHECK the external binding instrument on an ON body, where the
        # kernel-native stash exists: external _linear_logits CE must equal the
        # stashed-head CE to the digit, or the OFF readings below are untrusted.
        chk_path = f"{R}/fo-learnable-50k_step50000.checkpoint.pt"
        if os.path.exists(chk_path):
            chk = measure(chk_path, sample, external_binding=True)
            d = abs(chk["binding_bpb"] - chk["binding_bpb_ext"])
            print(f"[self-check] ON 50k binding: stashed={chk['binding_bpb']:.6f} "
                  f"external={chk['binding_bpb_ext']:.6f} |d|={d:.2e} "
                  f"{'PASS' if d < 1e-4 else 'FAIL — external instrument wrong, aborting'}")
            if d >= 1e-4:
                return

        rows = []
        for label, step, path, arm in ATTRIB:
            if not os.path.exists(path):
                print(f"  SKIP (missing): {os.path.basename(path)}")
                continue
            m = measure(path, sample, external_binding=True)
            m.update(label=label, step=step, arm=arm)
            rows.append(m)
            bb = f"{m['binding_bpb_ext']:.3f}" if m["binding_bpb_ext"] is not None else "  -  "
            lb = f"{m['local_bpb']:.3f}" if m["local_bpb"] is not None else "  -  "
            ref = learn.get(step)
            delta = ""
            if ref:
                delta = (f"  OFF-ON: full={m['full_bpb'] - ref['full_bpb']:+.3f}"
                         f" binding={m['binding_bpb_ext'] - ref['binding_bpb']:+.3f}"
                         f" local={m['local_bpb'] - ref['local_bpb']:+.3f}")
            print(f"  {label:11} step {step:6}  full={m['full_bpb']:.3f}  binding={bb}  local={lb}{delta}")
        json.dump(rows, open("docs/data/decepticon-attribution.json", "w"), indent=0)
        print("wrote docs/data/decepticon-attribution.json")
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
        # ON-minus-OFF full-bpb delta per step (the clean organ contribution).
        # Pair within a suite (s8 vs 50k are separate runs; step 10000 exists in both).
        for suite in sorted({r["label"].split("-", 1)[1] for r in rows}):
            srows = [r for r in rows if r["label"].split("-", 1)[1] == suite]
            for st in sorted({r["step"] for r in srows}):
                on = next((r for r in srows if r["arm"] == "adapt" and r["step"] == st), None)
                off = next((r for r in srows if r["arm"] == "noadapt" and r["step"] == st), None)
                if on and off:
                    print(f"  [{suite}] step {st:6}: organ delta (OFF - ON) full = {off['full_bpb'] - on['full_bpb']:+.3f} bpb")
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
