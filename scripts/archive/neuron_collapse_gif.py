#!/usr/bin/env python3
"""GIF: the raw-mode neuron collapse for two tokens, through depth.

The neuron field (gate x up = the down_proj input z) for a single raw token collapses
from many active neurons in early layers to ~one dominant neuron by ~L11 -- the
crystal (Session 6). This animates it for two tokens, one frame per layer, with the
participation ratio (effective number of active neurons) so the collapse is measured,
not just seen.

Reproduce: python scripts/neuron_collapse_gif.py [OUT.gif]
"""
from __future__ import annotations
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "HuggingFaceTB/SmolLM2-135M"
TOKENS = [(" capital", "#e8873a"), (" Paris", "#4a90d9")]   # (text, colour)
OUT = sys.argv[1] if len(sys.argv) > 1 else "neuron_collapse.gif"

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float32).eval()
LAYERS = model.model.layers
INTER = LAYERS[0].mlp.down_proj.weight.shape[1]

z = {}
for l, layer in enumerate(LAYERS):
    layer.mlp.down_proj.register_forward_pre_hook(
        (lambda L: (lambda m, i: z.__setitem__(L, i[0][0, -1, :].detach().cpu().numpy())))(l))


def acts(text):
    z.clear()
    ids = tok(text, return_tensors="pt").input_ids       # SmolLM2 adds no BOS
    with torch.no_grad():
        model(ids)
    return np.stack([z[l] for l in range(len(LAYERS))])   # [layers, INTER]


Z = [acts(t) for t, _ in TOKENS]
n = len(LAYERS)


def pr(v):                                                # effective # active neurons
    e = v ** 2
    return float((e.sum() ** 2) / ((e ** 2).sum() + 1e-12))


PR = [np.array([pr(Z[k][l]) for l in range(n)]) for k in range(len(TOKENS))]
# which neuron each collapses onto, late
crystal = [int(np.abs(Z[k][-6:]).mean(0).argmax()) for k in range(len(TOKENS))]
print("effective active neurons by layer:")
for l in range(n):
    print(f"  L{l:02d}  " + "  ".join(f"{t.strip()}={PR[k][l]:6.1f}" for k, (t, _) in enumerate(TOKENS)))
print("crystal neuron (late-layer argmax):",
      {TOKENS[k][0].strip(): crystal[k] for k in range(len(TOKENS))})

fig, (ax, axp) = plt.subplots(2, 1, figsize=(8, 4.8), height_ratios=[3, 1])
fig.set_facecolor("#0a0a0a")
x = np.arange(INTER)


def frame(l):
    ax.clear(); ax.set_facecolor("#0a0a0a")
    for k, (t, c) in enumerate(TOKENS):
        v = np.abs(Z[k][l]); v = v / (v.max() + 1e-12)
        ax.plot(x, v if k == 0 else -v, color=c, lw=0.7, alpha=0.95)
        ax.text(0.01, 0.92 - k * 0.86, t.strip(), color=c, transform=ax.transAxes,
                fontsize=10, weight="bold")
    ax.set_ylim(-1.08, 1.08); ax.set_xlim(0, INTER)
    ax.set_title(f"SmolLM2-135M  raw  L{l:02d}   effective active neurons:  "
                 + "   ".join(f"{t.strip()} {PR[k][l]:.0f}" for k, (t, _) in enumerate(TOKENS)),
                 color="#ddd", fontsize=10)
    ax.set_yticks([]); ax.set_xlabel("neuron (0..%d)" % INTER, color="#888", fontsize=8)
    ax.tick_params(colors="#888")
    for s in ax.spines.values():
        s.set_color("#333")

    axp.clear(); axp.set_facecolor("#0a0a0a")
    for k, (t, c) in enumerate(TOKENS):
        axp.semilogy(range(n), PR[k], color=c, lw=1.2, alpha=0.9)
        axp.scatter([l], [PR[k][l]], color=c, s=30, zorder=3)
    axp.axvline(l, color="#555", lw=0.6)
    axp.set_xlim(0, n - 1); axp.set_ylabel("eff. #\nneurons", color="#888", fontsize=8)
    axp.set_xlabel("layer", color="#888", fontsize=8)
    axp.tick_params(colors="#888", labelsize=7)
    for s in axp.spines.values():
        s.set_color("#333")
    fig.tight_layout()


FuncAnimation(fig, frame, frames=n, interval=300).save(OUT, writer=PillowWriter(fps=4))
print("wrote", OUT)
