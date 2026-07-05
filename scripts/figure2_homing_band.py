#!/usr/bin/env python3
"""Figure 2 of paper/heinrich_method.tex — the readout commit tracks relative depth.

For each model, the answer token's mean rank among the 16-token background panel
(0 = out-reads all of them) as a function of RELATIVE depth (layer / n_layers).
All three models collapse to rank 0 near three-quarters depth, despite different
layer counts and training (base / instruct / larger). Depth fraction, not
absolute layer, schedules the readout commit.

Reproduce: python scripts/figure2_homing_band.py  -> paper/figures/homing_band.png
Data: docs/data/homing-study-v3-corrected.json + docs/data/homing-study-v4-*.json
      (produced by scripts/homing_study_v4.py).
"""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCES = [
    ("smollm2-135m (base)", "docs/data/homing-study-v3-corrected.json", "#1f77b4"),
    ("smollm2-135m-instruct", "docs/data/homing-study-v4-smollm2-135m-instruct.json", "#d62728"),
    ("smollm2-360m (base)", "docs/data/homing-study-v4-smollm2-360m.json", "#2ca02c"),
]
OUT = "paper/figures/homing_band.png"


def mean_rank_curve(runs: list) -> list:
    curves = [r["rank_curve"] for r in runs if r.get("rank_curve")]
    if not curves:
        return []
    nL = min(len(c) for c in curves)
    return [sum(c[i] for c in curves) / len(curves) for i in range(nL)]


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, path, color in SOURCES:
        d = json.load(open(path))
        runs = d["runs"]
        mc = mean_rank_curve(runs)
        nL = len(mc)
        x = [i / (nL - 1) for i in range(nL)]  # relative depth 0..1
        ax.plot(x, mc, "-o", ms=3, lw=1.6, color=color,
                label=f"{label}  ({nL} layers)")

    ax.axhline(8, ls=":", lw=1.0, color="#888",
               label="chance (8 of 16)")
    ax.axvline(0.75, ls="--", lw=1.0, color="#444")
    ax.text(0.745, 3.0, "commit band ~¾ depth ", rotation=90,
            fontsize=8, color="#444", va="center", ha="right")
    ax.set_xlabel("relative depth  (layer / n_layers)")
    ax.set_ylabel("answer's mean rank among 16 background")
    ax.set_title("Readout commit tracks relative depth, not absolute layer")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, None)
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
