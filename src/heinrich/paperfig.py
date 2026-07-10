"""Regenerate the paper's data-derived figures from committed goldens.

The Frozen Frame's standard is captured figures with TRUE PROVENANCE: every
figure either regenerates deterministically from a committed docs/data
golden, or is a live capture whose provenance is stated and verified — never
redrawn from memory (omit, never fake).

The registry maps each figure in paper/figures/ to its kind:
  data     — rebuilt here from docs/data JSONs (matplotlib, Agg)
  capture  — produced by a live instrument (companion capture); this
             command verifies the file exists and reports its provenance,
             it does NOT regenerate it.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
FIGURES = REPO / "paper" / "figures"


def _mean_rank_curve(runs: list) -> list:
    curves = [r["rank_curve"] for r in runs if r.get("rank_curve")]
    if not curves:
        return []
    n = min(len(c) for c in curves)
    return [sum(c[i] for c in curves) / len(curves) for i in range(n)]


def build_homing_band(out: Path) -> dict:
    """Figure 2: the readout commit tracks relative depth, not absolute
    layer — four models' mean answer-rank curves collapse at ~3/4 depth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = [
        ("smollm2-135m (base)",
         "docs/data/homing-study-v3-corrected.json", "#1f77b4"),
        ("smollm2-135m-instruct",
         "docs/data/homing-study-v4-smollm2-135m-instruct.json", "#d62728"),
        ("smollm2-360m (base)",
         "docs/data/homing-study-v4-smollm2-360m.json", "#2ca02c"),
        ("qwen2.5-0.5b-instruct",
         "docs/data/homing-study-v4-qwen2.5-0.5b-instruct.json", "#9467bd"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    used = []
    for label, rel, color in sources:
        path = REPO / rel
        d = json.loads(path.read_text())
        mc = _mean_rank_curve(d["runs"])
        n = len(mc)
        x = [i / (n - 1) for i in range(n)]
        ax.plot(x, mc, "-o", ms=3, lw=1.6, color=color,
                label=f"{label}  ({n} layers)")
        used.append(rel)
    ax.axhline(8, ls=":", lw=1.0, color="#888", label="chance (8 of 16)")
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
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return {"sources": used}


REGISTRY = {
    "homing_band.png": {
        "kind": "data",
        "paper": "heinrich_method.tex (Figure 2)",
        "builder": build_homing_band,
    },
    "horn_smollm2-135m.png": {
        "kind": "capture",
        "paper": "heinrich_method.tex (Figure 1)",
        "provenance": ("live companion capture (paper/figures/"
                       "capture_horn.mjs against the running companion, "
                       ":8377) — a measurement image, not a plot; "
                       "regenerating requires the live instrument"),
    },
}


def paper_figures(only: list[str] | None = None) -> dict:
    """Rebuild every data-derived figure; verify + report captures."""
    report = []
    for name, spec in REGISTRY.items():
        if only and name not in only:
            continue
        out = FIGURES / name
        entry = {"figure": name, "kind": spec["kind"],
                 "paper": spec["paper"]}
        if spec["kind"] == "data":
            info = spec["builder"](out)
            entry["status"] = "rebuilt"
            entry["sources"] = info["sources"]
        else:
            entry["status"] = "present" if out.exists() else "MISSING"
            entry["provenance"] = spec["provenance"]
        report.append(entry)
    return {"figures_dir": str(FIGURES), "report": report}
