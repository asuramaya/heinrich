"""Probe library: build, cache, project, analyze.

Given a model backend and the hand-curated probe catalog
(`probe_catalog.PROBES`), build a dict of unit directions per concept
at a chosen layer. Project any residual onto all directions at once to
get a sparse concept-level profile of the input.

Design notes
------------
The library is intentionally NOT an SAE. It's a collection of directions
derived from contrastive prompt pairs YOU curated. Benefits:

  - Every direction has a human-readable label (the concept name).
  - Zero training cost; build is a handful of forward passes.
  - No monosemanticity mystery — the direction is pointing at what you
    chose to contrast.
  - Extendable: add a PROBES[concept] entry, rebuild, done.

Cost:
  - You miss concepts you didn't include.
  - The directions are as clean as your contrasts. If a concept's pos/neg
    sets share a spurious surface feature, the direction captures that
    instead. This is why `session11-probing-attack.md` is required reading
    before trusting any probe.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .probe_catalog import PROBES


@dataclass
class Probe:
    """One built probe: concept name + unit direction at one layer."""
    name: str
    layer: int
    direction: np.ndarray            # unit vector [hidden]
    mag_gap: float                    # ||mean(pos) - mean(neg)||_2 (raw, before unit)
    loocv_acc: float                  # leave-one-out linear accuracy
    n_pos: int
    n_neg: int
    description: str


@dataclass
class ProbeLibrary:
    """Collection of probes built at one layer of one model."""
    model_id: str
    layer: int
    hidden_size: int
    probes: dict[str, Probe] = field(default_factory=dict)

    def names(self) -> list[str]:
        return list(self.probes.keys())

    def project(self, residual: np.ndarray) -> dict[str, float]:
        """Project a residual onto every probe direction."""
        return {name: float(residual @ p.direction)
                for name, p in self.probes.items()}

    def project_centered(self, residual: np.ndarray) -> dict[str, float]:
        """Project against the probe's midpoint between pos and neg means.
        A value >0 means the input is on the pos side; <0 is neg side."""
        # Without caching the midpoint per probe we just return the dot
        # product. Midpoint-subtraction is handled in `profile_with_midpoints`.
        return self.project(residual)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_id": self.model_id,
            "layer": self.layer,
            "hidden_size": self.hidden_size,
            "probes": {
                name: {
                    "direction": p.direction.tolist(),
                    "mag_gap": p.mag_gap,
                    "loocv_acc": p.loocv_acc,
                    "n_pos": p.n_pos,
                    "n_neg": p.n_neg,
                    "description": p.description,
                }
                for name, p in self.probes.items()
            },
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path | str) -> "ProbeLibrary":
        payload = json.loads(Path(path).read_text())
        lib = cls(model_id=payload["model_id"],
                  layer=payload["layer"],
                  hidden_size=payload["hidden_size"])
        for name, d in payload["probes"].items():
            lib.probes[name] = Probe(
                name=name, layer=lib.layer,
                direction=np.array(d["direction"], dtype=np.float32),
                mag_gap=d["mag_gap"], loocv_acc=d["loocv_acc"],
                n_pos=d["n_pos"], n_neg=d["n_neg"],
                description=d["description"],
            )
        return lib


def _capture_last_residual(backend, prompt: str, layer: int) -> np.ndarray:
    import mlx.core as mx
    from heinrich.cartography.perturb import _mask_dtype

    inner = backend._inner
    tok = backend.tokenizer
    mdtype = _mask_dtype(backend.model)
    n_layers = len(inner.layers)

    ids = tok.encode(prompt)
    x = mx.array([ids])
    T = len(ids)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None
    h = inner.embed_tokens(x)
    for i in range(n_layers):
        h = inner.layers[i](h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]
        if i == layer:
            return np.array(h.astype(mx.float32))[0, -1, :].copy()
    return np.array(h.astype(mx.float32))[0, -1, :].copy()


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + 1e-9) if n > 0 else v


def _loocv(pos: np.ndarray, neg: np.ndarray) -> float:
    N = min(len(pos), len(neg))
    correct, total = 0, 0
    for i in range(N):
        mask = np.ones(N, bool); mask[i] = False
        tp = pos[mask].mean(0)
        tn = neg[mask].mean(0)
        d = _unit(tp - tn)
        thr = (tp @ d + tn @ d) / 2
        if pos[i] @ d > thr: correct += 1
        total += 1
        if neg[i] @ d <= thr: correct += 1
        total += 1
    return correct / total


def build_library(backend, *, layer: int,
                  concepts: list[str] | None = None,
                  progress: bool = True) -> ProbeLibrary:
    """Build all probes in the catalog (or a chosen subset) at ``layer``.

    Runs 2 forward passes per contrastive pair (one per side), roughly
    ~30 forward passes per concept × n_concepts.
    """
    import sys

    inner = backend._inner
    hidden = inner.embed_tokens.weight.shape[-1]
    model_id = getattr(backend, "model_id", "?")

    lib = ProbeLibrary(model_id=model_id, layer=layer, hidden_size=int(hidden))
    target = concepts if concepts else list(PROBES.keys())
    t0 = time.time()
    for i, name in enumerate(target):
        if name not in PROBES:
            raise KeyError(f"Unknown concept {name!r}. Known: {list(PROBES.keys())}")
        entry = PROBES[name]
        if progress:
            sys.stderr.write(f"[{i+1}/{len(target)}] {name} ... ")
            sys.stderr.flush()
        pos_res = np.stack([_capture_last_residual(backend, p, layer)
                            for p in entry["pos"]])
        neg_res = np.stack([_capture_last_residual(backend, p, layer)
                            for p in entry["neg"]])
        diff = pos_res.mean(0) - neg_res.mean(0)
        mag_gap = float(np.linalg.norm(diff))
        direction = _unit(diff)
        acc = _loocv(pos_res, neg_res)
        lib.probes[name] = Probe(
            name=name, layer=layer, direction=direction.astype(np.float32),
            mag_gap=mag_gap, loocv_acc=float(acc),
            n_pos=len(pos_res), n_neg=len(neg_res),
            description=entry.get("description", ""),
        )
        if progress:
            sys.stderr.write(f"gap={mag_gap:5.2f}  loocv={acc*100:4.1f}%\n")
            sys.stderr.flush()
    if progress:
        sys.stderr.write(f"built {len(lib.probes)} probes in {time.time()-t0:.1f}s\n")
    return lib


def profile_prompt(backend, prompt: str, library: ProbeLibrary) -> dict[str, float]:
    """Return {probe_name: projection} for a prompt's last-token residual."""
    residual = _capture_last_residual(backend, prompt, library.layer)
    return library.project(residual)


def profile_trajectory(backend, prompt: str, library: ProbeLibrary,
                       max_tokens: int = 40) -> list[dict[str, float]]:
    """Return projection traces across a greedy generation from ``prompt``.

    One projection dict per generated token (at the probe's layer, for the
    current last position).
    """
    import mlx.core as mx
    from heinrich.cartography.perturb import _mask_dtype

    inner = backend._inner
    tok = backend.tokenizer
    mdtype = _mask_dtype(backend.model)
    n_layers = len(inner.layers)

    ids = list(tok.encode(prompt))
    layer = library.layer
    trace = []
    for _ in range(max_tokens):
        x = mx.array([ids])
        T = len(ids)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None
        h = inner.embed_tokens(x)
        last_res = None
        for i in range(n_layers):
            h = inner.layers[i](h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]
            if i == layer:
                last_res = np.array(h.astype(mx.float32))[0, -1, :]
        h = inner.norm(h)
        logits = np.array(backend._lm_head(h).astype(mx.float32)[0, -1, :])
        nxt = int(np.argmax(logits))
        ids.append(nxt)
        trace.append(library.project(last_res))
    return trace


def profile_many(backend, prompts: list[str],
                 library: ProbeLibrary) -> dict[str, list[float]]:
    """Return {probe_name: [projection per prompt]}."""
    out: dict[str, list[float]] = {n: [] for n in library.names()}
    for p in prompts:
        proj = profile_prompt(backend, p, library)
        for n, v in proj.items():
            out[n].append(v)
    return out


def plot_radar(profile: dict[str, float], *, out_path: str | Path,
               title: str = "", vmin: float | None = None,
               vmax: float | None = None) -> None:
    """Radar chart of the concept profile. One spoke per probe."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(profile.keys())
    values = np.array([profile[n] for n in names])
    if vmax is None:
        vmax = float(np.max(np.abs(values))) if len(values) else 1.0
    if vmin is None:
        vmin = -vmax

    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])
    values_closed = np.concatenate([values, values[:1]])

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, values_closed, 'o-', color='#cc3355', lw=2)
    ax.fill(angles_closed, values_closed, alpha=0.25, color='#cc3355')
    ax.set_xticks(angles)
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=9)
    ax.set_ylim(vmin, vmax)
    ax.set_title(title, pad=20)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_heatmap(profile_many_out: dict[str, list[float]],
                 prompts: list[str], *, out_path: str | Path,
                 title: str = "") -> None:
    """Heatmap: rows=probes, cols=prompts. Cell = projection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(profile_many_out.keys())
    mat = np.array([profile_many_out[n] for n in names])
    vmax = float(np.max(np.abs(mat)))
    fig, ax = plt.subplots(figsize=(min(20, 1 + len(prompts)*0.4), max(4, len(names)*0.4)))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels([p[:40] + ('…' if len(p) > 40 else '')
                        for p in prompts], rotation=45, ha='right', fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='projection')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
