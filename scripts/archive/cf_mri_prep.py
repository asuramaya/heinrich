#!/usr/bin/env python3
"""Prep a real heinrich .mri for the Cloudflare worker.

Real `mri-decompose` writes tokens.npz (numpy zip); the worker wants a plain
tokens.json sidecar so it never parses numpy. This extracts it and rebuilds the
root models.json manifest. Run after `heinrich mri-decompose`.

  python3 scripts/cf_mri_prep.py --out web/.data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def write_tokens_sidecar(mri_dir: Path) -> bool:
    npz = mri_dir / "tokens.npz"
    decomp = mri_dir / "decomp"
    if not npz.exists() or not decomp.exists():
        return False
    tok = dict(np.load(str(npz), allow_pickle=True))
    out = {
        "token_ids": [int(x) for x in tok["token_ids"]] if "token_ids" in tok else [],
        "scripts": [str(s) for s in tok["scripts"]] if "scripts" in tok else [],
        "token_texts": [str(t) for t in tok["token_texts"]] if "token_texts" in tok else [],
    }
    (decomp / "tokens.json").write_text(json.dumps(out))
    return True


def write_norm_sidecars(mri_dir: Path) -> None:
    """norms.npz/baselines.npz → worker-native JSON sidecars (stats per key)."""
    np_ = np
    for fname, jname, want_norm in [("norms.npz", "norms.json", False),
                                     ("baselines.npz", "baselines.json", True)]:
        p = mri_dir / fname
        if not p.exists():
            continue
        z = np_.load(str(p))
        out = {}
        for k in z.files:
            a = np_.asarray(z[k], dtype=np_.float32).ravel()
            rec = {"shape": list(z[k].shape),
                   "mean": float(a.mean()) if a.size else 0.0,
                   "std": float(a.std()) if a.size else 0.0}
            if want_norm:
                rec["norm"] = float(np_.linalg.norm(a))
            else:
                rec["min"] = float(a.min()) if a.size else 0.0
                rec["max"] = float(a.max()) if a.size else 0.0
            out[k] = rec
        (mri_dir / jname).write_text(json.dumps(out))


def rebuild_manifest(out: Path):
    models = []
    for meta_path in sorted(out.glob("*/*.mri/metadata.json")):
        meta = json.loads(meta_path.read_text())
        mri_dir = meta_path.parent
        cap = meta.get("capture", {})
        mdl = meta.get("model", {})
        models.append({
            "model": mri_dir.parent.name,
            "mode": cap.get("mode", "?"),
            "n_layers": mdl.get("n_layers", 0),
            "n_tokens": cap.get("n_tokens", cap.get("n_index", 0)),
            "version": meta.get("version", "?"),
            "architecture": meta.get("architecture", "transformer"),
        })
    models.sort(key=lambda m: (0 if m["architecture"] == "transformer" else 1, m["model"]))
    (out / "models.json").write_text(json.dumps(models, indent=2))
    return models


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="web/.data")
    a = ap.parse_args()
    out = Path(a.out)
    for meta_path in sorted(out.glob("*/*.mri/metadata.json")):
        mri = meta_path.parent
        ok = write_tokens_sidecar(mri)
        write_norm_sidecars(mri)
        print(f"{'sidecar' if ok else 'no-tokens'}: {mri}")
    models = rebuild_manifest(out)
    print(f"manifest: {out/'models.json'} ({len(models)} mri)")
    for m in models:
        print(f"  {m['model']}/{m['mode']}  L={m['n_layers']} tok={m['n_tokens']}")


if __name__ == "__main__":
    main()
