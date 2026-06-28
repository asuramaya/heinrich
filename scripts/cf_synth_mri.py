#!/usr/bin/env python3
"""Synthetic MRI generator for the Cloudflare/WebGPU companion prototype.

Pure stdlib — no numpy, no model, no installs. Emits a small, structurally
valid `.mri/decomp/` directory that matches what `heinrich mri-decompose`
writes (HEI2/TOKS/PCSC binary indexes + per-layer .npy + tokens.npz), so the
companion SPA can render it through the Worker without any real capture.

Scores are cluster-structured (one cluster per script) with per-layer drift, so
the cloud and trajectory viewports show real geometry instead of noise.

Usage:
  python3 scripts/cf_synth_mri.py --out web/.data --model synth-mini --mode raw \
      --layers 6 --hidden 64 --tokens 2000 --k 24
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import zipfile
from pathlib import Path

# --- minimal .npy / .npz writers (NPY format v1.0) ------------------------

def _shape_repr(shape: tuple[int, ...]) -> str:
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(s) for s in shape) + ")"


def npy_bytes(payload: bytes, descr: str, shape: tuple[int, ...]) -> bytes:
    """Wrap raw little-endian payload bytes in an NPY v1.0 container."""
    hdr = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (
        descr, _shape_repr(shape))
    magic = b"\x93NUMPY\x01\x00"
    # header length must make total (magic + 2 + header) a multiple of 64
    base = len(magic) + 2
    pad = (64 - (base + len(hdr) + 1) % 64) % 64
    hdr = hdr + " " * pad + "\n"
    return magic + struct.pack("<H", len(hdr)) + hdr.encode("latin-1") + payload


def f16(vals) -> bytes:
    vals = list(vals)
    return struct.pack(f"<{len(vals)}e", *vals)


def f32(vals) -> bytes:
    vals = list(vals)
    return struct.pack(f"<{len(vals)}f", *vals)


def i32(vals) -> bytes:
    vals = list(vals)
    return struct.pack(f"<{len(vals)}i", *vals)


def write_npy(path: Path, payload: bytes, descr: str, shape):
    path.write_bytes(npy_bytes(payload, descr, shape))


def write_str_npy(path: Path, strings: list[str]) -> bytes:
    """Unicode array '<U{w}' — numpy stores UCS4 (4 bytes/char), little-endian."""
    w = max((len(s) for s in strings), default=1)
    buf = bytearray()
    for s in strings:
        for ch in s.ljust(w, "\x00"):
            buf += struct.pack("<I", ord(ch))
    payload = bytes(buf)
    descr = f"<U{w}"
    if path is not None:
        write_npy(path, payload, descr, (len(strings),))
    return npy_bytes(payload, descr, (len(strings),))


# --- synthetic geometry ---------------------------------------------------

SCRIPTS = ["Latin", "CJK", "Cyrillic", "Arabic", "code", "digits", "Greek", "Hebrew"]


def generate(out: Path, model: str, mode: str, n_layers: int, hidden: int,
             n_tokens: int, K: int, intermediate: int, seed: int) -> Path:
    random.seed(seed)
    n_scripts = min(len(SCRIPTS), max(3, n_tokens // 300 + 3))
    scripts = SCRIPTS[:n_scripts]

    # per-token assignment
    tok_script = [random.randrange(n_scripts) for _ in range(n_tokens)]
    tok_script_name = [scripts[c] for c in tok_script]
    tok_texts = [f"{scripts[tok_script[i]][:2]}_{i}" for i in range(n_tokens)]
    tok_ids = list(range(n_tokens))

    # cluster centers in K-space: each script gets a distinct direction in the
    # first ~6 PCs; remaining PCs are shared noise.
    centers = []
    for c in range(n_scripts):
        ctr = [0.0] * K
        for k in range(min(6, K)):
            ctr[k] = math.cos(2 * math.pi * (c / n_scripts) + k * 1.3) * (3.0 - k * 0.35)
        centers.append(ctr)

    mri_dir = out / model / f"{mode}.mri"
    decomp = mri_dir / "decomp"
    decomp.mkdir(parents=True, exist_ok=True)

    total_layers = n_layers  # no virtual layers in the synthetic set

    # per-layer score matrices [N x K], with drift across depth
    layer_scores: list[list[float]] = []  # flat row-major per layer
    for li in range(total_layers):
        drift = li / max(1, total_layers - 1)  # 0..1
        spread = 0.6 + 0.8 * (1 - abs(drift - 0.5) * 2)  # widest mid-stack
        rows = []
        for i in range(n_tokens):
            ctr = centers[tok_script[i]]
            for k in range(K):
                # rotate clusters slowly with depth + per-PC decaying noise
                rot = math.sin(drift * math.pi + k * 0.2)
                base = ctr[k] * (0.4 + 0.6 * drift) * (1 + 0.15 * rot)
                noise = random.gauss(0, spread * math.exp(-k / 6.0) + 0.05)
                rows.append(base + noise)
        layer_scores.append(rows)
        write_npy(decomp / f"L{li:02d}_scores.npy", f16(rows), "<f2", (n_tokens, K))
        # variance: decaying spectrum, normalized to sum 1
        raw = [math.exp(-k / 5.0) for k in range(K)]
        s = sum(raw)
        var = [r / s for r in raw]
        write_npy(decomp / f"L{li:02d}_variance.npy", f32(var), "<f4", (K,))
        # components [K x hidden], random unit-ish rows
        comp = []
        for k in range(K):
            row = [random.gauss(0, 1) for _ in range(hidden)]
            nrm = math.sqrt(sum(x * x for x in row)) or 1.0
            comp.extend(x / nrm for x in row)
        write_npy(decomp / f"L{li:02d}_components.npy", f32(comp), "<f4", (K, hidden))

    BIN_K = min(K, 50)
    VAR_K = K

    # all_scores.bin : [HEI2][tl][N][score_k][var_k][f32 var: tl*var_k][f16 scores: tl*N*score_k]
    with open(decomp / "all_scores.bin", "wb") as f:
        f.write(struct.pack("<4sIIII", b"HEI2", total_layers, n_tokens, BIN_K, VAR_K))
        for li in range(total_layers):
            raw = [math.exp(-k / 5.0) for k in range(VAR_K)]
            s = sum(raw)
            f.write(f32([r / s for r in raw]))
        for li in range(total_layers):
            rows = layer_scores[li]
            if BIN_K == K:
                f.write(f16(rows))
            else:
                trimmed = []
                for i in range(n_tokens):
                    trimmed.extend(rows[i * K:i * K + BIN_K])
                f.write(f16(trimmed))

    # pc_scores.bin : [PCSC][tl][N][K] then [K x tl x N] f16
    with open(decomp / "pc_scores.bin", "wb") as f:
        f.write(struct.pack("<4sIII", b"PCSC", total_layers, n_tokens, K))
        for pc in range(K):
            for li in range(total_layers):
                rows = layer_scores[li]
                f.write(f16(rows[pc::K]))  # column pc for all tokens, this layer

    # token_scores.bin : [TOKS][N][tl][K] then [N x tl x K] f16
    with open(decomp / "token_scores.bin", "wb") as f:
        f.write(struct.pack("<4sIII", b"TOKS", n_tokens, total_layers, K))
        for i in range(n_tokens):
            for li in range(total_layers):
                rows = layer_scores[li]
                f.write(f16(rows[i * K:i * K + K]))

    # decomp/meta.json
    (decomp / "meta.json").write_text(json.dumps({
        "n_layers": total_layers, "n_real_layers": n_layers, "n_sample": n_tokens,
        "n_tokens": n_tokens, "K": K, "score_k": BIN_K, "var_k": VAR_K,
        "hidden_size": hidden, "intermediate_size": intermediate, "mode": mode,
        "synthetic": True,
    }, indent=2))

    # tokens.npz : token_ids (i32), scripts (<U), token_texts (<U)
    with zipfile.ZipFile(mri_dir / "tokens.npz", "w", zipfile.ZIP_STORED) as z:
        z.writestr("token_ids.npy", npy_bytes(i32(tok_ids), "<i4", (n_tokens,)))
        z.writestr("scripts.npy", write_str_npy(None, tok_script_name))
        z.writestr("token_texts.npy", write_str_npy(None, tok_texts))

    # worker-native sidecar: token metadata as plain JSON (no npz parsing in JS)
    (decomp / "tokens.json").write_text(json.dumps({
        "token_ids": tok_ids, "scripts": tok_script_name, "token_texts": tok_texts,
    }))

    # .mri/metadata.json
    (mri_dir / "metadata.json").write_text(json.dumps({
        "architecture": "transformer", "version": "0.4-synth",
        "capture": {"mode": mode, "n_tokens": n_tokens, "intermediate_size": intermediate},
        "model": {"name": model, "n_layers": n_layers, "hidden_size": hidden},
    }, indent=2))

    return mri_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="web/.data")
    ap.add_argument("--model", default="synth-mini")
    ap.add_argument("--mode", default="raw")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=2000)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--intermediate", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)
    mri = generate(out, a.model, a.mode, a.layers, a.hidden,
                   a.tokens, a.k, a.intermediate, a.seed)
    total = sum(f.stat().st_size for f in mri.rglob("*") if f.is_file())
    print(f"wrote {mri}")
    print(f"  {sum(1 for _ in mri.rglob('*') if _.is_file())} files, "
          f"{total/1024:.0f} KB")
    rebuild_manifest(out)


def rebuild_manifest(out: Path):
    """Scan <out> for *.mri dirs and write models.json (R2 has no dir scan)."""
    models = []
    for meta_path in sorted(out.glob("*/*.mri/metadata.json")):
        meta = json.loads(meta_path.read_text())
        mri_dir = meta_path.parent
        models.append({
            "model": mri_dir.parent.name,
            "mode": meta.get("capture", {}).get("mode", "?"),
            "n_layers": meta.get("model", {}).get("n_layers", 0),
            "n_tokens": meta.get("capture", {}).get("n_tokens", 0),
            "version": meta.get("version", "?"),
            "architecture": meta.get("architecture", "transformer"),
        })
    models.sort(key=lambda m: (0 if m["architecture"] == "transformer" else 1, m["model"]))
    (out / "models.json").write_text(json.dumps(models, indent=2))
    print(f"manifest: {out/'models.json'} ({len(models)} mri)")


if __name__ == "__main__":
    main()
