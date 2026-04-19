"""Serve-path artifacts for the companion viewer.

The analysis path keeps exact raw/decomp outputs. This module builds and reads
query-shaped serving artifacts under ``<mri>/serve`` so the runtime can answer
UI requests without hydrating the full all-scores blob in the browser.
"""

from __future__ import annotations

import json
import shutil
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


_PC_MAGIC = b"PCSC"
_TOK_MAGIC = b"TOKS"
_SERVE_VERSION = 1
_FULL_PC_NAME = "pc_scores_full.bin"
_FULL_TOKEN_NAME = "token_scores_full.bin"
_SERVE_META_NAME = "meta.json"


def _read_pc_header(path: Path) -> tuple[int, int, int]:
    with open(path, "rb") as f:
        hdr = f.read(16)
    magic, n_layers, n_tok, full_k = struct.unpack("<4sIII", hdr)
    if magic != _PC_MAGIC:
        raise ValueError(f"Bad PC index magic at {path}: {magic!r}")
    return n_layers, n_tok, full_k


def _read_token_header(path: Path) -> tuple[int, int, int]:
    with open(path, "rb") as f:
        hdr = f.read(16)
    magic, n_tok, n_layers, full_k = struct.unpack("<4sIII", hdr)
    if magic != _TOK_MAGIC:
        raise ValueError(f"Bad token index magic at {path}: {magic!r}")
    return n_tok, n_layers, full_k


def _variance_path(decomp_dir: Path, layer_id: int | str) -> Path:
    if isinstance(layer_id, int):
        return decomp_dir / f"L{layer_id:02d}_variance.npy"
    return decomp_dir / f"{layer_id}_variance.npy"


def _load_pc_variances(mri_dir: Path, meta: dict[str, Any]) -> list[list[float]]:
    decomp_dir = mri_dir / "decomp"
    out: list[list[float]] = []
    for layer_meta in meta.get("layers", []):
        vp = _variance_path(decomp_dir, layer_meta.get("layer"))
        if vp.exists():
            out.append(np.load(str(vp)).astype(np.float32).tolist())
        else:
            out.append([])
    return out


def _write_step_index(full_pc_src: Path, out_path: Path, step: int) -> int:
    n_layers, n_tok, full_k = _read_pc_header(full_pc_src)
    n_sample = (n_tok + step - 1) // step
    stride = n_layers * n_tok * 2
    with open(full_pc_src, "rb") as src, open(out_path, "wb") as dst:
        dst.write(struct.pack("<4sIII", _PC_MAGIC, n_layers, n_sample, full_k))
        for pc in range(full_k):
            src.seek(16 + pc * stride)
            slab = np.frombuffer(src.read(stride), dtype=np.float16).reshape(n_layers, n_tok)
            dst.write(np.ascontiguousarray(slab[:, ::step]).tobytes())
    return n_sample


def build_serve_artifacts(
    mri_path: str,
    *,
    steps: tuple[int, ...] = (10, 25, 50),
    force: bool = False,
) -> dict[str, Any]:
    """Build companion serve artifacts from an existing MRI decomposition."""
    mri_dir = Path(mri_path)
    decomp_dir = mri_dir / "decomp"
    meta_path = decomp_dir / "meta.json"
    full_pc_src = decomp_dir / "pc_scores.bin"
    full_tok_src = decomp_dir / "token_scores.bin"

    if not meta_path.exists():
        return {"error": f"No decomposition metadata at {meta_path}. Run mri-decompose first."}
    if not full_pc_src.exists():
        return {"error": f"No PC-major score index at {full_pc_src}. Run mri-decompose first."}
    if not full_tok_src.exists():
        return {"error": f"No token-major score index at {full_tok_src}. Run mri-decompose first."}

    meta = json.loads(meta_path.read_text())
    n_layers_pc, n_tok_pc, full_k_pc = _read_pc_header(full_pc_src)
    n_tok_tok, n_layers_tok, full_k_tok = _read_token_header(full_tok_src)
    if (n_layers_pc, n_tok_pc, full_k_pc) != (n_layers_tok, n_tok_tok, full_k_tok):
        return {
            "error": "PC/token score indexes disagree on shape",
            "pc_index": [n_layers_pc, n_tok_pc, full_k_pc],
            "token_index": [n_tok_tok, n_layers_tok, full_k_tok],
        }

    serve_dir = mri_dir / "serve"
    serve_dir.mkdir(parents=True, exist_ok=True)

    full_pc_dst = serve_dir / _FULL_PC_NAME
    full_tok_dst = serve_dir / _FULL_TOKEN_NAME
    if force or not full_pc_dst.exists():
        shutil.copyfile(full_pc_src, full_pc_dst)
    if force or not full_tok_dst.exists():
        shutil.copyfile(full_tok_src, full_tok_dst)

    steps_meta: dict[str, dict[str, Any]] = {}
    for step in sorted({int(s) for s in steps if int(s) > 1}):
        out_path = serve_dir / f"pc_scores_step{step}.bin"
        if force or not out_path.exists():
            n_sample = _write_step_index(full_pc_src, out_path, step)
        else:
            _, n_sample, _ = _read_pc_header(out_path)
        steps_meta[str(step)] = {
            "pc_scores": out_path.name,
            "step": step,
            "n_sample": n_sample,
        }

    serve_meta = {
        "version": _SERVE_VERSION,
        "source": "serve",
        "mri_path": str(mri_dir),
        "n_layers": n_layers_pc,
        "n_real_layers": int(meta.get("n_real_layers", n_layers_pc)),
        "n_tokens": n_tok_pc,
        "full_k": full_k_pc,
        "pc_scores": full_pc_dst.name,
        "token_scores": full_tok_dst.name,
        "steps": steps_meta,
        "pc_vars": _load_pc_variances(mri_dir, meta),
    }
    (serve_dir / _SERVE_META_NAME).write_text(json.dumps(serve_meta, indent=2))
    load_serve_meta.cache_clear()
    return {
        "mri_path": str(mri_dir),
        "serve_dir": str(serve_dir),
        "n_layers": n_layers_pc,
        "n_tokens": n_tok_pc,
        "full_k": full_k_pc,
        "steps": steps_meta,
    }


def _synthesize_serve_meta(mri_path: str) -> dict[str, Any]:
    mri_dir = Path(mri_path)
    decomp_dir = mri_dir / "decomp"
    meta_path = decomp_dir / "meta.json"
    if not meta_path.exists():
        return {"error": "No decomposition. Run mri-decompose."}
    meta = json.loads(meta_path.read_text())
    full_pc_src = decomp_dir / "pc_scores.bin"
    full_tok_src = decomp_dir / "token_scores.bin"
    if full_pc_src.exists():
        n_layers_pc, n_tok_pc, full_k_pc = _read_pc_header(full_pc_src)
    else:
        n_layers_pc = int(meta.get("n_layers", 0))
        n_tok_pc = int(meta.get("n_sample", 0))
        full_k_pc = int(meta.get("n_components", 0))
    if full_tok_src.exists():
        n_tok_tok, n_layers_tok, full_k_tok = _read_token_header(full_tok_src)
        n_layers_pc = n_layers_tok
        n_tok_pc = n_tok_tok
        full_k_pc = max(full_k_pc, full_k_tok)
    return {
        "version": _SERVE_VERSION,
        "source": "decomp",
        "mri_path": str(mri_dir),
        "n_layers": n_layers_pc,
        "n_real_layers": int(meta.get("n_real_layers", n_layers_pc)),
        "n_tokens": n_tok_pc,
        "full_k": full_k_pc,
        "pc_scores": str(full_pc_src) if full_pc_src.exists() else None,
        "token_scores": str(full_tok_src) if full_tok_src.exists() else None,
        "steps": {},
        "pc_vars": _load_pc_variances(mri_dir, meta),
    }


@lru_cache(maxsize=64)
def load_serve_meta(mri_path: str) -> dict[str, Any]:
    serve_meta_path = Path(mri_path) / "serve" / _SERVE_META_NAME
    if serve_meta_path.exists():
        return json.loads(serve_meta_path.read_text())
    return _synthesize_serve_meta(mri_path)


def resolve_pc_index(mri_path: str, step: int | None = None) -> tuple[Path | None, int, int]:
    """Return (path, sampled_tokens, original_tokens) for a PC-major serve index."""
    meta = load_serve_meta(mri_path)
    if "error" in meta:
        return None, 0, 0
    n_tokens = int(meta.get("n_tokens", 0))
    serve_dir = Path(mri_path) / "serve"
    if step and step > 1:
        step_meta = meta.get("steps", {}).get(str(step))
        if step_meta:
            return serve_dir / step_meta["pc_scores"], int(step_meta["n_sample"]), n_tokens
    pc_name = meta.get("pc_scores")
    if meta.get("source") == "serve" and pc_name:
        return serve_dir / str(pc_name), n_tokens, n_tokens
    decomp_path = Path(mri_path) / "decomp" / "pc_scores.bin"
    return (decomp_path if decomp_path.exists() else None), n_tokens, n_tokens


def resolve_token_index(mri_path: str) -> Path | None:
    meta = load_serve_meta(mri_path)
    if "error" in meta:
        return None
    serve_dir = Path(mri_path) / "serve"
    tok_name = meta.get("token_scores")
    if meta.get("source") == "serve" and tok_name:
        return serve_dir / str(tok_name)
    decomp_path = Path(mri_path) / "decomp" / "token_scores.bin"
    return decomp_path if decomp_path.exists() else None
