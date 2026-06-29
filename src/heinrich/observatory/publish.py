"""Publish a decomposed .mri to an R2 bucket (or a local export dir).

The producer half writes a multi-GB .mri (raw weights, activations, attention).
The consumer (the Observatory viewer) reads only a few MB of decomposition
blobs + JSON sidecars. `publish` computes the sidecars, selects the lean
consumer subset, and uploads it — never the raw weights.

R2 speaks the S3 API, so this is pure Python (boto3); no Node/wrangler needed.
Credentials from env: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
(or pass account_id / endpoint_url explicitly).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

# decomp/ files the viewer actually reads (see web/ARTIFACT_FORMAT.md).
# Excludes *_variance.npy, *_components.npy, summaries — not consumer-served.
_DECOMP_EXACT = {
    "meta.json", "tokens.json",
    "all_scores.bin", "pc_scores.bin", "token_scores.bin", "token_neurons.bin",
    "gate_heatmap.npy", "weight_alignment.json", "neuron_importance.json",
}


def _model_mode(mri_dir: Path) -> tuple[str, str]:
    meta = json.loads((mri_dir / "metadata.json").read_text())
    mode = meta.get("capture", {}).get("mode", mri_dir.name.replace(".mri", "") or "raw")
    return mri_dir.parent.name, mode


def _ensure_intermediate(decomp: Path) -> None:
    """Back-fill intermediate_size into decomp/meta.json from the TOKN header.

    Decompositions made before the producer wrote intermediate_size lack it, so
    the Neurons viewport stays dark. Read it from token_neurons.bin (TOKN header,
    offset 12) and patch meta.json — keeps the artifact self-describing.
    """
    import struct
    meta_p = decomp / "meta.json"
    tn = decomp / "token_neurons.bin"
    if not (meta_p.exists() and tn.exists()):
        return
    meta = json.loads(meta_p.read_text())
    if meta.get("intermediate_size"):
        return
    with open(tn, "rb") as f:
        magic, _n_tok, _n_layers, inter = struct.unpack("<4sIII", f.read(16))
    if magic == b"TOKN" and inter:
        meta["intermediate_size"] = int(inter)
        meta_p.write_text(json.dumps(meta, indent=2))


def write_sidecars(mri_dir: Path) -> list[str]:
    """tokens.npz/norms.npz/baselines.npz → worker-native JSON sidecars.

    Idempotent; returns the sidecar filenames written.
    """
    import numpy as np
    written: list[str] = []
    decomp = mri_dir / "decomp"
    if decomp.exists():
        _ensure_intermediate(decomp)

    npz = mri_dir / "tokens.npz"
    if npz.exists() and decomp.exists():
        tok = dict(np.load(str(npz), allow_pickle=True))
        (decomp / "tokens.json").write_text(json.dumps({
            "token_ids": [int(x) for x in tok.get("token_ids", [])],
            "scripts": [str(s) for s in tok.get("scripts", [])],
            "token_texts": [str(t) for t in tok.get("token_texts", [])],
        }))
        written.append("decomp/tokens.json")

    for fname, jname, want_norm in [("norms.npz", "norms.json", False),
                                    ("baselines.npz", "baselines.json", True)]:
        p = mri_dir / fname
        if not p.exists():
            continue
        z = np.load(str(p))
        out = {}
        for k in z.files:
            a = np.asarray(z[k], dtype=np.float32).ravel()
            rec = {"shape": list(z[k].shape),
                   "mean": float(a.mean()) if a.size else 0.0,
                   "std": float(a.std()) if a.size else 0.0}
            if want_norm:
                rec["norm"] = float(np.linalg.norm(a))
            else:
                rec["min"] = float(a.min()) if a.size else 0.0
                rec["max"] = float(a.max()) if a.size else 0.0
            out[k] = rec
        (mri_dir / jname).write_text(json.dumps(out))
        written.append(jname)
    return written


def consumer_files(mri_dir: Path, with_hover: bool = False) -> Iterator[tuple[Path, str]]:
    """Yield (local_path, key_suffix) for the lean consumer subset.

    key_suffix is relative to the model prefix (<model>/<mode>.mri/...).
    """
    if (mri_dir / "metadata.json").exists():
        yield mri_dir / "metadata.json", "metadata.json"
    for side in ("norms.json", "baselines.json"):
        if (mri_dir / side).exists():
            yield mri_dir / side, side
    decomp = mri_dir / "decomp"
    if decomp.exists():
        for f in sorted(decomp.iterdir()):
            if not f.is_file():
                continue
            if f.name in _DECOMP_EXACT or f.name.endswith("_scores.npy"):
                yield f, f"decomp/{f.name}"
    if with_hover and (mri_dir / "mlp").exists():
        for f in sorted((mri_dir / "mlp").glob("L*_gate.npy")):
            yield f, f"mlp/{f.name}"
        for f in sorted((mri_dir / "mlp").glob("L*_up.npy")):
            yield f, f"mlp/{f.name}"


def manifest_entry(mri_dir: Path) -> dict:
    meta = json.loads((mri_dir / "metadata.json").read_text())
    model, mode = _model_mode(mri_dir)
    cap, mdl = meta.get("capture", {}), meta.get("model", {})
    return {
        "model": model, "mode": mode,
        "n_layers": mdl.get("n_layers", 0),
        "n_tokens": cap.get("n_tokens", cap.get("n_index", 0)),
        "version": meta.get("version", "?"),
        "architecture": meta.get("architecture", "transformer"),
    }


def _r2_client(account_id: str | None, endpoint_url: str | None,
               access_key: str | None, secret_key: str | None):
    import boto3
    account_id = account_id or os.environ.get("R2_ACCOUNT_ID")
    endpoint_url = endpoint_url or os.environ.get("R2_ENDPOINT") or (
        f"https://{account_id}.r2.cloudflarestorage.com" if account_id else None)
    if not endpoint_url:
        raise ValueError("R2 endpoint unknown: set R2_ACCOUNT_ID or pass endpoint_url")
    return boto3.client(
        "s3", endpoint_url=endpoint_url,
        aws_access_key_id=access_key or os.environ.get("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=secret_key or os.environ.get("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )


def _content_type(name: str) -> str:
    if name.endswith(".json"):
        return "application/json"
    if name.endswith(".npy") or name.endswith(".bin"):
        return "application/octet-stream"
    return "application/octet-stream"


def publish(mri_dir: str | Path, *, bucket: str | None = None,
            account_id: str | None = None, endpoint_url: str | None = None,
            access_key: str | None = None, secret_key: str | None = None,
            local_dir: str | Path | None = None, with_hover: bool = False,
            dry_run: bool = False) -> dict:
    """Publish a decomposed .mri.

    bucket    → upload to R2 over the S3 API.
    local_dir → export the same key layout to a directory (no network).
    dry_run   → only report what would be published.
    """
    mri_dir = Path(mri_dir)
    if not (mri_dir / "decomp" / "meta.json").exists():
        raise FileNotFoundError(f"no decomp/ — run `heinrich mri-decompose --mri {mri_dir}` first")

    sidecars = write_sidecars(mri_dir)
    model, mode = _model_mode(mri_dir)
    prefix = f"{model}/{mode}.mri"
    files = list(consumer_files(mri_dir, with_hover=with_hover))
    total = sum(p.stat().st_size for p, _ in files)
    plan = {"model": model, "mode": mode, "prefix": prefix, "sidecars": sidecars,
            "n_files": len(files), "bytes": total}

    if dry_run:
        plan["files"] = [k for _, k in files]
        return plan

    if local_dir:
        out = Path(local_dir)
        for p, suffix in files:
            dst = out / prefix / suffix
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(p.read_bytes())
        _update_manifest_local(out, manifest_entry(mri_dir))
        plan["target"] = str(out)
        return plan

    if not bucket:
        raise ValueError("pass bucket=... (R2) or local_dir=... (export)")
    client = _r2_client(account_id, endpoint_url, access_key, secret_key)
    for p, suffix in files:
        client.put_object(Bucket=bucket, Key=f"{prefix}/{suffix}",
                          Body=p.read_bytes(), ContentType=_content_type(suffix),
                          CacheControl="public, max-age=31536000, immutable")
    _update_manifest_r2(client, bucket, manifest_entry(mri_dir))
    plan["target"] = f"r2://{bucket}"
    return plan


def _upsert(models: list, entry: dict) -> list:
    models = [m for m in models if not (m.get("model") == entry["model"] and m.get("mode") == entry["mode"])]
    models.append(entry)
    models.sort(key=lambda m: (0 if m.get("architecture") == "transformer" else 1, m.get("model", "")))
    return models


def _update_manifest_local(out: Path, entry: dict) -> None:
    p = out / "models.json"
    models = json.loads(p.read_text()) if p.exists() else []
    p.write_text(json.dumps(_upsert(models, entry), indent=2))


def _update_manifest_r2(client, bucket: str, entry: dict) -> None:
    try:
        body = client.get_object(Bucket=bucket, Key="models.json")["Body"].read()
        models = json.loads(body)
    except Exception:
        models = []
    client.put_object(Bucket=bucket, Key="models.json",
                      Body=json.dumps(_upsert(models, entry), indent=2).encode(),
                      ContentType="application/json", CacheControl="no-cache")
