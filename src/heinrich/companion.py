"""Heinrich companion — live viewer and command runner.

HTTP server that serves the viewer frontend and provides API endpoints
for querying MRI data, running analysis commands, and reading signals
from the DB. Replaces viz.py for the MRI workflow.

Usage:
    heinrich companion                  # http://localhost:8377
    heinrich companion --port 9000      # custom port
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any

import numpy as np


# --- In-memory caches ---

_ui_html_cache: str | None = None
_decomp_meta_cache: dict[str, dict] = {}
_weight_align_cache: dict[str, list] = {}


# --- WebSocket for live sync ---
# Lightweight WebSocket implementation (no external deps)

_ws_clients: list = []  # active WebSocket connections
_ws_lock = threading.Lock()


def notify_companions(event: dict):
    """Push an event to all connected companion browsers.

    Called from emit_signals() when new signals are written to the DB.
    Thread-safe.
    """
    frame = _ws_encode(json.dumps(event))
    with _ws_lock:
        dead = []
        for sock in _ws_clients:
            try:
                sock.sendall(frame)
            except Exception:
                dead.append(sock)
        for s in dead:
            _ws_clients.remove(s)


def _ws_encode(text: str) -> bytes:
    """Encode a text string as a WebSocket frame."""
    data = text.encode('utf-8')
    frame = bytearray()
    frame.append(0x81)  # text frame, FIN
    length = len(data)
    if length < 126:
        frame.append(length)
    elif length < 65536:
        frame.append(126)
        frame.extend(length.to_bytes(2, 'big'))
    else:
        frame.append(127)
        frame.extend(length.to_bytes(8, 'big'))
    frame.extend(data)
    return bytes(frame)


def _ws_handshake(handler):
    """Perform WebSocket upgrade handshake."""
    import base64
    key = None
    for line in handler.headers.as_string().split('\r\n'):
        if line.lower().startswith('sec-websocket-key:'):
            key = line.split(':', 1)[1].strip()
    if not key:
        return False
    accept = base64.b64encode(
        hashlib.sha1((key + '258EAFA5-E914-47DA-95CA-5AB5CF11E235').encode()).digest()
    ).decode()
    handler.wfile.write(
        f"HTTP/1.1 101 Switching Protocols\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n".encode()
    )
    handler.wfile.flush()
    with _ws_lock:
        _ws_clients.append(handler.request)
    # Keep connection alive
    try:
        while True:
            data = handler.request.recv(1024)
            if not data:
                break
            # Check for close frame
            if data[0] == 0x88:
                break
    except Exception:
        pass
    finally:
        with _ws_lock:
            if handler.request in _ws_clients:
                _ws_clients.remove(handler.request)
    return True


# --- PCA cache ---

_pca_cache: dict[str, dict] = {}


def _compute_pca(mri_path: str, n_sample: int = 5000) -> dict:
    """Compute PCA projections at every layer.

    Caches to disk as pca_3d.npz inside the MRI directory.
    Subsequent loads are instant instead of minutes.
    """
    from .profile.mri import load_mri

    mri_dir = Path(mri_path)
    cache_file = mri_dir / "pca_3d.npz"

    mri = load_mri(mri_path)
    meta = mri['metadata']
    n_layers = meta['model']['n_layers']
    model_name = meta['model']['name']
    mode = meta.get('capture', {}).get('mode', '?')
    n_tok = len(mri['token_ids'])

    if n_sample is None or n_sample <= 0 or n_sample >= n_tok:
        idx = list(range(n_tok))
        n_sample = n_tok
    else:
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(n_tok, n_sample, replace=False))

    scripts = [str(s) for s in mri.get('scripts', np.array(['?'] * n_tok))[idx]]
    texts = [str(t) for t in mri.get('token_texts', np.array([''] * n_tok))[idx]]

    # Check disk cache
    if cache_file.exists():
        try:
            cached = dict(np.load(cache_file, allow_pickle=False))
            if int(cached.get('n_sample', 0)) == n_sample:
                layers_data = []
                for i in range(n_layers):
                    pk, vk = f"proj_L{i}", f"var_L{i}"
                    if pk in cached and vk in cached:
                        layers_data.append({
                            "layer": i,
                            "pc_var": cached[vk].tolist(),
                            "points": cached[pk].astype(np.float32).tolist(),
                        })
                if layers_data:
                    return {
                        "model": model_name, "model_dir": mri_dir.parent.name,
                        "mode": mode, "n_tokens": n_sample,
                        "n_layers": len(layers_data),
                        "scripts": scripts, "texts": texts, "layers": layers_data,
                    }
        except Exception:
            pass

    # Compute PCA
    import sys
    cache_arrays = {"n_sample": np.array(n_sample)}
    layers_data = []
    for i in range(n_layers):
        print(f"  PCA L{i}/{n_layers}...", end="\r", file=sys.stderr)
        exit_key = f"exit_L{i}"
        if exit_key not in mri:
            continue
        exits = mri[exit_key][idx].astype(np.float32)
        centered = exits - exits.mean(axis=0)
        U, S, _ = np.linalg.svd(centered, full_matrices=False)
        proj = U[:, :3] * S[:3]
        pmax = np.abs(proj).max() + 1e-8
        proj_norm = (proj / pmax).astype(np.float16)
        var_exp = (S[:3] ** 2) / (S ** 2).sum()
        cache_arrays[f"proj_L{i}"] = proj_norm
        cache_arrays[f"var_L{i}"] = var_exp.astype(np.float32)
        layers_data.append({
            "layer": i,
            "pc_var": var_exp.tolist(),
            "points": proj_norm.astype(np.float32).tolist(),
        })
    print(" " * 30, end="\r", file=sys.stderr)

    # Save cache
    try:
        np.savez_compressed(cache_file, **cache_arrays)
        print(f"  Cached PCA to {cache_file}", file=sys.stderr)
    except Exception:
        pass

    return {
        "model": model_name, "model_dir": mri_dir.parent.name,
        "mode": mode, "n_tokens": n_sample,
        "n_layers": len(layers_data),
        "scripts": scripts, "texts": texts, "layers": layers_data,
    }


def _load_decomp(mri_path: str, layer: int) -> dict:
    """Load pre-computed PCA decomposition for a single layer."""
    p = Path(mri_path) / "decomp"
    scores_path = p / f"L{layer:02d}_scores.npy"
    var_path = p / f"L{layer:02d}_variance.npy"
    if not scores_path.exists():
        return {"error": f"No decomposition at L{layer}. Run mri-decompose."}
    scores = np.load(scores_path).astype(np.float32)  # [N, K]
    variance = np.load(var_path).astype(np.float32)    # [K]
    # Token metadata — allow_pickle needed for variable-length string arrays in npz
    meta_mri = json.loads((Path(mri_path) / "metadata.json").read_text())
    tok = dict(np.load(Path(mri_path) / "tokens.npz", allow_pickle=True))
    meta_decomp = json.loads((p / "meta.json").read_text())
    n_sample = meta_decomp["n_sample"]
    sample_raw = meta_decomp.get("sample_indices")
    n_tok = len(tok["token_ids"])
    if sample_raw is None or sample_raw == "all":
        sample_idx = np.arange(n_tok)
    else:
        sample_idx = np.asarray(sample_raw)
    scripts = [str(s) for s in tok["scripts"][sample_idx]]
    texts = [str(t) for t in tok["token_texts"][sample_idx]]
    ids = tok["token_ids"][sample_idx].tolist()
    lm = meta_decomp["layers"][layer] if layer < len(meta_decomp["layers"]) else {}
    return {
        "model": meta_mri["model"]["name"],
        "model_dir": Path(mri_path).parent.name,
        "mode": meta_mri.get("capture", {}).get("mode", "?"),
        "layer": layer,
        "n_tokens": len(scores),
        "n_components": len(variance),
        "variance": variance.tolist(),
        "scores": scores.tolist(),
        "scripts": scripts,
        "texts": texts,
        "token_ids": ids,
        "pc1_pct": lm.get("pc1_pct", 0),
        "intrinsic_dim": lm.get("intrinsic_dim", 0),
        "neighbor_stability": lm.get("neighbor_stability", 0),
    }


def _load_decomp_meta(mri_path: str) -> dict:
    """Load decomposition metadata (variance spectrum, intrinsic dim per layer)."""
    if mri_path in _decomp_meta_cache:
        return _decomp_meta_cache[mri_path]
    p = Path(mri_path) / "decomp" / "meta.json"
    if not p.exists():
        return {"error": "No decomposition. Run mri-decompose."}
    meta = json.loads(p.read_text())
    meta["model_dir"] = Path(mri_path).parent.name
    meta["mode"] = json.loads((Path(mri_path) / "metadata.json").read_text()).get("capture", {}).get("mode", "?")
    _decomp_meta_cache[mri_path] = meta
    return meta


def _token_biography(mri_path: str, token_idx: int) -> dict:
    """Get precomputed gate heatmap for one token."""
    p = Path(mri_path) / "decomp" / "gate_heatmap.npy"
    if not p.exists():
        return {"error": "No gate heatmap. Re-run mri-decompose."}
    heat = np.load(str(p), mmap_mode='r')
    if token_idx >= heat.shape[0]:
        return {"error": f"Token {token_idx} out of range"}
    row = heat[token_idx].astype(np.float32)
    return {"token_idx": token_idx, "max_per_layer": [round(float(v), 2) for v in row]}


def _weight_alignment(mri_path: str, layer: int) -> dict:
    """Get precomputed weight-PC alignment."""
    all_data = _weight_alignment_all(mri_path)
    if isinstance(all_data, dict) and "error" in all_data:
        return all_data
    if layer < len(all_data):
        return all_data[layer]
    return {"error": f"Layer {layer} not found"}


def _weight_alignment_all(mri_path: str) -> list | dict:
    """Get all weight-PC alignment data, cached in memory."""
    if mri_path in _weight_align_cache:
        return _weight_align_cache[mri_path]
    p = Path(mri_path) / "decomp" / "weight_alignment.json"
    if not p.exists():
        return {"error": "No weight alignment. Re-run mri-decompose."}
    data = json.loads(p.read_text())
    _weight_align_cache[mri_path] = data
    return data


def _token_neurons(mri_path: str, token_idx: int) -> dict:
    """Get gate × up activations for one token at top neurons across all layers."""
    mri_dir = Path(mri_path)
    imp_path = mri_dir / "decomp" / "neuron_importance.json"
    if not imp_path.exists():
        return {"error": "No neuron data. Re-run mri-decompose."}
    importance = json.loads(imp_path.read_text())
    mlp_dir = mri_dir / "mlp"
    result = {"token_idx": token_idx, "layers": []}
    for li_data in importance:
        li = li_data["layer"]
        top_n = li_data["top_neurons"][:30]  # top 30 neurons
        gp = mlp_dir / f"L{li:02d}_gate.npy"
        up = mlp_dir / f"L{li:02d}_up.npy"
        if not gp.exists() or not up.exists() or not top_n:
            result["layers"].append({"layer": li, "neurons": []})
            continue
        g = np.load(str(gp), mmap_mode='r')
        u = np.load(str(up), mmap_mode='r')
        if token_idx >= g.shape[0]:
            result["layers"].append({"layer": li, "neurons": []})
            continue
        gt = g[token_idx].astype(np.float32)
        ut = u[token_idx].astype(np.float32)
        neurons = [{"idx": int(ni), "gate": round(float(gt[ni]), 3),
                     "up": round(float(ut[ni]), 3),
                     "contrib": round(float(gt[ni] * ut[ni]), 3)} for ni in top_n]
        result["layers"].append({"layer": li, "neurons": neurons})
    return result


def _token_attention(mri_path: str, token_idx: int) -> dict:
    """Get attention weights for one token across all layers and heads."""
    mri_dir = Path(mri_path)
    attn_dir = mri_dir / "attention"
    meta = json.loads((mri_dir / "metadata.json").read_text())
    n_layers = meta['model']['n_layers']
    result = {"token_idx": token_idx, "layers": []}
    for li in range(n_layers):
        wp = attn_dir / f"L{li:02d}_weights.npy"
        lp = attn_dir / f"L{li:02d}_logits.npy"
        fp = wp if wp.exists() else (lp if lp.exists() else None)
        if not fp:
            result["layers"].append({"layer": li, "heads": []})
            continue
        a = np.load(str(fp), mmap_mode='r')
        if token_idx >= a.shape[0]:
            result["layers"].append({"layer": li, "heads": []})
            continue
        row = a[token_idx].astype(np.float32)  # [heads, seq_len]
        result["layers"].append({"layer": li,
            "heads": [[round(float(v), 4) for v in head] for head in row]})
    return result


def _list_models(mri_root: str = "/Volumes/sharts") -> list[dict]:
    """List available MRI directories."""
    root = Path(mri_root)
    if not root.exists():
        return []
    models = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        for mri_dir in sorted(model_dir.glob("*.mri")):
            meta_path = mri_dir / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                arch = meta.get("architecture", "transformer")
                models.append({
                    "path": str(mri_dir),
                    "model": model_dir.name,
                    "mode": meta.get("capture", {}).get("mode", "?"),
                    "n_layers": meta.get("model", {}).get("n_layers", 0),
                    "n_tokens": meta.get("capture", {}).get("n_tokens", 0),
                    "version": meta.get("version", "?"),
                    "architecture": arch,
                })
    # Sort transformers first
    models.sort(key=lambda m: (0 if m["architecture"] == "transformer" else 1, m["model"]))
    return models


_signal_db = None

def _get_signal_db():
    """Get or create a shared SignalDB connection."""
    global _signal_db
    if _signal_db is None:
        from .core.db import SignalDB
        _signal_db = SignalDB()
    return _signal_db


def _query_signals(kind: str | None = None, model: str | None = None,
                   target: str | None = None) -> list[dict]:
    """Query signals from the DB."""
    db = _get_signal_db()
    clauses = []
    params = []
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    if model:
        clauses.append("model = ?")
        params.append(model)
    if target:
        clauses.append("target = ?")
        params.append(target)
    where = " AND ".join(clauses) if clauses else "1=1"
    rows = db._conn.execute(
        f"SELECT kind, source, model, target, value, metadata, created_at "
        f"FROM signals WHERE {where} ORDER BY kind, target",
        params,
    ).fetchall()
    return [
        {"kind": r[0], "source": r[1], "model": r[2], "target": r[3],
         "value": r[4], "metadata": json.loads(r[5] or "{}"), "created_at": r[6]}
        for r in rows
    ]


def _run_command(command: str, args: dict[str, str]) -> dict:
    """Run a heinrich CLI command via subprocess with --json."""
    cmd = [sys.executable, "-m", "heinrich.cli", "--json", command]
    for k, v in args.items():
        flag = "--" + k.replace("_", "-")
        cmd.extend([flag, str(v)])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return {"error": result.stderr[:500]}
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"output": result.stdout[:2000]}
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out (300s)"}


def _list_commands() -> list[dict]:
    """List available CLI commands with their descriptions."""
    from .cli import build_parser
    parser = build_parser()
    commands = []
    for action in parser._subparsers._actions:
        if not hasattr(action, '_parser_class'):
            continue
        for name, subparser in action.choices.items():
            params = []
            for a in subparser._actions:
                if a.dest in ('help',):
                    continue
                params.append({
                    "name": a.dest,
                    "flag": a.option_strings[0] if a.option_strings else a.dest,
                    "required": a.required if hasattr(a, 'required') else False,
                    "type": a.type.__name__ if a.type else "string",
                    "default": str(a.default) if a.default is not None else None,
                    "help": a.help or "",
                })
            commands.append({
                "name": name,
                "help": subparser.description or subparser.format_usage().strip(),
                "params": params,
            })
    return commands






class CompanionHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the companion API + viewer."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        # WebSocket upgrade
        if path == '/ws' and self.headers.get('Upgrade', '').lower() == 'websocket':
            _ws_handshake(self)
            return

        if path == '/' or path == '/index.html':
            ui_path = Path(__file__).parent / "companion_ui.html"
            self._send_html(ui_path.read_text())
        elif path == '/api/models':
            self._send_json(_list_models())
        elif path.startswith('/api/pca/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                n = int(qs.get('n', ['0'])[0])
                mri_path = f"/Volumes/sharts/{model}/{mode}.mri"
                if not Path(mri_path).exists():
                    self._send_json({"error": f"MRI not found: {mri_path}"})
                    return
                result = _compute_pca(mri_path, n_sample=n)
                if qs.get('fmt', [None])[0] == 'bin':
                    self._send_pca_binary(result)
                else:
                    self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/pca/<model>/<mode>?n=5000"})
        elif path.startswith('/api/scores/'):
            # /api/scores/model/mode — serves all_scores.bin (single binary, all layers)
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                decomp_dir = Path(f"/Volumes/sharts/{model}/{mode}.mri/decomp")
                bin_path = decomp_dir / "all_scores.bin"
                meta_path = decomp_dir / "meta.json"
                if bin_path.exists():
                    # Validate binary header matches metadata without reading full file
                    import struct
                    file_size = bin_path.stat().st_size
                    with open(bin_path, 'rb') as f:
                        header = f.read(16)
                    if len(header) >= 16 and meta_path.exists():
                        magic, bnL, bnN, bnK = struct.unpack_from('<4sIII', header)
                        # Accept both old (0x00000000) and new ('HEIN') magic
                        if magic not in (b'HEIN', b'\x00\x00\x00\x00'):
                            self._send_json({"error": f"Unknown binary format (magic: {magic!r})"})
                            return
                        meta = json.loads(meta_path.read_text())
                        expected = 16 + bnL * bnK * 4 + bnL * bnN * bnK * 2
                        if file_size != expected:
                            self._send_json({"error": f"Stale binary: size {file_size} != expected {expected}. Re-run decomposition."})
                            return
                        if meta.get("n_sample") != bnN or meta.get("n_components") != bnK:
                            self._send_json({"error": f"Binary/meta mismatch: bin has {bnN}tok/{bnK}pc, meta has {meta.get('n_sample')}/{meta.get('n_components')}. Re-run decomposition."})
                            return
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Content-Length', file_size)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Cache-Control', 'public, max-age=3600')
                    self.end_headers()
                    with open(bin_path, 'rb') as f:
                        while chunk := f.read(65536):
                            self.wfile.write(chunk)
                else:
                    self._send_json({"error": f"No scores at {bin_path}"})
            else:
                self._send_json({"error": "Usage: /api/scores/<model>/<mode>"})
        elif path.startswith('/api/decomp/'):
            # /api/decomp/model/mode?layer=5
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                layer = int(qs.get('layer', ['0'])[0])
                mri_path = f"/Volumes/sharts/{model}/{mode}.mri"
                result = _load_decomp(mri_path, layer)
                self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/decomp/<model>/<mode>?layer=N"})
        elif path.startswith('/api/decomp-meta/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                mri_path = f"/Volumes/sharts/{model}/{mode}.mri"
                result = _load_decomp_meta(mri_path)
                self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/decomp-meta/<model>/<mode>"})
        elif path.startswith('/api/token-bio/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                token = int(qs.get('token', ['0'])[0])
                mri_path = f"/Volumes/sharts/{model}/{mode}.mri"
                self._send_json(_token_biography(mri_path, token))
            else:
                self._send_json({"error": "Usage: /api/token-bio/<model>/<mode>?token=N"})
        elif path.startswith('/api/gate-heatmap/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                hp = Path(f"/Volumes/sharts/{model}/{mode}.mri/decomp/gate_heatmap.npy")
                if hp.exists():
                    self._send_file(hp)
                else:
                    self._send_json({"error": "No gate heatmap"})
        elif path.startswith('/api/token-neurons/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                token = int(qs.get('token', ['0'])[0])
                mri_path = f"/Volumes/sharts/{model}/{mode}.mri"
                self._send_json(_token_neurons(mri_path, token))
        elif path.startswith('/api/token-attn/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                token = int(qs.get('token', ['0'])[0])
                mri_path = f"/Volumes/sharts/{model}/{mode}.mri"
                self._send_json(_token_attention(mri_path, token))
        elif path.startswith('/api/norms/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                norms_path = Path(f"/Volumes/sharts/{model}/{mode}.mri/norms.npz")
                if norms_path.exists():
                    norms = dict(np.load(str(norms_path)))
                    result = {}
                    for k, v in sorted(norms.items()):
                        result[k] = {"mean": float(v.mean()), "std": float(v.std()),
                                     "min": float(v.min()), "max": float(v.max()),
                                     "shape": list(v.shape)}
                    self._send_json(result)
                else:
                    self._send_json({"error": "No norms.npz"})
        elif path.startswith('/api/baselines/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                bl_path = Path(f"/Volumes/sharts/{model}/{mode}.mri/baselines.npz")
                if bl_path.exists():
                    bl = dict(np.load(str(bl_path)))
                    result = {}
                    for k, v in sorted(bl.items()):
                        result[k] = {"norm": float(np.linalg.norm(v)),
                                     "mean": float(v.mean()), "std": float(v.std()),
                                     "shape": list(v.shape)}
                    self._send_json(result)
                else:
                    self._send_json({"error": "No baselines.npz"})
        elif path.startswith('/api/delta-scores/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                bin_path = Path(f"/Volumes/sharts/{model}/{mode}.mri/decomp/delta_scores.bin")
                if bin_path.exists():
                    self._send_file(bin_path)
                else:
                    self._send_json({"error": "No delta scores. Needs entry states."})
        elif path.startswith('/api/weight-align-all/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                mri_path = f"/Volumes/sharts/{model}/{mode}.mri"
                result = _weight_alignment_all(mri_path)
                self._send_json(result)
        elif path.startswith('/api/weight-align/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                layer = int(qs.get('layer', ['0'])[0])
                mri_path = f"/Volumes/sharts/{model}/{mode}.mri"
                self._send_json(_weight_alignment(mri_path, layer))
            else:
                self._send_json({"error": "Usage: /api/weight-align/<model>/<mode>?layer=N"})
        elif path == '/api/signals':
            kind = qs.get('kind', [None])[0]
            model = qs.get('model', [None])[0]
            target = qs.get('target', [None])[0]
            self._send_json(_query_signals(kind=kind, model=model, target=target))
        elif path == '/api/commands':
            self._send_json(_list_commands())
        elif path.startswith('/api/run/'):
            command = path.split('/api/run/')[1]
            args = {k: v[0] for k, v in qs.items()}
            result = _run_command(command, args)
            self._send_json(result)
        else:
            self.send_error(404)

    def _send_json(self, data: Any):
        body = json.dumps(data, default=str).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        """Stream a binary file without loading it all into memory."""
        size = path.stat().st_size
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', size)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'public, max-age=3600')
        self.end_headers()
        with open(path, 'rb') as f:
            while chunk := f.read(65536):
                self.wfile.write(chunk)

    def _send_pca_binary(self, result: dict):
        """Send PCA data as binary: JSON header + float32 point arrays.

        Format: 4-byte header_len (uint32) + JSON header + per-layer float32 arrays
        Each layer: n_tokens * 3 float32s (x,y,z)
        """
        import struct
        # Header: everything except points
        header = {k: v for k, v in result.items() if k != 'layers'}
        header['n_layers'] = len(result['layers'])
        header['pc_vars'] = [l['pc_var'] for l in result['layers']]
        header['layer_ids'] = [l['layer'] for l in result['layers']]
        header_bytes = json.dumps(header, default=str).encode()

        # Points: concatenated float32 arrays
        point_chunks = []
        for ld in result['layers']:
            arr = np.array(ld['points'], dtype=np.float32)
            point_chunks.append(arr.tobytes())

        body = struct.pack('<I', len(header_bytes)) + header_bytes + b''.join(point_chunks)
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # silence request logs


def run_companion(port: int = 8377):
    """Run the companion HTTP server (threaded)."""
    from socketserver import ThreadingMixIn

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(('0.0.0.0', port), CompanionHandler)
    print(f"Heinrich companion: http://localhost:{port}")
    print("  Cloud view: 3D PCA point cloud, scrub layers, orbit, hover tokens")
    print("  Depth view: signal DB charts per model")
    print("  Command view: run any heinrich command, see results")
    print("  WebSocket: live sync with MCP (signals push to browser)")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
