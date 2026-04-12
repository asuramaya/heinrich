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


def _compute_umap(mri_path: str, n_sample: int = 3000) -> dict:
    """Compute UMAP projections at every layer. Cached to disk."""
    import umap
    from .profile.mri import load_mri
    import sys

    mri_dir = Path(mri_path)
    cache_file = mri_dir / "umap_3d.npz"

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
                    if pk in cached:
                        layers_data.append({
                            "layer": i,
                            "pc_var": cached[vk].tolist() if vk in cached else [0, 0, 0],
                            "points": cached[pk].astype(np.float32).tolist(),
                        })
                if layers_data:
                    return {
                        "model": model_name, "model_dir": mri_dir.parent.name,
                        "mode": mode, "n_tokens": n_sample, "projection": "umap",
                        "n_layers": len(layers_data),
                        "scripts": scripts, "texts": texts, "layers": layers_data,
                    }
        except Exception:
            pass

    cache_arrays = {"n_sample": np.array(n_sample)}
    layers_data = []
    for i in range(n_layers):
        print(f"  UMAP L{i}/{n_layers}...", end="\r", file=sys.stderr)
        exit_key = f"exit_L{i}"
        if exit_key not in mri:
            continue
        exits = mri[exit_key][idx].astype(np.float32)
        reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42, verbose=False)
        proj = reducer.fit_transform(exits)
        pmax = np.abs(proj).max() + 1e-8
        proj_norm = (proj / pmax).astype(np.float16)

        # Also compute PCA variance for the HUD
        centered = exits - exits.mean(axis=0)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        var_exp = (S[:3] ** 2) / (S ** 2).sum()

        cache_arrays[f"proj_L{i}"] = proj_norm
        cache_arrays[f"var_L{i}"] = var_exp.astype(np.float32)
        layers_data.append({
            "layer": i, "pc_var": var_exp.tolist(),
            "points": proj_norm.astype(np.float32).tolist(),
        })
    print(" " * 30, end="\r", file=sys.stderr)

    try:
        np.savez_compressed(cache_file, **cache_arrays)
        print(f"  Cached UMAP to {cache_file}", file=sys.stderr)
    except Exception:
        pass

    return {
        "model": model_name, "model_dir": mri_dir.parent.name,
        "mode": mode, "n_tokens": n_sample, "projection": "umap",
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
    # Token metadata
    meta_mri = json.loads((Path(mri_path) / "metadata.json").read_text())
    tok = dict(np.load(Path(mri_path) / "tokens.npz", allow_pickle=True))
    meta_decomp = json.loads((p / "meta.json").read_text())
    n_sample = meta_decomp["n_sample"]
    sample_idx = meta_decomp.get("sample_indices", list(range(len(tok["token_ids"]))))
    if isinstance(sample_idx, str):
        sample_idx = list(range(len(tok["token_ids"])))
    scripts = [str(s) for s in tok["scripts"][sample_idx]]
    texts = [str(t) for t in tok["token_texts"][sample_idx]]
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
        "pc1_pct": lm.get("pc1_pct", 0),
        "intrinsic_dim": lm.get("intrinsic_dim", 0),
        "neighbor_stability": lm.get("neighbor_stability", 0),
    }


def _load_decomp_meta(mri_path: str) -> dict:
    """Load decomposition metadata (variance spectrum, intrinsic dim per layer)."""
    p = Path(mri_path) / "decomp" / "meta.json"
    if not p.exists():
        return {"error": "No decomposition. Run mri-decompose."}
    meta = json.loads(p.read_text())
    meta["model_dir"] = Path(mri_path).parent.name
    meta["mode"] = json.loads((Path(mri_path) / "metadata.json").read_text()).get("capture", {}).get("mode", "?")
    return meta


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


def _query_signals(kind: str | None = None, model: str | None = None,
                   target: str | None = None) -> list[dict]:
    """Query signals from the DB."""
    from .core.db import SignalDB
    db_path = str(Path(__file__).resolve().parent.parent.parent / "data" / "heinrich.db")
    db = SignalDB(db_path)
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
    db.close()
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


# --- Frontend ---

VIEWER_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Heinrich Companion</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a0a; color: #ccc; font-family: 'SF Mono','Menlo',monospace; font-size: 13px; }
#app { display: flex; flex-direction: column; height: 100vh; }
#toolbar {
  display: flex; gap: 12px; padding: 8px 16px; background: #111; border-bottom: 1px solid #222;
  align-items: center; flex-wrap: wrap;
}
#toolbar select, #toolbar input, #toolbar button {
  background: #1a1a1a; border: 1px solid #333; color: #ccc; padding: 4px 8px;
  font-family: inherit; font-size: 12px; border-radius: 3px;
}
#toolbar button { cursor: pointer; }
#toolbar button:hover { border-color: #888; }
#toolbar label { color: #888; font-size: 11px; }
#viewport { flex: 1; position: relative; overflow: hidden; }
#canvas3d { width: 100%; height: 100%; }
#overlay {
  position: absolute; top: 8px; left: 16px; z-index: 10;
  pointer-events: none;
}
#overlay-title { font-size: 20px; font-weight: bold; }
#overlay-status { font-weight: bold; margin-left: 8px; }
#overlay-stats { font-size: 12px; color: #888; margin-top: 4px; }
#legend {
  position: absolute; top: 8px; right: 16px; z-index: 10;
  font-size: 11px; line-height: 1.8;
}
.legend-item { display: flex; align-items: center; gap: 5px; }
.legend-dot { width: 7px; height: 7px; border-radius: 50%; }
#layer-bar {
  display: flex; align-items: center; gap: 12px; padding: 8px 16px;
  background: #111; border-top: 1px solid #222;
}
#layer-slider { flex: 1; accent-color: #4477AA; }
#result-panel {
  position: absolute; bottom: 48px; left: 16px; right: 16px; max-height: 40vh;
  background: #111; border: 1px solid #333; border-radius: 4px;
  overflow: auto; padding: 12px; display: none; z-index: 20; font-size: 12px;
}
#result-panel pre { white-space: pre-wrap; word-break: break-all; }
#result-panel .chart { height: 200px; }
#view-tabs { display: flex; gap: 0; }
#view-tabs button {
  border-radius: 0; border-right: none; padding: 4px 12px;
}
#view-tabs button:last-child { border-right: 1px solid #333; }
#view-tabs button.active { background: #333; color: #fff; }
#token-info {
  position: absolute; bottom: 56px; right: 16px; background: #1a1a1a;
  border: 1px solid #444; padding: 6px 10px; border-radius: 4px;
  display: none; z-index: 15; max-width: 300px;
}
#cmd-panel {
  position: absolute; top: 50px; left: 16px; background: #111;
  border: 1px solid #333; border-radius: 4px; padding: 12px;
  display: none; z-index: 25; min-width: 400px; max-height: 80vh; overflow: auto;
}
#cmd-params { display: flex; flex-direction: column; gap: 6px; margin: 8px 0; }
#cmd-params label { display: flex; justify-content: space-between; gap: 8px; }
#cmd-params input { flex: 1; }
</style>
</head>
<body>
<div id="app">
  <div id="toolbar">
    <label>Model <select id="sel-model"></select></label>
    <label>Mode <select id="sel-mode"><option>raw</option><option>naked</option><option>template</option></select></label>
    <label>n <input id="inp-n" type="number" value="5000" style="width:60px" title="0 = all tokens"></label>
    <button id="btn-load">Load</button>
    <span style="color:#333">|</span>
    <div id="view-tabs">
      <button class="active" data-view="cloud">Cloud</button>
      <button data-view="depth">Depth</button>
      <button data-view="cmd">Command</button>
    </div>
    <span style="flex:1"></span>
    <span id="status-text" style="color:#666"></span>
  </div>
  <div id="viewport">
    <canvas id="canvas3d"></canvas>
    <div id="overlay">
      <span id="overlay-title"></span><span id="overlay-status"></span>
      <div id="overlay-stats"></div>
    </div>
    <div id="legend"></div>
    <div id="token-info"></div>
    <div id="result-panel"></div>
    <div id="cmd-panel">
      <label>Command <select id="sel-cmd" style="width:100%"></select></label>
      <div id="cmd-params"></div>
      <button id="btn-run" style="width:100%;padding:6px;margin-top:4px">Run</button>
    </div>
  </div>
  <div id="layer-bar">
    <button id="btn-play" style="width:60px">Play</button>
    <span id="layer-label">L00</span>
    <input type="range" id="layer-slider" min="0" max="29" value="0">
    <span id="pc-label" style="color:#888"></span>
  </div>
</div>

<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const COLORS = {
  latin:0x4477AA, CJK:0xEE6677, Cyrillic:0x228833, code:0xCCBB44,
  Arabic:0x66CCEE, Japanese:0xAA3377, Korean:0x999999, Thai:0x44BB99,
  Hebrew:0xEE8866, other:0xAAAAAA, special:0x666666
};

let data = null, currentLayer = 0, playing = false, playTimer = null;
let scene, camera, renderer, controls, points, geometry;
let currentView = 'cloud';
let commands = [];

// --- Three.js setup ---
const canvas = document.getElementById('canvas3d');
scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);
camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.position.set(0, 0, 3);
renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
geometry = new THREE.BufferGeometry();
const mat = new THREE.PointsMaterial({
  size: 0.008, vertexColors: true, transparent: true, opacity: 0.6,
  sizeAttenuation: true, depthWrite: false, blending: THREE.AdditiveBlending
});
points = new THREE.Points(geometry, mat);
scene.add(points);

// Token text labels — generated as a texture atlas
let labelSprites = [];
function makeTextTexture(text, color) {
  const c = document.createElement('canvas');
  c.width = 128; c.height = 32;
  const ctx = c.getContext('2d');
  ctx.font = '14px monospace';
  ctx.fillStyle = color;
  ctx.fillText(text.slice(0, 12), 2, 20);
  const tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.LinearFilter;
  return tex;
}


function resize() {
  const vp = document.getElementById('viewport');
  const w = vp.clientWidth, h = vp.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio);
}
window.addEventListener('resize', resize);
resize();

// --- Raycaster for token hover ---
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.03;
const mouse = new THREE.Vector2();

canvas.addEventListener('mousemove', e => {
  if (!data) return;
  const rect = canvas.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(points);
  const info = document.getElementById('token-info');
  if (hits.length > 0) {
    const idx = hits[0].index;
    info.textContent = `${data.texts[idx]}  [${data.scripts[idx]}]`;
    info.style.display = 'block';
  } else {
    info.style.display = 'none';
  }
});

// --- API ---
async function api(path) {
  const r = await fetch('/api/' + path);
  return r.json();
}

// --- Init ---
async function init() {
  const models = await api('models');
  const sel = document.getElementById('sel-model');
  // Group by model, prefer those with raw/naked/template modes
  const byModel = {};
  models.forEach(m => {
    if (!byModel[m.model]) byModel[m.model] = [];
    byModel[m.model].push(m.mode);
  });
  // Sort: models with 'raw' mode first, then alphabetical
  const sorted = Object.keys(byModel).sort((a, b) => {
    const aHas = byModel[a].includes('raw') ? 0 : 1;
    const bHas = byModel[b].includes('raw') ? 0 : 1;
    return aHas - bHas || a.localeCompare(b);
  });
  sorted.forEach(model => {
    const opt = document.createElement('option');
    opt.value = model;
    opt.textContent = model + ' (' + byModel[model].join(',') + ')';
    sel.appendChild(opt);
  });

  commands = await api('commands');
  const cmdSel = document.getElementById('sel-cmd');
  commands.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.name;
    opt.textContent = c.name;
    cmdSel.appendChild(opt);
  });
  cmdSel.addEventListener('change', showCmdParams);
  showCmdParams();

  document.getElementById('btn-load').addEventListener('click', loadModel);
  loadModel();
}

async function loadModel() {
  const model = document.getElementById('sel-model').value;
  const mode = document.getElementById('sel-mode').value;
  const n = document.getElementById('inp-n').value;
  document.getElementById('status-text').textContent = 'Loading...';

  data = await api(`pca/${model}/${mode}?n=${n}`);
  if (data.error) {
    document.getElementById('status-text').textContent = data.error;
    return;
  }

  // Build points
  const nn = data.n_tokens;
  const positions = new Float32Array(nn * 3);
  const colors = new Float32Array(nn * 3);
  for (let i = 0; i < nn; i++) {
    const hex = COLORS[data.scripts[i]] || 0xAAAAAA;
    colors[i*3] = ((hex>>16)&0xFF)/255;
    colors[i*3+1] = ((hex>>8)&0xFF)/255;
    colors[i*3+2] = (hex&0xFF)/255;
  }
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  // Legend
  const legend = document.getElementById('legend');
  legend.textContent = '';
  const sc = {};
  data.scripts.forEach(s => sc[s] = (sc[s]||0)+1);
  Object.entries(sc).sort((a,b)=>b[1]-a[1]).filter(([,c])=>c>5).forEach(([s,c]) => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    const dot = document.createElement('span');
    dot.className = 'legend-dot';
    dot.style.background = '#'+(COLORS[s]||0xAAAAAA).toString(16).padStart(6,'0');
    item.appendChild(dot);
    item.appendChild(document.createTextNode(s+' ('+c+')'));
    legend.appendChild(item);
  });

  document.getElementById('layer-slider').max = data.n_layers - 1;
  setLayer(0);
  document.getElementById('status-text').textContent = `${data.model_dir}/${data.mode} ${nn} tokens`;
}

function setLayer(layer) {
  if (!data || !data.layers[layer]) return;
  currentLayer = layer;
  const ld = data.layers[layer];
  const pos = geometry.attributes.position.array;
  const pts = ld.points;
  for (let i = 0; i < data.n_tokens; i++) {
    pos[i*3]=pts[i][0]; pos[i*3+1]=pts[i][1]; pos[i*3+2]=pts[i][2];
  }
  geometry.attributes.position.needsUpdate = true;
  const pc1 = (ld.pc_var[0]*100).toFixed(1);
  let status, color;
  if (ld.pc_var[0]>0.95) { status='CRYSTAL'; color='#FF4444'; }
  else if (ld.pc_var[0]>0.5) { status='COLLAPSING'; color='#FFAA44'; }
  else { status='DIVERSE'; color='#44FF44'; }
  document.getElementById('overlay-title').textContent = data.model_dir+'  L'+String(layer).padStart(2,'0');
  const statusEl = document.getElementById('overlay-status');
  statusEl.textContent = status; statusEl.style.color = color;
  document.getElementById('overlay-stats').textContent =
    'PC1:'+pc1+'%  PC2:'+(ld.pc_var[1]*100).toFixed(1)+'%  PC3:'+(ld.pc_var[2]*100).toFixed(1)+'%';
  document.getElementById('layer-label').textContent = 'L'+String(layer).padStart(2,'0');
  document.getElementById('layer-slider').value = layer;
  document.getElementById('pc-label').textContent = 'PC1='+pc1+'%';
}

// --- Depth view ---
async function showDepthView() {
  const model = document.getElementById('sel-model').value;
  const panel = document.getElementById('result-panel');
  const signals = await api(`signals?model=${model}`);
  if (!signals.length) {
    panel.textContent = 'No signals in DB for ' + model + '. Run an analysis command first.';
    panel.style.display = 'block';
    return;
  }
  // Group by kind, render as simple text chart
  const byKind = {};
  signals.forEach(s => {
    if (!byKind[s.kind]) byKind[s.kind] = [];
    byKind[s.kind].push(s);
  });
  let html = '';
  for (const [kind, sigs] of Object.entries(byKind)) {
    const sorted = sigs.sort((a,b) => {
      const la = parseInt(a.target.replace('L','')) || 0;
      const lb = parseInt(b.target.replace('L','')) || 0;
      return la - lb;
    });
    html += kind + ':\\n';
    const maxVal = Math.max(...sorted.map(s => Math.abs(s.value)));
    sorted.forEach(s => {
      const barLen = Math.round(Math.abs(s.value) / maxVal * 40);
      const bar = '\\u2588'.repeat(barLen);
      html += '  ' + s.target.padEnd(5) + ' ' + s.value.toFixed(2).padStart(10) + ' ' + bar + '\\n';
    });
    html += '\\n';
  }
  const pre = document.createElement('pre');
  pre.textContent = html;
  panel.textContent = '';
  panel.appendChild(pre);
  panel.style.display = 'block';
}

// --- Command panel ---
function showCmdParams() {
  const name = document.getElementById('sel-cmd').value;
  const cmd = commands.find(c => c.name === name);
  const div = document.getElementById('cmd-params');
  div.textContent = '';
  if (!cmd) return;
  cmd.params.forEach(p => {
    if (p.name === 'json_output' || p.name === 'no_db' || p.name === 'command' || p.name === 'db_path') return;
    const label = document.createElement('label');
    label.textContent = p.name + (p.required ? ' *' : '');
    const input = document.createElement('input');
    input.name = p.name;
    input.placeholder = p.help || p.name;
    if (p.default) input.value = p.default;
    if (p.name === 'model') input.value = document.getElementById('sel-model').value || '';
    if (p.name === 'mri') {
      const m = document.getElementById('sel-model').value;
      const mode = document.getElementById('sel-mode').value;
      input.value = '/Volumes/sharts/' + m + '/' + mode + '.mri';
    }
    label.appendChild(input);
    div.appendChild(label);
  });
}

document.getElementById('btn-run').addEventListener('click', async () => {
  const name = document.getElementById('sel-cmd').value;
  const inputs = document.querySelectorAll('#cmd-params input');
  const args = {};
  inputs.forEach(inp => { if (inp.value) args[inp.name] = inp.value; });

  document.getElementById('status-text').textContent = 'Running ' + name + '...';
  const result = await api('run/' + name + '?' + new URLSearchParams(args));
  document.getElementById('status-text').textContent = '';

  // Render result
  const panel = document.getElementById('result-panel');
  panel.textContent = '';

  if (result.layers && Array.isArray(result.layers)) {
    // Layer chart
    const maxVal = Math.max(...result.layers.map(l => {
      const vals = Object.values(l).filter(v => typeof v === 'number' && v !== l.layer);
      return Math.max(...vals.map(Math.abs));
    }));
    let text = Object.keys(result).filter(k => k !== 'layers').map(k => k + ': ' + JSON.stringify(result[k])).join('\\n') + '\\n\\n';
    const fields = Object.keys(result.layers[0]).filter(k => k !== 'layer' && typeof result.layers[0][k] === 'number');
    text += 'layer  ' + fields.map(f => f.padStart(12)).join('') + '\\n';
    text += '-'.repeat(7 + fields.length * 13) + '\\n';
    result.layers.forEach(l => {
      text += ('L' + String(l.layer).padStart(2, '0')).padEnd(7);
      fields.forEach(f => { text += String(typeof l[f]==='number' ? l[f].toFixed(4) : l[f]).padStart(13); });
      text += '\\n';
    });
    const pre = document.createElement('pre');
    pre.textContent = text;
    panel.appendChild(pre);
  } else if (result.positive || result.negative) {
    // Ranked list
    let text = '';
    if (result.positive) {
      text += 'Aligned (+):\\n';
      result.positive.forEach(t => { text += '  ' + (t.score||0).toFixed(3).padStart(8) + '  ' + (t.token||'?') + '\\n'; });
    }
    if (result.negative) {
      text += '\\nOpposed (-):\\n';
      result.negative.forEach(t => { text += '  ' + (t.score||0).toFixed(3).padStart(8) + '  ' + (t.token||'?') + '\\n'; });
    }
    const pre = document.createElement('pre');
    pre.textContent = text;
    panel.appendChild(pre);
  } else {
    // JSON fallback
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(result, null, 2);
    panel.appendChild(pre);
  }
  panel.style.display = 'block';
});

// --- View tabs ---
document.querySelectorAll('#view-tabs button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#view-tabs button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentView = btn.dataset.view;
    document.getElementById('result-panel').style.display = 'none';
    document.getElementById('cmd-panel').style.display = 'none';
    canvas.style.display = currentView === 'cloud' ? 'block' : 'none';
    if (currentView === 'depth') showDepthView();
    if (currentView === 'cmd') document.getElementById('cmd-panel').style.display = 'block';
  });
});

// --- Layer controls ---
document.getElementById('layer-slider').addEventListener('input', e => setLayer(parseInt(e.target.value)));
document.getElementById('btn-play').addEventListener('click', () => {
  playing = !playing;
  document.getElementById('btn-play').textContent = playing ? 'Pause' : 'Play';
  if (playing) playTimer = setInterval(() => setLayer((currentLayer+1) % (data?.n_layers||1)), 500);
  else clearInterval(playTimer);
});
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowRight') setLayer(Math.min(currentLayer+1, (data?.n_layers||1)-1));
  if (e.key === 'ArrowLeft') setLayer(Math.max(currentLayer-1, 0));
  if (e.key === ' ') { e.preventDefault(); document.getElementById('btn-play').click(); }
});

// --- Render loop ---
(function animate() {
  requestAnimationFrame(animate);
  if (currentView === 'cloud') { controls.update(); renderer.render(scene, camera); }
})();

init();

// --- Live sync via WebSocket ---
let ws;
function connectWS() {
  ws = new WebSocket('ws://' + location.host + '/ws');
  ws.onmessage = (e) => {
    const event = JSON.parse(e.data);
    if (event.type === 'signals') {
      const msg = event.analysis + ' on ' + event.model + ' (' + event.count + ' signals)';
      document.getElementById('status-text').textContent = 'Live: ' + msg;
      document.getElementById('status-text').style.color = '#44FF44';
      setTimeout(() => { document.getElementById('status-text').style.color = '#666'; }, 3000);
      // Auto-refresh depth view if visible
      if (currentView === 'depth') showDepthView();
    }
  };
  ws.onclose = () => setTimeout(connectWS, 2000);
  ws.onerror = () => ws.close();
}
connectWS();
</script>
</body>
</html>"""


class CompanionHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the companion API + viewer."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        # WebSocket upgrade
        if path == '/ws' and 'upgrade' in self.headers.get('Connection', '').lower():
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
                # Send points as binary for speed, metadata as JSON header
                proj = qs.get('proj', ['pca'])[0]
                if proj == 'umap':
                    result = _compute_umap(mri_path, n_sample=n)
                if qs.get('fmt', [None])[0] == 'bin':
                    self._send_pca_binary(result)
                else:
                    self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/pca/<model>/<mode>?n=5000"})
        elif path.startswith('/api/decomp/'):
            # /api/decomp/model/mode?layer=5
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                layer = int(qs.get('layer', ['0'])[0])
                mri_path = f"/Volumes/sharts/{model}/{mode}.mri"
                result = _load_decomp(mri_path, layer)
                if qs.get('fmt', [None])[0] == 'bin':
                    self._send_decomp_binary(result)
                else:
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
        self.end_headers()
        self.wfile.write(body)

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

    server = ThreadedHTTPServer(('127.0.0.1', port), CompanionHandler)
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
