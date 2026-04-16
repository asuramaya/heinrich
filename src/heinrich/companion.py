"""Heinrich companion — live viewer and command runner.

HTTP server that serves the viewer frontend and provides API endpoints
for querying MRI data, running analysis commands, and reading signals
from the DB. Replaces viz.py for the MRI workflow.

Usage:
    heinrich companion                  # http://localhost:8377
    heinrich companion --port 9000      # custom port
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any

import numpy as np


# --- In-memory caches ---

_ui_html_cache: str | None = None
_decomp_meta_cache: dict[str, dict] = {}
_weight_align_cache: dict[str, list] = {}


# --- Capture relay: MCP → server → browser poll → result ---
_capture_pending: dict[str, threading.Event] = {}
_capture_results: dict[str, dict] = {}
_capture_lock = threading.Lock()
_CAPTURE_DIR = Path("/tmp/heinrich/captures")

# --- Long-poll command queue ---
_poll_commands: list[dict] = []  # pending commands for the browser
_poll_event = threading.Event()  # signals when a new command is available
_poll_lock = threading.Lock()


def _poll_push(cmd: dict):
    """Push a command for the browser to pick up."""
    with _poll_lock:
        _poll_commands.append(cmd)
    _poll_event.set()


def _poll_take(timeout: float = 30.0) -> dict | None:
    """Block until a command is available or timeout."""
    _poll_event.wait(timeout=timeout)
    with _poll_lock:
        if _poll_commands:
            cmd = _poll_commands.pop(0)
            if not _poll_commands:
                _poll_event.clear()
            return cmd
    _poll_event.clear()
    return None


def notify_companions(event: dict):
    """Push an event to all connected companion browsers.

    Called from emit_signals() when new signals are written to the DB.
    Thread-safe.
    """
    _poll_push(event)


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
        from sklearn.utils.extmath import randomized_svd
        U, S, _ = randomized_svd(centered, n_components=min(10, centered.shape[1]),
                                  random_state=42)
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
    """Load decomposition metadata for a layer. No scores (use binary blob)."""
    p = Path(mri_path) / "decomp"
    var_path = p / f"L{layer:02d}_variance.npy"
    if not var_path.exists():
        return {"error": f"No decomposition at L{layer}. Run mri-decompose."}
    variance = np.load(str(var_path)).astype(np.float32)
    meta_mri = json.loads((Path(mri_path) / "metadata.json").read_text())
    # allow_pickle needed for variable-length string arrays stored by numpy
    tok = dict(np.load(str(Path(mri_path) / "tokens.npz"), allow_pickle=True))
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
    scores_path = p / f"L{layer:02d}_scores.npy"
    n_components = len(variance)
    if scores_path.exists():
        s = np.load(str(scores_path), mmap_mode='r')
        n_components = s.shape[1]
    lm = meta_decomp["layers"][layer] if layer < len(meta_decomp["layers"]) else {}
    return {
        "model": meta_mri["model"]["name"],
        "model_dir": Path(mri_path).parent.name,
        "mode": meta_mri.get("capture", {}).get("mode", "?"),
        "layer": layer,
        "n_tokens": n_sample,
        "n_components": n_components,
        "variance": variance.tolist(),
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
    mri_meta = json.loads((Path(mri_path) / "metadata.json").read_text())
    meta["mode"] = mri_meta.get("capture", {}).get("mode", "?")
    meta["intermediate_size"] = mri_meta.get("capture", {}).get("intermediate_size", 0)
    _decomp_meta_cache[mri_path] = meta
    return meta


# --- Score cache: keeps last 3 layers' float32 arrays in RAM ---
# First load from USB is ~4s, cached access is ~5ms.
_score_cache: dict[str, np.ndarray] = {}  # key → float32 array
_score_cache_order: list[str] = []
_SCORE_CACHE_MAX = 3


def _get_scores_f32(mri_path: str, layer: int) -> np.ndarray | None:
    """Get float32 score array for a layer, cached in RAM."""
    key = f"{mri_path}:L{layer:02d}"
    if key in _score_cache:
        return _score_cache[key]
    score_path = Path(mri_path) / "decomp" / f"L{layer:02d}_scores.npy"
    if not score_path.exists():
        return None
    arr = np.load(str(score_path)).astype(np.float32)
    _score_cache[key] = arr
    _score_cache_order.append(key)
    while len(_score_cache_order) > _SCORE_CACHE_MAX:
        evict = _score_cache_order.pop(0)
        _score_cache.pop(evict, None)
    return arr


def _direction_quality(mri_path: str, a: int, b: int, layer: int) -> dict:
    """Full-K direction analysis between two tokens at a layer.

    Returns magnitude, concentration, bimodality, explained variance,
    and top tokens per side — all computed in the full hidden dimension.
    """
    decomp = Path(mri_path) / "decomp"
    score_path = decomp / f"L{layer:02d}_scores.npy"
    if not score_path.exists():
        return {"error": f"No scores at L{layer}"}

    tok = dict(np.load(str(Path(mri_path) / "tokens.npz"), allow_pickle=True))
    texts = [str(t) for t in tok["token_texts"]]
    scripts_arr = [str(s) for s in tok["scripts"]]

    scores = np.load(str(score_path), mmap_mode="r")
    N, K = scores.shape

    if a >= N or b >= N:
        return {"error": f"Token index out of range (max {N-1})"}

    # Direction vector (full K)
    diff = scores[a].astype(np.float32) - scores[b].astype(np.float32)
    mag = float(np.linalg.norm(diff))
    direction = diff / (mag + 1e-8)

    # Concentration: how many PCs for 50%, 80%, 95%
    diff2 = diff ** 2
    total_d2 = float(diff2.sum())
    order = np.argsort(diff2)[::-1]
    cumul = np.cumsum(diff2[order]) / (total_d2 + 1e-8)
    pcs_50 = int(np.searchsorted(cumul, 0.5)) + 1
    pcs_80 = int(np.searchsorted(cumul, 0.8)) + 1
    pcs_95 = int(np.searchsorted(cumul, 0.95)) + 1
    top_pcs = [{"pc": int(order[i]),
                "share": float(diff2[order[i]] / (total_d2 + 1e-8)),
                "delta": float(diff[order[i]])}
               for i in range(min(5, len(order)))]

    # Project all tokens onto direction
    proj = scores @ direction

    # Explained variance: var(projection) / total layer variance
    proj_var = float(np.var(proj))
    total_var = float(np.sum(np.var(scores, axis=0)))
    explained = proj_var / (total_var + 1e-8)

    # Bimodality: histogram valley test
    hist, edges = np.histogram(proj, bins=100)
    mid = len(hist) // 2
    left_peak = float(hist[:mid].max())
    right_peak = float(hist[mid:].max())
    valley = float(hist[max(0, mid - 5):mid + 5].min())
    bimodal_ratio = valley / (min(left_peak, right_peak) + 1e-8)

    # Top tokens per side
    top_a_idx = np.argsort(proj)[-10:][::-1]
    top_b_idx = np.argsort(proj)[:10]
    top_a_list = [{"idx": int(i), "text": texts[i], "script": scripts_arr[i],
                   "proj": float(proj[i])} for i in top_a_idx]
    top_b_list = [{"idx": int(i), "text": texts[i], "script": scripts_arr[i],
                   "proj": float(proj[i])} for i in top_b_idx]

    return {
        "layer": layer, "K": K, "N": N,
        "token_a": {"idx": a, "text": texts[a], "script": scripts_arr[a]},
        "token_b": {"idx": b, "text": texts[b], "script": scripts_arr[b]},
        "magnitude": mag,
        "concentration": {"pcs_50": pcs_50, "pcs_80": pcs_80, "pcs_95": pcs_95},
        "top_pcs": top_pcs,
        "explained_variance": explained,
        "bimodality": bimodal_ratio,
        "top_a": top_a_list,
        "top_b": top_b_list,
    }


def _direction_project(mri_path: str, a: int, b: int, layer: int) -> dict:
    """Project all tokens onto the full-K direction between two tokens.

    Returns normalized projections [-1, +1] centered on the A-B midpoint,
    suitable for direct use as color values in the browser.
    """
    decomp = Path(mri_path) / "decomp"
    scores = _get_scores_f32(mri_path, layer)
    if scores is None:
        return {"error": f"No scores at L{layer}"}
    N, K = scores.shape

    if a >= N or b >= N:
        return {"error": f"Token index out of range (max {N-1})"}

    diff = scores[a] - scores[b]
    mag = float(np.linalg.norm(diff))
    direction = diff / (mag + 1e-8)

    proj = (scores @ direction).tolist()

    return {"projections": proj, "magnitude": mag, "K": K,
            "proj_a": proj[a], "proj_b": proj[b]}


def _direction_nonlinear(mri_path: str, a: int, b: int, layer: int,
                         n_sample: int = 2000) -> dict:
    """Test whether a direction is linear or nonlinear.

    Splits tokens into A-side and B-side by projection sign.
    Compares linear probe accuracy vs k-NN accuracy.
    Large gap (knn >> linear) means the concept boundary is curved.
    """
    decomp = Path(mri_path) / "decomp"
    score_path = decomp / f"L{layer:02d}_scores.npy"
    if not score_path.exists():
        return {"error": f"No scores at L{layer}"}

    scores = np.load(str(score_path), mmap_mode="r")
    N, K = scores.shape

    if a >= N or b >= N:
        return {"error": f"Token index out of range (max {N-1})"}

    # Compute direction and project
    diff = scores[a].astype(np.float32) - scores[b].astype(np.float32)
    mag = float(np.linalg.norm(diff))
    direction = diff / (mag + 1e-8)
    proj = scores @ direction

    # Labels: 1 = A-side (positive projection), 0 = B-side
    pa, pb = float(proj[a]), float(proj[b])
    mid = (pa + pb) / 2
    labels = (proj > mid).astype(np.int32)

    # Sample for speed
    rng = np.random.RandomState(42)
    n_sample = min(n_sample, N)
    idx = rng.choice(N, n_sample, replace=False)
    X = scores[idx].astype(np.float32)
    y = labels[idx]

    # Skip if too imbalanced
    pos = y.sum()
    if pos < 10 or (n_sample - pos) < 10:
        return {"linear_acc": 0.5, "knn_acc": 0.5, "gap": 0.0,
                "verdict": "too_few_samples", "n_sample": n_sample}

    # Train/test split
    n_train = int(n_sample * 0.7)
    perm = rng.permutation(n_sample)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Linear probe: project onto direction, threshold at midpoint
    proj_test = X_test @ direction
    linear_pred = (proj_test > mid).astype(np.int32)
    linear_acc = float((linear_pred == y_test).mean())

    # k-NN: 5-nearest-neighbors in full K space
    from numpy.linalg import norm
    k = 5
    knn_correct = 0
    for i in range(len(X_test)):
        dists = norm(X_train - X_test[i], axis=1)
        nn_idx = np.argpartition(dists, k)[:k]
        nn_labels = y_train[nn_idx]
        pred = 1 if nn_labels.sum() > k / 2 else 0
        if pred == y_test[i]:
            knn_correct += 1
    knn_acc = knn_correct / len(X_test)

    gap = knn_acc - linear_acc
    if gap > 0.1:
        verdict = "nonlinear"
    elif gap > 0.03:
        verdict = "slightly_nonlinear"
    else:
        verdict = "linear"

    return {"linear_acc": linear_acc, "knn_acc": knn_acc,
            "gap": gap, "verdict": verdict, "n_sample": n_sample, "K": K}


def _direction_depth(mri_path: str, a: int, b: int) -> dict:
    """Compute full-K direction strength at every layer.

    Returns per-layer: magnitude, concentration (pcs_50), token A and B
    projections, and population percentiles (10th, 50th, 90th).
    """
    decomp = Path(mri_path) / "decomp"
    meta_path = decomp / "meta.json"
    if not meta_path.exists():
        return {"error": "No decomp/meta.json"}
    meta = json.loads(meta_path.read_text())
    n_layers = len(meta["layers"])

    layers = []
    for li in range(n_layers):
        scores = _get_scores_f32(mri_path, li)
        if scores is None:
            continue
        N, K = scores.shape
        if a >= N or b >= N:
            continue

        diff = scores[a] - scores[b]
        mag = float(np.linalg.norm(diff))
        direction = diff / (mag + 1e-8)

        pa = float(scores[a] @ direction)
        pb = float(scores[b] @ direction)

        step = max(1, N // 500)
        sample_proj = scores[::step] @ direction
        p10 = float(np.percentile(sample_proj, 10))
        p50 = float(np.percentile(sample_proj, 50))
        p90 = float(np.percentile(sample_proj, 90))

        # Concentration: how many PCs carry 50% of squared difference
        diff2 = diff ** 2
        total_d2 = float(diff2.sum())
        order = np.argsort(diff2)[::-1]
        cumul = np.cumsum(diff2[order]) / (total_d2 + 1e-8)
        pcs_50 = int(np.searchsorted(cumul, 0.5)) + 1

        # Top 8 PCs at this layer
        top_n = min(8, len(order))
        top_pcs = [{"pc": int(order[i]),
                    "share": float(diff2[order[i]] / (total_d2 + 1e-8))}
                   for i in range(top_n)]

        layers.append({
            "layer": li, "magnitude": mag,
            "proj_a": pa, "proj_b": pb,
            "p10": p10, "p50": p50, "p90": p90,
            "pcs_50": pcs_50, "top_pcs": top_pcs,
        })

    return {"layers": layers, "n_layers": n_layers}


def _direction_circuit(mri_path: str, a: int, b: int) -> dict:
    """Per-head and MLP write-attribution for the A-B concept direction.

    For each layer, measures how much of the concept direction each
    attention head's output subspace can reconstruct (via o_proj slices),
    plus the MLP's contribution (via down_proj).

    This is Sharkey et al. 2025 open problem 2.3: automatic circuit
    discovery from concept directions.
    """
    mri_dir = Path(mri_path)
    decomp = mri_dir / "decomp"
    meta_path = mri_dir / "metadata.json"
    decomp_meta_path = decomp / "meta.json"

    if not meta_path.exists():
        return {"error": "No metadata.json"}
    if not decomp_meta_path.exists():
        return {"error": "No decomp/meta.json"}

    mri_meta = json.loads(meta_path.read_text())
    decomp_meta = json.loads(decomp_meta_path.read_text())

    model_info = mri_meta.get("model", {})
    n_heads = model_info.get("n_heads")
    hidden_size = model_info.get("hidden_size")
    if not n_heads or not hidden_size:
        return {"error": "metadata.json missing n_heads or hidden_size"}

    head_dim = hidden_size // n_heads
    n_layers = len(decomp_meta.get("layers", []))

    layers_out = []
    for li in range(n_layers):
        scores = _get_scores_f32(mri_path, li)
        if scores is None:
            continue
        N, K = scores.shape
        if a >= N or b >= N:
            continue

        # Direction in PC space
        diff = scores[a] - scores[b]
        mag = float(np.linalg.norm(diff))
        if mag < 1e-12:
            continue
        dir_pc = diff / mag

        # Map to hidden space via PCA components
        comp_path = decomp / f"L{li:02d}_components.npy"
        if not comp_path.exists():
            continue
        components = np.load(str(comp_path))  # [K, hidden]
        dir_hidden = components.T @ dir_pc[:components.shape[0]]
        dir_norm = float(np.linalg.norm(dir_hidden))
        if dir_norm < 1e-12:
            continue
        dir_hidden = dir_hidden / dir_norm

        # Per-head attribution via o_proj
        o_proj_path = mri_dir / "weights" / f"L{li:02d}" / "o_proj.npy"
        heads_out = []
        if o_proj_path.exists():
            o_proj = np.load(str(o_proj_path))  # [hidden, hidden]
            for h in range(n_heads):
                W_h = o_proj[:, h * head_dim:(h + 1) * head_dim]  # [hidden, head_dim]
                proj = W_h @ (W_h.T @ dir_hidden)
                attrib = float(np.linalg.norm(proj))
                heads_out.append({"head": h, "attrib": attrib})
        else:
            heads_out = [{"head": h, "attrib": 0.0} for h in range(n_heads)]

        # MLP attribution via down_proj
        mlp_attrib = 0.0
        down_proj_path = mri_dir / "weights" / f"L{li:02d}" / "down_proj.npy"
        if down_proj_path.exists():
            down_proj = np.load(str(down_proj_path))  # [hidden, intermediate]
            proj_mlp = down_proj @ (down_proj.T @ dir_hidden)
            mlp_attrib = float(np.linalg.norm(proj_mlp))

        # Normalize: scale so max(heads + mlp) = 1 for readability
        all_attribs = [hd["attrib"] for hd in heads_out] + [mlp_attrib]
        max_attrib = max(all_attribs) if all_attribs else 1.0
        if max_attrib > 1e-12:
            for hd in heads_out:
                hd["attrib"] = hd["attrib"] / max_attrib
            mlp_attrib = mlp_attrib / max_attrib

        layers_out.append({
            "layer": li,
            "heads": heads_out,
            "mlp_attrib": mlp_attrib,
        })

    return {"layers": layers_out, "n_heads": n_heads, "n_layers": n_layers}


def _direction_weight_alignment(mri_path: str, a: int, b: int) -> dict:
    """Compute how each weight matrix at each layer aligns with the concept direction in hidden space.

    For each layer, maps the A-B concept direction from PC space into the
    full hidden space, then measures each weight matrix's ability to access
    that direction:  alignment = ||W @ (W^T @ dir)|| / ||dir||

    This is Sharkey et al. 2025 open problem 2.1.1: bridging PCA-space
    and weight-space representations.
    """
    mri_dir = Path(mri_path)
    decomp = mri_dir / "decomp"
    meta_path = mri_dir / "metadata.json"
    decomp_meta_path = decomp / "meta.json"

    if not meta_path.exists():
        return {"error": "No metadata.json"}
    if not decomp_meta_path.exists():
        return {"error": "No decomp/meta.json"}

    decomp_meta = json.loads(decomp_meta_path.read_text())
    n_layers = len(decomp_meta.get("layers", []))

    MATRICES = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

    layers_out = []
    for li in range(n_layers):
        scores = _get_scores_f32(mri_path, li)
        if scores is None:
            continue
        N, K = scores.shape
        if a >= N or b >= N:
            continue

        # Direction in PC space -> hidden space via PCA components
        diff = scores[a] - scores[b]
        mag = float(np.linalg.norm(diff))
        if mag < 1e-12:
            continue
        dir_pc = diff / mag

        comp_path = decomp / f"L{li:02d}_components.npy"
        if not comp_path.exists():
            continue
        components = np.load(str(comp_path))  # [K, hidden]
        dir_hidden = (components.T @ dir_pc[:components.shape[0]]).astype(np.float32)
        dir_norm = float(np.linalg.norm(dir_hidden))
        if dir_norm < 1e-12:
            continue
        dir_hidden = dir_hidden / dir_norm

        weights_dir = mri_dir / "weights" / f"L{li:02d}"
        matrices = {}
        for mname in MATRICES:
            wp = weights_dir / f"{mname}.npy"
            if not wp.exists():
                continue
            W = np.load(str(wp), mmap_mode='r').astype(np.float32)
            # W shape: [out_features, in_features].
            # Alignment = how much of the direction W's column space can reconstruct.
            # Case 1: in_features == hidden (e.g. q/k/v_proj: W projects FROM hidden)
            #   → W^T @ dir lives in out-space, W @ (W^T @ dir) back in hidden.
            # Case 2: out_features == hidden (e.g. down_proj: W projects TO hidden)
            #   → same formula works: W^T @ dir goes to in-space, W @ that returns.
            if W.shape[1] == len(dir_hidden):
                proj = W @ (W.T @ dir_hidden)
                alignment = float(np.linalg.norm(proj))
            elif W.shape[0] == len(dir_hidden):
                proj = W @ (W.T @ dir_hidden)
                alignment = float(np.linalg.norm(proj))
            else:
                alignment = 0.0
            matrices[mname] = alignment

        layers_out.append({"layer": li, "matrices": matrices})

    return {"layers": layers_out, "n_layers": n_layers, "matrix_names": MATRICES}


def _auto_discover_directions(mri_path: str, layer: int, n_top: int = 20) -> dict:
    """Automated direction discovery: find bimodal PCA components.

    Each PC is already a direction in hidden space.  The scores column IS
    the projection of all tokens onto that direction.  PCs with bimodal
    score distributions correspond to real features the model uses ---
    features humans haven't labeled.

    Addresses open problem 3.5 (Microscope AI) from Sharkey et al. 2025.
    """
    scores = _get_scores_f32(mri_path, layer)
    if scores is None:
        return {"error": f"No scores at L{layer}"}
    N, K = scores.shape

    # Load token texts and scripts
    tok_path = Path(mri_path) / "tokens.npz"
    if not tok_path.exists():
        return {"error": "No tokens.npz"}
    tok = dict(np.load(str(tok_path), allow_pickle=True))
    texts = [str(t) for t in tok["token_texts"]]
    scripts_arr = [str(s) for s in tok["scripts"]]

    # Load variance data for variance_pct
    var_path = Path(mri_path) / "decomp" / f"L{layer:02d}_variance.npy"
    if var_path.exists():
        variance = np.load(str(var_path)).astype(np.float32)
        total_var = float(variance.sum()) if variance.sum() > 0 else 1.0
    else:
        variance = None
        total_var = 1.0

    # Compute bimodality for each PC
    pc_bimodality = []
    for pc in range(K):
        proj = scores[:, pc]
        hist, _ = np.histogram(proj, bins=100)
        mid = len(hist) // 2
        left_peak = float(hist[:mid].max())
        right_peak = float(hist[mid:].max())
        valley = float(hist[max(0, mid - 5):mid + 5].min())
        bimodal = valley / (min(left_peak, right_peak) + 1e-8)
        pc_bimodality.append((pc, bimodal))

    # Rank by bimodality (lowest = most bimodal = strongest features)
    pc_bimodality.sort(key=lambda x: x[1])

    # Build results for top-N
    discovered = []
    for pc, bimodal in pc_bimodality[:n_top]:
        proj = scores[:, pc]
        top_pos = np.argsort(proj)[-5:][::-1]
        top_neg = np.argsort(proj)[:5]

        var_pct = float(variance[pc] / total_var) if variance is not None and pc < len(variance) else 0.0

        discovered.append({
            "pc": int(pc),
            "bimodality": float(bimodal),
            "variance_pct": var_pct,
            "top_pos": [{"idx": int(i), "text": texts[i], "script": scripts_arr[i],
                         "proj": float(proj[i])} for i in top_pos],
            "top_neg": [{"idx": int(i), "text": texts[i], "script": scripts_arr[i],
                         "proj": float(proj[i])} for i in top_neg],
        })

    return {"layer": int(layer), "discovered": discovered}


def _token_predicts(mri_path: str, token_idx: int, layer: int, top_k: int = 20) -> dict:
    """What does the model predict from this token at this layer?

    Projects the token's exit state through lmhead to get the prediction
    distribution. Returns top-K predicted tokens with logits.
    This is the logit lens for a single token — answers "what does the
    model know about this token at this depth?"
    """
    mri_dir = Path(mri_path)
    exit_path = mri_dir / f"L{layer:02d}_exit.npy"
    lmhead_path = mri_dir / "lmhead.npy"

    if not exit_path.exists():
        return {"error": f"No exit states at L{layer}"}
    if not lmhead_path.exists():
        return {"error": "No lmhead matrix"}

    # Load exit state for this token
    exits = np.load(str(exit_path), mmap_mode="r")
    if token_idx >= exits.shape[0]:
        return {"error": f"Token index {token_idx} out of range (max {exits.shape[0]-1})"}

    exit_vec = exits[token_idx].astype(np.float32)

    # Load lmhead (cached after first access)
    cache_key = f"{mri_path}:lmhead"
    if cache_key not in _score_cache:
        _score_cache[cache_key] = np.load(str(lmhead_path)).astype(np.float32)
    lmhead = _score_cache[cache_key]

    # Project: logits = lmhead @ exit_vec
    logits = lmhead @ exit_vec

    # Softmax for probabilities (numerically stable)
    logits_shifted = logits - logits.max()
    probs = np.exp(logits_shifted)
    probs = probs / probs.sum()

    # Top-K
    top_indices = np.argsort(logits)[-top_k:][::-1]

    # Map vocab indices to token texts via token_ids
    tok = dict(np.load(str(mri_dir / "tokens.npz"), allow_pickle=True))
    texts = [str(t) for t in tok["token_texts"]]
    token_ids = tok["token_ids"]
    # Build reverse map: vocab_id → mri_index
    vocab_to_mri = {}
    for i, vid in enumerate(token_ids):
        vocab_to_mri[int(vid)] = i

    predictions = []
    for vi in top_indices:
        vi = int(vi)
        mri_idx = vocab_to_mri.get(vi, -1)
        text = texts[mri_idx] if mri_idx >= 0 else f"[vocab:{vi}]"
        predictions.append({
            "vocab_id": vi,
            "mri_idx": mri_idx,
            "text": text,
            "logit": float(logits[vi]),
            "prob": float(probs[vi]),
        })

    return {
        "token_idx": token_idx,
        "token_text": texts[token_idx] if token_idx < len(texts) else "?",
        "layer": layer,
        "top_k": predictions,
    }


_steer_backend_cache = {}

def _get_steer_backend(model_id: str):
    """Get or create a model backend for steering tests."""
    if model_id not in _steer_backend_cache:
        from .cartography.runtime import get_backend
        _steer_backend_cache[model_id] = get_backend(model_id)
    return _steer_backend_cache[model_id]


def _direction_steer_test(mri_path: str, a: int, b: int, layer: int,
                          prompt: str, alpha: float = 2.0,
                          max_tokens: int = 30,
                          model_id: str = "") -> dict:
    """Test if steering along the A->B direction changes model output.

    Generates text with and without steering, compares outputs.
    If output changes meaningfully, the direction is causal.
    """
    decomp = Path(mri_path) / "decomp"
    score_path = decomp / f"L{layer:02d}_scores.npy"
    if not score_path.exists():
        return {"error": f"No scores at L{layer}"}

    # Load direction from full-K components
    comp_path = decomp / f"L{layer:02d}_components.npy"
    if not comp_path.exists():
        return {"error": f"No components at L{layer} — needed to map PC direction to hidden space"}

    scores = np.load(str(score_path), mmap_mode="r")
    components = np.load(str(comp_path))  # [K, hidden_dim]
    N, K = scores.shape

    # Direction in PC space
    diff = scores[a].astype(np.float32) - scores[b].astype(np.float32)
    mag = float(np.linalg.norm(diff))
    dir_pc = diff / (mag + 1e-8)

    # Map to hidden space: direction_hidden = components.T @ dir_pc
    direction = (components.T @ dir_pc[:components.shape[0]]).astype(np.float32)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # Get backend
    if not model_id:
        meta = json.loads((Path(mri_path) / "metadata.json").read_text())
        model_id = meta["model"]["name"]

    try:
        backend = _get_steer_backend(model_id)
    except Exception as e:
        return {"error": f"Cannot load model {model_id}: {e}"}

    # Generate clean
    clean = backend.generate(prompt, max_tokens=max_tokens)

    # Generate steered (A->B direction, positive alpha)
    steer_dirs = {layer: (direction, alpha)}
    steered_pos = backend.generate(prompt, steer_dirs=steer_dirs, max_tokens=max_tokens)

    # Simple change metric: character-level edit distance ratio
    def _change_ratio(a_text, b_text):
        if not a_text and not b_text:
            return 0.0
        common = sum(c1 == c2 for c1, c2 in zip(a_text, b_text))
        return 1.0 - common / max(len(a_text), len(b_text), 1)

    change_pos = _change_ratio(clean, steered_pos)

    return {
        "prompt": prompt, "layer": layer, "alpha": alpha,
        "clean": clean,
        "steered": steered_pos,
        "change": change_pos,
        "changed": change_pos > 0.2,
    }


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


# Cache mmap'd gate/up arrays per MRI path — avoids 60 file opens per request
_mlp_mmap_cache: dict[str, tuple] = {}  # mri_path → (n_layers, intermediate, gates[], ups[])
_NEURON_CACHE_MAX = 2000  # ~180MB at 90KB/entry
_neuron_result_cache: dict[str, bytes] = {}  # "mri_path:token_idx" → result bytes

def _get_mlp_mmaps(mri_path: str):
    if mri_path in _mlp_mmap_cache:
        return _mlp_mmap_cache[mri_path]
    mri_dir = Path(mri_path)
    meta_path = mri_dir / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    n_layers = meta['model']['n_layers']
    intermediate = meta['capture'].get('intermediate_size', 0)
    if not intermediate:
        return None
    mlp_dir = mri_dir / "mlp"
    if not mlp_dir.exists():
        return None
    gates, ups = [], []
    for li in range(n_layers):
        gp = mlp_dir / f"L{li:02d}_gate.npy"
        up = mlp_dir / f"L{li:02d}_up.npy"
        gates.append(np.load(str(gp), mmap_mode='r') if gp.exists() else None)
        ups.append(np.load(str(up), mmap_mode='r') if up.exists() else None)
    entry = (n_layers, intermediate, gates, ups)
    _mlp_mmap_cache[mri_path] = entry
    return entry


def _neuron_field(mri_path: str, token_idx: int) -> bytes | dict:
    """Full gate×up activation for one token across all layers.

    Returns raw bytes: float16 array [n_layers × intermediate_size], row-major.
    ~92KB for 30 layers × 1536 neurons.

    Uses transposed index (token_neurons.bin) for single-seek O(1) reads.
    Falls back to per-layer mmaps if index doesn't exist.
    """
    cache_key = f"{mri_path}:{token_idx}"
    if cache_key in _neuron_result_cache:
        return _neuron_result_cache[cache_key]

    # Fast path: transposed index
    tok_idx_path = Path(mri_path) / "decomp" / "token_neurons.bin"
    if tok_idx_path.exists():
        import struct as _st
        with open(tok_idx_path, 'rb') as f:
            hdr = f.read(16)
            magic, n_tok, n_layers, intermediate = _st.unpack('<4sIII', hdr)
            if magic == b'TOKN' and token_idx < n_tok:
                stride = n_layers * intermediate * 2
                f.seek(16 + token_idx * stride)
                data = f.read(stride)
                if len(_neuron_result_cache) >= _NEURON_CACHE_MAX:
                    keys = list(_neuron_result_cache.keys())
                    for k in keys[:len(keys) // 4]:
                        del _neuron_result_cache[k]
                _neuron_result_cache[cache_key] = data
                return data

    # Slow path: gather from per-layer mmaps
    entry = _get_mlp_mmaps(mri_path)
    if not entry:
        return {"error": "No MLP data"}
    n_layers, intermediate, gates, ups = entry
    result = np.zeros((n_layers, intermediate), dtype=np.float16)
    for li in range(n_layers):
        g = gates[li]
        if g is None:
            continue
        if token_idx >= g.shape[0]:
            return {"error": f"Token {token_idx} out of range (max {g.shape[0]-1})"}
        gt = g[token_idx].astype(np.float32)
        u = ups[li]
        if u is not None:
            result[li] = (gt * u[token_idx].astype(np.float32)).astype(np.float16)
        else:
            result[li] = gt.astype(np.float16)
    data = result.tobytes()
    if len(_neuron_result_cache) >= _NEURON_CACHE_MAX:
        keys = list(_neuron_result_cache.keys())
        for k in keys[:len(keys) // 4]:
            del _neuron_result_cache[k]
    _neuron_result_cache[cache_key] = data
    return data


# Cache decomp score mmaps: mri_path → {layer_idx: np.memmap}
_decomp_score_cache: dict[str, dict] = {}

def _get_score_mmap(mri_path: str, layer: int):
    """Get mmap'd score array for a layer, cached."""
    if mri_path not in _decomp_score_cache:
        _decomp_score_cache[mri_path] = {}
    cache = _decomp_score_cache[mri_path]
    if layer in cache:
        return cache[layer]
    decomp = Path(mri_path) / "decomp"
    meta_path = decomp / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    n_real = meta.get("n_real_layers", meta["n_layers"])
    if layer < n_real:
        sp = decomp / f"L{layer:02d}_scores.npy"
    elif layer == n_real:
        sp = decomp / "emb_scores.npy"
    elif layer == n_real + 1:
        sp = decomp / "lmh_scores.npy"
    else:
        return None
    if not sp.exists():
        return None
    arr = np.load(str(sp), mmap_mode='r')
    cache[layer] = arr
    return arr


def _pc_column_cached(mri_path: str, pc: int, layer: int) -> bytes | dict:
    """One PC's scores for all tokens at one layer. Cached mmaps = instant after first hit."""
    arr = _get_score_mmap(mri_path, layer)
    if arr is None:
        return {"error": f"No scores for layer {layer}"}
    if pc >= arr.shape[1]:
        return {"error": f"PC {pc} out of range (max {arr.shape[1]-1})"}
    return arr[:, pc].astype(np.float16).tobytes()


def _token_hover(mri_path: str, token: int, layer: int) -> bytes | dict:
    """Combined hover data: PC scores + neuron activations for one token at one layer.

    Returns: [4B K][4B inter][4B layer] [float16 × K] [float16 × inter]
    ~4KB total. One fetch feeds both spectrum and neuron viewports.
    """
    import struct
    # PC scores
    score_arr = _get_score_mmap(mri_path, layer)
    if score_arr is None:
        return {"error": f"No scores for layer {layer}"}
    if token >= score_arr.shape[0]:
        return {"error": f"Token {token} out of range (max {score_arr.shape[0]-1})"}
    K = score_arr.shape[1]
    pc_row = score_arr[token].astype(np.float16)

    # Neuron activations (gate × up at this layer)
    entry = _get_mlp_mmaps(mri_path)
    inter = 0
    neuron_row = np.array([], dtype=np.float16)
    if entry:
        n_layers, intermediate, gates, ups = entry
        inter = intermediate
        if layer < n_layers and gates[layer] is not None and token < gates[layer].shape[0]:
            gt = gates[layer][token].astype(np.float32)
            if ups[layer] is not None:
                neuron_row = (gt * ups[layer][token].astype(np.float32)).astype(np.float16)
            else:
                neuron_row = gt.astype(np.float16)
        else:
            neuron_row = np.zeros(intermediate, dtype=np.float16)

    header = struct.pack('<III', K, inter, layer)
    return header + pc_row.tobytes() + neuron_row.tobytes()


def _token_layer_scores(mri_path: str, token: int, layer: int) -> bytes | dict:
    """All PCs for one token at one layer. ~1KB, instant via mmap."""
    arr = _get_score_mmap(mri_path, layer)
    if arr is None:
        return {"error": f"No scores for layer {layer}"}
    if token >= arr.shape[0]:
        return {"error": f"Token {token} out of range (max {arr.shape[0]-1})"}
    import struct
    row = arr[token].astype(np.float16)
    header = struct.pack('<II', arr.shape[1], layer)  # K, layer
    return header + row.tobytes()


def _pc_full(mri_path: str, pc: int) -> bytes | dict:
    """One PC, all tokens, all layers. Returns float16 [nLayers × nTokens].
    ~3.1MB for SmolLM2-135M (32 layers × 48660 tokens × 2 bytes).

    Uses PC-major index (pc_scores.bin) for single-seek O(1) reads.
    Falls back to per-layer mmaps if index doesn't exist.
    """
    decomp = Path(mri_path) / "decomp"

    # Fast path: PC-major index — one seek, one sequential read
    pc_idx_path = decomp / "pc_scores.bin"
    if pc_idx_path.exists():
        import struct as _st
        with open(pc_idx_path, 'rb') as f:
            hdr = f.read(16)
            magic, n_layers, n_tok, full_k = _st.unpack('<4sIII', hdr)
            if magic == b'PCSC' and pc < full_k:
                stride = n_layers * n_tok * 2
                f.seek(16 + pc * stride)
                result = np.frombuffer(f.read(stride), dtype=np.float16).reshape(n_layers, n_tok)
                header = _st.pack('<III', n_layers, n_tok, pc)
                return header + result.tobytes()
            elif magic == b'PCSC' and pc >= full_k:
                return {"error": f"PC {pc} out of range (max {full_k-1})"}

    # Slow path: gather from per-layer mmaps (1.5M page faults on USB)
    meta_path = decomp / "meta.json"
    if not meta_path.exists():
        return {"error": "No decomposition"}
    meta = json.loads(meta_path.read_text())
    n_real = meta.get("n_real_layers", meta["n_layers"])
    n_total = n_real + 2
    first = _get_score_mmap(mri_path, 0)
    if first is None:
        return {"error": "No score data"}
    n_tok = first.shape[0]
    max_pc = first.shape[1]
    if pc >= max_pc:
        return {"error": f"PC {pc} out of range (max {max_pc-1})"}
    result = np.zeros((n_total, n_tok), dtype=np.float16)
    for li in range(n_total):
        arr = _get_score_mmap(mri_path, li)
        if arr is not None and pc < arr.shape[1]:
            result[li] = arr[:, pc].astype(np.float16)
    import struct
    header = struct.pack('<III', n_total, n_tok, pc)
    return header + result.tobytes()


def _pc_medium(mri_path: str, pcs: list[int], step: int = 10) -> bytes | dict:
    """Multiple PCs, sampled tokens, all layers.
    ~312KB per PC for SmolLM2-135M at step=10 (32 layers × 4866 tokens × 2 bytes).
    """
    decomp = Path(mri_path) / "decomp"
    meta_path = decomp / "meta.json"
    if not meta_path.exists():
        return {"error": "No decomposition"}
    meta = json.loads(meta_path.read_text())
    n_real = meta.get("n_real_layers", meta["n_layers"])
    n_total = n_real + 2
    first = _get_score_mmap(mri_path, 0)
    if first is None:
        return {"error": "No score data"}
    n_tok = first.shape[0]
    max_pc = first.shape[1]
    n_sample = (n_tok + step - 1) // step
    n_pcs = len(pcs)
    result = np.zeros((n_pcs, n_total, n_sample), dtype=np.float16)
    for pi, pc in enumerate(pcs):
        if pc >= max_pc:
            continue
        for li in range(n_total):
            arr = _get_score_mmap(mri_path, li)
            if arr is not None and pc < arr.shape[1]:
                result[pi, li] = arr[::step, pc].astype(np.float16)[:n_sample]
    import struct
    header = struct.pack('<IIIII', n_pcs, n_total, n_sample, step, n_tok)
    # Append PC indices
    pc_bytes = struct.pack(f'<{n_pcs}I', *pcs)
    return header + pc_bytes + result.tobytes()


def _token_pca_full(mri_path: str, token_idx: int) -> bytes | dict:
    """All PC scores for one token across all layers.

    Uses transposed index (token_scores.bin) for single-seek O(1) reads.
    Falls back to per-layer mmaps if index doesn't exist.
    """
    decomp = Path(mri_path) / "decomp"
    tok_idx_path = decomp / "token_scores.bin"

    # Fast path: transposed index — one seek, one read
    if tok_idx_path.exists():
        import struct as _st
        with open(tok_idx_path, 'rb') as f:
            hdr = f.read(16)
            magic, n_tok, n_layers, full_k = _st.unpack('<4sIII', hdr)
            if magic != b'TOKS':
                return {"error": f"Bad token index magic: {magic!r}"}
            if token_idx >= n_tok:
                return {"error": f"Token {token_idx} out of range (max {n_tok-1})"}
            stride = n_layers * full_k * 2  # float16
            f.seek(16 + token_idx * stride)
            row = np.frombuffer(f.read(stride), dtype=np.float16).reshape(n_layers, full_k)
        header = _st.pack('<II', n_layers, full_k)
        return header + row.astype(np.float32).tobytes()

    # Slow path: gather from per-layer mmaps
    meta_path = decomp / "meta.json"
    if not meta_path.exists():
        return {"error": "No decomposition"}
    meta = json.loads(meta_path.read_text())
    n_real = meta.get("n_real_layers", meta["n_layers"])
    n_total = n_real + 2
    full_k = 0
    for li in range(n_total):
        s = _get_score_mmap(mri_path, li)
        if s is not None:
            if token_idx >= s.shape[0]:
                return {"error": f"Token {token_idx} out of range"}
            full_k = max(full_k, s.shape[1])
    if not full_k:
        return {"error": "No score files found"}
    result = np.zeros((n_total, full_k), dtype=np.float32)
    for li in range(n_total):
        s = _get_score_mmap(mri_path, li)
        if s is not None:
            row = s[token_idx].astype(np.float32)
            result[li, :len(row)] = row
    import struct
    header = struct.pack('<II', n_total, full_k)
    return header + result.tobytes()


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

    mri_root = "/Volumes/sharts"

    def _mri_path(self, model: str, mode: str) -> str:
        return f"{self.mri_root}/{model}/{mode}.mri"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        # Long-poll: browser blocks here waiting for commands
        if path == '/api/poll':
            timeout = float(qs.get('timeout', ['30'])[0])
            cmd = _poll_take(timeout)
            if cmd:
                self._send_json(cmd)
            else:
                self._send_json({"cmd": "none"})
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
                mri_path = self._mri_path(model, mode)
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
                decomp_dir = Path(self._mri_path(model, mode)) / "decomp"
                bin_path = decomp_dir / "all_scores.bin"
                meta_path = decomp_dir / "meta.json"
                if bin_path.exists():
                    # Validate binary header matches metadata without reading full file
                    import struct
                    file_size = bin_path.stat().st_size
                    with open(bin_path, 'rb') as f:
                        header = f.read(20)
                    if len(header) >= 16 and meta_path.exists():
                        magic = header[:4]
                        if magic == b'HEI2':
                            # v2: separate var_k and score_k
                            bnL, bnN, bnK, bnVK = struct.unpack_from('<IIII', header, 4)
                            hdr_size = 20
                        elif magic in (b'HEIN', b'\x00\x00\x00\x00'):
                            # v1: shared k for variance and scores
                            bnL, bnN, bnK = struct.unpack_from('<III', header, 4)
                            bnVK = bnK
                            hdr_size = 16
                        else:
                            self._send_json({"error": f"Unknown binary format (magic: {magic!r})"})
                            return
                        meta = json.loads(meta_path.read_text())
                        expected = hdr_size + bnL * bnVK * 4 + bnL * bnN * bnK * 2
                        if file_size != expected:
                            self._send_json({"error": f"Stale binary: size {file_size} != expected {expected}. Re-run decomposition."})
                            return
                        if meta.get("n_sample") != bnN:
                            self._send_json({"error": f"Binary/meta mismatch: bin has {bnN} tokens, meta has {meta.get('n_sample')}. Re-run decomposition."})
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
                mri_path = self._mri_path(model, mode)
                result = _load_decomp(mri_path, layer)
                self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/decomp/<model>/<mode>?layer=N"})
        elif path.startswith('/api/decomp-meta/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                mri_path = self._mri_path(model, mode)
                result = _load_decomp_meta(mri_path)
                self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/decomp-meta/<model>/<mode>"})
        elif path.startswith('/api/token-bio/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                token = int(qs.get('token', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                self._send_json(_token_biography(mri_path, token))
            else:
                self._send_json({"error": "Usage: /api/token-bio/<model>/<mode>?token=N"})
        elif path.startswith('/api/gate-heatmap/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                hp = Path(self._mri_path(model, mode)) / "decomp" / "gate_heatmap.npy"
                if hp.exists():
                    self._send_file(hp)
                else:
                    self._send_json({"error": "No gate heatmap"})
        elif path.startswith('/api/pc-full/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                pc = int(qs.get('pc', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                result = _pc_full(mri_path, pc)
                if isinstance(result, dict):
                    self._send_json(result)
                else:
                    self._send_bytes(result)
        elif path.startswith('/api/pc-medium/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                pcs_str = qs.get('pcs', [''])[0]
                pcs = [int(p) for p in pcs_str.split(',') if p]
                step = int(qs.get('step', ['10'])[0])
                mri_path = self._mri_path(model, mode)
                result = _pc_medium(mri_path, pcs, step)
                if isinstance(result, dict):
                    self._send_json(result)
                else:
                    self._send_bytes(result)
        elif path.startswith('/api/pc-column/'):
            # Serve one PC's scores for all tokens at one layer: float16 [nTokens]
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                pc = int(qs.get('pc', ['0'])[0])
                layer = int(qs.get('layer', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                result = _pc_column_cached(mri_path, pc, layer)
                if isinstance(result, dict):
                    self._send_json(result)
                else:
                    self._send_bytes(result)
        elif path.startswith('/api/token-hover/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                token = int(qs.get('token', ['0'])[0])
                layer = int(qs.get('layer', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                result = _token_hover(mri_path, token, layer)
                if isinstance(result, dict):
                    self._send_json(result)
                else:
                    self._send_bytes(result)
        elif path.startswith('/api/token-layer/'):
            # All PCs for one token at one layer: [4B K][4B layer][float16 × K] ~1KB
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                token = int(qs.get('token', ['0'])[0])
                layer = int(qs.get('layer', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                result = _token_layer_scores(mri_path, token, layer)
                if isinstance(result, dict):
                    self._send_json(result)
                else:
                    self._send_bytes(result)
        elif path.startswith('/api/token-pca/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                token = int(qs.get('token', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                result = _token_pca_full(mri_path, token)
                if isinstance(result, dict):
                    self._send_json(result)
                else:
                    self._send_bytes(result)
        elif path.startswith('/api/neuron-field/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                token = int(qs.get('token', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                result = _neuron_field(mri_path, token)
                if isinstance(result, dict):
                    self._send_json(result)
                else:
                    self._send_bytes(result)
            else:
                self._send_json({"error": "Usage: /api/neuron-field/<model>/<mode>?token=N"})
        elif path.startswith('/api/token-neurons/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                token = int(qs.get('token', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                self._send_json(_token_neurons(mri_path, token))
        elif path.startswith('/api/token-attn/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                token = int(qs.get('token', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                self._send_json(_token_attention(mri_path, token))
        elif path.startswith('/api/norms/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                norms_path = Path(self._mri_path(model, mode)) / "norms.npz"
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
                bl_path = Path(self._mri_path(model, mode)) / "baselines.npz"
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
                bin_path = Path(self._mri_path(model, mode)) / "decomp" / "delta_scores.bin"
                if bin_path.exists():
                    self._send_file(bin_path)
                else:
                    self._send_json({"error": "No delta scores. Needs entry states."})
        elif path.startswith('/api/weight-align-all/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                mri_path = self._mri_path(model, mode)
                result = _weight_alignment_all(mri_path)
                self._send_json(result)
        elif path.startswith('/api/weight-align/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                layer = int(qs.get('layer', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                self._send_json(_weight_alignment(mri_path, layer))
            else:
                self._send_json({"error": "Usage: /api/weight-align/<model>/<mode>?layer=N"})
        elif path == '/api/signals':
            kind = qs.get('kind', [None])[0]
            model = qs.get('model', [None])[0]
            target = qs.get('target', [None])[0]
            self._send_json(_query_signals(kind=kind, model=model, target=target))
        elif path.startswith('/api/direction-quality/'):
            # /api/direction-quality/model/mode?a=N&b=N&layer=L
            # Full-K direction analysis between two tokens
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                a = int(qs.get('a', ['0'])[0])
                b = int(qs.get('b', ['0'])[0])
                layer = int(qs.get('layer', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                result = _direction_quality(mri_path, a, b, layer)
                self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/direction-quality/<model>/<mode>?a=N&b=N&layer=L"})
        elif path.startswith('/api/direction-project/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                a = int(qs.get('a', ['0'])[0])
                b = int(qs.get('b', ['0'])[0])
                layer = int(qs.get('layer', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                result = _direction_project(mri_path, a, b, layer)
                if "projections" in result:
                    # Send as binary float32 for efficiency (600KB vs 3MB JSON)
                    import struct
                    proj = np.array(result["projections"], dtype=np.float32)
                    header = json.dumps({k: v for k, v in result.items()
                                        if k != "projections"}).encode()
                    # Pad header to 4-byte alignment for Float32Array
                    pad = (4 - len(header) % 4) % 4
                    header_padded = header + b'\x00' * pad
                    body = struct.pack('<I', len(header_padded)) + header_padded + proj.tobytes()
                    self._send_bytes(body)
                else:
                    self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/direction-project/<model>/<mode>?a=N&b=N&layer=L"})
        elif path.startswith('/api/direction-nonlinear/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                a = int(qs.get('a', ['0'])[0])
                b = int(qs.get('b', ['0'])[0])
                layer = int(qs.get('layer', ['0'])[0])
                n_sample = int(qs.get('n_sample', ['2000'])[0])
                mri_path = self._mri_path(model, mode)
                result = _direction_nonlinear(mri_path, a, b, layer, n_sample)
                self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/direction-nonlinear/<model>/<mode>?a=N&b=N&layer=L"})
        elif path.startswith('/api/direction-depth/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                a = int(qs.get('a', ['0'])[0])
                b = int(qs.get('b', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                result = _direction_depth(mri_path, a, b)
                self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/direction-depth/<model>/<mode>?a=N&b=N"})
        elif path.startswith('/api/direction-circuit/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                a = int(qs.get('a', ['0'])[0])
                b = int(qs.get('b', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                result = _direction_circuit(mri_path, a, b)
                self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/direction-circuit/<model>/<mode>?a=N&b=N"})
        elif path.startswith('/api/direction-discover/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                layer = int(qs.get('layer', ['0'])[0])
                n = int(qs.get('n', ['20'])[0])
                mri_path = self._mri_path(model, mode)
                result = _auto_discover_directions(mri_path, layer, n_top=n)
                self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/direction-discover/<model>/<mode>?layer=L&n=20"})
        elif path.startswith('/api/direction-weights/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                a = int(qs.get('a', ['0'])[0])
                b = int(qs.get('b', ['1'])[0])
                mri_path = self._mri_path(model, mode)
                result = _direction_weight_alignment(mri_path, a, b)
                self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/direction-weights/<model>/<mode>?a=N&b=N"})
        elif path.startswith('/api/token-predicts/'):
            # /api/token-predicts/model/mode?token=N&layer=L&k=20
            parts = path.split('/')
            if len(parts) >= 5:
                model_name, mode_name = parts[3], parts[4]
                token_idx = int(qs.get('token', ['0'])[0])
                layer = int(qs.get('layer', ['0'])[0])
                top_k = int(qs.get('k', ['20'])[0])
                mri_path = self._mri_path(model_name, mode_name)
                result = _token_predicts(mri_path, token_idx, layer, top_k)
                self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/token-predicts/<model>/<mode>?token=N&layer=L"})
        elif path == '/api/commands':
            self._send_json(_list_commands())
        elif path.startswith('/api/run/'):
            command = path.split('/api/run/')[1]
            args = {k: v[0] for k, v in qs.items()}
            result = _run_command(command, args)
            self._send_json(result)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length else b'{}'
        try:
            args = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"})
            return

        if path == '/api/navigate':
            cmd = {"cmd": "navigate"}
            cmd.update(args)
            _poll_push(cmd)
            self._send_json({"ok": True})

        elif path == '/api/capture':
            import uuid
            rid = uuid.uuid4().hex[:12]
            viewport = args.get("viewport", "0")
            fmt = args.get("format", "png")
            timeout = 300.0 if fmt == "gif" else 10.0

            ev = threading.Event()
            with _capture_lock:
                _capture_pending[rid] = ev

            _poll_push({"cmd": "capture", "request_id": rid,
                        "viewport": viewport, "format": fmt})

            if not ev.wait(timeout=timeout):
                with _capture_lock:
                    _capture_pending.pop(rid, None)
                    _capture_results.pop(rid, None)
                self._send_json({"error": "capture timed out"})
                return

            with _capture_lock:
                result = _capture_results.pop(rid, None)
                _capture_pending.pop(rid, None)

            if not result:
                self._send_json({"error": "capture failed"})
                return
            if "error" in result:
                self._send_json({"error": result["error"]})
                return

            # Binary upload path (GIF) — file already saved by /api/capture-upload
            if "data_path" in result:
                filepath = Path(result["data_path"])
                self._send_json({"path": str(filepath),
                                 "size": filepath.stat().st_size,
                                 "filename": result.get("filename", filepath.name)})
                return

            # Base64 JSON path (PNG screenshots)
            if "data" not in result:
                self._send_json({"error": "capture failed", "detail": result})
                return
            import base64
            _CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            ext = "gif" if fmt == "gif" else "png"
            filename = result.get("filename", f"capture_{rid}.{ext}")
            filepath = _CAPTURE_DIR / filename
            img_data = base64.b64decode(result["data"])
            filepath.write_bytes(img_data)
            self._send_json({"path": str(filepath), "size": len(img_data),
                             "filename": filename})

        elif path == '/api/capture-upload':
            # Direct binary upload from browser
            _CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            qs = parse_qs(parsed.query)
            rid = qs.get("request_id", [""])[0]
            filename = qs.get("filename", [f"capture_{rid}.png"])[0]
            filepath = _CAPTURE_DIR / filename
            filepath.write_bytes(body)
            with _capture_lock:
                _capture_results[rid] = {"data_path": str(filepath),
                                         "filename": filename}
                ev = _capture_pending.get(rid)
                if ev:
                    ev.set()
            self._send_json({"path": str(filepath), "size": len(body)})

        elif path == '/api/capture-result':
            # JSON capture result from browser (base64 data or error)
            rid = args.get("request_id", "")
            with _capture_lock:
                _capture_results[rid] = args
                ev = _capture_pending.get(rid)
                if ev:
                    ev.set()
            self._send_json({"ok": True})

        elif path == '/api/direction-steer':
            model_name = args.get("model", "")
            mode_name = args.get("mode", "")
            a = int(args.get("a", 0))
            b = int(args.get("b", 0))
            layer = int(args.get("layer", 0))
            prompt = args.get("prompt", "Once upon a time")
            alpha = float(args.get("alpha", 2.0))
            max_tokens = int(args.get("max_tokens", 30))
            if not model_name:
                self._send_json({"error": "model required"})
                return
            mri_path = self._mri_path(model_name, mode_name or "raw")
            model_meta = json.loads((Path(mri_path) / "metadata.json").read_text())
            model_id = model_meta["model"]["name"]
            result = _direction_steer_test(
                mri_path, a, b, layer, prompt, alpha, max_tokens, model_id)
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

    def _send_bytes(self, data: bytes):
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', len(data))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        # Chunk writes to avoid broken pipe on large responses
        offset = 0
        while offset < len(data):
            chunk = data[offset:offset + 65536]
            self.wfile.write(chunk)
            offset += 65536

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
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # silence request logs


def run_companion(port: int = 8377, mri_root: str = "/Volumes/sharts"):
    """Run the companion HTTP server (threaded)."""
    from socketserver import ThreadingMixIn

    CompanionHandler.mri_root = mri_root

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(('0.0.0.0', port), CompanionHandler)
    print(f"Heinrich companion: http://localhost:{port}")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
