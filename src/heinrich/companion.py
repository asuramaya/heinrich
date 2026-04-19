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
import time
from collections import OrderedDict
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

# --- Chat inbox: browser → MCP client ---
# MCP clients drain with chat_drain(); the browser long-polls /api/chat-poll
# for replies posted via chat_reply(). Bounded so a disconnected MCP client
# does not leak memory.
_chat_inbox: list[dict] = []
_chat_outbox: list[dict] = []  # replies from MCP client → browser
_chat_lock = threading.Lock()
_chat_event = threading.Event()       # browser ← MCP (outbox)
_chat_inbox_event = threading.Event()  # browser → MCP (inbox)
_CHAT_MAX = 200


def chat_drain() -> list[dict]:
    """MCP client pulls pending user messages. Called by external tools."""
    with _chat_lock:
        msgs = list(_chat_inbox)
        _chat_inbox.clear()
        if not _chat_inbox:
            _chat_inbox_event.clear()
    return msgs


def chat_reply(text: str, request_id: str = "") -> None:
    """MCP client posts a reply. Browser long-poll delivers it."""
    with _chat_lock:
        _chat_outbox.append({"reply": text, "request_id": request_id})
        while len(_chat_outbox) > _CHAT_MAX:
            _chat_outbox.pop(0)
    _chat_event.set()


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


# --- Score cache: LRU by total bytes, drops entries for other MRIs on switch ---
# First load from USB is ~4s, cached access is ~5ms.
# OrderedDict: insertion order = LRU order, move_to_end() is O(1) on hit.
_score_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
# Budget in bytes. ~2 GB default covers 3 layers of Qwen-0.5B (~540 MB/layer) with
# headroom for lmhead (~250 MB). Override via HEINRICH_SCORE_CACHE_GB env var for
# larger models (a 7B model with hidden=4096, vocab=150K needs ~2.4 GB per layer).
import os as _os
_SCORE_CACHE_MAX_BYTES = int(
    float(_os.environ.get("HEINRICH_SCORE_CACHE_GB", "2")) * 1_000_000_000
)
_score_cache_bytes = 0
# Thread-safety for the cache — _score_cache_put/get race if two HTTP threads
# simultaneously insert new layers. Held only during mutation, not during
# file IO (IO happens before/after) so this doesn't serialize cold loads.
_score_cache_lock = threading.Lock()


def _score_cache_evict_to_fit(incoming_bytes: int,
                              active_mri: str | None = None) -> None:
    """Evict LRU entries until the cache has room for `incoming_bytes`.
    Prefers evicting entries from other MRIs first so a model switch reclaims
    memory from the previous model before trimming the current one.

    Caller must hold _score_cache_lock.
    """
    global _score_cache_bytes
    if active_mri:
        for key in list(_score_cache):
            if _score_cache_bytes + incoming_bytes <= _SCORE_CACHE_MAX_BYTES:
                return
            if not key.startswith(active_mri + ":"):
                arr = _score_cache.pop(key)
                _score_cache_bytes -= arr.nbytes
    while (_score_cache and
           _score_cache_bytes + incoming_bytes > _SCORE_CACHE_MAX_BYTES):
        _, arr = _score_cache.popitem(last=False)
        _score_cache_bytes -= arr.nbytes


def _score_cache_put(key: str, arr: np.ndarray, active_mri: str | None = None) -> None:
    global _score_cache_bytes
    with _score_cache_lock:
        if key in _score_cache:
            _score_cache.move_to_end(key)
            return
        _score_cache_evict_to_fit(arr.nbytes, active_mri=active_mri)
        _score_cache[key] = arr
        _score_cache_bytes += arr.nbytes


def _get_scores_f32(mri_path: str, layer: int) -> np.ndarray | None:
    """Get float32 score array for a layer, cached in RAM.

    Eagerly casts float16 → float32 because many callers index individual
    rows and numpy's mixed-dtype ops silently upcast. Large models (26+
    layers × 540 MB) can overflow the cache budget; callers that only
    need projection should prefer `_get_scores_mmap` + `_project_chunked`.
    """
    key = f"{mri_path}:L{layer:02d}"
    if key in _score_cache:
        _score_cache.move_to_end(key)
        return _score_cache[key]
    score_path = Path(mri_path) / "decomp" / f"L{layer:02d}_scores.npy"
    if not score_path.exists():
        return None
    arr = np.load(str(score_path)).astype(np.float32)
    _score_cache_put(key, arr, active_mri=mri_path)
    return arr


_scores_mmap_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
_SCORES_MMAP_MAX = 256  # handles only — mmap backing is OS-paged

# Weight matrix mmap cache (handles only, OS page cache handles warmth).
_weight_mmap_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
_WEIGHT_MMAP_MAX = 512


def _get_weight_mmap(path: Path) -> np.ndarray | None:
    """Mmap a weight matrix, caching the handle so repeated pins don't
    re-open the same file. OS page cache handles warmth. LRU-bounded
    so switching between many MRIs doesn't accumulate handles forever.
    """
    key = str(path)
    h = _weight_mmap_cache.get(key)
    if h is not None:
        _weight_mmap_cache.move_to_end(key)
        return h
    if not path.exists():
        return None
    h = np.load(key, mmap_mode="r")
    _weight_mmap_cache[key] = h
    while len(_weight_mmap_cache) > _WEIGHT_MMAP_MAX:
        _weight_mmap_cache.popitem(last=False)
    return h


# Gram matrix cache — for a weight W of shape [M, K] we cache the [D, D]
# Gram matrix on the side matching the direction length D. This lets
# ||W @ d|| = sqrt(d.T @ Gram @ d) bypass the O(MK) matmul per pin change.
# One cold pass builds the cache (full W read once); subsequent pin changes
# are pure O(D²) — hundreds of microseconds per matrix. LRU-evicted
# by byte budget (override via HEINRICH_GRAM_CACHE_GB).
_weight_gram_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_WEIGHT_GRAM_MAX_BYTES = int(
    float(_os.environ.get("HEINRICH_GRAM_CACHE_GB", "1.5")) * 1_000_000_000
)
_weight_gram_bytes = 0

# Token metadata cache — tokens.npz holds 150k-entry text/script/vocab_id
# arrays. Loading + decoding + building the vocab→mri reverse map was
# ~1s on repeat calls; do it once per MRI.
_vocab_map_cache: dict[str, tuple] = {}


def _get_vocab_map(mri_path: str) -> tuple:
    """Return (texts, vocab_to_mri_index) for the MRI, cached.

    `texts` is a list of str. `vocab_to_mri_index` is a dict from
    tokenizer vocab id → row index in the captured MRI.
    """
    cached = _vocab_map_cache.get(mri_path)
    if cached is not None:
        return cached
    tok = dict(np.load(str(Path(mri_path) / "tokens.npz"), allow_pickle=True))
    texts = [str(t) for t in tok["token_texts"]]
    token_ids = tok.get("token_ids")
    vocab_to_mri: dict[int, int] = {}
    if token_ids is not None:
        for i, vid in enumerate(token_ids):
            vocab_to_mri[int(vid)] = i
    entry = (texts, vocab_to_mri)
    _vocab_map_cache[mri_path] = entry
    return entry


def _get_weight_gram(path: Path, side: str) -> np.ndarray | None:
    """Gram matrix for a weight file on the requested side.

    side='col' → W.T @ W  (shape [K, K], for directions matching W.shape[1])
    side='row' → W @ W.T  (shape [M, M], for directions matching W.shape[0])

    Returns float32 Gram. None if the file doesn't exist.
    LRU-evicted when cache exceeds HEINRICH_GRAM_CACHE_GB (default 1.5 GB).
    """
    global _weight_gram_bytes
    key = (str(path), side)
    g = _weight_gram_cache.get(key)
    if g is not None:
        _weight_gram_cache.move_to_end(key)
        return g
    W = _get_weight_mmap(path)
    if W is None:
        return None
    Wf = W.astype(np.float32)
    g = (Wf.T @ Wf) if side == "col" else (Wf @ Wf.T)
    while _weight_gram_cache and (_weight_gram_bytes + g.nbytes) > _WEIGHT_GRAM_MAX_BYTES:
        _, evicted = _weight_gram_cache.popitem(last=False)
        _weight_gram_bytes -= evicted.nbytes
    _weight_gram_cache[key] = g
    _weight_gram_bytes += g.nbytes
    return g


def _get_scores_mmap(mri_path: str, layer: int) -> np.ndarray | None:
    """Zero-copy mmap handle to a layer's score matrix (usually float16).

    Cheap to keep open indefinitely — the OS page cache handles warmth.
    Combined with _project_chunked this lets large MRIs sweep all layers
    without the 2 GB score cache churning.
    """
    key = f"{mri_path}:L{layer:02d}"
    mmap = _scores_mmap_cache.get(key)
    if mmap is not None:
        _scores_mmap_cache.move_to_end(key)
        return mmap
    path = Path(mri_path) / "decomp" / f"L{layer:02d}_scores.npy"
    if not path.exists():
        return None
    mmap = np.load(str(path), mmap_mode="r")
    _scores_mmap_cache[key] = mmap
    while len(_scores_mmap_cache) > _SCORES_MMAP_MAX:
        _scores_mmap_cache.popitem(last=False)
    return mmap


def _project_chunked(scores: np.ndarray, direction: np.ndarray,
                     chunk: int = 10000) -> np.ndarray:
    """Compute scores @ direction in memory-bounded chunks.

    Avoids allocating a full float32 copy of the scores matrix. Direction
    must be 1-D (single vector) or 2-D (batched [K, M]). Output is always
    float32.
    """
    N = scores.shape[0]
    if direction.ndim == 1:
        out = np.empty(N, dtype=np.float32)
        for i in range(0, N, chunk):
            out[i:i + chunk] = scores[i:i + chunk].astype(np.float32) @ direction.astype(np.float32, copy=False)
        return out
    # 2-D: direction is [K, M]
    M = direction.shape[1]
    out = np.empty((N, M), dtype=np.float32)
    dir_f32 = direction.astype(np.float32, copy=False)
    for i in range(0, N, chunk):
        out[i:i + chunk] = scores[i:i + chunk].astype(np.float32) @ dir_f32
    return out


def _bimodality(proj: np.ndarray) -> float:
    """Histogram valley test on a 1-D projection. Lower = more bimodal.

    Finds the two largest peaks in the density histogram (separated by
    at least 10 bins), then the minimum bin between them. Returns
    valley / min(peak_heights). Near 0 = clear bimodal gap; near 1 =
    unimodal (no meaningful valley). Robust to skew / shift because it
    uses the data's actual min-max range and doesn't assume a center.
    """
    proj = proj.astype(np.float32, copy=False)
    lo, hi = float(proj.min()), float(proj.max())
    if hi - lo < 1e-12:
        return 1.0  # constant projection — not bimodal
    n_bins = 100
    hist, _ = np.histogram(proj, bins=n_bins, range=(lo, hi))
    # Global peak
    p1 = int(hist.argmax())
    p1_h = int(hist[p1])
    if p1_h == 0:
        return 1.0
    # Second peak: highest bin at least 10 bins away from p1
    min_sep = 10
    masked = hist.astype(np.int64).copy()
    lo_i = max(0, p1 - min_sep)
    hi_i = min(n_bins, p1 + min_sep + 1)
    masked[lo_i:hi_i] = -1
    if masked.max() <= 0:
        return 1.0  # only one density mass — unimodal
    p2 = int(masked.argmax())
    p2_h = int(hist[p2])
    # Valley: min bin strictly between the two peaks
    a, b = (p1, p2) if p1 < p2 else (p2, p1)
    valley_slice = hist[a + 1:b]
    if valley_slice.size == 0:
        return 1.0
    valley = float(valley_slice.min())
    min_peak = min(p1_h, p2_h)
    return valley / (min_peak + 1e-8)


_random_bm_cache: "OrderedDict[tuple, list[float]]" = OrderedDict()
_RANDOM_BM_CACHE_MAX = 64


def _random_bimodality_baseline(scores: np.ndarray, n_random: int = 50,
                                seed: int = 42) -> list[float]:
    """Bimodality of 50 directions between random token pairs. For percentile context.

    Cached by (array id, shape, n_random, seed) — called repeatedly per layer
    across pin-changes. Cache is LRU-bounded so model switches evict stale
    entries. Array id is the numpy data pointer so cache hits only when the
    same score matrix is reused.
    """
    key = (scores.ctypes.data, scores.shape, n_random, seed)
    if key in _random_bm_cache:
        _random_bm_cache.move_to_end(key)
        return _random_bm_cache[key]

    rng = np.random.RandomState(seed)
    N = scores.shape[0]
    # Batched path: build all random directions at once, do ONE [N,K]×[K,M] BLAS call.
    # Chunked so we don't allocate a full float32 copy of scores.
    idxs = rng.choice(N, (n_random, 2), replace=True).astype(np.int64)
    dirs = (scores[idxs[:, 0]].astype(np.float32)
            - scores[idxs[:, 1]].astype(np.float32))
    norms = np.linalg.norm(dirs, axis=1)
    valid = norms > 1e-8
    dirs = dirs[valid] / norms[valid, None]
    if dirs.shape[0] == 0:
        _random_bm_cache[key] = []
        return []
    projs = _project_chunked(scores, dirs.T)  # [N, M]
    out: list[float] = [_bimodality(projs[:, j]) for j in range(projs.shape[1])]

    _random_bm_cache[key] = out
    while len(_random_bm_cache) > _RANDOM_BM_CACHE_MAX:
        _random_bm_cache.popitem(last=False)
    return out


def _bimodality_percentile(bimodal_ratio: float,
                           random_bms: list[float]) -> float:
    """Percent of random pairs MORE unimodal than this. Lower = rarer signal."""
    if not random_bms:
        return 50.0
    return float(sum(1 for r in random_bms if r > bimodal_ratio)
                 / len(random_bms) * 100)


def _load_token_ids(mri_path: str) -> np.ndarray | None:
    """Load token_ids array from tokens.npz without allow_pickle (int array, safe)."""
    try:
        with np.load(str(Path(mri_path) / "tokens.npz")) as z:
            if "token_ids" in z.files:
                return np.asarray(z["token_ids"])
    except (FileNotFoundError, ValueError, OSError):
        pass
    return None


def _functional_hit_rate(mri_path: str, layer: int,
                         token_a: int, token_b: int,
                         top_a_idx, top_b_idx,
                         k_check: int = 5, top_k: int = 50) -> tuple:
    """Do geometrically A-like tokens actually predict A through lmhead?

    Returns (hit_rate, warning_bool) or (None, None) if files missing.
    """
    try:
        mri_dir = Path(mri_path)
        exit_path = mri_dir / f"L{layer:02d}_exit.npy"
        lmhead_path = mri_dir / "lmhead.npy"
        if not (exit_path.exists() and lmhead_path.exists()):
            return None, None
        token_ids = _load_token_ids(mri_path)
        if token_ids is None:
            return None, None
        exits = np.load(str(exit_path), mmap_mode="r")
        lmhead_key = f"{mri_path}:lmhead"
        if lmhead_key not in _score_cache:
            _score_cache_put(lmhead_key,
                             np.load(str(lmhead_path)).astype(np.float32),
                             active_mri=mri_path)
        else:
            _score_cache.move_to_end(lmhead_key)
        lmhead = _score_cache[lmhead_key]

        vocab_a = int(token_ids[token_a])
        vocab_b = int(token_ids[token_b])

        hits = 0
        checks = 0
        for idx in [int(i) for i in top_a_idx[:k_check]]:
            if idx >= exits.shape[0]:
                continue
            logits = lmhead @ exits[idx].astype(np.float32)
            if vocab_a in set(np.argsort(logits)[-top_k:].tolist()):
                hits += 1
            checks += 1
        for idx in [int(i) for i in top_b_idx[:k_check]]:
            if idx >= exits.shape[0]:
                continue
            logits = lmhead @ exits[idx].astype(np.float32)
            if vocab_b in set(np.argsort(logits)[-top_k:].tolist()):
                hits += 1
            checks += 1
        if checks == 0:
            return None, None
        rate = hits / checks
        return rate, rate < 0.2
    except (FileNotFoundError, ValueError, OSError):
        return None, None


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

    # Project all tokens onto direction (chunked, avoids 537MB float32 copy)
    proj = _project_chunked(scores, direction)

    # Explained variance: var(projection) / total layer variance.
    # Two-pass chunked variance: exact, no fp16 overflow.
    #   pass 1: mean via sum of float32 chunks
    #   pass 2: sum of squared deviations from that mean
    proj_var = float(np.var(proj))
    CV = 10000
    col_sum = np.zeros(K, dtype=np.float64)
    for i in range(0, N, CV):
        col_sum += scores[i:i + CV].astype(np.float32).sum(axis=0, dtype=np.float64)
    col_mean = col_sum / N
    ssd = np.zeros(K, dtype=np.float64)
    for i in range(0, N, CV):
        chunk = scores[i:i + CV].astype(np.float32) - col_mean.astype(np.float32)
        ssd += (chunk.astype(np.float64) ** 2).sum(axis=0)
    total_var = float(ssd.sum() / N)
    explained = proj_var / (total_var + 1e-8)

    # Bimodality: histogram valley test (shared helper)
    bimodal_ratio = _bimodality(proj)

    # Random baseline: percentile among 50 random token-pair directions
    random_bms = _random_bimodality_baseline(scores)
    percentile = _bimodality_percentile(bimodal_ratio, random_bms)

    # All-layer bimodality sweep (mmap reads, no cache eviction)
    import json as _json
    decomp_meta_path = Path(mri_path) / "decomp" / "meta.json"
    layer_bms: list[dict] = []
    if decomp_meta_path.exists():
        dm = _json.loads(decomp_meta_path.read_text())
        for li_meta in dm.get("layers", []):
            li = li_meta.get("layer", -1)
            # Skip virtual layers ('emb', 'lmh') — no L{NN}_scores.npy files
            if not isinstance(li, int):
                continue
            if li == layer:
                layer_bms.append({"layer": li, "bimodality": bimodal_ratio})
                continue
            li_scores = _get_scores_mmap(mri_path, li)
            if li_scores is None:
                continue
            if a >= li_scores.shape[0]:
                continue
            li_diff = li_scores[a].astype(np.float32) - li_scores[b].astype(np.float32)
            li_mag = float(np.linalg.norm(li_diff))
            if li_mag < 1e-8:
                continue
            li_proj = _project_chunked(li_scores, li_diff / li_mag)
            layer_bms.append({"layer": li, "bimodality": _bimodality(li_proj)})

    # Top tokens per side
    sorted_idx = np.argsort(proj)
    top_a_idx = sorted_idx[-10:][::-1]
    top_b_idx = sorted_idx[:10]
    top_a_list = [{"idx": int(i), "text": texts[i], "script": scripts_arr[i],
                   "proj": float(proj[i])} for i in top_a_idx]
    top_b_list = [{"idx": int(i), "text": texts[i], "script": scripts_arr[i],
                   "proj": float(proj[i])} for i in top_b_idx]

    # --- Anchor consistency check ---
    # Find 3 tokens most similar to A (highest proj, excluding A) and
    # 3 most similar to B (lowest proj, excluding B). For each of the 9
    # alt pairs, recompute bimodality. High variance = anchor-dependent.
    def _alt_bimodality(idx_a: int, idx_b: int) -> float:
        alt_diff = scores[idx_a].astype(np.float32) - scores[idx_b].astype(np.float32)
        alt_mag = float(np.linalg.norm(alt_diff))
        return _bimodality(_project_chunked(scores, alt_diff / (alt_mag + 1e-8)))

    # Top-3 A-like: highest projection, excluding A itself
    alt_a_candidates = sorted_idx[::-1]  # descending
    alt_a = [int(i) for i in alt_a_candidates if int(i) != a][:3]
    # Top-3 B-like: lowest projection, excluding B itself
    alt_b = [int(i) for i in sorted_idx if int(i) != b][:3]

    alt_bimodalities = []
    for aa in alt_a:
        for bb in alt_b:
            alt_bimodalities.append(_alt_bimodality(aa, bb))

    if len(alt_bimodalities) > 0:
        alt_mean = float(np.mean(alt_bimodalities))
        alt_std = float(np.std(alt_bimodalities))
        anchor_consistency = alt_std / (alt_mean + 1e-8)
        # Check if any alt pair flips bimodal<->unimodal (threshold 0.5)
        orig_is_bimodal = bimodal_ratio < 0.5
        anchor_warning = any(
            (br < 0.5) != orig_is_bimodal for br in alt_bimodalities
        )
    else:
        anchor_consistency = 0.0
        anchor_warning = False
        alt_bimodalities = []

    # --- Functional validation ---
    # Do top A-like tokens actually predict token A through lmhead?
    functional_hit_rate, functional_warning = _functional_hit_rate(
        mri_path, layer, a, b, top_a_idx, top_b_idx)

    result = {
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
        "anchor_consistency": anchor_consistency,
        "anchor_warning": anchor_warning,
        "alt_bimodalities": [float(b) for b in alt_bimodalities],
        "functional_hit_rate": functional_hit_rate,
        "functional_warning": functional_warning,
        "random_baseline_percentile": percentile,
        "layer_bimodalities": layer_bms,
        "best_layer": min(layer_bms, key=lambda x: x["bimodality"])["layer"] if layer_bms else layer,
        "worst_layer": max(layer_bms, key=lambda x: x["bimodality"])["layer"] if layer_bms else layer,
    }
    return result


def _direction_project(mri_path: str, a: int, b: int, layer: int) -> dict:
    """Project all tokens onto the full-K direction between two tokens.

    Returns normalized projections [-1, +1] centered on the A-B midpoint,
    suitable for direct use as color values in the browser.
    """
    scores = _get_scores_mmap(mri_path, layer)
    if scores is None:
        return {"error": f"No scores at L{layer}"}
    N, K = scores.shape

    if a >= N or b >= N:
        return {"error": f"Token index out of range (max {N-1})"}

    diff = scores[a].astype(np.float32) - scores[b].astype(np.float32)
    mag = float(np.linalg.norm(diff))
    direction = diff / (mag + 1e-8)

    proj = _project_chunked(scores, direction).tolist()

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


# _direction_depth, _direction_circuit, _direction_weight_alignment live in
# companion_directions.py. Imported at module bottom so all helpers they
# depend on (_get_scores_f32, _bimodality, etc.) are defined first.


def _auto_discover_directions(mri_path: str, layer: int, n_top: int = 20,
                              exclude_outliers: bool = True) -> dict:
    """Automated direction discovery: find bimodal PCA components.

    Each PC is already a direction in hidden space.  The scores column IS
    the projection of all tokens onto that direction.  PCs with bimodal
    score distributions correspond to real features the model uses ---
    features humans haven't labeled.

    When ``exclude_outliers`` is True, filters tokens with |z-score| > 6 on
    any PC before computing bimodality, so the crystal (뀔 in raw mode) and
    similar extreme-outlier tokens don't dominate every PC. The unfiltered
    population is still returned in ``top_pos``/``top_neg``.

    Each discovered PC carries:
      - bimodality (lower = more bimodal)
      - random_baseline_percentile: how rare this bimodality is
        vs 50 random token-pair directions (lower = rarer signal)
      - functional_hit_rate: fraction of top-A and top-B tokens that
        predict A/B through lmhead (None if missing files)
      - functional_warning: True when hit rate < 20%

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

    var_path = Path(mri_path) / "decomp" / f"L{layer:02d}_variance.npy"
    if var_path.exists():
        variance = np.load(str(var_path)).astype(np.float32)
        total_var = float(variance.sum()) if variance.sum() > 0 else 1.0
    else:
        variance = None
        total_var = 1.0

    # Outlier mask (crystal suppression): drop tokens with |z| > 6 on any PC.
    # Computed once on the full scores, used for bimodality calc only.
    # Std floor: float16 scores have PCs where std approaches zero on near-dead
    # axes; naive z = x/std blows up spuriously. Clamp std to the 5th percentile
    # of per-PC stds so near-zero axes can't manufacture outliers.
    outlier_mask = None
    excluded_count = 0
    if exclude_outliers and N > 20:
        raw_std = scores.std(axis=0)
        std_floor = float(np.percentile(raw_std[raw_std > 0], 5)) if np.any(raw_std > 0) else 1e-8
        std = np.maximum(raw_std, std_floor) + 1e-8
        z_all = np.abs(scores) / std[None, :]
        outlier_mask = (z_all.max(axis=1) > 6.0)
        excluded_count = int(outlier_mask.sum())
        # Only apply if it doesn't nuke the population
        if excluded_count > N // 2:
            outlier_mask = None
            excluded_count = 0

    scored = scores if outlier_mask is None else scores[~outlier_mask]

    # Per-PC bimodality over the filtered population
    pc_bimodality = [(pc, _bimodality(scored[:, pc])) for pc in range(K)]
    pc_bimodality.sort(key=lambda x: x[1])  # most bimodal first

    # Random baseline (computed once on filtered population — applies to all PCs)
    random_bms = _random_bimodality_baseline(scored)

    discovered = []
    for pc, bimodal in pc_bimodality[:n_top]:
        proj = scores[:, pc]
        top_pos = np.argsort(proj)[-5:][::-1]
        top_neg = np.argsort(proj)[:5]

        var_pct = (float(variance[pc] / total_var)
                   if variance is not None and pc < len(variance) else 0.0)

        pctile = _bimodality_percentile(bimodal, random_bms)

        # Functional validation: anchor = single top-A and top-B token
        top_a_anchor = int(top_pos[0]) if len(top_pos) else 0
        top_b_anchor = int(top_neg[0]) if len(top_neg) else 0
        hit_rate, warning = _functional_hit_rate(
            mri_path, layer, top_a_anchor, top_b_anchor,
            top_pos.tolist(), top_neg.tolist(),
        )

        discovered.append({
            "pc": int(pc),
            "bimodality": float(bimodal),
            "variance_pct": var_pct,
            "random_baseline_percentile": pctile,
            "functional_hit_rate": hit_rate,
            "functional_warning": warning,
            "top_pos": [{"idx": int(i), "text": texts[i], "script": scripts_arr[i],
                         "proj": float(proj[i])} for i in top_pos],
            "top_neg": [{"idx": int(i), "text": texts[i], "script": scripts_arr[i],
                         "proj": float(proj[i])} for i in top_neg],
        })

    return {
        "layer": int(layer),
        "discovered": discovered,
        "outliers_excluded": excluded_count,
        "n_population": int(scored.shape[0]),
    }


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
        _score_cache_put(cache_key,
                         np.load(str(lmhead_path)).astype(np.float32),
                         active_mri=mri_path)
    else:
        _score_cache.move_to_end(cache_key)
    lmhead = _score_cache[cache_key]

    # Project: logits = lmhead @ exit_vec
    logits = lmhead @ exit_vec

    # Softmax for probabilities (numerically stable)
    logits_shifted = logits - logits.max()
    probs = np.exp(logits_shifted)
    probs = probs / probs.sum()

    # Top-K
    top_indices = np.argsort(logits)[-top_k:][::-1]

    # Token metadata: cached per MRI — rebuilding the 150k-entry vocab→mri
    # dict per call was eating ~1s on warm repeats.
    texts, vocab_to_mri = _get_vocab_map(mri_path)

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


def _residual_trajectory(mri_path: str, prompt: str, layer: int,
                         n_generated: int = 10, model_id: str = "") -> dict:
    """Capture the residual at `layer` at every generated position.

    Answers "does belief/refusal/commitment emerge during generation,
    or is it decided at the prompt boundary?" — plumbing #2.

    Returns a list of residuals: position 0 is prompt-end, 1 is after
    first generated token, 2 is after the second, etc. Also returns the
    generated text so callers can correlate position → token meaning.
    """
    if not model_id:
        try:
            mri_meta = json.loads((Path(mri_path) / "metadata.json").read_text())
            model_id = mri_meta["model"].get("huggingface_id") or mri_meta["model"].get("name", "")
        except (FileNotFoundError, KeyError):
            return {"error": "No MRI metadata; supply model_id explicitly."}
    try:
        backend = _get_steer_backend(model_id)
    except Exception as e:
        return {"error": f"Cannot load model {model_id}: {e}"}

    # Position 0: prompt-end residual via a normal forward
    try:
        r0 = backend.forward(prompt, return_residual=True, residual_layers=[layer])
        pos0 = r0.residuals.get(layer) if r0.residuals else None
        if pos0 is None:
            return {"error": f"Backend did not return residual at layer {layer}"}
        if pos0.ndim > 1:
            pos0 = pos0[-1]
    except Exception as e:
        return {"error": f"Prompt forward failed: {e}"}

    residuals = [np.asarray(pos0, dtype=np.float32)]
    tokens_out: list[dict] = []

    # Positions 1..N: generate step by step, capture residual at each
    try:
        with backend.generation_context(prompt) as gen:
            gen.capture_at(layer)
            for tok in gen.tokens(max_tokens=n_generated):
                if tok.residual is not None:
                    residuals.append(np.asarray(tok.residual, dtype=np.float32).ravel())
                tokens_out.append({
                    "step": tok.step,
                    "token_id": tok.token_id,
                    "text": tok.token_text,
                })
                if len(residuals) - 1 >= n_generated:
                    break
    except Exception as e:
        return {"error": f"Generation failed: {e}"}

    arr = np.stack(residuals) if residuals else np.zeros((0, 1), dtype=np.float32)
    return {
        "model_id": model_id,
        "prompt": prompt,
        "layer": layer,
        "n_positions": arr.shape[0],
        "hidden_size": int(arr.shape[1]) if arr.ndim > 1 else 0,
        "generated_tokens": tokens_out,
        "residuals_shape": list(arr.shape),
        # Return the full residual tensor as a nested list. Small: [n+1, hidden] float32.
        # Compact enough at n=10, hidden=896 → ~35 KB JSON.
        "residuals": arr.tolist(),
    }


def _capture_residual_at_position(backend, prompt: str, layer: int,
                                   position: int = 0) -> "np.ndarray | None":
    """Capture residual at `layer` for `prompt`, at given position.

    Single-layer convenience wrapper. For multi-layer sweeps use
    `_capture_residuals_multilayer` — it does one forward pass per prompt
    regardless of how many layers you want, saving ~N× time.
    """
    return _capture_residuals_multilayer(
        backend, prompt, [layer], position
    ).get(layer)


def _capture_residuals_multilayer(backend, prompt: str, layers: list,
                                   position: int = 0) -> dict:
    """Capture residuals at multiple layers in ONE forward pass.

    Returns {layer: np.ndarray(hidden,)}.  Dramatic speedup for multi-layer
    sweeps: on Qwen-7B, a 6-layer sweep goes from ~6× forward passes per
    prompt to ~1× forward pass per prompt.

    position=0: last prompt token.
    position>0: step through generation, capture at position-th generated token.
    """
    if position == 0:
        try:
            r = backend.forward(prompt, return_residual=True,
                                residual_layers=list(layers))
            residuals = getattr(r, "residuals", None)
            out = {}
            if residuals:
                for L in layers:
                    vec = residuals.get(L)
                    if vec is None:
                        continue
                    if vec.ndim > 1:
                        vec = vec[-1]
                    out[L] = np.asarray(vec, dtype=np.float32)
            return out
        except Exception:
            return {}
    # Multi-position generation: pick up residual at the k-th generated step.
    # Current backend only captures one layer per GenerationContext — fall back
    # to N separate generation passes when position > 0. Rare path.
    out = {}
    for L in layers:
        try:
            with backend.generation_context(prompt) as gen:
                gen.capture_at(L)
                for tok in gen.tokens(max_tokens=position):
                    if tok.step + 1 == position and tok.residual is not None:
                        out[L] = np.asarray(tok.residual, dtype=np.float32).ravel()
                    if tok.step + 1 >= position:
                        break
        except Exception:
            pass
    return out


def _behavioral_direction(mri_path: str, pos_prompts: list, neg_prompts: list,
                          layer: int, model_id: str = "",
                          mri_match_k: int = 30,
                          position: int = 0) -> dict:
    """Extract a direction from contrastive prompt groups, verify it robustly.

    position: 0 = last prompt token; >0 = captured at the k-th generated
              position. For tests of "does the direction only appear during
              generation" (Berger 2026).

    Runs the model on each prompt, captures the residual at `layer` at the
    chosen position, computes direction = mean(pos) − mean(neg). This is
    the contrastive-behavior direction.

    Validates with:
    - Cohen's d between pos/neg projections
    - Bootstrap cosine: resample prompts with replacement, measure direction stability
    - Projects MRI vocab onto the direction for top-K sanity
    """
    if not pos_prompts or not neg_prompts:
        return {"error": "Need non-empty pos_prompts and neg_prompts"}
    if len(pos_prompts) < 3 or len(neg_prompts) < 3:
        return {"error": "Need at least 3 prompts per side for meaningful statistics"}

    if not model_id:
        mri_meta = json.loads((Path(mri_path) / "metadata.json").read_text())
        model_id = mri_meta["model"].get("huggingface_id") or mri_meta["model"].get("name", "")
    try:
        backend = _get_steer_backend(model_id)
    except Exception as e:
        return {"error": f"Cannot load model {model_id}: {e}"}

    pos_vecs = [_capture_residual_at_position(backend, p, layer, position) for p in pos_prompts]
    neg_vecs = [_capture_residual_at_position(backend, p, layer, position) for p in neg_prompts]
    pos_vecs = [v for v in pos_vecs if v is not None]
    neg_vecs = [v for v in neg_vecs if v is not None]
    if not pos_vecs or not neg_vecs:
        return {"error": "No residuals captured (backend may not support residual_layers)"}

    P = np.stack(pos_vecs)  # [n_pos, hidden]
    N = np.stack(neg_vecs)  # [n_neg, hidden]
    pos_mean = P.mean(axis=0)
    neg_mean = N.mean(axis=0)
    diff = pos_mean - neg_mean
    mag = float(np.linalg.norm(diff))
    direction = diff / (mag + 1e-8)

    # Cohen's d on the direction
    pos_proj = P @ direction
    neg_proj = N @ direction
    pooled_std = float(np.sqrt(
        (pos_proj.var(ddof=1) + neg_proj.var(ddof=1)) / 2 + 1e-12))
    cohen_d = float((pos_proj.mean() - neg_proj.mean()) / (pooled_std + 1e-8))

    # Bootstrap: resample pos/neg with replacement, recompute direction, measure cosine
    rng = np.random.RandomState(42)
    boot_cos = []
    for _ in range(50):
        bp_idx = rng.randint(0, len(pos_vecs), size=len(pos_vecs))
        bn_idx = rng.randint(0, len(neg_vecs), size=len(neg_vecs))
        bp = P[bp_idx].mean(axis=0)
        bn = N[bn_idx].mean(axis=0)
        bd = bp - bn
        bn_mag = float(np.linalg.norm(bd))
        if bn_mag < 1e-8:
            continue
        boot_cos.append(float((bd / bn_mag) @ direction))
    boot_cos = np.array(boot_cos, dtype=np.float32)
    boot_cos.sort()

    # Project direction onto vocabulary — "top_pos / top_neg" labels.
    # Primary path: MRI's captured score matrix (token-level residuals) —
    # gives projections in the model's actual residual geometry at this layer.
    # Fallback path: project direction through lm_head directly (logit lens).
    # This works for ANY model, even without a captured MRI.
    top_pos = []
    top_neg = []
    comp_path = Path(mri_path) / "decomp" / f"L{layer:02d}_components.npy"
    tok_path = Path(mri_path) / "tokens.npz"
    vocab_source = "none"
    if comp_path.exists() and tok_path.exists():
        components = np.load(str(comp_path))
        dir_pc = components @ direction
        scores = _get_scores_mmap(mri_path, layer)
        if scores is not None:
            proj_vocab = _project_chunked(scores, dir_pc[:scores.shape[1]])
            srt = np.argsort(proj_vocab)
            try:
                texts, _ = _get_vocab_map(mri_path)
                for i in srt[-10:][::-1]:
                    top_pos.append({"idx": int(i), "text": texts[i], "proj": float(proj_vocab[i])})
                for i in srt[:10]:
                    top_neg.append({"idx": int(i), "text": texts[i], "proj": float(proj_vocab[i])})
                vocab_source = "mri"
            except Exception:
                pass
    if not top_pos and hasattr(backend, "project_through_lm_head"):
        try:
            logits = backend.project_through_lm_head(direction)
            tokenizer = backend.tokenizer
            srt = np.argsort(logits)
            for i in srt[-10:][::-1]:
                text = tokenizer.decode([int(i)])
                top_pos.append({"idx": int(i), "text": text, "proj": float(logits[i])})
            for i in srt[:10]:
                text = tokenizer.decode([int(i)])
                top_neg.append({"idx": int(i), "text": text, "proj": float(logits[i])})
            vocab_source = "lm_head"
        except Exception:
            pass

    def _pctile(arr, p):
        if len(arr) == 0:
            return None
        idx = min(len(arr) - 1, int(len(arr) * p))
        return float(arr[idx])

    return {
        "layer": layer,
        "position": position,
        "n_pos": len(pos_vecs),
        "n_neg": len(neg_vecs),
        "magnitude": mag,
        "cohen_d": cohen_d,
        "direction_hidden": direction.tolist(),
        "bootstrap_cosine": {
            "p5":  _pctile(boot_cos, 0.05),
            "p50": _pctile(boot_cos, 0.50),
            "p95": _pctile(boot_cos, 0.95),
            "mean": float(boot_cos.mean()) if len(boot_cos) else None,
        },
        "top_pos": top_pos,
        "top_neg": top_neg,
        "model_id": model_id,
    }


def _replicate_probe(mri_path: str, pos_prompts: list, neg_prompts: list,
                     layer: int, model_id: str = "",
                     expected_tokens_pos: list | None = None,
                     expected_tokens_neg: list | None = None,
                     n_random_null: int = 100,
                     position: int = 0) -> dict:
    """Apply the 5-test falsification pipeline to a contrastive probe.

    Returns a verdict object any paper's claim must pass. See
    papers/lie-detection/ATTACK_PLAN.md.

    Tests:
      1. Bootstrap anchor stability — boot_cos p5 > 0.7
      2. Random-direction null — orig d beats null distribution
      3. Within-group control — d(pos_vs_neg) > 2.5× max(d_within_pos, d_within_neg)
      4. Vocab projection sanity — top tokens contain expected concept words
      5. (Causal ablation is the 5th test; runs via a separate steering endpoint)

    verdict ∈ {robust_feature, partial, falsified}
    """
    if len(pos_prompts) < 6 or len(neg_prompts) < 6:
        return {"error": "Need at least 6 prompts per side for within-group splits"}

    # --- Main extraction (Test 1 is inside: bootstrap_cosine) ---
    main = _behavioral_direction(
        mri_path, pos_prompts, neg_prompts, layer,
        model_id=model_id, position=position,
    )
    if "error" in main:
        return {"error": f"main extraction: {main['error']}"}

    direction = np.array(main["direction_hidden"], dtype=np.float32)

    # --- Test 2: random-direction null baseline ---
    # Generate n random unit vectors in hidden space. For each, compute Cohen's d
    # on the pos/neg projections. If original d is inside this null distribution,
    # the signal is not distinguishable from random directions.
    # We need pos/neg residual vectors — re-capture them (one more pass, cached).
    try:
        backend = _get_steer_backend(main["model_id"])
    except Exception as e:
        return {"error": f"Cannot load model for null test: {e}"}
    pos_vecs = [_capture_residual_at_position(backend, p, layer, position) for p in pos_prompts]
    neg_vecs = [_capture_residual_at_position(backend, p, layer, position) for p in neg_prompts]
    pos_vecs = [v for v in pos_vecs if v is not None]
    neg_vecs = [v for v in neg_vecs if v is not None]
    P = np.stack(pos_vecs).astype(np.float32)
    N = np.stack(neg_vecs).astype(np.float32)
    K_hidden = direction.shape[0]

    rng = np.random.RandomState(123)
    null_dirs = rng.randn(n_random_null, K_hidden).astype(np.float32)
    null_dirs /= np.linalg.norm(null_dirs, axis=1, keepdims=True) + 1e-8
    null_d = []
    for i in range(n_random_null):
        d = null_dirs[i]
        pp = P @ d; nn = N @ d
        ps = float(np.sqrt((pp.var(ddof=1) + nn.var(ddof=1)) / 2 + 1e-12))
        null_d.append(float((pp.mean() - nn.mean()) / (ps + 1e-8)))
    null_d = np.array(null_d, dtype=np.float32)
    # Direction sign is arbitrary in random, so use |d|
    abs_null_d = np.abs(null_d)
    abs_null_d.sort()
    null_p95 = float(abs_null_d[int(len(abs_null_d) * 0.95)])
    null_max = float(abs_null_d.max())
    orig_d_abs = abs(main["cohen_d"])
    null_pass = orig_d_abs > null_p95

    # --- Test 3: within-group control ---
    def _within_split_d(vecs: list) -> float:
        """Cohen's d from splitting one class in half and extracting a direction."""
        n = len(vecs)
        if n < 6:
            return 0.0
        half = n // 2
        # Random shuffle for fairness
        rng2 = np.random.RandomState(42)
        perm = rng2.permutation(n)
        a_idx = perm[:half]
        b_idx = perm[half:2 * half]
        A = np.stack([vecs[i] for i in a_idx])
        B = np.stack([vecs[i] for i in b_idx])
        d = A.mean(axis=0) - B.mean(axis=0)
        m = float(np.linalg.norm(d))
        if m < 1e-8:
            return 0.0
        dir_ = d / m
        ap = A @ dir_; bp = B @ dir_
        ps = float(np.sqrt((ap.var(ddof=1) + bp.var(ddof=1)) / 2 + 1e-12))
        return float((ap.mean() - bp.mean()) / (ps + 1e-8))

    within_pos_d = _within_split_d(pos_vecs)
    within_neg_d = _within_split_d(neg_vecs)
    within_max = max(abs(within_pos_d), abs(within_neg_d))
    signal_noise = orig_d_abs / (within_max + 1e-8)
    within_pass = signal_noise > 2.5

    # --- Test 1 verdict (already computed in main) ---
    boot_p5 = main.get("bootstrap_cosine", {}).get("p5", 0.0) or 0.0
    boot_pass = boot_p5 > 0.7

    # --- Test 4: vocab projection sanity ---
    vocab_match = None
    if expected_tokens_pos or expected_tokens_neg:
        pos_texts = {t["text"].strip().lower() for t in main.get("top_pos", [])}
        neg_texts = {t["text"].strip().lower() for t in main.get("top_neg", [])}
        exp_pos = {x.strip().lower() for x in (expected_tokens_pos or [])}
        exp_neg = {x.strip().lower() for x in (expected_tokens_neg or [])}

        def _match_any(got: set, expected: set) -> int:
            """Count expected tokens that appear in top-10 (substring match)."""
            n = 0
            for e in expected:
                for g in got:
                    if e in g or g in e:
                        n += 1
                        break
            return n
        pos_matches = _match_any(pos_texts, exp_pos)
        neg_matches = _match_any(neg_texts, exp_neg)
        vocab_match = {
            "pos_matches": pos_matches,
            "pos_expected": len(exp_pos),
            "neg_matches": neg_matches,
            "neg_expected": len(exp_neg),
            "pos_fraction": pos_matches / max(1, len(exp_pos)),
            "neg_fraction": neg_matches / max(1, len(exp_neg)),
        }
    vocab_pass = (
        vocab_match is None or  # no expected tokens given — skip
        ((vocab_match["pos_matches"] >= 3) or
         (vocab_match["neg_matches"] >= 3))
    )

    # --- Verdict ---
    all_pass = boot_pass and null_pass and within_pass and vocab_pass
    any_pass = boot_pass or null_pass or within_pass
    if all_pass:
        verdict = "robust_feature"
    elif any_pass:
        verdict = "partial"
    else:
        verdict = "falsified"

    return {
        "layer": layer,
        "position": position,
        "n_pos": main["n_pos"],
        "n_neg": main["n_neg"],
        "cohen_d": main["cohen_d"],
        "magnitude": main["magnitude"],
        "direction_hidden": main["direction_hidden"],
        "top_pos": main["top_pos"],
        "top_neg": main["top_neg"],
        "test_1_bootstrap": {
            "boot_cosine_p5": boot_p5,
            "boot_cosine_p50": main.get("bootstrap_cosine", {}).get("p50"),
            "pass": boot_pass,
        },
        "test_2_null": {
            "null_p95_abs_d": null_p95,
            "null_max_abs_d": null_max,
            "orig_abs_d": orig_d_abs,
            "pass": null_pass,
        },
        "test_3_within_group": {
            "within_pos_d": within_pos_d,
            "within_neg_d": within_neg_d,
            "signal_noise_ratio": signal_noise,
            "pass": within_pass,
        },
        "test_4_vocab": {
            "match": vocab_match,
            "pass": vocab_pass,
        },
        "verdict": verdict,
        "model_id": main["model_id"],
    }


def _replicate_probe_multilayer(
    mri_path: str, pos_prompts: list, neg_prompts: list,
    layers: list, model_id: str = "",
    expected_tokens_pos: list | None = None,
    expected_tokens_neg: list | None = None,
    n_random_null: int = 100,
    position: int = 0,
) -> dict:
    """Replicate-probe at multiple layers, ONE forward pass per prompt.

    ~N× faster than calling `_replicate_probe` N times for N layers:
    both prompt-sets go through the model once, we slice out the
    requested layers from the cached residuals, and run the 5-test
    pipeline per layer offline.

    On Qwen-7B: 6-layer sweep drops from ~60 min (6× the 10-min single-layer
    run) to ~10-12 min (one forward-pass batch + cheap per-layer analysis).
    """
    if len(pos_prompts) < 6 or len(neg_prompts) < 6:
        return {"error": "Need at least 6 prompts per side for within-group splits"}
    if not layers:
        return {"error": "No layers specified"}

    if not model_id:
        try:
            mri_meta = json.loads((Path(mri_path) / "metadata.json").read_text())
            model_id = mri_meta["model"].get("huggingface_id") or mri_meta["model"].get("name", "")
        except (FileNotFoundError, KeyError):
            return {"error": "No MRI metadata; supply model_id explicitly."}
    try:
        backend = _get_steer_backend(model_id)
    except Exception as e:
        return {"error": f"Cannot load model {model_id}: {e}"}

    # --- Single capture pass: one backend.forward per prompt, all layers ---
    import time as _t
    t_cap = _t.time()
    pos_captures = [_capture_residuals_multilayer(backend, p, layers, position)
                    for p in pos_prompts]
    neg_captures = [_capture_residuals_multilayer(backend, p, layers, position)
                    for p in neg_prompts]
    capture_sec = _t.time() - t_cap

    # --- Per-layer analysis (cheap; no model forward) ---
    results_per_layer = {}
    for L in layers:
        pos_vecs = [c.get(L) for c in pos_captures]
        pos_vecs = [v for v in pos_vecs if v is not None]
        neg_vecs = [c.get(L) for c in neg_captures]
        neg_vecs = [v for v in neg_vecs if v is not None]
        if len(pos_vecs) < 3 or len(neg_vecs) < 3:
            results_per_layer[L] = {"error": "Too few captured residuals"}
            continue

        P = np.stack(pos_vecs).astype(np.float32)
        N = np.stack(neg_vecs).astype(np.float32)
        diff = P.mean(axis=0) - N.mean(axis=0)
        mag = float(np.linalg.norm(diff))
        direction = diff / (mag + 1e-8)

        # Cohen's d
        pp = P @ direction; nn = N @ direction
        pooled = float(np.sqrt((pp.var(ddof=1) + nn.var(ddof=1)) / 2 + 1e-12))
        cohen_d = float((pp.mean() - nn.mean()) / (pooled + 1e-8))

        # Bootstrap cosine stability
        rng = np.random.RandomState(42 + L)
        boot_cos = []
        for _ in range(50):
            bp = P[rng.randint(0, len(pos_vecs), size=len(pos_vecs))].mean(axis=0)
            bn = N[rng.randint(0, len(neg_vecs), size=len(neg_vecs))].mean(axis=0)
            bd = bp - bn
            bnm = float(np.linalg.norm(bd))
            if bnm < 1e-8:
                continue
            boot_cos.append(float((bd / bnm) @ direction))
        boot_cos = np.array(boot_cos, dtype=np.float32) if boot_cos else np.array([0.0])
        boot_cos.sort()
        boot_p5 = float(boot_cos[int(len(boot_cos) * 0.05)])

        # Permutation null: shuffle (pos+neg) labels, re-extract the
        # direction the same way, recompute Cohen's d. The distribution
        # of permutation d's is the correct null — it accounts for the
        # optimization bias that the original random-unit-vector baseline
        # missed. Replaces broken Test 2 + Test 3.
        n_pos = len(pos_vecs); n_neg = len(neg_vecs)
        all_vecs = np.concatenate([P, N], axis=0)
        perm_rng = np.random.RandomState(777 + L)
        n_perm = max(100, n_random_null)
        perm_d_abs = []
        for _ in range(n_perm):
            idx = perm_rng.permutation(n_pos + n_neg)
            A = all_vecs[idx[:n_pos]]
            B = all_vecs[idx[n_pos:n_pos + n_neg]]
            d_ = A.mean(axis=0) - B.mean(axis=0)
            m = float(np.linalg.norm(d_))
            if m < 1e-8:
                continue
            dir_ = d_ / m
            ap = A @ dir_; bp = B @ dir_
            ps = float(np.sqrt((ap.var(ddof=1) + bp.var(ddof=1)) / 2 + 1e-12))
            perm_d_abs.append(abs(float((ap.mean() - bp.mean()) / (ps + 1e-8))))
        perm_arr = np.array(perm_d_abs, dtype=np.float32); perm_arr.sort()
        perm_p95 = float(perm_arr[int(len(perm_arr) * 0.95)])
        perm_p99 = float(perm_arr[int(len(perm_arr) * 0.99)])
        # Report SNR = d / perm_p95 for continuity with prior runs
        snr = abs(cohen_d) / (perm_p95 + 1e-8)
        # Legacy fields kept for backcompat — interpreted as permutation-based now
        null_p95 = perm_p95
        within_pos_d = 0.0  # deprecated
        within_neg_d = 0.0  # deprecated

        # Vocab projection — MRI first, lm_head fallback
        top_pos = []; top_neg = []
        vocab_source = "none"
        comp_path = Path(mri_path) / "decomp" / f"L{L:02d}_components.npy"
        if comp_path.exists() and (Path(mri_path) / "tokens.npz").exists():
            try:
                components = np.load(str(comp_path))
                dir_pc = components @ direction
                scores = _get_scores_mmap(mri_path, L)
                if scores is not None:
                    proj_vocab = _project_chunked(scores, dir_pc[:scores.shape[1]])
                    srt = np.argsort(proj_vocab)
                    texts, _ = _get_vocab_map(mri_path)
                    for i in srt[-10:][::-1]:
                        top_pos.append({"idx": int(i), "text": texts[i], "proj": float(proj_vocab[i])})
                    for i in srt[:10]:
                        top_neg.append({"idx": int(i), "text": texts[i], "proj": float(proj_vocab[i])})
                    vocab_source = "mri"
            except Exception:
                pass
        if not top_pos and hasattr(backend, "project_through_lm_head"):
            try:
                logits = backend.project_through_lm_head(direction)
                tokenizer = backend.tokenizer
                srt = np.argsort(logits)
                for i in srt[-10:][::-1]:
                    text = tokenizer.decode([int(i)])
                    top_pos.append({"idx": int(i), "text": text, "proj": float(logits[i])})
                for i in srt[:10]:
                    text = tokenizer.decode([int(i)])
                    top_neg.append({"idx": int(i), "text": text, "proj": float(logits[i])})
                vocab_source = "lm_head"
            except Exception:
                pass

        # Vocab match test
        vocab_match = None
        if expected_tokens_pos or expected_tokens_neg:
            pos_texts = {t["text"].strip().lower() for t in top_pos}
            neg_texts = {t["text"].strip().lower() for t in top_neg}
            exp_pos = {x.strip().lower() for x in (expected_tokens_pos or [])}
            exp_neg = {x.strip().lower() for x in (expected_tokens_neg or [])}
            def _mcount(got, expected):
                """Substring match. Ignore empty got (strip of whitespace
                tokens) — "" in anything is True and would spuriously
                match every expected token. Also ignore single-char got
                (punctuation) unless the expected is exactly that char."""
                n = 0
                clean_got = {g for g in got if len(g) >= 2}
                for e in expected:
                    for g in clean_got:
                        if e in g or g in e:
                            n += 1
                            break
                return n
            vocab_match = {
                "pos_matches": _mcount(pos_texts, exp_pos),
                "neg_matches": _mcount(neg_texts, exp_neg),
                "pos_expected": len(exp_pos),
                "neg_expected": len(exp_neg),
            }
        boot_pass = boot_p5 > 0.7
        # Permutation pass: orig d exceeds the 95th percentile of shuffled-label d's.
        # This is the correctly-calibrated null — replaces the broken Test 2 + Test 3.
        perm_pass = abs(cohen_d) > perm_p95
        # SNR threshold relaxed from 2.5 (for the broken within-group test) to 1.1
        # since perm_p95 already IS the 95% null floor. Anything meaningfully above
        # it is real signal.
        within_pass = snr > 1.1
        vocab_pass = (vocab_match is None or
                      vocab_match["pos_matches"] >= 3 or
                      vocab_match["neg_matches"] >= 3)
        all_pass = boot_pass and perm_pass and within_pass and vocab_pass
        any_pass = boot_pass or perm_pass
        verdict = ("robust_feature" if all_pass else
                   "partial" if any_pass else "falsified")

        results_per_layer[L] = {
            "layer": L,
            "cohen_d": cohen_d,
            "magnitude": mag,
            "direction_hidden": direction.tolist(),
            "test_1_bootstrap": {"boot_cosine_p5": boot_p5, "pass": boot_pass},
            "test_2_permutation": {
                "perm_p95": perm_p95,
                "perm_p99": perm_p99,
                "orig_abs_d": abs(cohen_d),
                "pass": perm_pass,
            },
            "test_3_signal_to_noise": {
                "signal_noise_ratio": snr,  # d / perm_p95
                "pass": within_pass,
            },
            "test_4_vocab": {"match": vocab_match, "pass": vocab_pass,
                             "vocab_source": vocab_source},
            "top_pos": top_pos, "top_neg": top_neg,
            "verdict": verdict,
        }

    return {
        "layers": list(layers),
        "n_pos": len(pos_prompts),
        "n_neg": len(neg_prompts),
        "position": position,
        "capture_sec": capture_sec,
        "model_id": model_id,
        "results": results_per_layer,
    }


def _causal_truth_test(
    direction: list,  # hidden-space unit vector
    layer: int,
    statements: list,  # [{"statement": str, "label": 0 or 1}, ...]
    model_id: str,
    alphas: list | None = None,
    eval_template: str = 'True or false: "{}" Answer:',
) -> dict:
    """Behavioral causal test: does steering with `direction` flip the
    model's truth judgment?

    For each statement, build an evaluation prompt. Run forward with and
    without steering. Extract logits for " True" and " False" tokens.
    Compute truth-score = logit(True) − logit(False).

    Predictions:
      - If `direction` is a true truth direction (M&T's claim): +α should
        raise truth-score for FALSE statements (bias toward asserting true)
        and −α should lower it for TRUE statements.
      - If `direction` is a flag-this-claim direction (Finding 02): α
        should shift the truth-score similarly for both, or differently in
        a way that doesn't track the statement's ground-truth label.

    The test reports the mean shift in truth-score per (label, alpha)
    cell. Clean answer to "is the direction causal-for-truth."
    """
    if not statements:
        return {"error": "Need statements"}
    if alphas is None:
        alphas = [-3.0, -1.5, 0.0, +1.5, +3.0]
    try:
        backend = _get_steer_backend(model_id)
    except Exception as e:
        return {"error": f"Cannot load model {model_id}: {e}"}

    d = np.asarray(direction, dtype=np.float32)
    d /= np.linalg.norm(d) + 1e-8

    # Resolve truth/falsity tokens. Try the most common variants.
    tokenizer = backend.tokenizer
    def _tok_id(s):
        ids = tokenizer.encode(s)
        # Skip BOS if present
        if len(ids) >= 2 and ids[0] == getattr(tokenizer, "bos_token_id", -1):
            return ids[1]
        return ids[0] if ids else -1
    # Try both leading-space and capitalization variants; pick best later
    candidates = {
        "True": [_tok_id(" True"), _tok_id("True")],
        "False": [_tok_id(" False"), _tok_id("False")],
        "true": [_tok_id(" true"), _tok_id("true")],
        "false": [_tok_id(" false"), _tok_id("false")],
    }

    def _truth_score(logits: np.ndarray) -> tuple:
        """Return (truth_score, p_true, p_false) using the best candidate pair."""
        # Use log-softmax-ish comparison: just logit differences
        best = None
        for t_var, f_var in [(" True", " False"), ("True", "False"),
                              (" true", " false"), ("true", "false")]:
            t_id = _tok_id(t_var); f_id = _tok_id(f_var)
            if t_id < 0 or f_id < 0 or t_id >= len(logits) or f_id >= len(logits):
                continue
            lt = float(logits[t_id]); lf = float(logits[f_id])
            if best is None or abs(lt - lf) > abs(best[0]):
                best = (lt - lf, lt, lf, t_var, f_var)
        if best is None:
            return None, None, None, None, None
        return best  # (score, logit_true, logit_false, t_var, f_var)

    # Run per (alpha, label) cell
    results = []
    for stmt in statements:
        s_text = stmt["statement"]
        label = int(stmt["label"])
        prompt = eval_template.format(s_text)
        row = {"statement": s_text, "label": label, "alphas": {}}
        # For clean α=0 we can run without steering
        for alpha in alphas:
            # steer_dirs tuple is (direction, mean_gap). Runtime applies
            # direction × mean_gap × alpha. We want linear α scaling, so
            # mean_gap=1.0 and alpha varies.
            steer_dirs = {layer: (d, 1.0)} if abs(alpha) > 1e-8 else None
            try:
                if steer_dirs:
                    r = backend.forward(prompt, steer_dirs=steer_dirs, alpha=alpha)
                else:
                    r = backend.forward(prompt)
            except Exception as e:
                row["alphas"][alpha] = {"error": str(e)}
                continue
            logits = r.logits
            score, lt, lf, tv, fv = _truth_score(logits)
            # Top-1 token under this steering
            top_id = int(np.argmax(logits))
            top_tok = tokenizer.decode([top_id])
            row["alphas"][alpha] = {
                "truth_score": score,
                "logit_true": lt, "logit_false": lf,
                "top_token": top_tok,
                "t_var": tv, "f_var": fv,
            }
        results.append(row)

    # Aggregate per (label, alpha)
    from collections import defaultdict
    agg = defaultdict(list)
    for row in results:
        for alpha, cell in row["alphas"].items():
            if "truth_score" in cell and cell["truth_score"] is not None:
                agg[(row["label"], alpha)].append(cell["truth_score"])
    summary = {}
    for (label, alpha), scores in agg.items():
        arr = np.array(scores, dtype=np.float32)
        summary[f"label={label},alpha={alpha}"] = {
            "n": len(scores),
            "mean_truth_score": float(arr.mean()),
            "std": float(arr.std()),
            "frac_positive": float((arr > 0).mean()),
        }

    # Compute the effect per label: truth_score at +α vs -α
    effects = {}
    for label in (0, 1):
        sorted_alphas = sorted(alphas)
        if len(sorted_alphas) >= 2:
            low = sorted_alphas[0]
            high = sorted_alphas[-1]
            low_scores = [c["truth_score"]
                          for r in results if r["label"] == label
                          for a, c in r["alphas"].items()
                          if a == low and "truth_score" in c and c["truth_score"] is not None]
            high_scores = [c["truth_score"]
                           for r in results if r["label"] == label
                           for a, c in r["alphas"].items()
                           if a == high and "truth_score" in c and c["truth_score"] is not None]
            if low_scores and high_scores:
                effects[f"label={label}"] = {
                    "low_alpha": low, "mean_score_at_low": float(np.mean(low_scores)),
                    "high_alpha": high, "mean_score_at_high": float(np.mean(high_scores)),
                    "causal_shift": float(np.mean(high_scores) - np.mean(low_scores)),
                }

    return {
        "layer": layer,
        "alphas": alphas,
        "n_statements": len(statements),
        "per_statement": results,
        "aggregate_by_label_alpha": summary,
        "causal_effects": effects,
        "model_id": model_id,
    }


def _live_forward(mri_path: str, prompt: str, model_id: str = "") -> dict:
    """Run a prompt through the live model, project residuals into MRI's PCA space.

    Returns per-layer PCA scores for each token position in the prompt,
    in the same format as the stored MRI scores. These can be compared
    directly with the pre-captured vocabulary.
    """
    decomp = Path(mri_path) / "decomp"
    meta_path = decomp / "meta.json"
    if not meta_path.exists():
        return {"error": "No decomposition metadata"}

    meta = json.loads(meta_path.read_text())
    n_layers = meta.get("n_real_layers", len(meta["layers"]))

    # Get model ID from MRI metadata
    if not model_id:
        mri_meta = json.loads((Path(mri_path) / "metadata.json").read_text())
        model_id = mri_meta["model"]["name"]

    # Load model
    try:
        backend = _get_steer_backend(model_id)
    except Exception as e:
        return {"error": f"Cannot load model {model_id}: {e}"}

    # Forward with residual capture at all layers
    all_layers = list(range(n_layers))
    try:
        result = backend.forward(prompt, return_residual=True, residual_layers=all_layers)
    except Exception as e:
        return {"error": f"Forward pass failed: {e}"}

    # Get residuals
    residuals = getattr(result, 'residuals', None)
    if not residuals:
        return {"error": "Backend did not return residuals. Check residual_layers support."}

    # Project each layer's residual through stored PCA components
    live_scores = {}
    for layer in range(n_layers):
        comp_path = decomp / f"L{layer:02d}_components.npy"
        if not comp_path.exists() or layer not in residuals:
            continue
        components = np.load(str(comp_path))  # [K, hidden_dim]
        residual = residuals[layer]  # [hidden_dim] (last position)
        if residual.ndim == 1:
            residual = residual.reshape(1, -1)
        scores = (residual.astype(np.float32) @ components.T)  # [1, K] or [seq_len, K]
        live_scores[layer] = scores.tolist()

    # Tokenize the prompt for display
    token_texts = []
    if hasattr(backend, 'tokenizer'):
        try:
            ids = backend.tokenizer.encode(prompt)
            token_texts = [backend.tokenizer.decode([tid]) for tid in ids]
        except Exception:
            token_texts = [prompt]

    return {
        "prompt": prompt,
        "model_id": model_id,
        "n_layers": n_layers,
        "token_texts": token_texts,
        "scores": live_scores,
        "logits_top5": [
            {"id": int(result.probs.argsort()[-1-i]),
             "prob": float(result.probs[result.probs.argsort()[-1-i]])}
            for i in range(min(5, len(result.probs)))
        ],
        "prediction": result.top_token,
        "entropy": float(result.entropy),
    }


_steer_backend_cache = {}

def _get_steer_backend(model_id: str):
    """Get or create a model backend for steering tests.

    Dispatch order:
      - path ending in .checkpoint.pt → decepticon backend (causal bank)
      - existing local directory → HF backend (local snapshot)
      - anything else → MLX backend (HF hub ID or local HF dir)

    Raises on load failure — caller wraps in try/except.
    """
    if model_id in _steer_backend_cache:
        return _steer_backend_cache[model_id]
    path = Path(model_id)
    if model_id.endswith(".checkpoint.pt") and path.exists():
        from .backend.decepticon import DecepticonBackend
        backend = DecepticonBackend(str(path))
    elif path.is_dir() and (path / "config.json").exists():
        from .backend.hf import HFBackend
        backend = HFBackend(str(path))
    else:
        from .backend.mlx import MLXBackend
        backend = MLXBackend(model_id)
    _steer_backend_cache[model_id] = backend
    return backend


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




# Direction-analysis functions extracted to sibling module. Import at bottom
# so their late-bound `_c.*` lookups see a fully-loaded companion module.
from .companion_directions import (  # noqa: E402
    _direction_depth,
    _direction_circuit,
    _direction_weight_alignment,
    _direction_cross_model,
    _direction_brief,
    _direction_bootstrap,
    _find_token_by_text,
    _pca_reconstruction,
)


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

        # Chat outbox long-poll: browser waits for MCP replies
        if path == '/api/chat-poll':
            timeout = float(qs.get('timeout', ['25'])[0])
            _chat_event.wait(timeout=timeout)
            with _chat_lock:
                if _chat_outbox:
                    reply = _chat_outbox.pop(0)
                    if not _chat_outbox:
                        _chat_event.clear()
                    self._send_json(reply)
                else:
                    _chat_event.clear()
                    self._send_json({"reply": None})
            return

        # Chat inbox drain: MCP client pulls pending user messages.
        # Add ?timeout=N to long-poll; wakes on first message instead of polling.
        if path == '/api/chat-drain':
            timeout = float(qs.get('timeout', ['0'])[0])
            if timeout > 0:
                with _chat_lock:
                    have_now = bool(_chat_inbox)
                if not have_now:
                    _chat_inbox_event.wait(timeout=min(timeout, 60.0))
            self._send_json({"messages": chat_drain()})
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
        elif path.startswith('/api/token-resolve/'):
            # /api/token-resolve/<model>/<mode>?text=...
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                text = qs.get('text', [''])[0]
                mri_path = self._mri_path(model, mode)
                idx = _find_token_by_text(mri_path, text)
                self._send_json({"text": text, "idx": idx})
            else:
                self._send_json({"error": "Usage: /api/token-resolve/<model>/<mode>?text=..."})
        elif path.startswith('/api/direction-bootstrap/'):
            # /api/direction-bootstrap/<model>/<mode>?a=N&b=N&layer=L&n_boot=100
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                a = int(qs.get('a', ['0'])[0])
                b = int(qs.get('b', ['0'])[0])
                layer = int(qs.get('layer', ['0'])[0])
                n_boot = int(qs.get('n_boot', ['100'])[0])
                n_random = int(qs.get('n_random', ['100'])[0])
                neighborhood = int(qs.get('neighborhood', ['20'])[0])
                mri_path = self._mri_path(model, mode)
                self._send_json(_direction_bootstrap(
                    mri_path, a, b, layer,
                    n_boot=n_boot, n_random=n_random, neighborhood=neighborhood))
            else:
                self._send_json({"error": "Usage: /api/direction-bootstrap/<model>/<mode>?a=N&b=N&layer=L"})
        elif path.startswith('/api/direction-brief/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                a = int(qs.get('a', ['0'])[0])
                b = int(qs.get('b', ['0'])[0])
                layer = int(qs.get('layer', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                self._send_json(_direction_brief(mri_path, a, b, layer))
            else:
                self._send_json({"error": "Usage: /api/direction-brief/<model>/<mode>?a=N&b=N&layer=L"})
        elif path.startswith('/api/pca-reconstruction/'):
            # PCA faithfulness at a layer. /<model>/<mode>?layer=L&top_k=50
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                layer = int(qs.get('layer', ['0'])[0])
                top_k = int(qs.get('top_k', ['50'])[0])
                mri_path = self._mri_path(model, mode)
                self._send_json(_pca_reconstruction(mri_path, layer, top_k=top_k))
            else:
                self._send_json({"error": "Usage: /api/pca-reconstruction/<model>/<mode>?layer=L"})
        elif path == '/api/direction-cross':
            # Cross-model direction comparison. Query params:
            #   model_a, mode_a, a_a, b_a, model_b, mode_b, a_b, b_b
            try:
                mri_a = self._mri_path(qs['model_a'][0], qs['mode_a'][0])
                mri_b = self._mri_path(qs['model_b'][0], qs['mode_b'][0])
                result = _direction_cross_model(
                    mri_a, mri_b,
                    int(qs['a_a'][0]), int(qs['b_a'][0]),
                    int(qs['a_b'][0]), int(qs['b_b'][0]),
                )
                self._send_json(result)
            except (KeyError, ValueError) as e:
                self._send_json({"error": f"Usage: /api/direction-cross?model_a=...&mode_a=...&a_a=N&b_a=N&model_b=...&mode_b=...&a_b=N&b_b=N. {e}"})
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

        # Binary endpoints — body is blob bytes, not JSON. Handle before json.loads
        # so PNG/GIF uploads don't get rejected.
        if path == '/api/capture-upload':
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
            return

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

        elif path == '/api/chat-reply':
            # External MCP client posts a reply; browser picks it up via /api/chat-poll.
            text = args.get("reply", "")
            rid = args.get("request_id", "")
            if not text:
                self._send_json({"error": "empty reply"})
                return
            chat_reply(text, request_id=rid)
            self._send_json({"ok": True})

        elif path == '/api/chat':
            # Browser posts a chat message. Queues for MCP client drain via chat_drain().
            msg = args.get("message", "").strip()
            if not msg:
                self._send_json({"error": "empty message"})
                return
            import uuid
            rid = uuid.uuid4().hex[:8]
            with _chat_lock:
                _chat_inbox.append({
                    "request_id": rid,
                    "message": msg,
                    "pinnedA": args.get("pinnedA"),
                    "pinnedB": args.get("pinnedB"),
                    "layer": args.get("layer"),
                    "model": args.get("model"),
                    "mode": args.get("mode"),
                    "ts": time.time(),
                })
                while len(_chat_inbox) > _CHAT_MAX:
                    _chat_inbox.pop(0)
            _chat_inbox_event.set()
            self._send_json({"ok": True, "request_id": rid,
                             "pending": "queued for MCP client"})

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

        elif path == '/api/capture-result':
            # JSON capture result from browser (base64 data or error)
            rid = args.get("request_id", "")
            with _capture_lock:
                _capture_results[rid] = args
                ev = _capture_pending.get(rid)
                if ev:
                    ev.set()
            self._send_json({"ok": True})

        elif path == '/api/residual-trajectory':
            # POST body: {model, mode, prompt, layer, n_generated, model_id}
            model = args.get("model", "")
            mode = args.get("mode", "raw")
            prompt = args.get("prompt", "")
            layer = int(args.get("layer", 0))
            n_generated = int(args.get("n_generated", 10))
            model_id = args.get("model_id", "")
            if not prompt:
                self._send_json({"error": "prompt required"})
                return
            if not model:
                self._send_json({"error": "model required"})
                return
            mri_path = self._mri_path(model, mode)
            self._send_json(_residual_trajectory(
                mri_path, prompt, layer, n_generated=n_generated, model_id=model_id))
        elif path == '/api/causal-truth-test':
            # POST: {direction:[...], layer, statements:[{statement, label}], model_id, alphas}
            direction = args.get("direction") or []
            layer = int(args.get("layer", 20))
            stmts = args.get("statements") or []
            model_id = args.get("model_id", "")
            alphas = args.get("alphas")
            if not direction or not stmts or not model_id:
                self._send_json({"error": "direction, statements, model_id required"})
                return
            self._send_json(_causal_truth_test(
                direction, layer, stmts, model_id,
                alphas=alphas,
            ))
        elif path == '/api/replicate-probe-multilayer':
            # POST: {model, mode, layers:[...], pos_prompts, neg_prompts,
            #        expected_tokens_pos, expected_tokens_neg, model_id, n_random_null, position}
            model = args.get("model", "")
            mode = args.get("mode", "raw")
            layers = [int(x) for x in (args.get("layers") or [])]
            pos = args.get("pos_prompts") or []
            neg = args.get("neg_prompts") or []
            model_id = args.get("model_id", "")
            exp_pos = args.get("expected_tokens_pos")
            exp_neg = args.get("expected_tokens_neg")
            n_null = int(args.get("n_random_null", 100))
            position = int(args.get("position", 0))
            if not model:
                self._send_json({"error": "model required"})
                return
            mri_path = self._mri_path(model, mode)
            self._send_json(_replicate_probe_multilayer(
                mri_path, pos, neg, layers,
                model_id=model_id,
                expected_tokens_pos=exp_pos,
                expected_tokens_neg=exp_neg,
                n_random_null=n_null,
                position=position,
            ))
        elif path == '/api/replicate-probe':
            # POST body: {model, mode, layer, pos_prompts, neg_prompts, model_id,
            #             expected_tokens_pos, expected_tokens_neg, n_random_null, position}
            model = args.get("model", "")
            mode = args.get("mode", "raw")
            layer = int(args.get("layer", 0))
            pos = args.get("pos_prompts") or []
            neg = args.get("neg_prompts") or []
            model_id = args.get("model_id", "")
            exp_pos = args.get("expected_tokens_pos")
            exp_neg = args.get("expected_tokens_neg")
            n_null = int(args.get("n_random_null", 100))
            position = int(args.get("position", 0))
            if not model:
                self._send_json({"error": "model required"})
                return
            mri_path = self._mri_path(model, mode)
            self._send_json(_replicate_probe(
                mri_path, pos, neg, layer,
                model_id=model_id,
                expected_tokens_pos=exp_pos,
                expected_tokens_neg=exp_neg,
                n_random_null=n_null,
                position=position,
            ))
        elif path == '/api/behavioral-direction':
            # POST body: {model, mode, layer, pos_prompts:[...], neg_prompts:[...], model_id:"..."}
            model = args.get("model", "")
            mode = args.get("mode", "raw")
            layer = int(args.get("layer", 0))
            pos = args.get("pos_prompts") or []
            neg = args.get("neg_prompts") or []
            model_id = args.get("model_id", "")
            if not model:
                self._send_json({"error": "model required"})
                return
            mri_path = self._mri_path(model, mode)
            result = _behavioral_direction(
                mri_path, pos, neg, layer, model_id=model_id)
            self._send_json(result)

        elif path == '/api/live-forward':
            prompt = args.get("prompt", "")
            model_id = args.get("model_id", "")  # HF model ID for loading
            mri_model = args.get("mri_model", "qwen-0.5b")  # MRI directory name
            mode_name = args.get("mode", "raw")
            if not prompt:
                self._send_json({"error": "prompt required"})
                return
            mri_path = self._mri_path(mri_model, mode_name)
            result = _live_forward(mri_path, prompt, model_id=model_id)
            self._send_json(result)

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
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Browser closed the socket mid-response (navigation, abort,
            # poll cancellation). Normal HTTP lifecycle event — don't
            # let it bubble up and crash the request thread.
            pass

    def _send_bytes(self, data: bytes):
        try:
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
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

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
