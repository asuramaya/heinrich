"""Direction-analysis functions for the companion viewer.

Extracted from companion.py to keep the main module focused on HTTP routing
and data loading. These functions analyze the concept direction between two
pinned tokens (A - B) across all layers.

Shared state (score cache, helpers) lives in companion.py and is accessed
via late-bound module reference (`_c.*` at call time) to avoid circular
import issues.

_direction_quality stays in companion.py because it needs token-text/script
metadata from tokens.npz (object arrays). The three functions here operate
on cached score matrices only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _c():
    """Late-bound companion module — avoids circular import at load time.

    Resolves via sys.modules so either load order works (companion imports
    this, or this imported first). After both modules finish loading, all
    attributes (`_c()._bimodality`, `_c()._get_scores_f32`, etc.) are live.
    """
    mod = sys.modules.get("heinrich.companion")
    if mod is None:
        import heinrich.companion  # noqa: F401 — populate sys.modules
        mod = sys.modules["heinrich.companion"]
    return mod


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
        # Zero-copy mmap — avoids 537 MB float32 allocation per layer.
        scores = _c()._get_scores_mmap(mri_path, li)
        if scores is None:
            continue
        N, K = scores.shape
        if a >= N or b >= N:
            continue

        # Upcast only the two anchor rows (~4 KB each), not the whole matrix.
        row_a = scores[a].astype(np.float32)
        row_b = scores[b].astype(np.float32)
        diff = row_a - row_b
        mag = float(np.linalg.norm(diff))
        direction = diff / (mag + 1e-8)

        # Chunked projection: memory-bounded, ~3× faster than astype+matmul.
        full_proj = _c()._project_chunked(scores, direction)
        pa = float(full_proj[a])
        pb = float(full_proj[b])

        step = max(1, N // 500)
        sample_proj = full_proj[::step]
        p10 = float(np.percentile(sample_proj, 10))
        p50 = float(np.percentile(sample_proj, 50))
        p90 = float(np.percentile(sample_proj, 90))

        diff2 = diff ** 2
        total_d2 = float(diff2.sum())
        order = np.argsort(diff2)[::-1]
        cumul = np.cumsum(diff2[order]) / (total_d2 + 1e-8)
        pcs_50 = int(np.searchsorted(cumul, 0.5)) + 1

        top_n = min(8, len(order))
        top_pcs = [{"pc": int(order[i]),
                    "share": float(diff2[order[i]] / (total_d2 + 1e-8))}
                   for i in range(top_n)]

        bm = _c()._bimodality(full_proj)
        random_bms = _c()._random_bimodality_baseline(scores)
        pctile = _c()._bimodality_percentile(bm, random_bms)

        layers.append({
            "layer": li, "magnitude": mag,
            "proj_a": pa, "proj_b": pb,
            "p10": p10, "p50": p50, "p90": p90,
            "pcs_50": pcs_50, "top_pcs": top_pcs,
            "bimodality": bm,
            "random_baseline_percentile": pctile,
        })

    best_layer = None
    worst_layer = None
    bimodality_range = None
    if layers:
        best = min(layers, key=lambda l: l["bimodality"])
        worst = max(layers, key=lambda l: l["bimodality"])
        best_layer = best["layer"]
        worst_layer = worst["layer"]
        bimodality_range = [best["bimodality"], worst["bimodality"]]

    return {
        "layers": layers, "n_layers": n_layers,
        "best_layer": best_layer, "worst_layer": worst_layer,
        "bimodality_range": bimodality_range,
    }


def _direction_circuit(mri_path: str, a: int, b: int) -> dict:
    """Per-head and MLP write-attribution for the A-B concept direction.

    For each layer, measures how much of the concept direction each
    attention head's output subspace can reconstruct (via o_proj slices),
    plus the MLP's contribution (via down_proj).

    Sharkey et al. 2025 open problem 2.3: automatic circuit discovery
    from concept directions.
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
        # mmap — we only need rows a and b for this function.
        scores = _c()._get_scores_mmap(mri_path, li)
        if scores is None:
            continue
        N, K = scores.shape
        if a >= N or b >= N:
            continue

        diff = scores[a].astype(np.float32) - scores[b].astype(np.float32)
        mag = float(np.linalg.norm(diff))
        if mag < 1e-12:
            continue
        dir_pc = diff / mag

        comp_path = decomp / f"L{li:02d}_components.npy"
        components = _c()._get_weight_mmap(comp_path)
        if components is None:
            continue
        dir_hidden = components.T @ dir_pc[:components.shape[0]]
        dir_norm = float(np.linalg.norm(dir_hidden))
        if dir_norm < 1e-12:
            continue
        dir_hidden = dir_hidden / dir_norm

        o_proj = _c()._get_weight_mmap(mri_dir / "weights" / f"L{li:02d}" / "o_proj.npy")
        heads_out = []
        if o_proj is not None:
            for h in range(n_heads):
                W_h = o_proj[:, h * head_dim:(h + 1) * head_dim]
                proj = W_h @ (W_h.T @ dir_hidden)
                attrib = float(np.linalg.norm(proj))
                heads_out.append({"head": h, "attrib": attrib})
        else:
            heads_out = [{"head": h, "attrib": 0.0} for h in range(n_heads)]

        mlp_attrib = 0.0
        down_proj = _c()._get_weight_mmap(mri_dir / "weights" / f"L{li:02d}" / "down_proj.npy")
        if down_proj is not None:
            proj_mlp = down_proj @ (down_proj.T @ dir_hidden)
            mlp_attrib = float(np.linalg.norm(proj_mlp))

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

    peak_zscore = None
    peak_layer = None
    if layers_out:
        best_li = max(layers_out, key=lambda ld: max(
            h["attrib"] for h in ld["heads"]))
        peak_layer = best_li["layer"]

        scores = _c()._get_scores_mmap(mri_path, peak_layer)
        comp_path = decomp / f"L{peak_layer:02d}_components.npy"
        o_proj_path = mri_dir / "weights" / f"L{peak_layer:02d}" / "o_proj.npy"
        down_proj_path = mri_dir / "weights" / f"L{peak_layer:02d}" / "down_proj.npy"

        if (scores is not None and comp_path.exists()):
            diff = scores[a].astype(np.float32) - scores[b].astype(np.float32)
            mag = float(np.linalg.norm(diff))
            if mag > 1e-12:
                dir_pc = diff / mag
                components = np.load(str(comp_path))
                dir_hidden = components.T @ dir_pc[:components.shape[0]]
                dir_norm = float(np.linalg.norm(dir_hidden))
                if dir_norm > 1e-12:
                    dir_hidden = dir_hidden / dir_norm

                    o_proj = np.load(str(o_proj_path)) if o_proj_path.exists() else None
                    down_proj = np.load(str(down_proj_path)) if down_proj_path.exists() else None

                    def _attribs_for_dir(d):
                        h_out = []
                        if o_proj is not None:
                            for h in range(n_heads):
                                W_h = o_proj[:, h * head_dim:(h + 1) * head_dim]
                                p = W_h @ (W_h.T @ d)
                                h_out.append(float(np.linalg.norm(p)))
                        else:
                            h_out = [0.0] * n_heads
                        m_attr = 0.0
                        if down_proj is not None:
                            p = down_proj @ (down_proj.T @ d)
                            m_attr = float(np.linalg.norm(p))
                        return h_out, m_attr

                    concept_heads, concept_mlp = _attribs_for_dir(dir_hidden)

                    rng = np.random.RandomState(42)
                    n_random = 50
                    random_heads = []
                    random_mlps = []
                    for _ in range(n_random):
                        rv = rng.randn(hidden_size).astype(np.float32)
                        rv = rv / (float(np.linalg.norm(rv)) + 1e-8)
                        rh, rm = _attribs_for_dir(rv)
                        random_heads.append(rh)
                        random_mlps.append(rm)

                    random_heads = np.array(random_heads)
                    random_mlps = np.array(random_mlps)

                    head_mean = random_heads.mean(axis=0)
                    head_std = random_heads.std(axis=0)
                    mlp_mean = float(random_mlps.mean())
                    mlp_std = float(random_mlps.std())

                    head_zscores = []
                    for h in range(n_heads):
                        z = ((concept_heads[h] - head_mean[h])
                             / (head_std[h] + 1e-8))
                        head_zscores.append({
                            "head": h,
                            "attrib": concept_heads[h],
                            "zscore": float(z),
                        })

                    mlp_z = (concept_mlp - mlp_mean) / (mlp_std + 1e-8)
                    peak_zscore = {
                        "heads": head_zscores,
                        "mlp": {
                            "attrib": concept_mlp,
                            "zscore": float(mlp_z),
                        },
                    }

    result = {"layers": layers_out, "n_heads": n_heads, "n_layers": n_layers}
    if peak_layer is not None:
        result["peak_layer"] = peak_layer
    if peak_zscore is not None:
        result["peak_zscore"] = peak_zscore
    return result


def _direction_bootstrap(mri_path: str, a: int, b: int, layer: int,
                         n_boot: int = 100, neighborhood: int = 20,
                         n_random: int = 100, seed: int = 42) -> dict:
    """Anchor-stability + random-null baselines for a direction at one layer.

    Two distributions that together falsify or ratify a direction claim:

    1. **Bootstrap**: resample anchors from the two endpoints' nearest
       neighborhoods, re-extract the direction, measure cosine to the
       original. A tight distribution (p5 > 0.9) means the direction is
       robust to which specific tokens you pinned; a wide one (p5 < 0.5)
       means you've picked an anchor-specific artifact.

    2. **Random unit null**: 100 random unit vectors in hidden space.
       Compute the projection bimodality of each. If the original
       bimodality is inside this null distribution, "BIMODAL" is not
       evidence of a real feature — it's what you'd get from noise.

    Returns both distributions as percentile summaries + the verdict:
      verdict in {"robust_feature", "anchor_sensitive", "not_distinguishable_from_noise"}.
    """
    scores = _c()._get_scores_mmap(mri_path, layer)
    if scores is None:
        return {"error": f"No scores at L{layer}"}
    N, K = scores.shape
    if a >= N or b >= N:
        return {"error": f"Token index out of range (max {N-1})"}

    # Original direction + its bimodality
    orig_diff = scores[a].astype(np.float32) - scores[b].astype(np.float32)
    orig_mag = float(np.linalg.norm(orig_diff))
    if orig_mag < 1e-8:
        return {"error": "Anchor tokens identical"}
    orig_dir = orig_diff / orig_mag
    orig_proj = _c()._project_chunked(scores, orig_dir)
    orig_bm = _c()._bimodality(orig_proj)

    # --- Bootstrap: top-N near A, top-N near B, pick pairs ---
    sorted_idx = np.argsort(orig_proj)
    a_neighbors = sorted_idx[-neighborhood:][::-1]
    b_neighbors = sorted_idx[:neighborhood]
    rng = np.random.RandomState(seed)
    boot_cos = np.empty(n_boot, dtype=np.float32)
    boot_bm = np.empty(n_boot, dtype=np.float32)
    for i in range(n_boot):
        aa = int(rng.choice(a_neighbors))
        bb = int(rng.choice(b_neighbors))
        if aa == bb:
            boot_cos[i] = 1.0
            boot_bm[i] = orig_bm
            continue
        alt_diff = scores[aa].astype(np.float32) - scores[bb].astype(np.float32)
        alt_mag = float(np.linalg.norm(alt_diff))
        if alt_mag < 1e-8:
            boot_cos[i] = 0.0
            boot_bm[i] = 1.0
            continue
        alt_dir = alt_diff / alt_mag
        boot_cos[i] = float(alt_dir @ orig_dir)
        boot_bm[i] = _c()._bimodality(_c()._project_chunked(scores, alt_dir))

    # --- Random unit null: unit vectors in PCA score space ---
    # Faster than random token-pair directions and more principled for
    # "would any random direction look this bimodal?"
    null_dirs = rng.randn(n_random, K).astype(np.float32)
    null_dirs /= np.linalg.norm(null_dirs, axis=1, keepdims=True) + 1e-8
    null_projs = _c()._project_chunked(scores, null_dirs.T)  # [N, n_random]
    null_bm = np.array([_c()._bimodality(null_projs[:, j])
                        for j in range(n_random)], dtype=np.float32)

    def _summary(arr):
        s = np.sort(arr)
        return {
            "p5":  float(s[int(len(s) * 0.05)]),
            "p25": float(s[int(len(s) * 0.25)]),
            "p50": float(s[int(len(s) * 0.50)]),
            "p75": float(s[int(len(s) * 0.75)]),
            "p95": float(s[int(len(s) * 0.95)]),
            "mean": float(s.mean()),
            "std":  float(s.std()),
        }

    cos_stats = _summary(boot_cos)
    boot_bm_stats = _summary(boot_bm)
    null_bm_stats = _summary(null_bm)
    # Verdict:
    #   anchor_sensitive: p5 cosine below 0.5 (direction flips on small anchor perturbation)
    #   not_distinguishable: original bimodality >= null p5 (more unimodal than 5% of noise)
    #   robust_feature: both checks pass
    anchor_sensitive = cos_stats["p5"] < 0.5
    null_p5 = null_bm_stats["p5"]
    not_distinguishable = orig_bm >= null_p5
    if anchor_sensitive:
        verdict = "anchor_sensitive"
    elif not_distinguishable:
        verdict = "not_distinguishable_from_noise"
    else:
        verdict = "robust_feature"

    return {
        "layer": layer, "K": K, "N": N,
        "orig_bimodality": orig_bm,
        "bootstrap_cosine": cos_stats,
        "bootstrap_bimodality": boot_bm_stats,
        "null_bimodality": null_bm_stats,
        "anchor_sensitive": anchor_sensitive,
        "not_distinguishable_from_noise": not_distinguishable,
        "verdict": verdict,
        "n_boot": n_boot,
        "n_random": n_random,
        "neighborhood": neighborhood,
    }


def _direction_brief(mri_path: str, a: int, b: int, layer: int) -> dict:
    """Cheap falsification signals for the critical path.

    Returns just the fast signals the UI needs to warn users about
    geometric-only / weak directions, without doing the full all-layer
    sweep. Computed once per pin change on the focal layer.

    - bimodality: 100-bin histogram valley/peak ratio on the full projection.
    - random_baseline_percentile: % of 50 random token-pair directions that
      are MORE unimodal than this one (cached per MRI layer).
    - functional_hit_rate / functional_warning: do top A/B tokens actually
      predict A/B through lmhead? (5 checks, ~50 ms warm.)
    """
    scores = _c()._get_scores_mmap(mri_path, layer)
    if scores is None:
        return {"error": f"No scores at L{layer}"}
    N, K = scores.shape
    if a >= N or b >= N:
        return {"error": f"Token index out of range (max {N-1})"}

    diff = scores[a].astype(np.float32) - scores[b].astype(np.float32)
    mag = float(np.linalg.norm(diff))
    direction = diff / (mag + 1e-8)
    full_proj = _c()._project_chunked(scores, direction)
    bm = _c()._bimodality(full_proj)
    random_bms = _c()._random_bimodality_baseline(scores)
    pctile = _c()._bimodality_percentile(bm, random_bms)

    sorted_idx = np.argsort(full_proj)
    top_a_idx = sorted_idx[-5:][::-1]
    top_b_idx = sorted_idx[:5]
    hit_rate, warning = _c()._functional_hit_rate(
        mri_path, layer, a, b, top_a_idx, top_b_idx, k_check=5)

    return {
        "layer": layer, "K": K, "N": N,
        "magnitude": mag,
        "bimodality": bm,
        "random_baseline_percentile": pctile,
        "functional_hit_rate": hit_rate,
        "functional_warning": warning,
    }


def _pca_reconstruction(mri_path: str, layer: int, top_k: int = 50,
                        n_sample: int = 1000) -> dict:
    """Measure PCA faithfulness by reconstruction error at a layer.

    For each K in [1, 5, 10, 50, top_k, full], reconstruct residuals from
    top-K PCA components and measure mean cosine + relative MSE against the
    original residuals at sampled token positions.

    Answers "how much does the PCA projection lose?" at this layer.
    """
    mri_dir = Path(mri_path)
    decomp = mri_dir / "decomp"
    exit_path = mri_dir / f"L{layer:02d}_exit.npy"
    scores_path = decomp / f"L{layer:02d}_scores.npy"
    comp_path = decomp / f"L{layer:02d}_components.npy"
    if not exit_path.exists():
        return {"error": f"No exit at L{layer}"}
    if not (scores_path.exists() and comp_path.exists()):
        return {"error": f"No decomp at L{layer}"}
    exits = np.load(str(exit_path), mmap_mode="r")
    scores = np.load(str(scores_path), mmap_mode="r")
    components = np.load(str(comp_path))  # [K, hidden]
    N = exits.shape[0]
    K_full = min(scores.shape[1], components.shape[0])

    # Sample token rows (deterministic for repeatability)
    rng = np.random.RandomState(42)
    n = min(n_sample, N)
    idx = rng.choice(N, n, replace=False) if n < N else np.arange(N)
    orig = exits[idx].astype(np.float32)
    scored = scores[idx].astype(np.float32)
    # Center: PCA reconstruction = mean + scores[:,:k] @ components[:k]
    mean = orig.mean(axis=0, keepdims=True)
    centered_norms = np.linalg.norm(orig - mean, axis=1) + 1e-8

    ks = sorted(set([1, 5, 10, 50, top_k, K_full]))
    ks = [k for k in ks if 0 < k <= K_full]
    out = []
    for k in ks:
        recon = mean + scored[:, :k] @ components[:k]
        err = np.linalg.norm(orig - recon, axis=1)
        cos = np.sum((orig - mean) * (recon - mean), axis=1) / (
            centered_norms * (np.linalg.norm(recon - mean, axis=1) + 1e-8)
        )
        out.append({
            "k": int(k),
            "rel_mse": float(np.mean((err / centered_norms) ** 2)),
            "mean_cosine": float(np.mean(cos)),
            "k_share": k / K_full,
        })
    return {"layer": layer, "n_sample": n, "K_full": K_full, "curves": out}


def _find_token_by_text(mri_path: str, text: str) -> int:
    """Best-effort text match in another MRI's captured vocabulary.

    Returns the row index whose token text equals `text`, or -1 if no
    match. Exact match only — tokenizer-specific whitespace / byte-pair
    edges will prevent matches across different tokenizers, and that's
    the honest answer: there's no equivalent token.
    """
    texts, _ = _c()._get_vocab_map(mri_path)
    try:
        return texts.index(text)
    except ValueError:
        return -1


def _direction_cross_model(
    mri_path_a: str, mri_path_b: str,
    a_a: int, b_a: int, a_b: int, b_b: int,
) -> dict:
    """Compare the same concept direction across two MRI captures.

    Each model has its own token indices (different tokenizers), so the
    caller supplies one (a, b) pair per model. Returns per-model depth
    profile (magnitude, bimodality, best_layer) side-by-side, aligned on
    normalized layer fraction so curves are visually comparable even when
    layer counts differ.

    Runs A and B in parallel threads — numpy releases the GIL during
    matmul, so wall-clock drops to ~max(A, B) from ~A+B.
    """
    import threading
    results: dict = {}

    def _run(key, mri, a, b):
        results[key] = _direction_depth(mri, a, b)

    ta = threading.Thread(target=_run, args=("a", mri_path_a, a_a, b_a))
    tb = threading.Thread(target=_run, args=("b", mri_path_b, a_b, b_b))
    ta.start(); tb.start()
    ta.join(); tb.join()
    depth_a = results["a"]
    depth_b = results["b"]
    if "error" in depth_a:
        return {"error": f"Model A: {depth_a['error']}"}
    if "error" in depth_b:
        return {"error": f"Model B: {depth_b['error']}"}
    # Text equivalence check: comparable only if the tokens at the supplied
    # indices have matching text in both tokenizers. Flag mismatches so the
    # UI can warn; tokenizers fragment differently so raw index equality is
    # almost never the same concept.
    texts_a, _va = _c()._get_vocab_map(mri_path_a)
    texts_b, _vb = _c()._get_vocab_map(mri_path_b)
    ta_a = texts_a[a_a] if 0 <= a_a < len(texts_a) else None
    tb_a = texts_a[b_a] if 0 <= b_a < len(texts_a) else None
    ta_b = texts_b[a_b] if 0 <= a_b < len(texts_b) else None
    tb_b = texts_b[b_b] if 0 <= b_b < len(texts_b) else None
    text_match_a = ta_a is not None and ta_a == ta_b
    text_match_b = tb_a is not None and tb_a == tb_b
    # Summary fields per side
    def _summary(d):
        layers = d.get("layers", [])
        return {
            "n_layers": d.get("n_layers"),
            "best_layer": d.get("best_layer"),
            "worst_layer": d.get("worst_layer"),
            "bimodality_range": d.get("bimodality_range"),
            "peak_magnitude": max((l.get("magnitude", 0) for l in layers), default=0),
            "min_bimodality": min((l.get("bimodality", 1) for l in layers), default=1),
            "layers": [{
                "layer": l["layer"],
                "layer_frac": l["layer"] / max(1, d["n_layers"] - 1),
                "magnitude": l["magnitude"],
                "bimodality": l["bimodality"],
                "random_baseline_percentile": l.get("random_baseline_percentile"),
            } for l in layers],
        }
    return {
        "model_a": _summary(depth_a),
        "model_b": _summary(depth_b),
        "tokens": {
            "a_a": {"idx": a_a, "text": ta_a},
            "b_a": {"idx": b_a, "text": tb_a},
            "a_b": {"idx": a_b, "text": ta_b},
            "b_b": {"idx": b_b, "text": tb_b},
        },
        "text_match_a": text_match_a,
        "text_match_b": text_match_b,
        "comparable": text_match_a and text_match_b,
    }


def _direction_weight_alignment(mri_path: str, a: int, b: int) -> dict:
    """Per-matrix alignment of each layer's weights with the A-B concept direction.

    Maps the direction from PC space into hidden space via PCA components,
    then measures each weight matrix's ability to access that direction.
    alignment = ||W @ dir|| (or ||W^T @ dir|| for GQA non-square shapes).

    Sharkey et al. 2025 open problem 2.1.1: bridging PCA-space and
    weight-space representations.
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
        scores = _c()._get_scores_mmap(mri_path, li)
        if scores is None:
            continue
        N, K = scores.shape
        if a >= N or b >= N:
            continue

        diff = scores[a].astype(np.float32) - scores[b].astype(np.float32)
        mag = float(np.linalg.norm(diff))
        if mag < 1e-12:
            continue
        dir_pc = diff / mag

        comp_path = decomp / f"L{li:02d}_components.npy"
        components = _c()._get_weight_mmap(comp_path)
        if components is None:
            continue
        dir_hidden = (components.T @ dir_pc[:components.shape[0]]).astype(np.float32)
        dir_norm = float(np.linalg.norm(dir_hidden))
        if dir_norm < 1e-12:
            continue
        dir_hidden = dir_hidden / dir_norm

        weights_dir = mri_dir / "weights" / f"L{li:02d}"
        matrices = {}
        D = len(dir_hidden)
        for mname in MATRICES:
            wp = weights_dir / f"{mname}.npy"
            W = _c()._get_weight_mmap(wp)
            if W is None:
                continue
            # Prefer Gram cache: ||W @ d||² = d.T @ (W.T W) @ d  (side='col',
            # shape [K,K]) when W.shape[1]==D, else ||W.T @ d||² uses W@W.T.
            if W.shape[1] == D:
                gram = _c()._get_weight_gram(wp, "col")
                alignment = float(np.sqrt(max(0.0, float(dir_hidden @ gram @ dir_hidden)))) if gram is not None else 0.0
            elif W.shape[0] == D:
                gram = _c()._get_weight_gram(wp, "row")
                alignment = float(np.sqrt(max(0.0, float(dir_hidden @ gram @ dir_hidden)))) if gram is not None else 0.0
            else:
                alignment = 0.0
            matrices[mname] = alignment

        layers_out.append({"layer": li, "matrices": matrices})

    return {"layers": layers_out, "n_layers": n_layers, "matrix_names": MATRICES}
