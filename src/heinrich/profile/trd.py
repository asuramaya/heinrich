"""Generate a .trd file — the per-head thread map.

The .trd decomposes the attention component of displacement into per-head
contributions using the o_proj weight matrix. For each sampled token at
each measured layer, reports how much each attention head contributed to
the residual displacement.

No additional forward passes beyond the .shrt — uses the displacement
vectors already captured plus the o_proj weights (extracted once per layer).

Usage:
    heinrich trd-profile --model X --shrt Y.shrt.npz --frt Z.frt.npz
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def generate_trd(
    model,
    shrt_path: str,
    frt_path: str,
    *,
    layers: list[int] | None = None,
    output: str | None = None,
) -> dict:
    """Generate a .trd per-head attribution profile.

    Uses the o_proj weight matrix to project displacement vectors onto
    per-head subspaces. The projection magnitude indicates how much each
    head's output subspace contributes to the token's displacement.

    Args:
        model: loaded model (MLX)
        shrt_path: .shrt.npz with displacement vectors
        frt_path: .frt.npz for script labels
        layers: which layers to decompose (default: sample 6 evenly)
        output: path for .trd.npz output

    Returns:
        metadata dict
    """
    from .shrt import load_shrt
    from .frt import load_frt
    from ..cartography.oproj import extract_oproj_weight

    t0 = time.time()

    shrt = load_shrt(shrt_path)
    frt = load_frt(frt_path)
    vectors = shrt['vectors'].astype(np.float32)
    token_ids = shrt['token_ids']
    meta = shrt['metadata']

    n_tokens, hidden_size = vectors.shape
    inner = getattr(model, 'model', model)
    n_layers = len(inner.layers)
    n_heads = inner.layers[0].self_attn.n_heads
    head_dim = hidden_size // n_heads

    # Build script lookup
    fl = {int(frt['token_ids'][i]): str(frt['scripts'][i])
          for i in range(len(frt['token_ids']))}
    scripts = np.array([fl.get(int(tid), 'unknown') for tid in token_ids])

    # Select layers
    if layers is None:
        indices = np.linspace(0, n_layers - 1, 6, dtype=int)
        layers = sorted(set(indices.tolist()))

    print(f"Extracting per-head attribution: {n_tokens} tokens x "
          f"{len(layers)} layers x {n_heads} heads...")

    save_data = {
        "layers": np.array(layers, dtype=np.int32),
        "token_ids": token_ids,
        "scripts": scripts,
    }

    layer_head_data = {}
    for layer in layers:
        t_layer = time.time()

        # Extract o_proj weight: [hidden_size, n_heads * head_dim]
        W = extract_oproj_weight(model, layer)

        # W reshaped to [hidden_size, n_heads, head_dim]
        W_heads = W.reshape(hidden_size, n_heads, head_dim)

        # Project each displacement vector onto each head's output subspace
        # vectors: [n_tokens, hidden_size]
        # W_heads: [hidden_size, n_heads, head_dim]
        # Result: [n_tokens, n_heads, head_dim]
        projections = np.einsum('th,hnk->tnk', vectors, W_heads)
        # Per-head contribution magnitude: [n_tokens, n_heads]
        head_contribs = np.linalg.norm(projections, axis=2).astype(np.float32)

        save_data[f"heads_L{layer}"] = head_contribs
        layer_head_data[layer] = head_contribs

        elapsed_layer = time.time() - t_layer
        print(f"  L{layer}: {elapsed_layer:.1f}s")

    elapsed = time.time() - t0

    # Per-layer, per-script, per-head summary
    layer_summaries = {}
    for layer in layers:
        hc = layer_head_data[layer]
        total_per_token = hc.sum(axis=1, keepdims=True)
        fractions = hc / np.maximum(total_per_token, 1e-8)

        script_heads = {}
        for s in sorted(set(scripts)):
            if s in ('special', 'unknown'):
                continue
            mask = scripts == s
            if mask.sum() < 5:
                continue
            mean_frac = fractions[mask].mean(axis=0)
            top_heads = np.argsort(-mean_frac)[:3]
            script_heads[s] = {
                "n": int(mask.sum()),
                "top_heads": [
                    {"head": int(h), "fraction": round(float(mean_frac[h]), 4)}
                    for h in top_heads
                ],
                "head_entropy": round(float(
                    -np.sum(mean_frac * np.log2(mean_frac + 1e-12))
                ), 4),
            }

        mean_contrib = hc.mean(axis=0)
        top_overall = np.argsort(-mean_contrib)[:5]

        layer_summaries[str(layer)] = {
            "mean_contrib_per_head": [round(float(x), 2) for x in mean_contrib],
            "top_heads": [{"head": int(h), "mean_contrib": round(float(mean_contrib[h]), 2)}
                          for h in top_overall],
            "script_heads": script_heads,
        }

    metadata = {
        "version": "0.1",
        "type": "trd",
        "generated_at": time.time(),
        "elapsed_s": round(elapsed, 1),
        "model": {
            "name": meta['model']['name'],
            "n_layers": n_layers,
            "hidden_size": hidden_size,
            "n_heads": n_heads,
            "head_dim": head_dim,
        },
        "source_shrt": shrt_path,
        "layers": layers,
        "n_tokens": n_tokens,
        "layer_summaries": layer_summaries,
    }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        save_data["metadata"] = np.array([json.dumps(metadata, ensure_ascii=False)])
        np.savez_compressed(output, **save_data)
        print(f"\n  Saved to {output}")

    return metadata


def load_trd(path: str) -> dict:
    """Load a .trd file. Returns dict with arrays and metadata."""
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d['metadata'][0]))
    result = {
        "metadata": meta,
        "layers": d["layers"],
        "token_ids": d["token_ids"],
        "scripts": d["scripts"],
    }
    for key in d.files:
        if key.startswith("heads_L"):
            result[key] = d[key]
    return result
