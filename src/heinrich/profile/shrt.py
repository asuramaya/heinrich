"""Generate a .shrt file — complete shart autopsy for a model.

The .shrt is the model's shart profile: which tokens matter, how much,
measured against the model's own silence baseline. Generated automatically
from the partial index (15K tokens, converged) plus the full tool suite.

Usage:
    python -m heinrich.discover.shrt --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
    # or via CLI: heinrich shart-profile --model X
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def generate_shrt(
    backend,
    *,
    db=None,
    n_index: int = 15000,
    seed: int = 42,
    output: str | None = None,
) -> dict:
    """Generate a complete .shrt profile for a model.

    Args:
        backend: loaded model backend (MLX or HF)
        db: SignalDB (optional, for prompts and directions)
        n_index: number of tokens for the index (default 15K, converges by 7K)
        seed: random seed for token sampling
        output: path to write the .shrt JSON file

    Returns:
        The .shrt dict (also written to file if output is specified)
    """
    from heinrich.cartography.templates import build_prompt

    cfg = backend.config
    t0 = time.time()

    # === 1. Extract baseline: model's response to silence ===
    silence = backend.tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=False,
        add_generation_prompt=True,
    )
    best_layer = cfg.safety_layers[-1] if cfg.safety_layers else cfg.n_layers - 2
    baseline_fwd = backend.forward(silence, return_residual=True, residual_layer=best_layer)
    baseline_residual = baseline_fwd.residual
    baseline_top = baseline_fwd.top_token
    baseline_entropy = baseline_fwd.entropy

    # === 2. Safety direction (if DB available) ===
    safety_dir = None
    direction_accuracy = None
    if db is not None:
        try:
            from heinrich.discover.directions import find_direction

            harmful = db.require_prompts(is_benign=False, min_count=10, limit=30)
            benign = db.require_prompts(is_benign=True, min_count=10, limit=30)

            import random
            rng = random.Random(seed)
            rng.shuffle(harmful)
            rng.shuffle(benign)

            h_fmt = [build_prompt(r["text"], model_config=cfg) for r in harmful[:20]]
            b_fmt = [build_prompt(r["text"], model_config=cfg) for r in benign[:20]]

            states = backend.capture_residual_states(h_fmt + b_fmt, layers=[best_layer])
            if best_layer in states:
                sl = states[best_layer]
                dr = find_direction(sl[:len(h_fmt)], sl[len(h_fmt):],
                                    name="safety", layer=best_layer)
                safety_dir = dr.direction / np.linalg.norm(dr.direction)
                direction_accuracy = dr.separation_accuracy
        except Exception:
            pass

    # === 3. Build the index: scan n_index random tokens ===
    vocab_size = backend.tokenizer.vocab_size

    # Build real token list
    real_tokens = []
    for tid in range(vocab_size):
        tok = backend.tokenizer.decode([tid])
        if tok.strip() and len(tok) > 0 and not tok.startswith('[control') and not tok.startswith('<'):
            real_tokens.append((tid, tok))

    # Sample
    rng_np = np.random.RandomState(seed)
    n_sample = min(n_index, len(real_tokens))
    sample_idx = rng_np.choice(len(real_tokens), n_sample, replace=False)
    sample = [real_tokens[i] for i in sample_idx]

    tokens = []
    vectors = []  # full delta vectors — never discard
    for tid, tok in sample:
        try:
            fmt = backend.tokenizer.apply_chat_template(
                [{"role": "user", "content": tok}],
                tokenize=False,
                add_generation_prompt=True,
            )
            fwd = backend.forward(fmt, return_residual=True, residual_layer=best_layer)
            if fwd.residual is None:
                continue

            delta_vec = fwd.residual - baseline_residual
            delta = float(np.linalg.norm(delta_vec))
            entry = {
                "id": tid,
                "token": tok,
                "delta": round(delta, 2),
                "top_changed": fwd.top_token != baseline_top,
            }

            # Safety projection if direction available
            if safety_dir is not None:
                entry["safety_shift"] = round(
                    float(np.dot(delta_vec, safety_dir)), 4)

            tokens.append(entry)
            vectors.append(delta_vec.astype(np.float16))
        except Exception:
            pass

    elapsed_index = time.time() - t0

    # === 4. Compute statistics ===
    deltas = np.array([t["delta"] for t in tokens])

    # Per-type breakdown
    def classify_token(tok):
        if any('\u4e00' <= c <= '\u9fff' for c in tok):
            return 'CJK'
        if any('\u0e00' <= c <= '\u0e7f' for c in tok):
            return 'Thai'
        if any('\u0400' <= c <= '\u04ff' for c in tok):
            return 'Cyrillic'
        if any('\u0600' <= c <= '\u06ff' for c in tok):
            return 'Arabic'
        if any('\u0590' <= c <= '\u05ff' for c in tok):
            return 'Hebrew'
        if any('\uac00' <= c <= '\ud7af' for c in tok):
            return 'Korean'
        if any('\u3040' <= c <= '\u30ff' for c in tok):
            return 'Japanese'
        if any(c in tok for c in '{}()[];_\\\n\t'):
            return 'code'
        if tok.strip().isascii() and tok.strip().isalpha():
            return 'ascii_alpha'
        if tok.strip().isascii():
            return 'ascii_other'
        return 'other'

    by_type = defaultdict(list)
    for t in tokens:
        by_type[classify_token(t["token"])].append(t["delta"])

    type_stats = {}
    for typ, vals in sorted(by_type.items(), key=lambda x: np.mean(x[1])):
        v = np.array(vals)
        type_stats[typ] = {
            "n": len(v),
            "mean": round(float(v.mean()), 2),
            "std": round(float(v.std()), 2),
            "min": round(float(v.min()), 2),
            "max": round(float(v.max()), 2),
        }

    # Convergence check
    half = len(deltas) // 2
    first_half = deltas[:half]
    second_half = deltas[half:]
    convergence = float(np.corrcoef(
        [np.mean([d for d, t in zip(first_half, tokens[:half]) if classify_token(t["token"]) == typ])
         for typ in by_type if len(by_type[typ]) > 5],
        [np.mean([d for d, t in zip(second_half, tokens[half:]) if classify_token(t["token"]) == typ])
         for typ in by_type if len(by_type[typ]) > 5],
    )[0, 1]) if len(by_type) > 2 else 0.0

    # === 5. Build the .shrt ===
    sorted_tokens = sorted(tokens, key=lambda x: x["delta"], reverse=True)

    shrt = {
        "version": "0.1",
        "generated_at": time.time(),
        "elapsed_s": round(time.time() - t0),

        "model": {
            "name": cfg.model_type,
            "n_layers": cfg.n_layers,
            "hidden_size": cfg.hidden_size,
            "n_heads": cfg.n_heads,
            "vocab_size": vocab_size,
            "real_tokens": len(real_tokens),
        },

        "baseline": {
            "type": "silence (extracted)",
            "top_token": baseline_top,
            "entropy": round(baseline_entropy, 4),
            "layer": best_layer,
        },

        "direction": {
            "available": safety_dir is not None,
            "accuracy": direction_accuracy,
            "layer": best_layer,
        } if safety_dir is not None else None,

        "index": {
            "n_sampled": len(tokens),
            "n_vocab": vocab_size,
            "coverage": round(len(tokens) / len(real_tokens) * 100, 1),
            "convergence_r": round(convergence, 4),
            "elapsed_s": round(elapsed_index),
        },

        "distribution": {
            "mean": round(float(deltas.mean()), 2),
            "std": round(float(deltas.std()), 2),
            "min": round(float(deltas.min()), 2),
            "max": round(float(deltas.max()), 2),
            "pct_top_changed": round(
                sum(1 for t in tokens if t["top_changed"]) / len(tokens) * 100, 1),
        },

        "by_type": type_stats,

        "top_sharts": sorted_tokens[:50],
        "bottom_sharts": sorted_tokens[-50:],

        "tokens": sorted_tokens,
    }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

        # One file. Metadata and vectors together. Nothing drifts.
        token_ids_arr = np.array([t["id"] for t in tokens], dtype=np.int32)
        vec_array = np.array(vectors) if vectors else np.array([])  # [n_tokens, hidden_dim] float16

        # Remove tokens list from JSON (it's in the npz as arrays)
        shrt_meta = {k: v for k, v in shrt.items() if k != "tokens"}
        shrt_meta["top_sharts"] = sorted_tokens[:50]
        shrt_meta["bottom_sharts"] = sorted_tokens[-50:]

        np.savez_compressed(
            output,
            metadata=np.array([json.dumps(shrt_meta, ensure_ascii=False)]),
            vectors=vec_array,
            token_ids=token_ids_arr,
            token_texts=np.array([t["token"] for t in tokens]),
            deltas=np.array([t["delta"] for t in tokens], dtype=np.float32),
            layer=np.array([best_layer]),
        )

        shrt["vectors_shape"] = list(vec_array.shape) if len(vec_array) > 0 else []
        shrt["output"] = output

    return shrt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a .shrt shart profile")
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--n-index", type=int, default=15000, help="Index size (default 15K)")
    parser.add_argument("--output", "-o", default=None, help="Output .shrt file path")
    parser.add_argument("--db", default=None, help="Database path for prompts/directions")
    args = parser.parse_args()

    from heinrich.backend.protocol import load_backend
    backend = load_backend(args.model)

    db = None
    if args.db:
        from heinrich.core.db import SignalDB
        db = SignalDB(args.db)

    output = args.output or f"data/runs/{args.model.split('/')[-1]}.shrt.npz"

    shrt = generate_shrt(backend, db=db, n_index=args.n_index, output=output)

    print(f"\n=== {shrt['model']['name']} ===")
    print(f"  {shrt['index']['n_sampled']} tokens indexed in {shrt['index']['elapsed_s']}s")
    print(f"  convergence: r={shrt['index']['convergence_r']}")
    print(f"  mean delta: {shrt['distribution']['mean']} +/- {shrt['distribution']['std']}")
    print(f"  top changed: {shrt['distribution']['pct_top_changed']}%")
    print(f"\n  By type:")
    for typ, stats in shrt["by_type"].items():
        print(f"    {typ:<15} n={stats['n']:>5}  mean={stats['mean']:>6.1f}")
    print(f"\n  Saved to {output}")

    if db:
        db.close()
