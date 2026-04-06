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


def _write_checkpoint(path, tokens, vectors, layer_deltas, measure_layers, n_done, n_total):
    """Write partial results to a checkpoint file. Overwrites previous checkpoint."""
    try:
        checkpoint = {
            "n_done": n_done,
            "n_total": n_total,
            "n_tokens": len(tokens),
            "token_ids": np.array([t["id"] for t in tokens], dtype=np.int32),
            "deltas": np.array([t["delta"] for t in tokens], dtype=np.float32),
        }
        for layer in measure_layers:
            if layer_deltas[layer]:
                checkpoint[f"deltas_L{layer}"] = np.array(layer_deltas[layer], dtype=np.float32)
        np.savez_compressed(path, **checkpoint)
    except Exception:
        pass  # checkpoint failure should never kill the run


def _extract_clean_baseline(tokenizer) -> str:
    """Build a structural-only baseline from the model's own chat template.

    Renders the template with an empty user message and no system content,
    then strips any injected system block. Works across template formats
    (ChatML, Llama, Mistral, etc.)
    """
    if not hasattr(tokenizer, 'apply_chat_template'):
        return ""

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=False,
        add_generation_prompt=True,
    )

    import re
    patterns = [
        r'<\|im_start\|>system\n.*?<\|im_end\|>\n?',
        r'<\|start_header_id\|>system<\|end_header_id\|>\n.*?<\|eot_id\|>\n?',
        r'\[INST\]\s*<<SYS>>.*?<</SYS>>\s*',
    ]
    for pat in patterns:
        rendered = re.sub(pat, '', rendered, flags=re.DOTALL)

    return rendered


def _extract_template_parts(tokenizer):
    """Extract prefix and suffix token IDs for token splicing.

    Uses the model's chat template to find the structural tokens that
    surround user content. Returns (prefix_ids, suffix_ids) for splicing
    any token ID as: prefix_ids + [tid] + suffix_ids.
    """
    # Render with a marker we can split on
    marker = "HEINRICH_SPLIT_MARKER"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
    )

    import re
    patterns = [
        r'<\|im_start\|>system\n.*?<\|im_end\|>\n?',
        r'<\|start_header_id\|>system<\|end_header_id\|>\n.*?<\|eot_id\|>\n?',
        r'\[INST\]\s*<<SYS>>.*?<</SYS>>\s*',
    ]
    for pat in patterns:
        rendered = re.sub(pat, '', rendered, flags=re.DOTALL)

    if marker not in rendered:
        # Fallback: can't find marker, use the baseline as-is
        baseline = _extract_clean_baseline(tokenizer)
        ids = tokenizer.encode(baseline)
        mid = len(ids) // 2
        return ids[:mid], ids[mid:]

    prefix_str, suffix_str = rendered.split(marker, 1)
    prefix_ids = tokenizer.encode(prefix_str)
    suffix_ids = tokenizer.encode(suffix_str)
    return prefix_ids, suffix_ids


def generate_shrt(
    backend,
    *,
    db=None,
    n_index: int = 15000,
    seed: int = 42,
    layers: list[int] | None = None,
    output: str | None = None,
) -> dict:
    """Generate a complete .shrt profile for a model.

    Args:
        backend: loaded model backend (MLX or HF)
        db: SignalDB (optional, for prompts and directions)
        n_index: number of tokens for the index (default 15K, converges by 7K)
        seed: random seed for token sampling
        layers: layers to measure at (default: [best_safety_layer])
                use "all" sentinel via [-1] to measure every layer
        output: path to write the .shrt JSON file

    Returns:
        The .shrt dict (also written to file if output is specified)
    """
    from heinrich.cartography.templates import build_prompt

    cfg = backend.config
    t0 = time.time()

    # === 1. Determine which layers to measure ===
    best_layer = cfg.safety_layers[-1] if cfg.safety_layers else cfg.n_layers - 2
    if layers is None:
        measure_layers = [best_layer]
    elif layers == [-1]:
        measure_layers = list(range(cfg.n_layers))
    else:
        measure_layers = layers
    primary_layer = measure_layers[0] if len(measure_layers) == 1 else best_layer

    # === 2. Extract baselines at each layer (single pass) ===
    clean_baseline = _extract_clean_baseline(backend.tokenizer)
    if len(measure_layers) == 1:
        baseline_fwd = backend.forward(clean_baseline, return_residual=True, residual_layer=measure_layers[0])
        baseline_residuals = {measure_layers[0]: baseline_fwd.residual}
    else:
        baseline_fwd = backend.forward(clean_baseline, return_residual=True, residual_layers=measure_layers)
        baseline_residuals = getattr(baseline_fwd, 'residuals', {})
    baseline_top = baseline_fwd.top_token
    baseline_entropy = baseline_fwd.entropy

    # === 3. Safety direction (if DB available, primary layer only) ===
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

            states = backend.capture_residual_states(h_fmt + b_fmt, layers=[primary_layer])
            if primary_layer in states:
                sl = states[primary_layer]
                dr = find_direction(sl[:len(h_fmt)], sl[len(h_fmt):],
                                    name="safety", layer=primary_layer)
                safety_dir = dr.direction / np.linalg.norm(dr.direction)
                direction_accuracy = dr.separation_accuracy
        except Exception:
            pass

    # === 4. Build the index: scan n_index random tokens ===
    vocab_size = backend.tokenizer.vocab_size

    # Extract template parts dynamically. Encode once. Splice forever.
    prefix_ids, suffix_ids = _extract_template_parts(backend.tokenizer)

    # Build real token list — filter by decoded text, deduplicate
    real_tokens = []
    seen_texts = set()
    for tid in range(vocab_size):
        tok = backend.tokenizer.decode([tid])
        if tok.strip() and len(tok) > 0 and not tok.startswith('[control') and not tok.startswith('<'):
            if tok not in seen_texts:
                seen_texts.add(tok)
                real_tokens.append((tid, tok))

    # Sample — random draw, plus exhaustive scan of small scripts.
    # When a script has <100 tokens in the vocab, random sampling gives
    # unreliable estimates. Include ALL tokens from small scripts.
    rng_np = np.random.RandomState(seed)
    n_sample = min(n_index, len(real_tokens))
    sample_idx = set(rng_np.choice(len(real_tokens), n_sample, replace=False))

    # Detect small-script tokens and ensure complete coverage
    from .frt import _detect_script
    script_pools = defaultdict(list)
    for i, (tid, tok) in enumerate(real_tokens):
        script_pools[_detect_script(tok)].append(i)
    for script, indices in script_pools.items():
        if 0 < len(indices) < 100:
            sample_idx.update(indices)  # add all tokens from small scripts

    sample = [real_tokens[i] for i in sorted(sample_idx)]

    tokens = []
    vectors = []  # full delta vectors at primary layer — never discard
    layer_deltas = {layer: [] for layer in measure_layers}  # delta per layer per token
    token_times = []  # wall time per token for throughput reporting
    checkpoint_interval = max(len(sample) // 10, 100)  # checkpoint every 10% or 100 tokens
    checkpoint_path = (output + ".checkpoint") if output else None
    n_processed = 0

    for tid, tok in sample:
        t_tok = time.time()
        try:
            input_ids = prefix_ids + [tid] + suffix_ids

            # Measure at all layers in one forward pass
            token_layer_deltas = {}
            primary_vec = None
            if len(measure_layers) == 1:
                fwd = backend.forward(
                    "", token_ids=input_ids,
                    return_residual=True, residual_layer=measure_layers[0])
                if fwd.residual is None:
                    continue
                layer = measure_layers[0]
                delta_vec = fwd.residual - baseline_residuals[layer]
                token_layer_deltas[layer] = float(np.linalg.norm(delta_vec))
                primary_vec = delta_vec
            else:
                fwd = backend.forward(
                    "", token_ids=input_ids,
                    return_residual=True, residual_layers=measure_layers)
                residuals = getattr(fwd, 'residuals', {})
                for layer in measure_layers:
                    if layer not in residuals:
                        continue
                    delta_vec = residuals[layer] - baseline_residuals[layer]
                    token_layer_deltas[layer] = float(np.linalg.norm(delta_vec))
                    if layer == primary_layer:
                        primary_vec = delta_vec

            if len(token_layer_deltas) != len(measure_layers):
                continue

            primary_delta = token_layer_deltas.get(primary_layer,
                            token_layer_deltas[measure_layers[0]])
            entry = {
                "id": tid,
                "token": tok,
                "delta": round(primary_delta, 2),
                "top_changed": fwd.top_token != baseline_top,
            }

            # Per-layer deltas
            if len(measure_layers) > 1:
                entry["layer_deltas"] = {
                    str(l): round(d, 2) for l, d in token_layer_deltas.items()
                }

            # Safety projection if direction available
            if safety_dir is not None and primary_vec is not None:
                entry["safety_shift"] = round(
                    float(np.dot(primary_vec, safety_dir)), 4)

            tokens.append(entry)
            if primary_vec is not None:
                vectors.append(primary_vec.astype(np.float16))
            for layer in measure_layers:
                layer_deltas[layer].append(token_layer_deltas[layer])
            token_times.append(time.time() - t_tok)
            n_processed += 1

            # Checkpoint: save partial results periodically
            if checkpoint_path and n_processed % checkpoint_interval == 0:
                _write_checkpoint(checkpoint_path, tokens, vectors, layer_deltas,
                                  measure_layers, n_processed, len(sample))
        except Exception:
            pass

    elapsed_index = time.time() - t0

    # Clean up checkpoint on successful completion
    if checkpoint_path:
        import os
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass

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

    # Convergence check: cumulative statistics stability.
    # Track mean and std at checkpoints. Converged when the last
    # checkpoint is within 1% of mean and 5% of std of the final value.
    convergence_detail = {"mean_stable_at": len(deltas), "std_stable_at": len(deltas)}
    if len(deltas) >= 100:
        checkpoints = list(range(100, len(deltas) + 1, max(len(deltas) // 20, 50)))
        if checkpoints[-1] != len(deltas):
            checkpoints.append(len(deltas))
        final_mean = float(deltas.mean())
        final_std = float(deltas.std())
        for cp in checkpoints:
            sub = deltas[:cp]
            if abs(float(sub.mean()) - final_mean) / max(abs(final_mean), 1e-8) < 0.01:
                if convergence_detail["mean_stable_at"] == len(deltas):
                    convergence_detail["mean_stable_at"] = cp
            else:
                convergence_detail["mean_stable_at"] = len(deltas)
            if abs(float(sub.std()) - final_std) / max(abs(final_std), 1e-8) < 0.05:
                if convergence_detail["std_stable_at"] == len(deltas):
                    convergence_detail["std_stable_at"] = cp
            else:
                convergence_detail["std_stable_at"] = len(deltas)
    converged_at = max(convergence_detail["mean_stable_at"],
                       convergence_detail["std_stable_at"])
    convergence_pct = round(converged_at / len(deltas) * 100, 1) if len(deltas) > 0 else 100

    # === 5. Build the .shrt — ranks are the primary signal ===
    sorted_tokens = sorted(tokens, key=lambda x: x["delta"], reverse=True)
    for rank, t in enumerate(sorted_tokens):
        t["rank"] = rank + 1

    # === Warnings: protect the next user ===
    warnings = []
    if baseline_entropy > 5.0:
        warnings.append(f"High baseline entropy ({baseline_entropy:.2f}). "
                        "Check if system prompt was stripped correctly.")
    if convergence_pct > 80:
        warnings.append(f"Not converged: statistics stabilize at {converged_at}/"
                        f"{len(deltas)} tokens ({convergence_pct}% of sample).")
    for typ, stats in type_stats.items():
        if stats["n"] < 10 and stats["n"] > 0:
            warnings.append(f"Script '{typ}' has only {stats['n']} tokens — "
                            "treat results as provisional.")

    shrt = {
        "version": "0.2",
        "generated_at": time.time(),
        "elapsed_s": round(time.time() - t0),
        "warnings": warnings,

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
            "layer": primary_layer,
        },

        "layers": measure_layers,

        "direction": {
            "available": safety_dir is not None,
            "accuracy": direction_accuracy,
            "layer": primary_layer,
        } if safety_dir is not None else None,

        "index": {
            "n_sampled": len(tokens),
            "n_vocab": vocab_size,
            "coverage": round(len(tokens) / len(real_tokens) * 100, 1),
            "converged_at_n": converged_at,
            "converged_at_pct": convergence_pct,
            "convergence_detail": convergence_detail,
            "elapsed_s": round(elapsed_index),
            "throughput": {
                "tokens_per_sec": round(len(tokens) / max(elapsed_index, 0.01), 1),
                "cold_ms": round(token_times[0] * 1000, 1) if token_times else 0,
                "warm_ms": round(np.median(token_times[10:]) * 1000, 1) if len(token_times) > 10 else 0,
                "cv": round(float(np.std(token_times[10:]) / max(np.mean(token_times[10:]), 1e-8)), 4) if len(token_times) > 10 else 0,
            },
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

    # Per-layer distribution stats
    if len(measure_layers) > 1:
        layer_stats = {}
        for layer in measure_layers:
            ld = np.array(layer_deltas[layer])
            if len(ld) > 0:
                layer_stats[str(layer)] = {
                    "mean": round(float(ld.mean()), 2),
                    "std": round(float(ld.std()), 2),
                    "cv": round(float(ld.std() / max(ld.mean(), 1e-8)), 4),
                }
        shrt["layer_stats"] = layer_stats

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

        token_ids_arr = np.array([t["id"] for t in tokens], dtype=np.int32)
        vec_array = np.array(vectors) if vectors else np.array([])

        # Remove tokens list from JSON (it's in the npz as arrays)
        shrt_meta = {k: v for k, v in shrt.items() if k != "tokens"}
        shrt_meta["top_sharts"] = sorted_tokens[:50]
        shrt_meta["bottom_sharts"] = sorted_tokens[-50:]

        save_dict = {
            "metadata": np.array([json.dumps(shrt_meta, ensure_ascii=False)]),
            "vectors": vec_array,
            "token_ids": token_ids_arr,
            "token_texts": np.array([t["token"] for t in tokens]),
            "deltas": np.array([t["delta"] for t in tokens], dtype=np.float32),
            "layer": np.array(measure_layers),
        }

        # Store per-layer deltas as layer_deltas_L{n} arrays
        if len(measure_layers) > 1:
            for layer in measure_layers:
                save_dict[f"deltas_L{layer}"] = np.array(
                    layer_deltas[layer], dtype=np.float32)

        np.savez_compressed(output, **save_dict)

        shrt["vectors_shape"] = list(vec_array.shape) if len(vec_array) > 0 else []
        shrt["output"] = output

    return shrt


def load_shrt(path: str) -> dict:
    """Load a .shrt file. Returns dict with arrays and metadata.
    Includes per-layer delta arrays (deltas_L{n}) if present.
    Warns if baseline looks suspicious."""
    import warnings as _warnings
    d = np.load(path, allow_pickle=False)  # safe: npz arrays only
    meta = json.loads(str(d['metadata'][0]))

    # Check baseline health
    baseline_ent = meta.get('baseline', {}).get('entropy', 0)
    if baseline_ent > 5.0:
        _warnings.warn(
            f"{path}: baseline entropy {baseline_ent:.2f} is high. "
            f"System prompt may not have been stripped. "
            f"Compare with caution.",
            stacklevel=2,
        )

    result = {
        "metadata": meta,
        "token_ids": d["token_ids"],
        "token_texts": d["token_texts"],
        "deltas": d["deltas"],
        "vectors": d["vectors"],
        "layer": d["layer"],
    }
    for key in d.files:
        if key.startswith("deltas_L"):
            result[key] = d[key]
    return result


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
