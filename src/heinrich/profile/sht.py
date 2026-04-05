"""Generate a .sht file — the model's output profile.

The .sht captures what comes out: the output distribution when each
token enters. The final product after the tokenizer farts atoms and
the model sharts on them.

Measures KL divergence from silence, top token change, entropy shift.
Model-agnostic: any model that produces a probability distribution.

Usage:
    python -m heinrich.profile.sht --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def generate_sht(
    backend,
    *,
    n_index: int = 15000,
    seed: int = 42,
    output: str | None = None,
) -> dict:
    """Generate a .sht output profile.

    Measures the output distribution change when each token enters.
    KL divergence from silence baseline. Model-agnostic.
    """
    cfg = backend.config
    t0 = time.time()

    # Clean baseline: structural template, system prompt stripped dynamically
    from .shrt import _extract_clean_baseline
    clean_baseline = _extract_clean_baseline(backend.tokenizer)
    baseline_fwd = backend.forward(clean_baseline)
    baseline_probs = baseline_fwd.probs
    baseline_top = baseline_fwd.top_token
    baseline_entropy = baseline_fwd.entropy

    # Extract template parts dynamically. Encode once. Splice forever.
    from .shrt import _extract_template_parts
    prefix_ids, suffix_ids = _extract_template_parts(backend.tokenizer)

    vocab_size = backend.tokenizer.vocab_size
    real_tokens = []
    seen_texts = set()
    for tid in range(vocab_size):
        tok = backend.tokenizer.decode([tid])
        if tok.strip() and len(tok) > 0 and not tok.startswith('[control') and not tok.startswith('<'):
            if tok not in seen_texts:
                seen_texts.add(tok)
                real_tokens.append((tid, tok))

    rng = np.random.RandomState(seed)
    n_sample = min(n_index, len(real_tokens))
    sample_idx = rng.choice(len(real_tokens), n_sample, replace=False)
    sample = [real_tokens[i] for i in sample_idx]

    tokens = []
    kl_divs = []
    entropies = []

    for tid, tok in sample:
        try:
            # Splice token ID directly — no decode round-trip
            input_ids = prefix_ids + [tid] + suffix_ids
            fwd = backend.forward("", token_ids=input_ids)

            p = fwd.probs
            q = baseline_probs
            mask = (p > 1e-10) & (q > 1e-10)
            kl = float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

            tokens.append({
                "id": tid, "token": tok, "kl": round(kl, 4),
                "entropy": round(fwd.entropy, 4),
                "top_changed": fwd.top_token != baseline_top,
                "top_token": fwd.top_token,
            })
            kl_divs.append(kl)
            entropies.append(fwd.entropy)
        except Exception:
            pass

    elapsed = time.time() - t0
    kl_arr = np.array(kl_divs)
    ent_arr = np.array(entropies)
    sorted_tokens = sorted(tokens, key=lambda x: x["kl"], reverse=True)

    meta = {
        "version": "0.1", "type": "sht",
        "generated_at": time.time(), "elapsed_s": round(elapsed),
        "model": {"name": cfg.model_type, "n_layers": cfg.n_layers,
                  "hidden_size": cfg.hidden_size, "vocab_size": vocab_size},
        "baseline": {"type": "silence", "top_token": baseline_top,
                     "entropy": round(baseline_entropy, 4)},
        "index": {"n_sampled": len(tokens), "n_vocab": vocab_size},
        "distribution": {
            "kl_mean": round(float(kl_arr.mean()), 4),
            "kl_std": round(float(kl_arr.std()), 4),
            "kl_max": round(float(kl_arr.max()), 4),
            "entropy_mean": round(float(ent_arr.mean()), 4),
            "pct_top_changed": round(
                sum(1 for t in tokens if t["top_changed"]) / len(tokens) * 100, 1),
        },
        "top_sharts": sorted_tokens[:50],
        "bottom_sharts": sorted_tokens[-50:],
    }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            metadata=np.array([json.dumps(meta, ensure_ascii=False)]),
            token_ids=np.array([t["id"] for t in sorted_tokens], dtype=np.int32),
            token_texts=np.array([t["token"] for t in sorted_tokens]),
            kl_divs=np.array([t["kl"] for t in sorted_tokens], dtype=np.float32),
            entropies=np.array([t["entropy"] for t in sorted_tokens], dtype=np.float32),
            top_changed=np.array([t["top_changed"] for t in sorted_tokens], dtype=np.bool_),
        )

    return meta


def load_sht(path: str) -> dict:
    """Load a .sht file."""
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d['metadata'][0]))
    return {"metadata": meta, "token_ids": d["token_ids"],
            "token_texts": d["token_texts"], "kl_divs": d["kl_divs"],
            "entropies": d["entropies"], "top_changed": d["top_changed"]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a .sht output profile")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-index", type=int, default=15000)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    from heinrich.backend.protocol import load_backend
    backend = load_backend(args.model)
    output = args.output or f"data/runs/{args.model.split('/')[-1]}.sht.npz"
    meta = generate_sht(backend, n_index=args.n_index, output=output)

    print(f"\n=== .sht: {meta['model']['name']} ===")
    print(f"  {meta['index']['n_sampled']} tokens, {meta['distribution']['pct_top_changed']}% changed top")
    print(f"  KL: mean={meta['distribution']['kl_mean']:.2f}")
    print(f"  Saved to {output}")
