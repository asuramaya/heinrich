"""Generate a .frt file — the tokenizer's atomic profile.

The .frt maps the tokenizer's vocabulary: which byte sequences became
atoms, how many bytes each atom costs, and what script each atom belongs to.

No model needed. No forward passes. Pure tokenizer analysis.

Usage:
    python -m heinrich.profile.frt --tokenizer Qwen/Qwen2.5-7B-Instruct
    # or via CLI: heinrich frt-profile --tokenizer X
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def generate_frt(
    tokenizer,
    *,
    output: str | None = None,
) -> dict:
    """Generate a .frt tokenizer profile.

    Args:
        tokenizer: a tokenizer with encode/decode methods
        output: path to write the .frt.npz file

    Returns:
        The .frt metadata dict
    """
    t0 = time.time()

    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)

    token_ids = []
    token_texts = []
    byte_counts = []
    is_special = []
    char_counts = []
    scripts = []

    for tid in range(vocab_size):
        text = tokenizer.decode([tid])
        raw_bytes = text.encode('utf-8', errors='replace')

        token_ids.append(tid)
        token_texts.append(text)
        byte_counts.append(len(raw_bytes))
        char_counts.append(len(text))

        special = (not text.strip()
                   or text.startswith('<')
                   or text.startswith('[control'))
        is_special.append(special)
        scripts.append(_detect_script(text))

    elapsed = time.time() - t0

    bc = np.array(byte_counts)
    script_counts = {}
    for s in scripts:
        script_counts[s] = script_counts.get(s, 0) + 1

    n_special = sum(is_special)
    n_real = vocab_size - n_special

    import hashlib
    vocab_hash = hashlib.sha256(
        '|'.join(token_texts).encode('utf-8')
    ).hexdigest()[:16]

    # Extract the default system prompt from the chat template
    default_system_prompt = None
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=False, add_generation_prompt=True,
            )
            # The system block is between <|im_start|>system\n and <|im_end|>
            if 'system\n' in rendered:
                start = rendered.index('system\n') + len('system\n')
                end = rendered.index('<|im_end|>', start) if '<|im_end|>' in rendered[start:] else len(rendered)
                default_system_prompt = rendered[start:end].strip()
        except Exception:
            pass

    meta = {
        "version": "0.1",
        "type": "frt",
        "generated_at": time.time(),
        "elapsed_s": round(elapsed, 1),
        "tokenizer": {
            "vocab_size": vocab_size,
            "n_real": n_real,
            "n_special": n_special,
            "vocab_hash": vocab_hash,
        },
        "byte_stats": {
            "mean": round(float(bc.mean()), 2),
            "std": round(float(bc.std()), 2),
            "min": int(bc.min()),
            "max": int(bc.max()),
        },
        "scripts": script_counts,

        "system_prompt": {
            "default": default_system_prompt,
            "injected": default_system_prompt is not None,
        },
    }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            metadata=np.array([json.dumps(meta, ensure_ascii=False)]),
            token_ids=np.array(token_ids, dtype=np.int32),
            token_texts=np.array(token_texts),
            byte_counts=np.array(byte_counts, dtype=np.int16),
            char_counts=np.array(char_counts, dtype=np.int16),
            is_special=np.array(is_special, dtype=np.bool_),
            scripts=np.array(scripts),
        )

    return meta


def load_frt(path: str) -> dict:
    """Load a .frt file. Returns dict with arrays and metadata."""
    d = np.load(path, allow_pickle=False)
    # metadata is stored as a string array, not pickle
    meta = json.loads(str(d['metadata'][0]))
    return {
        "metadata": meta,
        "token_ids": d["token_ids"],
        "token_texts": d["token_texts"],
        "byte_counts": d["byte_counts"],
        "char_counts": d["char_counts"],
        "is_special": d["is_special"],
        "scripts": d["scripts"],
    }


def _detect_script(text: str) -> str:
    """Detect the primary script of a token."""
    if not text.strip():
        return "special"
    for c in text:
        if '\u4e00' <= c <= '\u9fff': return "CJK"
        if '\u3040' <= c <= '\u30ff': return "Japanese"
        if '\uac00' <= c <= '\ud7af': return "Korean"
        if '\u0e00' <= c <= '\u0e7f': return "Thai"
        if '\u0600' <= c <= '\u06ff': return "Arabic"
        if '\u0590' <= c <= '\u05ff': return "Hebrew"
        if '\u0400' <= c <= '\u04ff': return "Cyrillic"
        if '\u0900' <= c <= '\u097f': return "Devanagari"
        if '\u0370' <= c <= '\u03ff': return "Greek"
    if any(c in text for c in '{}()[];_\\\n\t\r'):
        return "code"
    # Latin includes ASCII and Latin-extended (accented characters)
    stripped = text.strip()
    if stripped and all(c.isascii() or '\u00c0' <= c <= '\u024f' for c in stripped):
        return "latin"
    return "other"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a .frt tokenizer profile")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or path")
    parser.add_argument("--output", "-o", default=None, help="Output .frt.npz file path")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    output = args.output or f"data/runs/{args.tokenizer.split('/')[-1]}.frt.npz"
    meta = generate_frt(tokenizer, output=output)

    print(f"\n=== .frt: {args.tokenizer} ===")
    print(f"  vocab: {meta['tokenizer']['vocab_size']} ({meta['tokenizer']['n_real']} real)")
    print(f"  hash: {meta['tokenizer']['vocab_hash']}")
    print(f"  bytes/token: {meta['byte_stats']['mean']:.1f} +/- {meta['byte_stats']['std']:.1f}")
    print(f"  scripts: {meta['scripts']}")
    print(f"  saved to {output}")
