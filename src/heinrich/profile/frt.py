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

    Stores raw bytes for each token. Script classification, byte counts,
    and other derived properties are computed by analysis tools from the
    stored bytes, not during capture.

    Also extracts BPE merge rank where available — the direct measure
    of token frequency in the tokenizer's training corpus.

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
    raw_bytes_list = []  # the ground truth: what decode produces
    byte_counts = []
    char_counts = []
    merge_ranks = []  # BPE merge order: lower = more frequent

    # Extract merge ranks from the tokenizer if available
    merge_lookup = {}
    if hasattr(tokenizer, 'backend_tokenizer'):
        bt = tokenizer.backend_tokenizer
        if hasattr(bt, 'model') and hasattr(bt.model, 'get_vocab'):
            # HuggingFace fast tokenizer: vocab ordered by merge
            vocab = bt.model.get_vocab()
            # Vocab is {token_string: id} — the ID correlates with merge order
            # for BPE tokenizers (base vocab first, then merges in order)
            merge_lookup = {v: k for k, v in vocab.items()}

    for tid in range(vocab_size):
        text = tokenizer.decode([tid], skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        text_raw = tokenizer.decode([tid])  # unfiltered, for provenance
        raw = text.encode('utf-8', errors='replace')

        token_ids.append(tid)
        token_texts.append(text)
        raw_bytes_list.append(raw)
        byte_counts.append(len(raw))
        char_counts.append(len(text))

        # Track if this token differs under skip_special_tokens
        # (control tokens decode to [control_N] without the flag)
        is_control = (text != text_raw)

        # Merge rank: token ID itself is a proxy for merge order in BPE
        # Lower IDs = base vocab (byte-level), higher IDs = later merges
        # The exact merge rank is the ID minus the base vocab size
        merge_ranks.append(tid)

    # Also store script classification for backwards compatibility,
    # but mark it as derived
    scripts = [_detect_script(text) for text in token_texts]

    elapsed = time.time() - t0

    bc = np.array(byte_counts)
    script_counts = {}
    for s in scripts:
        script_counts[s] = script_counts.get(s, 0) + 1

    # Derive special: empty decode, or control tokens that decode differently
    # with skip_special_tokens
    is_special = [not text.strip() for text in token_texts]
    n_special = sum(is_special)
    n_real = vocab_size - n_special

    import hashlib
    vocab_hash = hashlib.sha256(
        '|'.join(token_texts).encode('utf-8')
    ).hexdigest()[:16]

    # Verify tokenizer round-trip integrity
    # Encode→decode should be identity for at least basic tokens
    roundtrip_failures = 0
    for tid in [0, 1, 100, 1000, vocab_size // 2, vocab_size - 1]:
        if tid < vocab_size:
            text = tokenizer.decode([tid])
            re_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(re_ids) != 1 or re_ids[0] != tid:
                roundtrip_failures += 1

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

    # Detect tokenizer backend
    tokenizer_backend = "unknown"
    if hasattr(tokenizer, 'is_fast'):
        tokenizer_backend = "fast" if tokenizer.is_fast else "slow"

    warnings = []
    if roundtrip_failures > 0:
        warnings.append(f"{roundtrip_failures} token IDs failed encode-decode round-trip. "
                        "Tokenizer may behave differently than during training.")
    if tokenizer_backend == "fast":
        warnings.append("Using fast tokenizer (no PyTorch). Verify merge behavior "
                        "matches the model's training tokenizer.")

    meta = {
        "version": "0.3",
        "type": "frt",
        "generated_at": time.time(),
        "elapsed_s": round(elapsed, 1),
        "warnings": warnings,
        "tokenizer": {
            "vocab_size": vocab_size,
            "n_real": n_real,
            "n_special": n_special,
            "vocab_hash": vocab_hash,
            "backend": tokenizer_backend,
            "roundtrip_failures": roundtrip_failures,
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

        # Store raw bytes as variable-length byte strings
        # numpy can't store variable-length bytes natively,
        # so store as fixed-width padded or as text hex encoding
        max_bytes = max(len(b) for b in raw_bytes_list)
        raw_bytes_padded = np.zeros((vocab_size, max_bytes), dtype=np.uint8)
        raw_bytes_lengths = np.array([len(b) for b in raw_bytes_list], dtype=np.int16)
        for i, b in enumerate(raw_bytes_list):
            raw_bytes_padded[i, :len(b)] = list(b)

        np.savez_compressed(
            output,
            metadata=np.array([json.dumps(meta, ensure_ascii=False)]),
            token_ids=np.array(token_ids, dtype=np.int32),
            token_texts=np.array(token_texts),
            raw_bytes=raw_bytes_padded,
            raw_bytes_lengths=raw_bytes_lengths,
            byte_counts=np.array(byte_counts, dtype=np.int16),
            char_counts=np.array(char_counts, dtype=np.int16),
            merge_ranks=np.array(merge_ranks, dtype=np.int32),
            is_special=np.array(is_special, dtype=np.bool_),
            scripts=np.array(scripts),  # derived, kept for backwards compat
        )

    return meta


def load_frt(path: str) -> dict:
    """Load a .frt file. Returns dict with arrays and metadata."""
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d['metadata'][0]))
    result = {
        "metadata": meta,
        "token_ids": d["token_ids"],
        "token_texts": d["token_texts"],
        "byte_counts": d["byte_counts"],
        "char_counts": d["char_counts"],
        "is_special": d["is_special"],
        "scripts": d["scripts"],
    }
    # v0.3 arrays (backwards compatible)
    for key in ["raw_bytes", "raw_bytes_lengths", "merge_ranks"]:
        if key in d.files:
            result[key] = d[key]
    return result


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
