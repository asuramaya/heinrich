"""Prompt-mode MRI: per-prompt × per-layer × per-position residual capture.

The standard `mri` command iterates over the vocabulary — one token per row,
single-token forward pass, captures the impulse-response of each token in
isolation. That's the right tool for "what does this token do to the residual
stream?" but the wrong tool for studying multi-token *prompts* (sentences,
phrases) where context matters.

This module captures full-context residuals for a fixed list of prompts:

  - For each prompt: tokenize, single forward pass with output_hidden_states
  - Save residual stream at every (layer, token_position) cell
  - Save token_ids, decoded text per token, original metadata

Sized for sets of ~10–500 prompts, not full vocab. Storage is ~tiny:
36 layers × ~16 tokens × 2560 hidden × 4 bytes ≈ 6 MB per prompt.

The first use case is the text-encoder side of image diffusion pipelines
(Qwen3 inside Z-Image, T5/CLIP inside Flux/SDXL): asking what the encoder
does with identity tokens (`subject1`, etc.) before they ever reach the DiT.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


PROMPT_MRI_VERSION = "0.1"


def _load_prompts(path: str) -> list[dict]:
    """Load prompts from a JSONL file. Each line: {"text": ..., ...metadata...}."""
    prompts = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj:
                raise ValueError(f"prompt entry missing 'text' field: {obj}")
            prompts.append(obj)
    return prompts


def capture_prompt_mri(
    backend,
    prompts_path: str,
    output: str,
    *,
    dtype: str = "float16",
) -> dict:
    """Capture per-prompt × per-layer × per-position residuals.

    Args:
        backend: HF or MLX backend (must expose `capture_all_positions`)
        prompts_path: JSONL file with {"text": ..., ...} per line
        output: output directory path
        dtype: storage dtype for residuals — "float16" (default) or "float32"

    Output structure:
        <output>/metadata.json    — model, prompt source, layer/dim info
        <output>/prompts.jsonl    — per-prompt metadata + token_ids
        <output>/residuals.npz    — keys 'r{idx:04d}' → [n_layers+1, n_tokens, hidden]
    """
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = _load_prompts(prompts_path)
    cfg = backend.config
    n_layers = cfg.n_layers
    hidden = cfg.hidden_size

    # Layers: 0..n_layers-1, captured via existing capture_all_positions which
    # uses output_hidden_states[layer+1] (post-layer state). Add a -1 sentinel
    # for the embedding layer (pre-first-layer state).
    layers = list(range(n_layers))

    np_dtype = np.float16 if dtype == "float16" else np.float32

    print(f"prompt-mri: {len(prompts)} prompts, {n_layers} layers, hidden={hidden}")
    t0 = time.time()

    residual_arrays: dict[str, np.ndarray] = {}
    enriched_prompts = []

    for i, p in enumerate(prompts):
        text = p["text"]
        token_ids = backend.tokenize(text)
        n_tokens = len(token_ids)

        # capture_all_positions returns {layer: [T, hidden]}.
        # Stack layers in order to [n_layers, T, hidden].
        per_layer = backend.capture_all_positions(text, layers=layers)
        stacked = np.stack(
            [per_layer[l].astype(np_dtype) for l in layers],
            axis=0,
        )  # [n_layers, T, hidden]

        # Embedding layer (input to first layer) is hidden_states[0] in HF.
        # Backend doesn't currently expose this directly, so we re-extract:
        # for now, just capture post-layer states. Future: extend backend to
        # also return the embedding state.

        residual_arrays[f"r{i:04d}"] = stacked

        decoded_tokens = [backend.decode([tid]) for tid in token_ids]
        enriched_prompts.append({
            "idx": i,
            "text": text,
            "n_tokens": n_tokens,
            "token_ids": [int(t) for t in token_ids],
            "decoded_tokens": decoded_tokens,
            **{k: v for k, v in p.items() if k != "text"},
        })

        if (i + 1) % 5 == 0 or i == len(prompts) - 1:
            print(f"  [{i + 1:3d}/{len(prompts):3d}] {text[:60]}  ({n_tokens} tok)")

    elapsed = time.time() - t0

    # Save residuals as npz (different shapes per prompt — npz handles fine)
    np.savez(out_dir / "residuals.npz", **residual_arrays)

    # Save prompt metadata
    with open(out_dir / "prompts.jsonl", "w") as fh:
        for p in enriched_prompts:
            fh.write(json.dumps(p) + "\n")

    # Save global metadata
    metadata = {
        "version": PROMPT_MRI_VERSION,
        "model_type": cfg.model_type,
        "n_layers": n_layers,
        "hidden_size": hidden,
        "intermediate_size": cfg.intermediate_size,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "vocab_size": cfg.vocab_size,
        "n_prompts": len(prompts),
        "prompts_source": str(prompts_path),
        "dtype": dtype,
        "elapsed_s": round(elapsed, 2),
        "captured_states": "post_layer_residual",  # future: + embedding, + per-head
    }
    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"  done. {elapsed:.1f}s. wrote {out_dir}")
    return metadata


def load_prompt_mri(path: str) -> dict:
    """Load a prompt-mri directory back into memory."""
    p = Path(path)
    with open(p / "metadata.json") as fh:
        metadata = json.load(fh)
    with open(p / "prompts.jsonl") as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]
    residuals = dict(np.load(p / "residuals.npz"))
    return {
        "metadata": metadata,
        "prompts": prompts,
        "residuals": residuals,
    }
