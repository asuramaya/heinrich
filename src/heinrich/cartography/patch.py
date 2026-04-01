"""Activation patching — causal intervention via component swapping between runs."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from ..signal import Signal, SignalStore


@dataclass
class PatchResult:
    layer: int
    component: str
    band: int | None
    clean_top: int
    corrupt_top: int
    patched_top: int
    kl_corrupt_to_patched: float
    recovery_fraction: float
    top_recovered: bool


def capture_all_states(
    model: Any, tokenizer: Any, prompt: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Forward pass capturing hidden states after every layer. Returns (states, logits)."""
    import mlx.core as mx
    from .perturb import _mask_dtype

    inner = getattr(model, "model", model)
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mdtype = _mask_dtype(model)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

    h = inner.embed_tokens(input_ids)
    states = []
    for ly in inner.layers:
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]
        states.append(np.array(h.astype(mx.float32)))

    h = inner.norm(h)
    logits = np.array(model.lm_head(h).astype(mx.float32)[0, -1, :])
    return states, logits


def patch_and_run(
    model: Any, tokenizer: Any,
    base_prompt: str,
    donor_states: list[np.ndarray],
    patch_layer: int,
    dims: tuple[int, int] | None = None,
) -> np.ndarray:
    """Run base_prompt's forward pass, but at patch_layer, swap in donor dims. Returns logits."""
    import mlx.core as mx
    from .perturb import _mask_dtype

    inner = getattr(model, "model", model)
    tokens = tokenizer.encode(base_prompt)
    input_ids = mx.array([tokens])
    T = len(tokens)
    mdtype = _mask_dtype(model)
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

    h = inner.embed_tokens(input_ids)
    for i, ly in enumerate(inner.layers):
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]

        if i == patch_layer:
            h_np = np.array(h.astype(mx.float32))
            donor = donor_states[i]
            if dims is not None:
                start, end = dims
                # Only patch at last token position
                h_np[0, -1, start:end] = donor[0, -1, start:end]
            else:
                h_np[0, -1, :] = donor[0, -1, :]
            h = mx.array(h_np.astype(np.float16))

    h = inner.norm(h)
    return np.array(model.lm_head(h).astype(mx.float32)[0, -1, :])


def sweep_band_patches(
    model: Any, tokenizer: Any,
    clean_prompt: str, corrupt_prompt: str,
    patch_layer: int,
    *, n_bands: int = 28,
    store: SignalStore | None = None,
) -> list[PatchResult]:
    """Patch each residual stream band from clean into corrupt run at patch_layer."""
    from ..inspect.self_analysis import _softmax

    clean_states, clean_logits = capture_all_states(model, tokenizer, clean_prompt)
    corrupt_states, corrupt_logits = capture_all_states(model, tokenizer, corrupt_prompt)

    clean_probs = _softmax(clean_logits)
    corrupt_probs = _softmax(corrupt_logits)
    clean_top = int(np.argmax(clean_probs))
    corrupt_top = int(np.argmax(corrupt_probs))

    kl_clean_corrupt = float(np.sum(clean_probs * np.log((clean_probs + 1e-12) / (corrupt_probs + 1e-12))))

    hidden_size = clean_states[0].shape[-1]
    band_size = hidden_size // n_bands
    results = []

    for band in range(n_bands):
        start = band * band_size
        end = start + band_size
        patched_logits = patch_and_run(
            model, tokenizer, corrupt_prompt, clean_states, patch_layer, dims=(start, end))
        patched_probs = _softmax(patched_logits)
        patched_top = int(np.argmax(patched_probs))

        kl_corrupt_patched = float(np.sum(corrupt_probs * np.log(
            (corrupt_probs + 1e-12) / (patched_probs + 1e-12))))
        recovery = kl_corrupt_patched / (kl_clean_corrupt + 1e-12)

        r = PatchResult(
            layer=patch_layer, component="residual_band", band=band,
            clean_top=clean_top, corrupt_top=corrupt_top, patched_top=patched_top,
            kl_corrupt_to_patched=kl_corrupt_patched, recovery_fraction=float(recovery),
            top_recovered=(patched_top == clean_top),
        )
        results.append(r)
        if store:
            store.add(Signal("patch_recovery", "cartography", "model",
                             f"L{patch_layer}.band{band}", r.recovery_fraction,
                             {"top_recovered": r.top_recovered, "kl": r.kl_corrupt_to_patched}))

    results.sort(key=lambda r: r.recovery_fraction, reverse=True)
    return results


def sweep_layer_patches(
    model: Any, tokenizer: Any,
    clean_prompt: str, corrupt_prompt: str,
    *, store: SignalStore | None = None,
) -> list[PatchResult]:
    """Patch entire residual stream from clean into corrupt at each layer."""
    from ..inspect.self_analysis import _softmax

    clean_states, clean_logits = capture_all_states(model, tokenizer, clean_prompt)
    _, corrupt_logits = capture_all_states(model, tokenizer, corrupt_prompt)

    clean_probs = _softmax(clean_logits)
    corrupt_probs = _softmax(corrupt_logits)
    clean_top = int(np.argmax(clean_probs))
    corrupt_top = int(np.argmax(corrupt_probs))
    kl_cc = float(np.sum(clean_probs * np.log((clean_probs + 1e-12) / (corrupt_probs + 1e-12))))

    results = []
    for layer in range(len(clean_states)):
        patched_logits = patch_and_run(model, tokenizer, corrupt_prompt, clean_states, layer)
        patched_probs = _softmax(patched_logits)
        patched_top = int(np.argmax(patched_probs))
        kl_cp = float(np.sum(corrupt_probs * np.log((corrupt_probs + 1e-12) / (patched_probs + 1e-12))))
        recovery = kl_cp / (kl_cc + 1e-12)

        r = PatchResult(
            layer=layer, component="full_residual", band=None,
            clean_top=clean_top, corrupt_top=corrupt_top, patched_top=patched_top,
            kl_corrupt_to_patched=kl_cp, recovery_fraction=float(recovery),
            top_recovered=(patched_top == clean_top),
        )
        results.append(r)
        if store:
            store.add(Signal("patch_layer_recovery", "cartography", "model",
                             f"L{layer}", r.recovery_fraction,
                             {"top_recovered": r.top_recovered}))

    return results
