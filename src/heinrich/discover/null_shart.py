"""Null-shart inventory.

Per-residual-dimension ablation measurement. For each dim d at a chosen
layer L, we zero that dim at the last token position and re-run the tail
of the forward pass. The KL from the baseline distribution is the dim's
ablation effect. The mean |residual[d]| across prompts is its activation
magnitude. The joint distribution tells us whether dims exist that carry
signal but have no downstream effect — the "null sharts" the theory
predicts.

The falsifiable claim (theory_of_sharts.tex):
    "The 47% of residual dimensions that are inert to ablation but still
    active during computation."

This module produces the data that either confirms or refutes that claim.

Cost model (Qwen-0.5B, Apple Silicon):
    For each prompt, we do 1 "head" forward (embed + L0..L) and N_dim
    "tail" forwards (L+1..end + lmhead). The head is shared; only tails
    are per-dim. Tail cost ≈ (n_layers - L - 1) × tokens × ~1ms.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NullShartRow:
    """One dim's row in the inventory."""
    dim: int
    mag: float          # mean |residual[dim]| across prompts
    kl: float           # mean KL(baseline || ablated)
    logit_l2: float     # mean ||baseline_logits - ablated_logits||_2
    top1_flip_rate: float  # fraction of prompts whose argmax flipped


@dataclass
class NullShartReport:
    """Full per-dim inventory at a single layer."""
    model_id: str
    layer: int
    n_prompts: int
    hidden_size: int
    baseline_noise_kl: float  # baseline→baseline KL (numeric noise floor)
    rows: list[NullShartRow] = field(default_factory=list)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) in nats."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def measure_null_sharts(
    backend,
    prompts: list[str],
    layer: int,
    *,
    dims: list[int] | None = None,
    progress: bool = True,
) -> NullShartReport:
    """Per-dim ablation at ``layer`` across ``prompts``.

    Zeroes one residual dimension at the last-token position after layer
    ``layer``, then runs the tail of the forward pass. Compares resulting
    next-token distribution to the unablated baseline.

    Args:
        backend: MLXBackend (needs _inner, tokenizer, _lm_head)
        prompts: list of prompt strings
        layer: residual layer to ablate at (post-layer output)
        dims: subset of dims to measure (None = all)
        progress: print dots to stderr

    Returns:
        NullShartReport with one row per dim.
    """
    import sys
    import mlx.core as mx
    from heinrich.cartography.perturb import _mask_dtype

    inner = backend._inner
    model = backend.model
    tok = backend.tokenizer
    mdtype = _mask_dtype(model)

    hidden_size = inner.embed_tokens.weight.shape[-1]
    n_layers = len(inner.layers)
    if dims is None:
        dims = list(range(hidden_size))

    if layer < 0 or layer >= n_layers:
        raise ValueError(f"layer {layer} out of range [0, {n_layers})")

    # Per-prompt accumulators over dims
    mag_sum = np.zeros(len(dims), dtype=np.float64)
    kl_sum = np.zeros(len(dims), dtype=np.float64)
    l2_sum = np.zeros(len(dims), dtype=np.float64)
    flip_sum = np.zeros(len(dims), dtype=np.int64)
    baseline_noise_sum = 0.0
    n_valid = 0

    for p_idx, prompt in enumerate(prompts):
        token_ids = tok.encode(prompt)
        if not token_ids:
            continue
        input_ids = mx.array([token_ids])
        T = len(token_ids)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        # --- head forward: embed + layers[0..layer] ---
        h = inner.embed_tokens(input_ids)
        for i in range(layer + 1):
            ly = inner.layers[i]
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]
        mx.eval(h)  # MLX graph evaluation, not Python eval
        # Copy post-layer state for branching. Float32 for numerics; we'll
        # cast back to model dtype before re-entering later layers.
        head_state = np.array(h.astype(mx.float32))  # [1, T, hidden]
        last_residual = head_state[0, -1, :].copy()  # [hidden]

        # --- run tail once for the BASELINE ---
        baseline_probs, baseline_logits = _run_tail(
            inner, model, backend, head_state, mask, mdtype, layer, n_layers,
        )
        baseline_top1 = int(np.argmax(baseline_probs))

        # --- noise-floor estimate: run tail AGAIN, compare to first baseline ---
        # f16 kernels are not bitwise deterministic. The KL of a second
        # identical run establishes the numeric floor.
        noise_probs, _ = _run_tail(
            inner, model, backend, head_state, mask, mdtype, layer, n_layers,
        )
        baseline_noise_sum += _kl_divergence(baseline_probs, noise_probs)
        n_valid += 1

        # --- per-dim ablation ---
        for di, d in enumerate(dims):
            mag_sum[di] += abs(last_residual[d])

            ablated_state = head_state.copy()
            ablated_state[0, -1, d] = 0.0

            abl_probs, abl_logits = _run_tail(
                inner, model, backend, ablated_state, mask, mdtype, layer, n_layers,
            )
            kl_sum[di] += _kl_divergence(baseline_probs, abl_probs)
            l2_sum[di] += float(np.linalg.norm(baseline_logits - abl_logits))
            if int(np.argmax(abl_probs)) != baseline_top1:
                flip_sum[di] += 1

            if progress and di % 100 == 0:
                sys.stderr.write('.')
                sys.stderr.flush()
        if progress:
            sys.stderr.write(f' [{p_idx+1}/{len(prompts)}]\n')
            sys.stderr.flush()

    n = max(1, n_valid)
    rows = [
        NullShartRow(
            dim=int(dims[di]),
            mag=float(mag_sum[di] / n),
            kl=float(kl_sum[di] / n),
            logit_l2=float(l2_sum[di] / n),
            top1_flip_rate=float(flip_sum[di] / n),
        )
        for di in range(len(dims))
    ]
    return NullShartReport(
        model_id=getattr(backend, "model_id", "?"),
        layer=layer,
        n_prompts=n_valid,
        hidden_size=hidden_size,
        baseline_noise_kl=float(baseline_noise_sum / n),
        rows=rows,
    )


def _run_tail(inner, model, backend, head_state, mask, mdtype, layer, n_layers):
    """Continue forward from after ``layer`` through lmhead.

    Returns (probs, logits) for the last token.
    """
    import mlx.core as mx

    h = mx.array(head_state.astype(np.float16))
    for i in range(layer + 1, n_layers):
        ly = inner.layers[i]
        h = ly(h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]
    h = inner.norm(h)
    logits = np.array(backend._lm_head(h).astype(mx.float32)[0, -1, :])
    probs = _softmax(logits)
    return probs, logits


def to_rows(report: NullShartReport) -> list[dict]:
    """Serializable list of row dicts. Use for JSON/CSV export."""
    return [
        {
            "dim": r.dim, "mag": r.mag, "kl": r.kl,
            "logit_l2": r.logit_l2, "top1_flip_rate": r.top1_flip_rate,
        }
        for r in report.rows
    ]


def summarize(report: NullShartReport, mag_thresh: float | None = None,
              kl_thresh: float | None = None) -> dict:
    """Quadrant counts and top-candidates.

    Thresholds default to medians, which gives 4 roughly-equal quadrants.
    """
    mags = np.array([r.mag for r in report.rows])
    kls = np.array([r.kl for r in report.rows])
    if mag_thresh is None:
        mag_thresh = float(np.median(mags))
    if kl_thresh is None:
        kl_thresh = float(np.median(kls))

    hi_mag = mags >= mag_thresh
    hi_kl = kls >= kl_thresh
    null_sharts = hi_mag & ~hi_kl       # active but inert — the theory claim
    key = hi_mag & hi_kl                # active and effective
    effective_without_mag = ~hi_mag & hi_kl   # small but impactful (surprising)
    inert = ~hi_mag & ~hi_kl

    floor = report.baseline_noise_kl + 1e-9
    null_score = mags / (kls + floor)
    top_null = np.argsort(null_score)[::-1][:20].tolist()

    return {
        "n_dims": int(len(report.rows)),
        "mag_thresh": mag_thresh,
        "kl_thresh": kl_thresh,
        "noise_floor_kl": report.baseline_noise_kl,
        "quadrants": {
            "null_shart_candidates": int(null_sharts.sum()),
            "key": int(key.sum()),
            "low_mag_high_kl": int(effective_without_mag.sum()),
            "inert": int(inert.sum()),
        },
        "null_shart_fraction": float(null_sharts.sum() / max(1, len(report.rows))),
        "top_null_dims": [
            {"dim": int(d), "mag": float(mags[d]), "kl": float(kls[d]),
             "ratio": float(null_score[d])}
            for d in top_null
        ],
    }
