"""Fossil vs. formation: does a model traversing frozen weights look
different from one whose weights are updating concurrent with traversal?

Measurement apparatus comparing FROZEN inference and ONLINE-SGD inference.

Added (follow-up): multi-layer updates (all MLP down_projs simultaneously)
and path-dependence testing (does the order of training texts matter?).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from heinrich.cartography.perturb import _mask_dtype


@dataclass
class TraceResult:
    mode: str
    prompt: str
    tokens_generated: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    residuals: list[np.ndarray] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    layer: int = 0
    elapsed_s: float = 0.0

    def residual_matrix(self) -> np.ndarray:
        return np.stack(self.residuals) if self.residuals else np.zeros((0, 0))


def _forward_and_logits(inner, lm_head_fn, ids_mx, mask, capture_layer: int):
    h = inner.embed_tokens(ids_mx)
    captured = None
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]
        if i == capture_layer:
            captured = h[:, -1, :]
    h = inner.norm(h)
    logits = lm_head_fn(h)[:, -1, :]
    return logits, captured


def frozen_trace(backend, prompt: str, *, layer: int, max_tokens: int = 40,
                 teacher_text: str | None = None) -> TraceResult:
    inner = backend._inner
    tok = backend.tokenizer
    mdtype = _mask_dtype(backend.model)
    lm_head_fn = backend._lm_head

    ids = list(tok.encode(prompt))
    teacher_ids = list(tok.encode(teacher_text)) if teacher_text else []

    result = TraceResult(mode="frozen", prompt=prompt, layer=layer)
    t0 = time.time()

    for step in range(max_tokens):
        x = mx.array([ids])
        T = len(ids)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        logits, captured = _forward_and_logits(inner, lm_head_fn, x, mask, layer)
        residual = np.array(captured.astype(mx.float32))[0]

        if teacher_ids and step < len(teacher_ids):
            nxt = teacher_ids[step]
        else:
            nxt = int(mx.argmax(logits[0]))

        result.residuals.append(residual)
        result.token_ids.append(nxt)
        result.tokens_generated.append(tok.decode([nxt]))
        ids.append(nxt)

    result.elapsed_s = time.time() - t0
    return result


def online_trace(backend, prompt: str, teacher_text: str, *, layer: int,
                 max_tokens: int = 40, lr: float = 1e-3,
                 update_layers: list[int] | str = "one") -> TraceResult:
    """Online SGD during inference.

    update_layers:
      - "one"  : single layer (== `layer`)
      - "all"  : all transformer layers' MLP down_proj
      - list[int]: specific layer indices
    """
    inner = backend._inner
    tok = backend.tokenizer
    mdtype = _mask_dtype(backend.model)
    lm_head_fn = backend._lm_head
    model = backend.model

    n_layers = len(inner.layers)
    if update_layers == "one":
        target_layers = [layer]
    elif update_layers == "all":
        target_layers = list(range(n_layers))
    elif isinstance(update_layers, list):
        target_layers = update_layers
    else:
        raise ValueError(f"bad update_layers: {update_layers}")

    teacher_ids = list(tok.encode(teacher_text))
    if not teacher_ids:
        raise ValueError("teacher_text must encode to >= 1 token")

    ids = list(tok.encode(prompt))
    result = TraceResult(mode="online", prompt=prompt, layer=layer)
    t0 = time.time()

    def loss_fn(model_local, ids_mx, mask_local, target):
        h = model_local.model.embed_tokens(ids_mx)
        for ly in model_local.model.layers:
            h = ly(h, mask=mask_local, cache=None)
            if isinstance(h, tuple):
                h = h[0]
        h = model_local.model.norm(h)
        logits = lm_head_fn(h)[:, -1, :]
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -logp[0, target]

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for step in range(max_tokens):
        x = mx.array([ids])
        T = len(ids)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        logits, captured = _forward_and_logits(inner, lm_head_fn, x, mask, layer)
        residual = np.array(captured.astype(mx.float32))[0]

        if step < len(teacher_ids):
            target_tok = teacher_ids[step]
        else:
            target_tok = int(mx.argmax(logits[0]))

        loss_val, grads = loss_and_grad(model, x, mask, target_tok)
        mx.eval(loss_val)  # MLX graph evaluation, not Python eval
        loss_scalar = float(loss_val)

        # Update targeted layers' MLP down_proj
        for li in target_layers:
            try:
                g = grads["model"]["layers"][li]["mlp"]["down_proj"]["weight"]
                w = model.model.layers[li].mlp.down_proj.weight
                model.model.layers[li].mlp.down_proj.weight = w - lr * g
                mx.eval(model.model.layers[li].mlp.down_proj.weight)  # MLX graph evaluation, not Python eval
            except (KeyError, AttributeError, TypeError):
                pass  # grad path missing; skip

        result.residuals.append(residual)
        result.token_ids.append(target_tok)
        result.tokens_generated.append(tok.decode([target_tok]))
        result.losses.append(loss_scalar)
        ids.append(target_tok)

    result.elapsed_s = time.time() - t0
    return result


def snapshot_weights(backend, layers: list[int]) -> list[mx.array]:
    """Save copies of the targeted layers' MLP down_proj weights."""
    return [mx.array(backend.model.model.layers[li].mlp.down_proj.weight)
            for li in layers]


def restore_weights(backend, layers: list[int], snapshots: list[mx.array]):
    """Restore layer weights from snapshots."""
    for li, snap in zip(layers, snapshots):
        backend.model.model.layers[li].mlp.down_proj.weight = mx.array(snap)


def compute_signatures(trace: TraceResult) -> dict:
    R = trace.residual_matrix()
    if R.ndim != 2 or len(R) < 2:
        return {"error": "not enough steps"}
    norms = np.linalg.norm(R, axis=1)
    diffs = np.diff(R, axis=0)
    velocity = np.linalg.norm(diffs, axis=1)
    curvature = np.zeros(len(diffs) - 1)
    for i in range(len(diffs) - 1):
        a, b = diffs[i], diffs[i + 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-9 and nb > 1e-9:
            cosang = float(np.clip(a @ b / (na * nb), -1.0, 1.0))
            curvature[i] = float(np.arccos(cosang))
    r0 = R[0] / (np.linalg.norm(R[0]) + 1e-9)
    self_sim = np.array([float((R[i] / (np.linalg.norm(R[i]) + 1e-9)) @ r0)
                         for i in range(len(R))])
    return {
        "norm_trajectory": norms.tolist(),
        "velocity": velocity.tolist(),
        "curvature": curvature.tolist(),
        "self_similarity": self_sim.tolist(),
        "mean_norm": float(norms.mean()),
        "mean_velocity": float(velocity.mean()),
        "mean_curvature": float(curvature.mean()),
    }


def plot_comparison(frozen: TraceResult, online: TraceResult,
                    out_path: str, *, probe_library=None) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fs = compute_signatures(frozen)
    os_ = compute_signatures(online)
    n_rows = 4 + (1 if probe_library else 0) + (1 if online.losses else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.5 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]
    r = 0
    axes[r].plot(fs["velocity"], label="frozen", color="#4477aa", lw=2)
    axes[r].plot(os_["velocity"], label="online", color="#cc3355", lw=2)
    axes[r].set_ylabel("velocity"); axes[r].legend(); axes[r].grid(alpha=0.3)
    axes[r].set_title("manifold speed per step ||Δresidual||")
    r += 1
    axes[r].plot(fs["curvature"], label="frozen", color="#4477aa", lw=2)
    axes[r].plot(os_["curvature"], label="online", color="#cc3355", lw=2)
    axes[r].set_ylabel("curvature (rad)"); axes[r].legend(); axes[r].grid(alpha=0.3)
    axes[r].set_title("trajectory curvature per step")
    r += 1
    axes[r].plot(fs["self_similarity"], label="frozen", color="#4477aa", lw=2)
    axes[r].plot(os_["self_similarity"], label="online", color="#cc3355", lw=2)
    axes[r].set_ylabel("cos(r[t], r[0])"); axes[r].legend(); axes[r].grid(alpha=0.3)
    axes[r].set_title("drift from initial residual")
    r += 1
    axes[r].plot(fs["norm_trajectory"], label="frozen", color="#4477aa", lw=2)
    axes[r].plot(os_["norm_trajectory"], label="online", color="#cc3355", lw=2)
    axes[r].set_ylabel("||residual||"); axes[r].legend(); axes[r].grid(alpha=0.3)
    axes[r].set_title("residual norm trajectory")
    r += 1
    if probe_library:
        names = probe_library.names()
        fm = np.array([[rr @ probe_library.probes[n].direction for n in names]
                       for rr in frozen.residuals])
        om = np.array([[rr @ probe_library.probes[n].direction for n in names]
                       for rr in online.residuals])
        axes[r].plot(np.abs(fm).sum(1), label="frozen", color="#4477aa", lw=2)
        axes[r].plot(np.abs(om).sum(1), label="online", color="#cc3355", lw=2)
        axes[r].set_ylabel("Σ|probe proj|"); axes[r].legend(); axes[r].grid(alpha=0.3)
        axes[r].set_title("total concept engagement per step")
        r += 1
    if online.losses:
        axes[r].plot(online.losses, color="#cc3355", lw=2)
        axes[r].set_ylabel("loss"); axes[r].grid(alpha=0.3)
        axes[r].set_title("online SGD loss per step (teacher-forced cross-entropy)")
    axes[-1].set_xlabel("generation step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
