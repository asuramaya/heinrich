"""O-Proj decomposition — find the real functional subspaces after projection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
import numpy as np
from ..signal import Signal, SignalStore

if TYPE_CHECKING:
    from .backend import Backend


@dataclass
class OProjDecomposition:
    layer: int
    n_heads: int
    head_dim: int
    hidden_size: int
    head_output_norms: list[float]
    head_overlap_matrix: np.ndarray  # [n_heads, n_heads] cosine similarity
    effective_rank: int
    top_singular_values: list[float]


def extract_oproj_weight(model: Any, layer: int, *, backend: Backend | None = None) -> np.ndarray:
    """Extract dequantized o_proj weight matrix [hidden_size, n_heads*head_dim].

    When backend is provided, uses backend.config for dimensions but still needs
    the raw model for direct weight probing. This is weight analysis, not inference,
    so the core extraction must touch model internals.
    """
    import mlx.core as mx
    inner = getattr(model, "model", model)
    attn = inner.layers[layer].self_attn
    hidden_size = inner.norm.weight.shape[0]
    # Probe by projecting identity-like vectors through quantized layer
    n_in = attn.n_heads * (hidden_size // attn.n_heads)
    cols = []
    batch = 64
    for start in range(0, n_in, batch):
        end = min(start + batch, n_in)
        inp = np.zeros((1, end - start, n_in), dtype=np.float16)
        for j in range(end - start):
            inp[0, j, start + j] = 1.0
        out = attn.o_proj(mx.array(inp))
        cols.append(np.array(out.astype(mx.float32)[0]))  # [batch, hidden_size]
    W = np.concatenate(cols, axis=0).T  # [hidden_size, n_in]
    return W


def decompose_oproj(model: Any, layer: int, *, top_k: int = 16,
                     backend: Backend | None = None) -> OProjDecomposition:
    """SVD of o_proj, compute head overlap matrix.

    When backend is provided, uses backend.config for model dimensions.
    The core SVD computation and weight extraction remain unchanged as
    this is weight analysis, not inference.
    """
    if backend is not None:
        n_heads = backend.config.n_heads
        hidden_size = backend.config.hidden_size
        head_dim = backend.config.head_dim
    else:
        inner = getattr(model, "model", model)
        n_heads = inner.layers[layer].self_attn.n_heads
        hidden_size = inner.norm.weight.shape[0]
        head_dim = hidden_size // n_heads

    W = extract_oproj_weight(model, layer, backend=backend)  # [hidden_size, n_heads * head_dim]

    # Per-head output norms and vectors
    head_vectors = []
    head_norms = []
    for h in range(n_heads):
        # Column slice for head h
        W_h = W[:, h * head_dim:(h + 1) * head_dim]  # [hidden_size, head_dim]
        # The "output direction" of this head is the dominant singular vector
        # But for overlap, use the Frobenius norm and the flattened column space
        norm = float(np.linalg.norm(W_h))
        head_norms.append(norm)
        # Flatten to get a single vector representing this head's output pattern
        # Use the mean of column vectors (crude but effective)
        vec = W_h.mean(axis=1)  # [hidden_size]
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 0:
            vec = vec / vec_norm
        head_vectors.append(vec)

    # Overlap matrix
    vecs = np.array(head_vectors)  # [n_heads, hidden_size]
    overlap = vecs @ vecs.T  # [n_heads, n_heads] cosine similarity

    # SVD of full o_proj
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    s_list = S[:top_k].tolist()
    # Effective rank: number of singular values > 1% of max
    threshold = S[0] * 0.01
    eff_rank = int(np.sum(S > threshold))

    return OProjDecomposition(
        layer=layer, n_heads=n_heads, head_dim=head_dim, hidden_size=hidden_size,
        head_output_norms=head_norms, head_overlap_matrix=overlap,
        effective_rank=eff_rank, top_singular_values=s_list,
    )


def scan_all_layers(
    model: Any, *, top_k: int = 16, store: SignalStore | None = None,
    backend: Backend | None = None,
) -> list[OProjDecomposition]:
    if backend is not None:
        n_layers = backend.config.n_layers
    else:
        inner = getattr(model, "model", model)
        n_layers = len(inner.layers)

    results = []
    for i in range(n_layers):
        d = decompose_oproj(model, i, top_k=top_k, backend=backend)
        results.append(d)
        if store:
            store.add(Signal("oproj_rank", "cartography", "model", f"layer.{i}",
                             d.effective_rank, {"top_sv": d.top_singular_values[:3]}))
    return results
