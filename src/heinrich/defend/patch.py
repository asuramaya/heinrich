"""Safety patching — generate LoRA-style weight modifications to amplify safety.

Given a safety direction and layer from discovery, compute weight deltas that
amplify the model's refusal signal. Two methods:

1. direction_amplify: modify o_proj so the safety direction's projection is
   stronger in the output.
2. neuron_boost: increase weights connecting to the top safety neurons.

Patches can be tested in-place (temporarily modifying weights) and exported
as low-rank LoRA (A, B) matrices via SVD decomposition.
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from heinrich.cartography.backend import Backend


def compute_safety_patch(
    backend: Any,
    safety_direction: np.ndarray,
    layer: int,
    *,
    strength: float = 1.0,
    method: str = "direction_amplify",
) -> dict[str, np.ndarray]:
    """Compute weight modifications that amplify the safety direction.

    method="direction_amplify":
        Modify o_proj to amplify safety direction projection.
        delta = strength * outer(safety_direction, safety_direction)
        This makes the o_proj output more aligned with the safety direction
        when the input already has a safety-direction component.

    method="neuron_boost":
        Increase weights connecting to top safety neurons.
        Extracts top activated dimensions from the safety direction and
        boosts the corresponding rows in o_proj.

    Returns {"layer.{layer}.self_attn.o_proj.weight": delta_matrix, ...}
    """
    d = safety_direction / (np.linalg.norm(safety_direction) + 1e-12)

    if method == "direction_amplify":
        # Rank-1 update: amplify projection onto safety direction
        # delta_W = strength * d @ d^T
        delta = strength * np.outer(d, d)

    elif method == "neuron_boost":
        # Boost top-k dimensions of the safety direction
        k = max(1, int(0.01 * len(d)))  # top 1% of dimensions
        top_dims = np.argsort(np.abs(d))[-k:]
        delta = np.zeros((len(d), len(d)), dtype=np.float32)
        for dim in top_dims:
            # Boost the row corresponding to this dimension
            delta[dim, :] += strength * d[dim] * d

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'direction_amplify' or 'neuron_boost'.")

    key = f"layer.{layer}.self_attn.o_proj.weight"
    return {key: delta.astype(np.float32)}


def apply_patch_and_test(
    backend: Any,
    patch: dict[str, np.ndarray],
    test_prompts: list[str],
) -> dict:
    """Apply a weight patch temporarily and measure impact on refusal rates.

    For each test prompt, runs forward pass before and after patching,
    compares refusal probability shift.

    Returns {
        "before": {"mean_refuse_prob": float, "mean_comply_prob": float},
        "after":  {"mean_refuse_prob": float, "mean_comply_prob": float},
        "per_prompt": [...],
    }
    """
    from heinrich.cartography.classify import classify_response

    # Measure baseline
    before_results = []
    for prompt in test_prompts:
        result = backend.forward(prompt)
        text = backend.generate(prompt, max_tokens=30)
        cls = classify_response(text)
        before_results.append({
            "prompt": prompt,
            "top_token": result.top_token,
            "entropy": result.entropy,
            "classification": cls.label,
        })

    # Apply patch
    _apply_weight_patch(backend, patch)

    # Measure after patch
    after_results = []
    try:
        for prompt in test_prompts:
            result = backend.forward(prompt)
            text = backend.generate(prompt, max_tokens=30)
            cls = classify_response(text)
            after_results.append({
                "prompt": prompt,
                "top_token": result.top_token,
                "entropy": result.entropy,
                "classification": cls.label,
            })
    finally:
        # Restore original weights
        _remove_weight_patch(backend, patch)

    # Aggregate
    before_refuse = sum(1 for r in before_results if r["classification"] == "REFUSES") / max(len(before_results), 1)
    after_refuse = sum(1 for r in after_results if r["classification"] == "REFUSES") / max(len(after_results), 1)

    return {
        "before": {
            "refusal_rate": round(before_refuse, 3),
            "n_prompts": len(before_results),
        },
        "after": {
            "refusal_rate": round(after_refuse, 3),
            "n_prompts": len(after_results),
        },
        "per_prompt": [
            {
                "prompt": before_results[i]["prompt"],
                "before_class": before_results[i]["classification"],
                "after_class": after_results[i]["classification"],
                "before_entropy": before_results[i]["entropy"],
                "after_entropy": after_results[i]["entropy"],
            }
            for i in range(len(before_results))
        ],
    }


def export_lora(
    patch: dict[str, np.ndarray],
    rank: int = 8,
    output_path: str | None = None,
) -> dict:
    """Decompose a weight patch into low-rank LoRA matrices (A, B).

    Uses SVD: delta ~= B @ A where A is [rank, in_dim], B is [out_dim, rank].

    If output_path is given, saves as .npz file.

    Returns {
        key: {"A": ndarray[rank, in], "B": ndarray[out, rank], "rank": int,
              "reconstruction_error": float}
        for each key in patch
    }.
    """
    result = {}
    for key, delta in patch.items():
        U, S, Vt = np.linalg.svd(delta, full_matrices=False)
        # Take top-rank components
        effective_rank = min(rank, len(S))
        B = U[:, :effective_rank] * S[:effective_rank]  # [out, rank]
        A = Vt[:effective_rank, :]  # [rank, in]

        # Reconstruction error
        reconstructed = B @ A
        error = float(np.linalg.norm(delta - reconstructed) / (np.linalg.norm(delta) + 1e-12))

        result[key] = {
            "A": A.astype(np.float32),
            "B": B.astype(np.float32),
            "rank": effective_rank,
            "reconstruction_error": round(error, 6),
        }

    if output_path is not None:
        save_dict = {}
        for key, data in result.items():
            safe_key = key.replace(".", "_")
            save_dict[f"{safe_key}_A"] = data["A"]
            save_dict[f"{safe_key}_B"] = data["B"]
        np.savez(output_path, **save_dict)

    return result


# --- Internal helpers ---

def _apply_weight_patch(backend: Any, patch: dict[str, np.ndarray]) -> dict[str, Any]:
    """Apply weight deltas to the backend model. Stores originals on backend._patch_originals."""
    originals = {}
    model = _get_inner_model(backend)

    for key, delta in patch.items():
        parts = key.split(".")
        # Parse: layer.{N}.self_attn.o_proj.weight
        layer_idx = int(parts[1])
        attr_path = parts[2:]  # e.g. ["self_attn", "o_proj", "weight"]

        obj = model.layers[layer_idx]
        for attr in attr_path[:-1]:
            obj = getattr(obj, attr)

        weight_attr = attr_path[-1]
        orig_weight = getattr(obj, weight_attr)

        # Store original
        originals[key] = orig_weight

        # Apply: convert to numpy, add delta, set back
        if hasattr(orig_weight, '__array__'):
            w_np = np.array(orig_weight, dtype=np.float32)
        else:
            w_np = orig_weight
        new_weight = w_np + delta

        # Try to convert back to framework tensor
        try:
            import mlx.core as mx
            new_weight = mx.array(new_weight)
        except ImportError:
            pass

        setattr(obj, weight_attr, new_weight)

    backend._patch_originals = originals
    return originals


def _remove_weight_patch(backend: Any, patch: dict[str, np.ndarray]) -> None:
    """Restore original weights after patching."""
    originals = getattr(backend, "_patch_originals", {})
    model = _get_inner_model(backend)

    for key in patch:
        if key not in originals:
            continue
        parts = key.split(".")
        layer_idx = int(parts[1])
        attr_path = parts[2:]

        obj = model.layers[layer_idx]
        for attr in attr_path[:-1]:
            obj = getattr(obj, attr)

        setattr(obj, attr_path[-1], originals[key])

    backend._patch_originals = {}


def _get_inner_model(backend: Any) -> Any:
    """Get the inner model object (with .layers) from a backend."""
    if hasattr(backend, "_inner"):
        return backend._inner
    model = getattr(backend, "model", backend)
    inner = getattr(model, "model", model)
    return inner


# Public alias for backward compatibility
direction_amplify = compute_safety_patch
