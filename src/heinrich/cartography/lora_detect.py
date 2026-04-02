"""LoRA safety-signature detection — detect safety-stripping adapters.

Question #19: Can the LoRA attack vector be detected?

A LoRA adapter targeting late layers (e.g., L24-27) could be distributed as
a "style adapter." This module detects signatures in LoRA weight matrices
that distinguish a safety-stripping adapter from a legitimate fine-tune.

Key insight: a safety-stripping LoRA has high cosine similarity with the
safety direction when projected through the affected layers. A style LoRA
is orthogonal to it.

Usage:
    from heinrich.cartography.lora_detect import analyze_lora_weights
    result = analyze_lora_weights("adapter.npz", backend=backend)
    print(result["risk_level"])  # "high", "medium", "low"
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import cosine


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_lora_weights(
    lora_path: str,
    backend: Any = None,
) -> dict:
    """Analyze a LoRA adapter for safety-relevant signatures.

    Checks:
    1. Which layers are modified (late-layer concentration = suspicious)
    2. SVD spectrum of the LoRA delta (low-rank modifications)
    3. Projection of LoRA delta onto known safety directions
    4. Magnitude relative to original weights

    Parameters
    ----------
    lora_path : str
        Path to a .npz file containing LoRA weight matrices.
        Expected keys follow the convention:
            base_model.model.layers.{N}.{submodule}.lora_A.weight
            base_model.model.layers.{N}.{submodule}.lora_B.weight
    backend : Any, optional
        A Backend instance (from heinrich.cartography.backend). If provided
        and it has a ``config`` attribute, used to determine total layer count
        and hidden size for proportional checks.

    Returns
    -------
    dict with keys:
        layers_modified : list[int]
        n_layers_modified : int
        late_layer_fraction : float
        svd_spectra : dict[int, list[float]]
        safety_alignment : dict[int, dict]   (only if safety_direction available)
        magnitude_stats : dict[int, dict]
        risk_level : str   ("high", "medium", "low")
        risk_reasons : list[str]
    """
    from ..inspect.lora import load_lora_deltas

    deltas, signals = load_lora_deltas(lora_path)
    if not deltas:
        return {
            "layers_modified": [],
            "n_layers_modified": 0,
            "late_layer_fraction": 0.0,
            "svd_spectra": {},
            "safety_alignment": {},
            "magnitude_stats": {},
            "risk_level": "low",
            "risk_reasons": ["no LoRA deltas found"],
            "load_signals": [s.kind for s in signals],
        }

    # Parse layer indices from delta key names
    layer_deltas: dict[int, list[np.ndarray]] = {}
    for key, delta in deltas.items():
        layer_idx = _extract_layer_index(key)
        if layer_idx is not None:
            layer_deltas.setdefault(layer_idx, []).append(delta)

    layers_modified = sorted(layer_deltas.keys())
    n_layers = _get_n_layers(backend)

    # --- Check 1: Late-layer concentration ---
    late_layer_fraction, late_layer_suspicious = _check_late_layer_concentration(
        layers_modified, n_layers
    )

    # --- Check 2: SVD spectrum of each delta ---
    svd_spectra: dict[int, list[float]] = {}
    for layer_idx, layer_delta_list in layer_deltas.items():
        combined = sum(
            np.abs(d) for d in layer_delta_list
        ) / len(layer_delta_list)
        try:
            sv = np.linalg.svd(combined, compute_uv=False)
            svd_spectra[layer_idx] = [round(float(s), 6) for s in sv[:20]]
        except np.linalg.LinAlgError:
            svd_spectra[layer_idx] = []

    # --- Check 3: Projection onto safety direction ---
    safety_direction = _get_safety_direction(backend)
    safety_alignment: dict[int, dict] = {}
    high_alignment_count = 0
    if safety_direction is not None:
        for layer_idx, layer_delta_list in layer_deltas.items():
            for delta in layer_delta_list:
                alignment = compare_lora_to_safety(delta, safety_direction)
                safety_alignment[layer_idx] = alignment
                if alignment["is_safety_relevant"]:
                    high_alignment_count += 1

    # --- Check 4: Magnitude stats ---
    magnitude_stats: dict[int, dict] = {}
    for layer_idx, layer_delta_list in layer_deltas.items():
        norms = [float(np.linalg.norm(d)) for d in layer_delta_list]
        fro_norms = [
            float(np.linalg.norm(d, "fro")) for d in layer_delta_list
        ]
        magnitude_stats[layer_idx] = {
            "spectral_norms": norms,
            "frobenius_norms": fro_norms,
            "max_spectral": max(norms) if norms else 0.0,
            "mean_frobenius": float(np.mean(fro_norms)) if fro_norms else 0.0,
        }

    # --- Risk assessment ---
    risk_reasons: list[str] = []

    if late_layer_suspicious:
        risk_reasons.append(
            f"late-layer concentration: {late_layer_fraction:.0%} of modified "
            f"layers are in the last quarter"
        )

    if high_alignment_count > 0:
        risk_reasons.append(
            f"{high_alignment_count} layer delta(s) align with safety direction"
        )

    # Low-rank signature: if top singular value dominates
    for layer_idx, spectrum in svd_spectra.items():
        if len(spectrum) >= 2 and spectrum[0] > 0:
            concentration = spectrum[0] / (sum(spectrum) + 1e-12)
            if concentration > 0.8:
                risk_reasons.append(
                    f"L{layer_idx}: top singular value captures "
                    f"{concentration:.0%} of spectral energy (rank-1-like)"
                )

    if len(risk_reasons) >= 2:
        risk_level = "high"
    elif len(risk_reasons) == 1:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "layers_modified": layers_modified,
        "n_layers_modified": len(layers_modified),
        "late_layer_fraction": round(late_layer_fraction, 3),
        "svd_spectra": svd_spectra,
        "safety_alignment": safety_alignment,
        "magnitude_stats": magnitude_stats,
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
    }


def simulate_safety_lora(
    backend: Any,
    target_layers: list[int],
    rank: int = 8,
) -> dict[str, np.ndarray]:
    """Generate a simulated safety-stripping LoRA.

    Creates a low-rank delta that would amplify the compliance direction
    (i.e., suppress the safety/refusal direction). This is for detection
    testing -- shows what a malicious LoRA looks like.

    Parameters
    ----------
    backend : Any
        Backend with a discovered safety direction (either via
        ``backend.safety_direction`` or fetched from a ModelProfile).
    target_layers : list[int]
        Layers to target (e.g. [24, 25, 26, 27]).
    rank : int
        Rank of the simulated LoRA adapter (default 8).

    Returns
    -------
    dict mapping LoRA key names to delta matrices (B @ A form).
    """
    safety_direction = _get_safety_direction(backend)
    if safety_direction is None:
        raise ValueError(
            "backend has no safety_direction; run discover_profile first"
        )

    d = safety_direction / (np.linalg.norm(safety_direction) + 1e-12)
    hidden_size = len(d)
    rng = np.random.default_rng(42)

    result: dict[str, np.ndarray] = {}
    for layer in target_layers:
        # The core: a rank-1 component aligned with -safety_direction
        # (negative = suppress safety). Pad to requested rank with noise.
        A = np.zeros((rank, hidden_size), dtype=np.float32)
        B = np.zeros((hidden_size, rank), dtype=np.float32)

        # Primary component: flip the safety direction
        A[0] = d
        B[:, 0] = -d  # negative = stripping safety

        # Fill remaining rank with small noise (looks like a real adapter)
        if rank > 1:
            noise_scale = 0.1
            A[1:] = rng.standard_normal((rank - 1, hidden_size)).astype(
                np.float32
            ) * noise_scale
            B[:, 1:] = rng.standard_normal((hidden_size, rank - 1)).astype(
                np.float32
            ) * noise_scale

        delta = B @ A
        key = (
            f"base_model.model.layers.{layer}"
            f".self_attn.o_proj.lora_delta"
        )
        result[key] = delta.astype(np.float32)

    return result


def compare_lora_to_safety(
    lora_delta: np.ndarray,
    safety_direction: np.ndarray,
    *,
    threshold: float = 0.3,
) -> dict:
    """Check if a LoRA delta aligns with the safety direction.

    High alignment (cos > threshold) suggests the LoRA modifies safety
    behavior. A style LoRA will be orthogonal to the safety direction.

    The check projects the safety direction through the delta matrix:
        projected = delta @ safety_direction
    Then measures cosine similarity between the projected vector and the
    safety direction itself.

    Parameters
    ----------
    lora_delta : np.ndarray
        The LoRA delta weight matrix (B @ A), shape [out_dim, in_dim].
    safety_direction : np.ndarray
        Unit vector for the safety direction, shape [hidden_size].
    threshold : float
        Cosine similarity above which the delta is flagged as
        safety-relevant.

    Returns
    -------
    dict with keys:
        cosine_similarity : float
        projection_norm : float
        is_safety_relevant : bool
        amplifies_safety : bool  (True if same direction, False if opposing)
    """
    d = safety_direction / (np.linalg.norm(safety_direction) + 1e-12)

    # Handle shape mismatch: delta might be [out, in] where out != in
    # or [hidden, hidden]. We project safely.
    if lora_delta.ndim != 2:
        return {
            "cosine_similarity": 0.0,
            "projection_norm": 0.0,
            "is_safety_relevant": False,
            "amplifies_safety": False,
        }

    out_dim, in_dim = lora_delta.shape

    # We can only meaningfully project if dimensions match the safety direction
    if in_dim != len(d) and out_dim != len(d):
        return {
            "cosine_similarity": 0.0,
            "projection_norm": 0.0,
            "is_safety_relevant": False,
            "amplifies_safety": False,
        }

    # Project safety direction through the delta
    if in_dim == len(d):
        projected = lora_delta @ d
    else:
        # out_dim matches, use transpose
        projected = lora_delta.T @ d

    proj_norm = float(np.linalg.norm(projected))

    # Compare projected vector to safety direction (truncate/pad as needed)
    if len(projected) == len(d):
        cos_sim = cosine(projected, d)
    elif len(projected) > len(d):
        cos_sim = cosine(projected[: len(d)], d)
    else:
        cos_sim = cosine(projected, d[: len(projected)])

    return {
        "cosine_similarity": round(float(cos_sim), 4),
        "projection_norm": round(proj_norm, 6),
        "is_safety_relevant": abs(cos_sim) > threshold,
        "amplifies_safety": cos_sim > threshold,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_layer_index(key: str) -> int | None:
    """Extract layer number from a LoRA key like 'layers.24.self_attn...'."""
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return None


def _get_n_layers(backend: Any) -> int:
    """Get total layer count from backend config, or use a default."""
    if backend is not None:
        cfg = getattr(backend, "config", None)
        if cfg is not None:
            return getattr(cfg, "n_layers", 32)
    return 32  # reasonable default


def _get_safety_direction(backend: Any) -> np.ndarray | None:
    """Get the safety direction from a backend or profile."""
    if backend is None:
        return None

    # Direct attribute
    sd = getattr(backend, "safety_direction", None)
    if sd is not None:
        return sd

    # From a profile attached to the backend
    profile = getattr(backend, "profile", None)
    if profile is not None:
        return getattr(profile, "safety_direction", None)

    return None


def _check_late_layer_concentration(
    layers_modified: list[int],
    n_layers: int,
) -> tuple[float, bool]:
    """Check if modifications concentrate in late layers.

    Returns (fraction_in_last_quarter, is_suspicious).
    """
    if not layers_modified or n_layers <= 0:
        return 0.0, False

    threshold_layer = n_layers * 3 // 4
    late_count = sum(1 for l in layers_modified if l >= threshold_layer)
    fraction = late_count / len(layers_modified)

    # Suspicious if >60% of modifications are in the last quarter
    return fraction, fraction > 0.6
