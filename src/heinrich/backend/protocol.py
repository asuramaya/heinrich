"""Backend protocol, ForwardResult, and load_backend factory."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import numpy as np


@dataclass
class ForwardResult:
    """Result of a forward pass through the model."""
    logits: np.ndarray         # [vocab_size] last-position logits
    probs: np.ndarray          # [vocab_size] softmax probabilities
    top_id: int
    top_token: str
    entropy: float
    n_tokens: int
    residual: np.ndarray | None = None  # [hidden_size] if requested
    per_layer: list[dict] | None = None  # optional per-layer data from instrumented forward


@runtime_checkable
class Backend(Protocol):
    """Protocol for model inference backends."""

    config: "ModelConfig"  # noqa: F821

    def forward(
        self,
        prompt: str,
        *,
        steer_dirs: dict[int, tuple[np.ndarray, float]] | None = None,
        alpha: float = 0.0,
        return_residual: bool = False,
        residual_layer: int = -1,
    ) -> ForwardResult: ...

    def generate(
        self,
        prompt: str,
        *,
        steer_dirs: dict[int, tuple[np.ndarray, float]] | None = None,
        alpha: float = 0.0,
        max_tokens: int = 30,
    ) -> str: ...

    def capture_residual_states(
        self,
        prompts: list[str],
        *,
        layers: list[int],
    ) -> dict[int, np.ndarray]: ...

    def capture_mlp_activations(
        self,
        prompt: str,
        layer: int,
    ) -> np.ndarray: ...

    def capture_all_positions(
        self,
        prompt: str,
        *,
        layers: list[int],
    ) -> dict[int, np.ndarray]: ...
    """Capture residual stream at ALL token positions.
    Returns {layer: array[n_tokens, hidden_size]}.
    """

    def capture_mlp_detail(
        self,
        prompt: str,
        layer: int,
    ) -> dict[str, np.ndarray]: ...
    """Capture gate, up, activated, and down projections separately.
    Returns {"gate": [...], "up": [...], "activated": [...], "output": [...]}.
    """

    def forward_with_neuron_mask(
        self,
        prompt: str,
        layer: int,
        neuron_indices: list[int],
        *,
        return_residual: bool = False,
    ) -> ForwardResult: ...
    """Forward pass with specific MLP neurons zeroed at target layer."""

    def perturb_head(
        self,
        prompt: str,
        layer: int,
        head: int,
        *,
        mode: str = "zero",
        scale: float = 0.0,
    ) -> ForwardResult: ...
    """Forward pass with a single attention head zeroed/scaled.
    mode: "zero", "scale", "negate", "double".
    """

    def weight_projection(
        self,
        layer: int,
        neuron_index: int,
    ) -> np.ndarray: ...
    """Extract the gate_proj weight row for a neuron -- no forward pass.
    Returns [hidden_size] vector for embedding-space scoring.
    """

    def capture_attention_patterns(
        self,
        prompt: str,
        *,
        layers: list[int],
    ) -> dict[int, np.ndarray]: ...
    """Capture attention weights after softmax.
    Returns {layer: attention_weights[n_heads, n_tokens, n_tokens]}.
    """

    def capture_per_layer_delta(
        self,
        prompt: str,
    ) -> list[tuple[int, float]]: ...
    """Compute residual stream delta norm at every layer.
    Returns [(layer, delta_norm)] where delta = norm(residual_out - residual_in).
    """

    def steer_and_generate(
        self,
        prompt: str,
        *,
        alpha: float,
        layers: list[int],
        direction: np.ndarray,
        max_tokens: int = 30,
    ) -> tuple[str, list[dict]]: ...
    """Generate with steering applied to specified layers.
    Returns (generated_text, per_token_metadata) where metadata includes
    refuse_prob, top_token, dampening at each step.
    """

    def instrumented_forward(
        self,
        prompt: str,
        *,
        directions: dict[str, np.ndarray] | None = None,
    ) -> list[dict]: ...
    """Run a forward pass and capture everything at every layer.
    Returns a list of dicts, one per layer, each containing:
      - layer: int
      - residual_norm: float
      - delta_norm: float
      - projections: dict[str, float] (projection onto each named direction)
      - top_neurons: list[tuple[int, float]] (top 10 active neurons)
      - attention_pos2_weight: float (attention weight on position 2)
    """

    def tokenize(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...

    def capabilities(self) -> "Capabilities": ...  # noqa: F821

    def forward_context(self) -> "ForwardContext": ...  # noqa: F821

    def generation_context(self, prompt: str) -> "GenerationContext": ...  # noqa: F821


def load_backend(model_id: str, *, backend: str = "auto", **kwargs):
    """Load a model with the appropriate backend.

    backend: "mlx", "hf", or "auto" (tries MLX first on macOS, falls back to HF).
    """
    if backend == "mlx":
        from .mlx import MLXBackend
        return MLXBackend(model_id)
    elif backend == "hf":
        from .hf import HFBackend
        return HFBackend(model_id, **kwargs)
    elif backend == "auto":
        import platform
        if platform.system() == "Darwin":
            try:
                from .mlx import MLXBackend
                return MLXBackend(model_id)
            except (ImportError, Exception):
                pass
        from .hf import HFBackend
        return HFBackend(model_id, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'mlx', 'hf', or 'auto'.")
