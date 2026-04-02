"""Intervention context — compose multiple interventions in a single forward pass.

Solves the 5 limitations of the basic Backend:
1. Compositional intervention (multiple steer + capture + ablate in one pass)
2. Conditional mid-generation intervention (per-token control loop)
3. Sub-layer access (pre-o_proj heads, gate/up MLP split)
4. Stateful generation (KV cache between tokens)
5. Capability declaration (what can this backend do?)

Usage:
    with backend.forward_context() as ctx:
        ctx.steer(layer=24, direction=d1, alpha=-0.15)
        ctx.zero_neurons(layer=27, neurons=[1934])
        ctx.capture_residual(layers=[15, 27])
        ctx.capture_attention(layer=27)
        result = ctx.run(prompt)

    with backend.generation_context(prompt) as gen:
        gen.steer(layer=24, direction=d, alpha=-0.15)
        for token in gen.tokens(max_tokens=200):
            proj = cosine(token.residual, safety_dir)
            if proj > 0.5:
                gen.inject_once(layer=20, vector=nudge)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator
import numpy as np


# === Capabilities ===

@dataclass
class Capabilities:
    """What a backend can do. The audit adapts based on this."""
    can_steer: bool = False
    can_capture_residual: bool = False
    can_capture_attention: bool = False
    can_capture_mlp_detail: bool = False
    can_neuron_mask: bool = False
    can_perturb_head: bool = False
    can_weight_access: bool = False
    can_kv_cache: bool = False
    can_gradient: bool = False
    can_all_positions: bool = False
    can_compose: bool = False       # ForwardContext support
    can_gen_control: bool = False    # GenerationContext support

    @property
    def depth_level(self) -> str:
        """What audit depth this backend supports."""
        if self.can_compose and self.can_gen_control:
            return "deep"
        if self.can_steer and self.can_capture_residual:
            return "standard"
        if self.can_capture_residual:
            return "quick"
        return "surface"


# === Intervention declarations ===

@dataclass
class SteerOp:
    layer: int
    direction: np.ndarray
    mean_gap: float
    alpha: float


@dataclass
class NeuronMaskOp:
    layer: int
    neurons: list[int]


@dataclass
class CaptureResidualOp:
    layers: list[int]
    all_positions: bool = False


@dataclass
class CaptureAttentionOp:
    layer: int


@dataclass
class CaptureMlpDetailOp:
    layer: int


# === Forward context result ===

@dataclass
class ContextResult:
    """Result from a ForwardContext.run() call."""
    logits: np.ndarray
    probs: np.ndarray
    top_id: int
    top_token: str
    entropy: float
    n_tokens: int
    residuals: dict[int, np.ndarray] = field(default_factory=dict)
    all_position_residuals: dict[int, np.ndarray] = field(default_factory=dict)
    attention: dict[int, np.ndarray] = field(default_factory=dict)
    mlp_detail: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)


# === Generation token result ===

@dataclass
class TokenResult:
    """Per-token result during GenerationContext iteration."""
    step: int
    token_id: int
    token_text: str
    residual: np.ndarray | None = None
    logits: np.ndarray | None = None


# === Forward Context ===

class ForwardContext:
    """Compose multiple interventions for a single forward pass.

    Collects declarations, then compiles them into backend-specific
    execution when run() is called.
    """

    def __init__(self, backend: Any):
        self._backend = backend
        self._steers: list[SteerOp] = []
        self._neuron_masks: list[NeuronMaskOp] = []
        self._capture_residuals: list[CaptureResidualOp] = []
        self._capture_attentions: list[CaptureAttentionOp] = []
        self._capture_mlp_details: list[CaptureMlpDetailOp] = []

    def steer(self, layer: int, direction: np.ndarray, mean_gap: float = 1.0, alpha: float = 1.0):
        self._steers.append(SteerOp(layer, direction, mean_gap, alpha))
        return self

    def zero_neurons(self, layer: int, neurons: list[int]):
        self._neuron_masks.append(NeuronMaskOp(layer, neurons))
        return self

    def capture_residual(self, layers: list[int], *, all_positions: bool = False):
        self._capture_residuals.append(CaptureResidualOp(layers, all_positions))
        return self

    def capture_attention(self, layer: int):
        self._capture_attentions.append(CaptureAttentionOp(layer))
        return self

    def capture_mlp_detail(self, layer: int):
        self._capture_mlp_details.append(CaptureMlpDetailOp(layer))
        return self

    def run(self, prompt: str) -> ContextResult:
        """Execute all declared interventions in a single forward pass."""
        return self._backend._execute_forward_context(prompt, self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# === Generation Context ===

class GenerationContext:
    """Stateful generation with per-token intervention control."""

    def __init__(self, backend: Any, prompt: str):
        self._backend = backend
        self._prompt = prompt
        self._steers: list[SteerOp] = []
        self._one_shot_injections: list[tuple[int, np.ndarray]] = []
        self._capture_layer: int | None = None

    def steer(self, layer: int, direction: np.ndarray, mean_gap: float = 1.0, alpha: float = 1.0):
        """Add persistent steering (applied at every generated token)."""
        self._steers.append(SteerOp(layer, direction, mean_gap, alpha))
        return self

    def inject_once(self, layer: int, vector: np.ndarray):
        """Inject a vector at next token only, then auto-remove."""
        self._one_shot_injections.append((layer, vector))

    def capture_at(self, layer: int):
        """Set which layer's residual to capture per token."""
        self._capture_layer = layer

    def tokens(self, *, max_tokens: int = 200) -> Iterator[TokenResult]:
        """Iterate generated tokens with per-token state."""
        return self._backend._execute_generation_context(
            self._prompt, self, max_tokens=max_tokens
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
