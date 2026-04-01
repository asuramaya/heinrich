"""Enumerate the control surface of any transformer model."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Knob:
    """A single adjustable axis in a model."""
    id: str
    kind: str           # "head", "mlp_neuron", "residual_dim", "layer_mlp", "layer_attn"
    layer: int
    index: int = 0
    granularity: str = "coarse"   # "coarse", "medium", "fine"
    dim: int = 0        # dimension of the knob (head_dim for heads, 1 for neurons)


class ControlSurface:
    """Complete control surface of a model."""

    def __init__(self, knobs: list[Knob] | None = None) -> None:
        self.knobs = knobs or []
        self._index()

    def _index(self) -> None:
        self.by_kind: dict[str, list[Knob]] = {}
        self.by_layer: dict[int, list[Knob]] = {}
        self.by_granularity: dict[str, list[Knob]] = {}
        for k in self.knobs:
            self.by_kind.setdefault(k.kind, []).append(k)
            self.by_layer.setdefault(k.layer, []).append(k)
            self.by_granularity.setdefault(k.granularity, []).append(k)

    @classmethod
    def from_config(
        cls,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        intermediate_size: int,
        hidden_size: int,
    ) -> ControlSurface:
        """Build control surface from architecture parameters."""
        knobs = []
        for layer in range(n_layers):
            # Coarse: whole heads
            for head in range(n_heads):
                knobs.append(Knob(f"head.{layer}.{head}", "head", layer, head, "coarse", head_dim))
            # Coarse: whole MLP layer
            knobs.append(Knob(f"mlp.{layer}", "layer_mlp", layer, 0, "coarse", intermediate_size))
            # Coarse: whole attention layer
            knobs.append(Knob(f"attn.{layer}", "layer_attn", layer, 0, "coarse", n_heads * head_dim))
            # Medium: individual MLP neurons (top-k only, not all)
            # We'll populate these during the sweep based on activation magnitude
        return cls(knobs)

    @classmethod
    def from_mlx_model(cls, model: Any) -> ControlSurface:
        """Discover control surface from a loaded MLX model."""
        inner = getattr(model, "model", model)
        layers = getattr(inner, "layers", [])
        n_layers = len(layers)
        if n_layers == 0:
            return cls([])

        layer0 = layers[0]
        attn = getattr(layer0, "self_attn", None)
        n_heads = getattr(attn, "n_heads", 28) if attn else 28
        n_kv_heads = getattr(attn, "n_kv_heads", 4) if attn else 4

        # Get hidden size from embed
        embed = getattr(inner, "embed_tokens", None)
        if embed is not None and hasattr(embed, "weight"):
            hidden_size = embed.weight.shape[1]
        else:
            hidden_size = 3584

        head_dim = hidden_size // n_heads

        # Get intermediate size from MLP
        mlp = getattr(layer0, "mlp", None)
        intermediate_size = 18944  # default for Qwen 7B
        if mlp is not None:
            import mlx.nn as nn
            for k, v in nn.utils.tree_flatten(mlp.parameters()):
                if "gate" in k and "weight" in k:
                    intermediate_size = v.shape[0]
                    break

        return cls.from_config(n_layers, n_heads, head_dim, intermediate_size, hidden_size)

    def summary(self) -> dict[str, Any]:
        return {
            "total_knobs": len(self.knobs),
            "by_kind": {k: len(v) for k, v in self.by_kind.items()},
            "by_granularity": {k: len(v) for k, v in self.by_granularity.items()},
            "n_layers": len(self.by_layer),
        }
