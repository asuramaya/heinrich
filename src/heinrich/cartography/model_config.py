"""Model configuration — auto-detect architecture from any loaded model.

Eliminates hardcoded layer=27, layers=[24,25,26,27], hidden_size=3584, etc.
Works with MLX models (via model.args) and HuggingFace transformers (via model.config).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelConfig:
    """Architecture parameters extracted from a loaded model."""
    model_type: str           # e.g. "qwen2", "llama", "mistral"
    n_layers: int             # total number of transformer layers
    hidden_size: int          # residual stream dimension
    intermediate_size: int    # MLP intermediate dimension
    n_heads: int              # number of attention heads
    n_kv_heads: int           # number of KV heads (for GQA)
    head_dim: int             # hidden_size // n_heads
    vocab_size: int
    max_position_embeddings: int
    chat_format: str          # "qwen", "llama", "mistral", "chatml", "base"

    @property
    def last_layer(self) -> int:
        """Index of the final transformer layer."""
        return self.n_layers - 1

    @property
    def safety_layers(self) -> list[int]:
        """Late layers where safety signals typically concentrate.
        Last 4 layers of the model.
        """
        return list(range(self.n_layers - 4, self.n_layers))

    @property
    def all_layers(self) -> list[int]:
        return list(range(self.n_layers))

    @property
    def probe_layers(self) -> list[int]:
        """Representative layers for lightweight probing.
        Early (10%), mid (50%), late (75%), final.
        """
        return sorted(set([
            self.n_layers // 10,
            self.n_layers // 2,
            int(self.n_layers * 0.75),
            self.last_layer,
        ]))


def detect_config(model: Any, tokenizer: Any = None) -> ModelConfig:
    """Auto-detect model configuration from a loaded model.

    Supports:
    - MLX models (mlx_lm): reads model.args
    - HuggingFace transformers: reads model.config
    - Fallback: introspects layer structure
    """
    # Try MLX model.args first
    args = getattr(model, "args", None)
    if args is not None:
        return _from_mlx_args(args, tokenizer)

    # Try HuggingFace model.config
    config = getattr(model, "config", None)
    if config is not None:
        return _from_hf_config(config, tokenizer)

    # Fallback: introspect structure
    return _from_introspection(model, tokenizer)


def _from_mlx_args(args: Any, tokenizer: Any = None) -> ModelConfig:
    """Extract config from MLX model args."""
    model_type = getattr(args, "model_type", "unknown")
    n_layers = getattr(args, "num_hidden_layers", 28)
    hidden_size = getattr(args, "hidden_size", 3584)
    intermediate_size = getattr(args, "intermediate_size", 18944)
    n_heads = getattr(args, "num_attention_heads", 28)
    n_kv_heads = getattr(args, "num_key_value_heads", n_heads)
    vocab_size = getattr(args, "vocab_size", 152064)
    max_pos = getattr(args, "max_position_embeddings", 32768)

    chat_format = _detect_chat_format(model_type, tokenizer)

    return ModelConfig(
        model_type=model_type,
        n_layers=n_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=hidden_size // n_heads,
        vocab_size=vocab_size,
        max_position_embeddings=max_pos,
        chat_format=chat_format,
    )


def _from_hf_config(config: Any, tokenizer: Any = None) -> ModelConfig:
    """Extract config from HuggingFace model config."""
    model_type = getattr(config, "model_type", "unknown")
    n_layers = getattr(config, "num_hidden_layers", 32)
    hidden_size = getattr(config, "hidden_size", 4096)
    intermediate_size = getattr(config, "intermediate_size", 11008)
    n_heads = getattr(config, "num_attention_heads", 32)
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    vocab_size = getattr(config, "vocab_size", 32000)
    max_pos = getattr(config, "max_position_embeddings", 4096)

    chat_format = _detect_chat_format(model_type, tokenizer)

    return ModelConfig(
        model_type=model_type,
        n_layers=n_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=hidden_size // n_heads,
        vocab_size=vocab_size,
        max_position_embeddings=max_pos,
        chat_format=chat_format,
    )


def _from_introspection(model: Any, tokenizer: Any = None) -> ModelConfig:
    """Fallback: extract config by inspecting model structure."""
    inner = getattr(model, "model", model)

    # Count layers
    layers = getattr(inner, "layers", [])
    n_layers = len(layers)

    # Hidden size from norm weight
    norm = getattr(inner, "norm", None)
    if norm is not None and hasattr(norm, "weight"):
        hidden_size = norm.weight.shape[0]
    else:
        hidden_size = 4096  # common default

    # Attention config from first layer
    n_heads = 32
    n_kv_heads = 32
    intermediate_size = hidden_size * 4
    if n_layers > 0:
        attn = getattr(layers[0], "self_attn", None)
        if attn is not None:
            n_heads = getattr(attn, "n_heads", getattr(attn, "num_heads", 32))
            n_kv_heads = getattr(attn, "n_kv_heads",
                                 getattr(attn, "num_key_value_heads", n_heads))
        mlp = getattr(layers[0], "mlp", None)
        if mlp is not None:
            gate = getattr(mlp, "gate_proj", None)
            if gate is not None and hasattr(gate, "weight"):
                intermediate_size = gate.weight.shape[0]

    # Vocab from embeddings
    embed = getattr(inner, "embed_tokens", None)
    vocab_size = embed.weight.shape[0] if embed is not None and hasattr(embed, "weight") else 32000

    model_type = type(inner).__name__.lower().replace("model", "")
    chat_format = _detect_chat_format(model_type, tokenizer)

    return ModelConfig(
        model_type=model_type,
        n_layers=n_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=hidden_size // n_heads,
        vocab_size=vocab_size,
        max_position_embeddings=32768,
        chat_format=chat_format,
    )


def _detect_chat_format(model_type: str, tokenizer: Any = None) -> str:
    """Detect chat format from model type and tokenizer."""
    # Check tokenizer for explicit chat template
    if tokenizer is not None:
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            if "<|im_start|>" in chat_template:
                return "chatml"  # ChatML format (Qwen, Yi, etc.)
            if "[INST]" in chat_template:
                return "llama"   # Llama 2/3 format
            if "<|start_header_id|>" in chat_template:
                return "llama3"  # Llama 3 format

        # Check special tokens
        eos = getattr(tokenizer, "eos_token", "")
        if "<|im_end|>" in str(eos):
            return "chatml"

    # Fall back to model type
    format_map = {
        "qwen": "chatml",
        "qwen2": "chatml",
        "yi": "chatml",
        "llama": "llama",
        "mistral": "mistral",
        "mixtral": "mistral",
        "gemma": "gemma",
        "phi": "chatml",
        "deepseek": "chatml",
    }

    for key, fmt in format_map.items():
        if key in model_type.lower():
            return fmt

    return "base"  # no chat format detected
