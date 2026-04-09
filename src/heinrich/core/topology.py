"""Model topology discovery — streams, interfaces, and structure.

Inspects a model's configuration to discover its computational topology:
which streams (layer stacks) exist, how they connect, and what modality
each processes. No weights loaded — pure metadata inspection.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Stream:
    """A named stack of transformer layers within a model."""

    name: str
    n_layers: int
    hidden_size: int
    intermediate_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    modality: str  # "text", "audio", "vision"


@dataclass(frozen=True, slots=True)
class Interface:
    """A connection between two streams."""

    source: str  # stream name
    target: str  # stream name
    mechanism: str  # "cross_attention", "projection", "concatenation"
    target_layers: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ModelTopology:
    """Complete structural description of a model."""

    model_type: str
    streams: dict[str, Stream]
    interfaces: list[Interface]
    default_stream: str  # the stream existing tools target implicitly

    @property
    def stream_names(self) -> list[str]:
        return list(self.streams.keys())

    @property
    def is_multistream(self) -> bool:
        return len(self.streams) > 1

    def stream(self, name: str) -> Stream:
        return self.streams[name]


# ---------------------------------------------------------------------------
# Discovery: config → topology
# ---------------------------------------------------------------------------

# Modality hints by model_type / sub-config model_type.
_AUDIO_TYPES = {"whisper"}
_VISION_TYPES = {"clip_vision_model", "vit", "siglip_vision_model", "dinov2"}

_ENCODER_DECODER_TYPES = {"whisper", "bart", "mbart", "t5", "mt5", "longt5", "pegasus", "marian", "blenderbot"}


def _infer_modality(model_type: str) -> str:
    mt = model_type.lower()
    if mt in _AUDIO_TYPES:
        return "audio"
    if mt in _VISION_TYPES:
        return "vision"
    return "text"


def _safe_int(config, *attrs, default: int = 0) -> int:
    for attr in attrs:
        val = getattr(config, attr, None)
        if val is not None:
            return int(val)
    return default


def _make_stream(
    name: str,
    config,
    modality: str,
    *,
    n_layers_attrs: tuple[str, ...] = ("num_hidden_layers",),
    hidden_size_attrs: tuple[str, ...] = ("hidden_size", "d_model"),
    intermediate_attrs: tuple[str, ...] = ("intermediate_size", "encoder_ffn_dim", "decoder_ffn_dim"),
    n_heads_attrs: tuple[str, ...] = ("num_attention_heads",),
    n_kv_heads_attrs: tuple[str, ...] = ("num_key_value_heads", "num_attention_heads"),
) -> Stream:
    n_layers = _safe_int(config, *n_layers_attrs)
    hidden_size = _safe_int(config, *hidden_size_attrs)
    intermediate_size = _safe_int(config, *intermediate_attrs)
    n_heads = _safe_int(config, *n_heads_attrs)
    n_kv_heads = _safe_int(config, *n_kv_heads_attrs, default=n_heads)
    head_dim = hidden_size // n_heads if n_heads > 0 else 0
    return Stream(
        name=name,
        n_layers=n_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        modality=modality,
    )


def discover_topology(config) -> ModelTopology:
    """Discover model topology from a HuggingFace config object.

    Handles:
    - Decoder-only (Llama, Qwen, Phi, GPT-2, Mistral)
    - Encoder-decoder (Whisper, BART, T5, Marian)
    - Dual encoder (CLIP, SigLIP)
    - Vision + LLM (LLaVA, and similar)
    """
    model_type = getattr(config, "model_type", "unknown")
    streams: dict[str, Stream] = {}
    interfaces: list[Interface] = []
    default_stream = "decoder"

    # --- Dual sub-config models (CLIP, LLaVA, etc.) ---
    vision_config = getattr(config, "vision_config", None)
    text_config = getattr(config, "text_config", None)

    if vision_config is not None and text_config is not None:
        vision_type = getattr(vision_config, "model_type", "vision")
        text_type = getattr(text_config, "model_type", "text")

        streams["vision"] = _make_stream(
            "vision", vision_config,
            modality="vision",
        )
        streams["text"] = _make_stream(
            "text", text_config,
            modality="text",
        )

        # Distinguish dual-encoder (CLIP) from vision-tower+LLM (LLaVA)
        architectures = getattr(config, "architectures", []) or []
        arch_str = " ".join(architectures).lower()

        if "clip" in model_type.lower() or "siglip" in model_type.lower():
            interfaces.append(Interface(
                source="vision",
                target="text",
                mechanism="projection",
            ))
            default_stream = "text"
        else:
            # Vision tower feeding LLM decoder (LLaVA pattern)
            interfaces.append(Interface(
                source="vision",
                target="text",
                mechanism="projection",
            ))
            default_stream = "text"

        return ModelTopology(
            model_type=model_type,
            streams=streams,
            interfaces=interfaces,
            default_stream=default_stream,
        )

    # --- Encoder-decoder models (Whisper, BART, T5) ---
    if model_type.lower() in _ENCODER_DECODER_TYPES or (
        hasattr(config, "encoder_layers") and hasattr(config, "decoder_layers")
    ):
        encoder_modality = _infer_modality(model_type)

        streams["encoder"] = _make_stream(
            "encoder", config,
            modality=encoder_modality,
            n_layers_attrs=("encoder_layers", "num_hidden_layers"),
            hidden_size_attrs=("hidden_size", "d_model"),
            intermediate_attrs=("encoder_ffn_dim", "intermediate_size"),
            n_heads_attrs=("encoder_attention_heads", "num_attention_heads"),
            n_kv_heads_attrs=("encoder_attention_heads", "num_attention_heads"),
        )
        streams["decoder"] = _make_stream(
            "decoder", config,
            modality="text",
            n_layers_attrs=("decoder_layers", "num_hidden_layers"),
            hidden_size_attrs=("hidden_size", "d_model"),
            intermediate_attrs=("decoder_ffn_dim", "intermediate_size"),
            n_heads_attrs=("decoder_attention_heads", "num_attention_heads"),
            n_kv_heads_attrs=("decoder_attention_heads", "num_attention_heads"),
        )

        decoder_n_layers = streams["decoder"].n_layers
        interfaces.append(Interface(
            source="encoder",
            target="decoder",
            mechanism="cross_attention",
            target_layers=list(range(decoder_n_layers)),
        ))
        default_stream = "decoder"

        return ModelTopology(
            model_type=model_type,
            streams=streams,
            interfaces=interfaces,
            default_stream=default_stream,
        )

    # --- Decoder-only models (Llama, Qwen, Phi, GPT-2, Mistral) ---
    streams["decoder"] = _make_stream(
        "decoder", config,
        modality="text",
    )
    default_stream = "decoder"

    return ModelTopology(
        model_type=model_type,
        streams=streams,
        interfaces=interfaces,
        default_stream=default_stream,
    )
