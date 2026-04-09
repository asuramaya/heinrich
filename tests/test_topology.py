"""Tests for model topology discovery."""
import pytest
from heinrich.core.topology import (
    Stream, Interface, ModelTopology, discover_topology,
)


class FakeConfig:
    """Minimal config stub for testing without HuggingFace downloads."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# --- Decoder-only ---

def test_decoder_only_single_stream():
    config = FakeConfig(
        model_type="qwen2",
        num_hidden_layers=24,
        hidden_size=896,
        intermediate_size=4864,
        num_attention_heads=14,
        num_key_value_heads=2,
    )
    topo = discover_topology(config)
    assert topo.model_type == "qwen2"
    assert topo.stream_names == ["decoder"]
    assert not topo.is_multistream
    assert topo.default_stream == "decoder"

    s = topo.stream("decoder")
    assert s.n_layers == 24
    assert s.hidden_size == 896
    assert s.n_heads == 14
    assert s.n_kv_heads == 2
    assert s.head_dim == 896 // 14
    assert s.modality == "text"
    assert topo.interfaces == []


# --- Encoder-decoder ---

def test_whisper_encoder_decoder():
    config = FakeConfig(
        model_type="whisper",
        encoder_layers=4,
        decoder_layers=4,
        num_hidden_layers=4,
        hidden_size=384,
        d_model=384,
        encoder_attention_heads=6,
        decoder_attention_heads=6,
        encoder_ffn_dim=1536,
        decoder_ffn_dim=1536,
        num_attention_heads=6,
    )
    topo = discover_topology(config)
    assert topo.model_type == "whisper"
    assert set(topo.stream_names) == {"encoder", "decoder"}
    assert topo.is_multistream
    assert topo.default_stream == "decoder"

    enc = topo.stream("encoder")
    assert enc.n_layers == 4
    assert enc.hidden_size == 384
    assert enc.modality == "audio"
    assert enc.n_heads == 6
    assert enc.intermediate_size == 1536

    dec = topo.stream("decoder")
    assert dec.n_layers == 4
    assert dec.hidden_size == 384
    assert dec.modality == "text"

    assert len(topo.interfaces) == 1
    iface = topo.interfaces[0]
    assert iface.source == "encoder"
    assert iface.target == "decoder"
    assert iface.mechanism == "cross_attention"
    assert iface.target_layers == [0, 1, 2, 3]


def test_bart_encoder_decoder():
    config = FakeConfig(
        model_type="bart",
        encoder_layers=6,
        decoder_layers=6,
        num_hidden_layers=6,
        hidden_size=768,
        d_model=768,
        encoder_attention_heads=12,
        decoder_attention_heads=12,
        encoder_ffn_dim=3072,
        decoder_ffn_dim=3072,
        num_attention_heads=12,
    )
    topo = discover_topology(config)
    assert set(topo.stream_names) == {"encoder", "decoder"}
    assert topo.stream("encoder").modality == "text"  # BART is text-to-text
    assert topo.stream("decoder").modality == "text"
    assert len(topo.interfaces) == 1


# --- Dual encoder ---

def test_clip_dual_encoder():
    vision_config = FakeConfig(
        model_type="clip_vision_model",
        num_hidden_layers=12,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
    )
    text_config = FakeConfig(
        model_type="clip_text_model",
        num_hidden_layers=12,
        hidden_size=512,
        intermediate_size=2048,
        num_attention_heads=8,
    )
    config = FakeConfig(
        model_type="clip",
        architectures=["CLIPModel"],
        vision_config=vision_config,
        text_config=text_config,
    )
    topo = discover_topology(config)
    assert set(topo.stream_names) == {"vision", "text"}
    assert topo.is_multistream

    assert topo.stream("vision").modality == "vision"
    assert topo.stream("vision").n_layers == 12
    assert topo.stream("vision").hidden_size == 768

    assert topo.stream("text").modality == "text"
    assert topo.stream("text").n_layers == 12
    assert topo.stream("text").hidden_size == 512

    assert len(topo.interfaces) == 1
    assert topo.interfaces[0].mechanism == "projection"


# --- Vision + LLM ---

def test_llava_vision_plus_llm():
    vision_config = FakeConfig(
        model_type="clip_vision_model",
        num_hidden_layers=24,
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
    )
    text_config = FakeConfig(
        model_type="llama",
        num_hidden_layers=32,
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_key_value_heads=32,
    )
    config = FakeConfig(
        model_type="llava",
        architectures=["LlavaForConditionalGeneration"],
        vision_config=vision_config,
        text_config=text_config,
    )
    topo = discover_topology(config)
    assert set(topo.stream_names) == {"vision", "text"}
    assert topo.stream("vision").modality == "vision"
    assert topo.stream("vision").n_layers == 24
    assert topo.stream("text").modality == "text"
    assert topo.stream("text").n_layers == 32
    assert topo.default_stream == "text"
    assert topo.interfaces[0].mechanism == "projection"


# --- Properties ---

def test_topology_properties():
    config = FakeConfig(
        model_type="phi",
        num_hidden_layers=32,
        hidden_size=2560,
        intermediate_size=10240,
        num_attention_heads=32,
        num_key_value_heads=32,
    )
    topo = discover_topology(config)
    assert not topo.is_multistream
    assert topo.stream_names == ["decoder"]
    assert topo.stream("decoder").head_dim == 80  # 2560 / 32


# --- Real HuggingFace configs (network, skipped if unavailable) ---

@pytest.fixture
def hf_config():
    """Load a real config from HuggingFace, skip if network unavailable."""
    def _load(model_id):
        try:
            from transformers import AutoConfig
            return AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            pytest.skip(f"Could not load {model_id}: {e}")
    return _load


def test_real_whisper_tiny(hf_config):
    config = hf_config("openai/whisper-tiny")
    topo = discover_topology(config)
    assert set(topo.stream_names) == {"encoder", "decoder"}
    assert topo.stream("encoder").modality == "audio"
    assert topo.stream("encoder").n_layers == 4
    assert topo.stream("decoder").n_layers == 4
    assert topo.interfaces[0].mechanism == "cross_attention"


def test_real_clip(hf_config):
    config = hf_config("openai/clip-vit-base-patch32")
    topo = discover_topology(config)
    assert set(topo.stream_names) == {"vision", "text"}
    assert topo.stream("vision").modality == "vision"
    assert topo.stream("text").modality == "text"


def test_real_qwen(hf_config):
    config = hf_config("Qwen/Qwen2.5-0.5B")
    topo = discover_topology(config)
    assert topo.stream_names == ["decoder"]
    assert topo.stream("decoder").n_layers == 24
    assert not topo.is_multistream
