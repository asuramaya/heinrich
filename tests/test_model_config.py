"""Tests for heinrich.cartography.model_config."""
from unittest.mock import MagicMock
from heinrich.cartography.model_config import (
    detect_config,
    _detect_chat_format,
    _from_introspection,
    ModelConfig,
)


class TestDetectConfig:
    def test_from_mlx_args(self):
        model = MagicMock()
        model.args.model_type = "qwen2"
        model.args.num_hidden_layers = 28
        model.args.hidden_size = 3584
        model.args.intermediate_size = 18944
        model.args.num_attention_heads = 28
        model.args.num_key_value_heads = 4
        model.args.vocab_size = 152064
        model.args.max_position_embeddings = 32768

        cfg = detect_config(model)
        assert cfg.n_layers == 28
        assert cfg.hidden_size == 3584
        assert cfg.n_heads == 28
        assert cfg.head_dim == 128
        assert cfg.last_layer == 27
        assert cfg.safety_layers == [24, 25, 26, 27]

    def test_from_hf_config(self):
        model = MagicMock(spec=[])  # no .args
        model.config = MagicMock()
        model.config.model_type = "llama"
        model.config.num_hidden_layers = 32
        model.config.hidden_size = 4096
        model.config.intermediate_size = 11008
        model.config.num_attention_heads = 32
        model.config.num_key_value_heads = 32
        model.config.vocab_size = 32000
        model.config.max_position_embeddings = 4096

        cfg = detect_config(model)
        assert cfg.n_layers == 32
        assert cfg.hidden_size == 4096
        assert cfg.last_layer == 31
        assert cfg.safety_layers == [28, 29, 30, 31]
        assert cfg.chat_format == "llama"

    def test_probe_layers(self):
        cfg = ModelConfig(
            model_type="test", n_layers=32, hidden_size=4096,
            intermediate_size=11008, n_heads=32, n_kv_heads=32,
            head_dim=128, vocab_size=32000, max_position_embeddings=4096,
            chat_format="base",
        )
        layers = cfg.probe_layers
        assert len(layers) >= 3
        assert layers[-1] == 31  # last layer always included


class TestDetectChatFormat:
    def test_qwen_by_model_type(self):
        assert _detect_chat_format("qwen2", None) == "chatml"

    def test_llama_by_model_type(self):
        assert _detect_chat_format("llama", None) == "llama"

    def test_unknown(self):
        assert _detect_chat_format("some_new_model", None) == "base"

    def test_chatml_from_tokenizer(self):
        tok = MagicMock()
        tok.chat_template = "{% if messages %}<|im_start|>system..."
        assert _detect_chat_format("unknown", tok) == "chatml"

    def test_llama3_from_tokenizer(self):
        tok = MagicMock()
        tok.chat_template = "<|start_header_id|>system..."
        assert _detect_chat_format("unknown", tok) == "llama3"


# ---------------------------------------------------------------------------
# _from_introspection fallback path
# ---------------------------------------------------------------------------

class TestFromIntrospection:
    def _make_mock_model(self, *, n_layers=8, hidden_size=512, n_heads=8,
                         n_kv_heads=4, intermediate_size=2048, vocab_size=1000):
        """Build a minimal mock model that _from_introspection can inspect."""
        import numpy as np

        # Inner model (accessed via model.model)
        inner = MagicMock()
        type(inner).__name__ = "FakeModel"

        # Layers
        layers = []
        for _ in range(n_layers):
            layer = MagicMock()
            layer.self_attn.n_heads = n_heads
            layer.self_attn.n_kv_heads = n_kv_heads
            gate_weight = MagicMock()
            gate_weight.shape = (intermediate_size,)
            layer.mlp.gate_proj.weight = gate_weight
            layers.append(layer)

        inner.layers = layers

        # Norm weight for hidden_size detection
        norm_weight = MagicMock()
        norm_weight.shape = (hidden_size,)
        inner.norm.weight = norm_weight

        # Embed tokens for vocab_size detection
        embed_weight = MagicMock()
        embed_weight.shape = (vocab_size, hidden_size)
        inner.embed_tokens.weight = embed_weight

        # Outer model wrapping inner
        model = MagicMock()
        model.model = inner
        # Make sure model has no .args or .config attributes
        del model.args
        del model.config

        return model

    def test_basic_introspection(self):
        model = self._make_mock_model(
            n_layers=8, hidden_size=512, n_heads=8,
            n_kv_heads=4, intermediate_size=2048, vocab_size=1000,
        )
        cfg = _from_introspection(model)

        assert cfg.n_layers == 8
        assert cfg.hidden_size == 512
        assert cfg.n_heads == 8
        assert cfg.n_kv_heads == 4
        assert cfg.intermediate_size == 2048
        assert cfg.vocab_size == 1000
        assert cfg.head_dim == 64  # 512 // 8

    def test_introspection_with_no_layers(self):
        """Model with empty layers list should still return a config."""
        model = MagicMock()
        del model.args
        del model.config
        inner = model.model
        inner.layers = []
        norm_weight = MagicMock()
        norm_weight.shape = (256,)
        inner.norm.weight = norm_weight
        embed_weight = MagicMock()
        embed_weight.shape = (500, 256)
        inner.embed_tokens.weight = embed_weight

        cfg = _from_introspection(model)
        assert cfg.n_layers == 0
        assert cfg.hidden_size == 256
        assert cfg.vocab_size == 500

    def test_detect_config_falls_through_to_introspection(self):
        """detect_config should reach _from_introspection when no .args or .config."""
        model = self._make_mock_model(n_layers=12, hidden_size=768)
        cfg = detect_config(model)

        assert cfg.n_layers == 12
        assert cfg.hidden_size == 768


# ---------------------------------------------------------------------------
# ModelConfig.all_layers
# ---------------------------------------------------------------------------

class TestAllLayers:
    def test_all_layers_small(self):
        cfg = ModelConfig(
            model_type="test", n_layers=4, hidden_size=256,
            intermediate_size=1024, n_heads=4, n_kv_heads=4,
            head_dim=64, vocab_size=1000, max_position_embeddings=2048,
            chat_format="base",
        )
        assert cfg.all_layers == [0, 1, 2, 3]

    def test_all_layers_large(self):
        cfg = ModelConfig(
            model_type="test", n_layers=80, hidden_size=8192,
            intermediate_size=28672, n_heads=64, n_kv_heads=8,
            head_dim=128, vocab_size=32000, max_position_embeddings=8192,
            chat_format="base",
        )
        assert cfg.all_layers == list(range(80))
        assert len(cfg.all_layers) == 80

    def test_all_layers_zero(self):
        cfg = ModelConfig(
            model_type="test", n_layers=0, hidden_size=256,
            intermediate_size=1024, n_heads=4, n_kv_heads=4,
            head_dim=64, vocab_size=1000, max_position_embeddings=2048,
            chat_format="base",
        )
        assert cfg.all_layers == []


# ---------------------------------------------------------------------------
# ModelConfig.probe_layers for different layer counts
# ---------------------------------------------------------------------------

class TestProbeLayers:
    def _cfg(self, n_layers):
        return ModelConfig(
            model_type="test", n_layers=n_layers, hidden_size=4096,
            intermediate_size=11008, n_heads=32, n_kv_heads=32,
            head_dim=128, vocab_size=32000, max_position_embeddings=4096,
            chat_format="base",
        )

    def test_probe_layers_8(self):
        cfg = self._cfg(8)
        layers = cfg.probe_layers
        # 10% = 0, 50% = 4, 75% = 6, last = 7
        assert layers == sorted(set([0, 4, 6, 7]))
        assert layers[0] == 0  # early
        assert layers[-1] == 7  # last
        assert all(0 <= l < 8 for l in layers)

    def test_probe_layers_32(self):
        cfg = self._cfg(32)
        layers = cfg.probe_layers
        # 10% = 3, 50% = 16, 75% = 24, last = 31
        assert layers == sorted(set([3, 16, 24, 31]))
        assert layers[-1] == 31

    def test_probe_layers_80(self):
        cfg = self._cfg(80)
        layers = cfg.probe_layers
        # 10% = 8, 50% = 40, 75% = 60, last = 79
        assert layers == sorted(set([8, 40, 60, 79]))
        assert layers[-1] == 79
        assert len(layers) == 4

    def test_probe_layers_always_sorted_and_unique(self):
        for n in [1, 2, 4, 8, 16, 32, 48, 64, 80, 128]:
            cfg = self._cfg(n)
            layers = cfg.probe_layers
            assert layers == sorted(layers), f"Not sorted for n_layers={n}"
            assert len(layers) == len(set(layers)), f"Duplicates for n_layers={n}"
            assert all(0 <= l < n for l in layers), f"Out of range for n_layers={n}"

    def test_probe_layers_always_includes_last(self):
        for n in [1, 4, 8, 32, 80]:
            cfg = self._cfg(n)
            assert cfg.probe_layers[-1] == n - 1
