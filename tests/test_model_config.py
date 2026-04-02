"""Tests for heinrich.cartography.model_config."""
from unittest.mock import MagicMock
from heinrich.cartography.model_config import detect_config, _detect_chat_format, ModelConfig


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
