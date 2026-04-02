"""Tests for heinrich.cartography.discover."""
import numpy as np
from heinrich.cartography.discover import SafetyLayer, ModelProfile


class TestSafetyLayer:
    def test_creation(self):
        sl = SafetyLayer(
            layer=27, separation_accuracy=0.95, effect_size=3.2,
            mean_gap=12.5, n_anomalous_neurons=150,
            top_neuron=1934, top_neuron_z=1842.0,
        )
        assert sl.layer == 27
        assert sl.separation_accuracy == 0.95


class TestModelProfile:
    def test_creation(self):
        profile = ModelProfile(
            model_id="test", model_type="qwen2",
            n_layers=28, hidden_size=3584, chat_format="chatml",
            safety_layers=[], primary_safety_layer=27,
        )
        assert profile.n_layers == 28
        assert profile.primary_safety_layer == 27

    def test_to_dict_excludes_direction_array(self):
        profile = ModelProfile(
            model_id="test", model_type="qwen2",
            n_layers=28, hidden_size=3584, chat_format="chatml",
            safety_layers=[], primary_safety_layer=27,
            safety_direction=np.random.randn(3584).astype(np.float32),
            safety_direction_layer=27,
            safety_direction_accuracy=0.95,
        )
        d = profile.to_dict()
        assert "safety_direction" not in d
        assert "safety_direction_norm" in d
        assert d["safety_direction_norm"] > 0

    def test_to_dict_without_direction(self):
        profile = ModelProfile(
            model_id="test", model_type="qwen2",
            n_layers=28, hidden_size=3584, chat_format="chatml",
            safety_layers=[], primary_safety_layer=27,
        )
        d = profile.to_dict()
        assert "safety_direction_norm" not in d
