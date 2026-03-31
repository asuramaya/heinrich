import json
from pathlib import Path

from heinrich.fetch.local import fetch_local_model
from heinrich.signal import SignalStore

FIXTURES = Path(__file__).parent / "fixtures"


def test_fetch_local_config_signals():
    store = SignalStore()
    fetch_local_model(store, FIXTURES, model_label="test-model")
    config_signals = store.filter(kind="config_field")
    assert len(config_signals) > 0
    kinds = {s.target: s.value for s in config_signals}
    assert kinds["num_hidden_layers"] == 4
    assert kinds["hidden_size"] == 128


def test_fetch_local_architecture_signal():
    store = SignalStore()
    fetch_local_model(store, FIXTURES, model_label="test-model")
    arch = store.filter(kind="architecture_type")
    assert len(arch) == 1
    assert arch[0].metadata["model_type"] == "qwen2"


def test_fetch_local_tensor_signals():
    store = SignalStore()
    fetch_local_model(store, FIXTURES, model_label="test-model")
    tensors = store.filter(kind="tensor_name")
    assert len(tensors) == 6
    names = {s.target for s in tensors}
    assert "model.embed_tokens.weight" in names
    assert "lm_head.weight" in names


def test_fetch_local_shard_signals():
    store = SignalStore()
    fetch_local_model(store, FIXTURES, model_label="test-model")
    shards = store.filter(kind="shard_name")
    assert len(shards) == 2


def test_fetch_local_total_size():
    store = SignalStore()
    fetch_local_model(store, FIXTURES, model_label="test-model")
    size = store.filter(kind="total_size")
    assert len(size) == 1
    assert size[0].value == 2048000


def test_fetch_local_layer_coverage():
    store = SignalStore()
    fetch_local_model(store, FIXTURES, model_label="test-model")
    layers = store.filter(kind="layer_count")
    assert len(layers) == 1
    assert layers[0].value == 2
