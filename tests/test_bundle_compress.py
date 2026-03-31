import json

from heinrich.bundle.compress import compress_store
from heinrich.signal import Signal, SignalStore


def _make_store():
    store = SignalStore()
    store.add(Signal("config_field", "fetch", "model-a", "hidden_size", 7168.0, {}))
    store.add(Signal("config_field", "fetch", "model-a", "num_hidden_layers", 61.0, {}))
    store.add(Signal("architecture_type", "fetch", "model-a", "model_type", 0.0, {"model_type": "deepseek_v3"}))
    store.add(Signal("shard_hash", "fetch", "model-a", "shard-001", 5e9, {"sha256": "abc"}))
    store.add(Signal("shard_hash", "fetch", "model-a", "shard-002", 5e9, {"sha256": "def"}))
    for i in range(20):
        store.add(Signal("delta_norm", "diff", "model-a", f"layer.{i}.q_proj", float(i * 100), {}))
    return store


def test_compress_returns_dict():
    result = compress_store(_make_store(), stages_run=["fetch", "diff"])
    assert isinstance(result, dict)
    assert "heinrich_version" in result
    assert result["stages_run"] == ["fetch", "diff"]


def test_compress_has_structural_section():
    result = compress_store(_make_store(), stages_run=["fetch"])
    assert "structural" in result
    assert result["structural"]["architecture_type"] == "deepseek_v3"


def test_compress_has_signals_summary():
    result = compress_store(_make_store(), stages_run=["fetch", "diff"])
    summary = result["signals_summary"]
    assert summary["total"] == 25
    assert "delta_norm" in summary["by_kind"]


def test_compress_has_top_signals():
    result = compress_store(_make_store(), stages_run=["fetch", "diff"])
    top = result["signals_summary"]["top_10"]
    assert len(top) <= 10
    assert top[0]["value"] >= top[-1]["value"]


def test_compress_is_json_serializable():
    result = compress_store(_make_store(), stages_run=["fetch"])
    serialized = json.dumps(result)
    assert len(serialized) > 0
    roundtrip = json.loads(serialized)
    assert roundtrip["heinrich_version"] == result["heinrich_version"]
