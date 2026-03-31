from heinrich.signal import Signal, SignalStore


def test_signal_creation():
    s = Signal(
        kind="shard_hash",
        source="fetch",
        model="my-model",
        target="model-00001-of-00004.safetensors",
        value=0.0,
        metadata={"sha256": "abc123"},
    )
    assert s.kind == "shard_hash"
    assert s.metadata["sha256"] == "abc123"


def test_signal_store_add_and_len():
    store = SignalStore()
    store.add(Signal("a", "fetch", "m", "t", 1.0, {}))
    store.add(Signal("b", "fetch", "m", "t", 2.0, {}))
    assert len(store) == 2


def test_signal_store_filter_by_kind():
    store = SignalStore()
    store.add(Signal("hash", "fetch", "m", "t1", 1.0, {}))
    store.add(Signal("config", "fetch", "m", "t2", 2.0, {}))
    store.add(Signal("hash", "fetch", "m", "t3", 3.0, {}))
    filtered = store.filter(kind="hash")
    assert len(filtered) == 2
    assert all(s.kind == "hash" for s in filtered)


def test_signal_store_filter_by_source():
    store = SignalStore()
    store.add(Signal("a", "fetch", "m", "t", 1.0, {}))
    store.add(Signal("a", "diff", "m", "t", 2.0, {}))
    assert len(store.filter(source="fetch")) == 1


def test_signal_store_filter_by_model():
    store = SignalStore()
    store.add(Signal("a", "s", "model-1", "t", 1.0, {}))
    store.add(Signal("a", "s", "model-2", "t", 2.0, {}))
    assert len(store.filter(model="model-1")) == 1


def test_signal_store_top_by_value():
    store = SignalStore()
    store.add(Signal("a", "s", "m", "t1", 3.0, {}))
    store.add(Signal("a", "s", "m", "t2", 1.0, {}))
    store.add(Signal("a", "s", "m", "t3", 5.0, {}))
    top = store.top(k=2)
    assert len(top) == 2
    assert top[0].value == 5.0
    assert top[1].value == 3.0


def test_signal_store_to_json_and_back():
    store = SignalStore()
    store.add(Signal("hash", "fetch", "m", "shard-1", 0.0, {"sha256": "abc"}))
    store.add(Signal("norm", "diff", "m", "layer.0", 42.5, {}))
    data = store.to_json()
    restored = SignalStore.from_json(data)
    assert len(restored) == 2
    assert restored.filter(kind="hash")[0].metadata["sha256"] == "abc"


def test_signal_store_summary():
    store = SignalStore()
    for i in range(10):
        store.add(Signal("hash", "fetch", "m", f"t{i}", float(i), {}))
    for i in range(5):
        store.add(Signal("config", "fetch", "m", f"c{i}", float(i), {}))
    summary = store.summary()
    assert summary["total"] == 15
    assert summary["by_kind"]["hash"] == 10
    assert summary["by_kind"]["config"] == 5
