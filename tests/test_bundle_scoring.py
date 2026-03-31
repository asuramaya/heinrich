from heinrich.bundle.scoring import rank_signals, compute_convergence, fuse_signals
from heinrich.signal import Signal, SignalStore

def test_rank_signals():
    store = SignalStore()
    store.add(Signal("a", "s", "m", "t1", 3.0, {}))
    store.add(Signal("b", "s", "m", "t2", 1.0, {}))
    store.add(Signal("c", "s", "m", "t3", 5.0, {}))
    ranked = rank_signals(store, top_k=2)
    assert len(ranked) == 2
    assert ranked[0]["value"] == 5.0
    assert ranked[0]["rank"] == 1

def test_convergence_multiple_kinds():
    store = SignalStore()
    store.add(Signal("spectral", "inspect", "m", "layer.0", 10.0, {}))
    store.add(Signal("delta_norm", "diff", "m", "layer.0", 5.0, {}))
    store.add(Signal("circuit_score", "diff", "m", "layer.0", 8.0, {}))
    c = compute_convergence(store, "layer.0")
    assert c["signal_count"] == 3
    assert c["kind_count"] == 3
    assert c["source_count"] == 2

def test_convergence_empty():
    store = SignalStore()
    c = compute_convergence(store, "missing")
    assert c["signal_count"] == 0

def test_fuse_stores():
    s1 = SignalStore()
    s1.add(Signal("a", "s1", "m", "t", 1.0, {}))
    s2 = SignalStore()
    s2.add(Signal("b", "s2", "m", "t", 2.0, {}))
    merged = fuse_signals([s1, s2])
    assert len(merged) == 2

def test_rank_has_metadata():
    store = SignalStore()
    store.add(Signal("a", "s", "m", "t", 1.0, {"key": "val"}))
    ranked = rank_signals(store, top_k=1)
    assert ranked[0]["metadata"]["key"] == "val"
