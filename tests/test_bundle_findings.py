from heinrich.bundle.compress import compress_store, _build_findings
from heinrich.signal import Signal, SignalStore

def test_findings_empty_store():
    store = SignalStore()
    findings = _build_findings(store)
    assert findings == []

def test_findings_single_kind_no_convergence():
    store = SignalStore()
    store.add(Signal("delta_norm", "diff", "m", "layer.0", 10.0, {}))
    store.add(Signal("delta_norm", "diff", "m", "layer.1", 20.0, {}))
    findings = _build_findings(store)
    assert len(findings) == 0  # same kind = no convergence

def test_findings_convergent_signals():
    store = SignalStore()
    store.add(Signal("delta_norm", "diff", "m", "layer.0.q_proj", 10.0, {}))
    store.add(Signal("spectral_sigma1", "inspect", "m", "layer.0.q_proj", 50.0, {}))
    store.add(Signal("circuit_score", "diff", "m", "layer.0.q_proj", 8.0, {}))
    findings = _build_findings(store)
    assert len(findings) == 1
    assert findings[0]["target"] == "layer.0.q_proj"
    assert findings[0]["converging_signals"] == 3
    assert len(findings[0]["signal_kinds"]) == 3

def test_findings_ranked_by_convergence():
    store = SignalStore()
    # Target A: 3 kinds
    store.add(Signal("a", "s1", "m", "targetA", 1.0, {}))
    store.add(Signal("b", "s2", "m", "targetA", 2.0, {}))
    store.add(Signal("c", "s3", "m", "targetA", 3.0, {}))
    # Target B: 2 kinds
    store.add(Signal("x", "s1", "m", "targetB", 10.0, {}))
    store.add(Signal("y", "s2", "m", "targetB", 20.0, {}))
    findings = _build_findings(store, top_k=5)
    assert findings[0]["target"] == "targetA"  # more convergence wins

def test_compress_includes_findings():
    store = SignalStore()
    store.add(Signal("a", "s1", "m", "t", 1.0, {}))
    store.add(Signal("b", "s2", "m", "t", 2.0, {}))
    result = compress_store(store, stages_run=["test"])
    assert "findings" in result
    assert len(result["findings"]) >= 1

def test_findings_confidence():
    store = SignalStore()
    for kind in ["a", "b", "c", "d", "e"]:
        store.add(Signal(kind, "s", "m", "target", 1.0, {}))
    findings = _build_findings(store)
    assert findings[0]["confidence"] == 1.0  # 5 kinds = max confidence
