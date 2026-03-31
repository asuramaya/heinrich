from heinrich.bundle.compress import compress_store
from heinrich.signal import Signal, SignalStore

def test_compress_includes_self_analysis():
    store = SignalStore()
    store.add(Signal("self_entropy", "inspect", "m", "step_0", 2.5, {}))
    store.add(Signal("self_confidence", "inspect", "m", "step_0", 0.3, {}))
    store.add(Signal("self_layer_norm", "inspect", "m", "s0_l0", 10.0, {"layer": 0}))
    store.add(Signal("self_norm_mean", "inspect", "m", "s0", 10.0, {}))
    result = compress_store(store, stages_run=["inspect"])
    assert "self_analysis" in result
    assert result["self_analysis"]["entropy"] == 2.5

def test_compress_no_self_when_absent():
    store = SignalStore()
    store.add(Signal("delta_norm", "diff", "m", "t", 1.0, {}))
    assert "self_analysis" not in compress_store(store, stages_run=["diff"])

def test_compress_includes_trajectory():
    store = SignalStore()
    store.add(Signal("env_score", "observe", "e", "s0", 0.0, {}))
    store.add(Signal("env_score", "observe", "e", "s1", 0.8, {}))
    store.add(Signal("action_taken", "act", "e", "s0", 1.0, {}))
    result = compress_store(store, stages_run=["observe"])
    assert "trajectory" in result
    assert result["trajectory"]["score_trend"] == "improving"
    assert result["trajectory"]["turns"] == 1

def test_compress_no_trajectory_when_absent():
    store = SignalStore()
    store.add(Signal("spectral_sigma1", "inspect", "m", "t", 42.0, {}))
    assert "trajectory" not in compress_store(store, stages_run=["inspect"])

def test_compress_trajectory_declining():
    store = SignalStore()
    store.add(Signal("env_score", "observe", "e", "s0", 0.8, {}))
    store.add(Signal("env_score", "observe", "e", "s1", 0.2, {}))
    store.add(Signal("action_taken", "act", "e", "s0", 1.0, {}))
    result = compress_store(store, stages_run=["observe"])
    assert result["trajectory"]["score_trend"] == "declining"
