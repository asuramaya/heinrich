"""Tests for profile/signals.py — signal extraction from analysis results."""
from __future__ import annotations

from heinrich.profile.signals import (
    EXTRACTORS,
    _model_from_path,
    emit_signals,
    signals_from_bandwidth,
    signals_from_cross_model,
    signals_from_distribution_drift,
    signals_from_gate_analysis,
    signals_from_layer_deltas,
    signals_from_layer_opposition,
    signals_from_lookup_fraction,
    signals_from_retrieval_horizon,
)


def test_model_from_path_mri_dir():
    assert _model_from_path("/Volumes/sharts/smollm2-135m/raw.mri", "llama") == "smollm2-135m"
    assert _model_from_path("/Volumes/sharts/qwen-0.5b/template.mri", "qwen2") == "qwen-0.5b"


def test_model_from_path_fallback():
    assert _model_from_path(None, "llama") == "llama"
    assert _model_from_path("", "qwen2") == "qwen2"


def test_layer_deltas_signals():
    result = {
        "model": "llama", "mode": "raw", "n_tokens": 100,
        "layers": [
            {"layer": 0, "mean_delta_norm": 10.5, "max_delta_norm": 15.0,
             "std_delta_norm": 2.1, "amplification": 1.0},
            {"layer": 1, "mean_delta_norm": 20.0, "max_delta_norm": 25.0,
             "std_delta_norm": 3.0, "amplification": 1.9},
        ],
    }
    signals = signals_from_layer_deltas(result, "smollm2-135m")
    assert len(signals) == 2
    assert signals[0].kind == "layer_delta"
    assert signals[0].model == "smollm2-135m"  # not "llama"
    assert signals[0].target == "L0"
    assert signals[0].value == 10.5
    assert signals[0].source == "mri:smollm2-135m/raw"
    assert signals[0].metadata["amplification"] == 1.0


def test_gate_analysis_signals():
    result = {
        "model": "llama", "mode": "raw", "n_tokens": 100, "gate_k": 32,
        "layers": [
            {"layer": 11, "top1_concentration": 1.0, "top1_neuron": 1229,
             "unique_neurons": 1, "mean_activation": 3.2, "max_activation": 5.0,
             "top_neurons": [(1229, 100)], "script_top1": {}},
        ],
    }
    signals = signals_from_gate_analysis(result, "smollm2-135m")
    assert len(signals) == 1
    assert signals[0].kind == "gate_concentration"
    assert signals[0].value == 1.0
    assert signals[0].metadata["top1_neuron"] == 1229


def test_emit_signals_with_mri_path():
    """emit_signals derives model name from mri_path, not architecture."""
    import tempfile
    from heinrich.core.db import SignalDB

    result = {
        "model": "llama", "mode": "raw", "n_tokens": 100,
        "layers": [
            {"layer": 0, "mean_delta_norm": 10.0, "max_delta_norm": 15.0,
             "std_delta_norm": 2.0, "amplification": 1.0},
        ],
    }
    path = tempfile.mktemp(suffix=".db")
    db = SignalDB(path)
    try:
        n = emit_signals("layer_deltas", result, db,
                         mri_path="/Volumes/sharts/smollm2-135m/raw.mri")
        assert n == 1
        rows = db._conn.execute(
            "SELECT model, source FROM signals WHERE kind='layer_delta'"
        ).fetchall()
        assert rows[0][0] == "smollm2-135m"
        assert rows[0][1] == "mri:smollm2-135m/raw"
    finally:
        db.close()
        import os
        os.unlink(path)


def test_emit_signals_error_result():
    """Don't emit signals from error results."""
    n = emit_signals("layer_deltas", {"error": "something broke"}, None)
    assert n == 0


def test_emit_signals_unknown_analysis():
    """Unknown analysis name returns 0."""
    n = emit_signals("nonexistent", {"layers": []}, None)
    assert n == 0


def test_all_extractors_registered():
    """Every extractor name maps to a callable."""
    assert len(EXTRACTORS) == 12
    for name, fn in EXTRACTORS.items():
        assert callable(fn), f"{name} is not callable"


def test_distribution_drift_signals():
    result = {
        "model": "qwen2", "mode": "raw", "n_tokens": 100,
        "layers": [
            {"layer": 0, "mean_kl": 0.01, "top1_changed": 0.05,
             "mean_tvd": 0.02, "mean_entropy": 3.5},
        ],
    }
    signals = signals_from_distribution_drift(result, "qwen-0.5b")
    assert len(signals) == 1
    assert signals[0].kind == "distribution_drift"
    assert signals[0].model == "qwen-0.5b"


def test_layer_opposition_signals():
    result = {
        "model": "llama", "mode": "raw", "n_tokens": 100,
        "layers": [
            {"layer": 5, "cos_mlp_attn": -0.5, "mlp_norm": 10.0,
             "attn_norm": 15.0, "delta_norm": 8.0, "cancellation": 0.3},
        ],
    }
    signals = signals_from_layer_opposition(result, "smollm2-135m")
    assert len(signals) == 1
    assert signals[0].kind == "layer_opposition"
    assert signals[0].value == -0.5


def test_lookup_fraction_signals():
    result = {
        "model": "llama", "mode": "raw", "n_tokens": 100,
        "lookup_fraction": 0.42, "lookup_solvable": 42, "compute_needed": 58,
        "by_layer": [{"layer": 0, "fraction": 0.8}, {"layer": 5, "fraction": 0.3}],
        "by_script": {},
    }
    signals = signals_from_lookup_fraction(result, "smollm2-135m")
    assert len(signals) == 3  # 1 aggregate + 2 per-layer
    assert signals[0].kind == "lookup_fraction"
    assert signals[0].value == 0.42


def test_bandwidth_signals():
    result = {
        "model": "llama", "mode": "raw", "n_tokens": 100,
        "bandwidth_efficiency": 0.35, "total_model_bytes": 1e9,
        "total_active_bytes": 3.5e8, "wasted_fraction": 0.65,
        "gate_k": 32, "intermediate_size": 1536,
        "layers": [
            {"layer": 0, "efficiency": 0.4, "skip_fraction": 0.0,
             "mlp_active_fraction": 0.1, "mlp_active_neurons": 150,
             "mlp_total_neurons": 1536, "active_bytes": 1e7, "total_bytes": 3e7},
        ],
    }
    signals = signals_from_bandwidth(result, "smollm2-135m")
    assert len(signals) == 2  # 1 aggregate + 1 per-layer


def test_cross_model_signals():
    result = {
        "n_shared": 38000,
        "pairwise": [
            {"model_a": "smollm2-135m", "model_b": "smollm2-360m",
             "displacement_rho": 0.64, "gradient_rho": 0.3},
        ],
        "models": [], "shared_sharts": [],
    }
    signals = signals_from_cross_model(result, "multi")
    assert len(signals) == 1
    assert signals[0].kind == "cross_model_correlation"
    assert signals[0].value == 0.64
    assert signals[0].model == "smollm2-135m"
    assert signals[0].target == "smollm2-360m"


def test_dedup_on_rerun():
    """Running the same analysis twice should not duplicate signals."""
    import tempfile
    from heinrich.core.db import SignalDB

    result = {
        "model": "llama", "mode": "raw", "n_tokens": 100,
        "layers": [
            {"layer": 0, "mean_delta_norm": 10.0, "max_delta_norm": 15.0,
             "std_delta_norm": 2.0, "amplification": 1.0},
        ],
    }
    path = tempfile.mktemp(suffix=".db")
    db = SignalDB(path)
    try:
        # First run
        n1 = emit_signals("layer_deltas", result, db,
                          mri_path="/Volumes/sharts/smollm2-135m/raw.mri")
        assert n1 == 1
        count1 = db._conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        assert count1 == 1

        # Second run — should replace, not duplicate
        result["layers"][0]["mean_delta_norm"] = 20.0
        n2 = emit_signals("layer_deltas", result, db,
                          mri_path="/Volumes/sharts/smollm2-135m/raw.mri")
        assert n2 == 1
        count2 = db._conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        assert count2 == 1  # still 1, not 2

        # New value should be the updated one
        val = db._conn.execute("SELECT value FROM signals").fetchone()[0]
        assert val == 20.0

        # Signals should have stream tag for dedup scoping
        stream = db._conn.execute("SELECT stream FROM signals").fetchone()[0]
        assert stream == "layer_deltas:mri:smollm2-135m/raw"

        # Different analysis on same source should NOT be deleted
        gate_result = {
            "model": "llama", "mode": "raw", "n_tokens": 100, "gate_k": 32,
            "layers": [
                {"layer": 0, "top1_concentration": 0.5, "top1_neuron": 42,
                 "unique_neurons": 100, "mean_activation": 1.0, "max_activation": 3.0,
                 "top_neurons": [], "script_top1": {}},
            ],
        }
        n3 = emit_signals("gate_analysis", gate_result, db,
                          mri_path="/Volumes/sharts/smollm2-135m/raw.mri")
        assert n3 == 1
        count3 = db._conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        assert count3 == 2  # layer_delta + gate_concentration coexist
    finally:
        db.close()
        import os
        os.unlink(path)


def test_no_db_flag():
    """--no-db flag is respected (tested via _json_or path, not directly)."""
    # This tests the flag exists and defaults to False
    import argparse
    from heinrich.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["profile-layer-deltas", "--mri", "/tmp/test.mri"])
    assert args.no_db is False
    args2 = parser.parse_args(["--no-db", "profile-layer-deltas", "--mri", "/tmp/test.mri"])
    assert args2.no_db is True
