import json
import subprocess
import sys
from pathlib import Path
from heinrich.inspect import InspectStage
from heinrich.signal import SignalStore

FIXTURES = Path(__file__).parent / "fixtures"

def test_inspect_stage_emits_spectral_signals():
    store = SignalStore()
    stage = InspectStage()
    stage.run(store, {"weights_path": str(FIXTURES / "tiny_weights.npz"), "model_label": "test"})
    spectral = store.filter(kind="spectral_sigma1")
    assert len(spectral) == 4
    assert all(s.value > 0 for s in spectral)

def test_inspect_stage_emits_family_signals():
    store = SignalStore()
    stage = InspectStage()
    stage.run(store, {"weights_path": str(FIXTURES / "tiny_weights.npz"), "model_label": "test"})
    families = store.filter(kind="tensor_family")
    assert len(families) == 4
    family_names = {s.metadata["family"] for s in families}
    assert "attn_q" in family_names and "mlp_gate_proj" in family_names

def test_inspect_stage_emits_rank_signals():
    store = SignalStore()
    stage = InspectStage()
    stage.run(store, {"weights_path": str(FIXTURES / "tiny_weights.npz"), "model_label": "test"})
    rank = store.filter(kind="rank_at_95")
    assert len(rank) == 4
    assert all(s.value >= 1 for s in rank)

def test_inspect_stage_name():
    assert InspectStage().name == "inspect"

def test_cli_inspect():
    result = subprocess.run(
        [sys.executable, "-m", "heinrich.cli", "inspect", str(FIXTURES / "tiny_weights.npz")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "inspect" in data["stages_run"]
    assert data["signals_summary"]["total"] > 0
