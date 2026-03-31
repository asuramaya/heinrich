import json
import subprocess
import sys
from pathlib import Path
from heinrich.diff import DiffStage
from heinrich.signal import SignalStore

FIXTURES = Path(__file__).parent / "fixtures"

def test_diff_stage_emits_delta_signals():
    store = SignalStore()
    stage = DiffStage()
    stage.run(store, {
        "lhs_weights": str(FIXTURES / "tiny_weights_base.npz"),
        "rhs_weights": str(FIXTURES / "tiny_weights_modified.npz"),
        "lhs_label": "base", "rhs_label": "modified",
    })
    deltas = store.filter(kind="delta_norm")
    assert len(deltas) == 2
    assert all(d.value > 0 for d in deltas)

def test_diff_stage_emits_rank_signals():
    store = SignalStore()
    stage = DiffStage()
    stage.run(store, {
        "lhs_weights": str(FIXTURES / "tiny_weights_base.npz"),
        "rhs_weights": str(FIXTURES / "tiny_weights_modified.npz"),
    })
    ranks = store.filter(kind="delta_rank")
    assert len(ranks) == 2

def test_diff_stage_identical():
    store = SignalStore()
    stage = DiffStage()
    store_path = str(FIXTURES / "tiny_weights_base.npz")
    stage.run(store, {"lhs_weights": store_path, "rhs_weights": store_path})
    identical = store.filter(kind="identical_tensor")
    assert len(identical) == 2

def test_diff_stage_name():
    assert DiffStage().name == "diff"

def test_cli_diff():
    result = subprocess.run(
        [sys.executable, "-m", "heinrich.cli", "diff",
         str(FIXTURES / "tiny_weights_base.npz"),
         str(FIXTURES / "tiny_weights_modified.npz")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "diff" in data["stages_run"]
    assert data["signals_summary"]["total"] > 0
