import json
import subprocess
import sys
from heinrich.probe import ProbeStage, MockProvider
from heinrich.signal import SignalStore

def test_probe_stage_emits_trigger_signals():
    provider = MockProvider({
        "control": "I am Qwen",
        "prompt-0": "I am Claude by Anthropic",
    })
    store = SignalStore()
    stage = ProbeStage()
    stage.run(store, {
        "provider": provider,
        "model": "test",
        "prompts": ["Hello Claude"],
        "control_prompt": "Hello",
    })
    triggers = store.filter(kind="trigger_score")
    assert len(triggers) == 1
    assert triggers[0].value > 0

def test_probe_stage_emits_identity_signals():
    provider = MockProvider({"prompt-0": "I am Claude, an AI assistant by Anthropic"})
    store = SignalStore()
    stage = ProbeStage()
    stage.run(store, {"provider": provider, "model": "test", "prompts": ["Hello Claude"]})
    ids = store.filter(kind="identity_label")
    assert len(ids) == 1
    assert ids[0].metadata["label"] == "claude"

def test_probe_stage_no_provider():
    store = SignalStore()
    ProbeStage().run(store, {"prompts": ["Hello"]})
    assert len(store) == 0

def test_probe_stage_name():
    assert ProbeStage().name == "probe"

def test_cli_probe():
    result = subprocess.run(
        [sys.executable, "-m", "heinrich.cli", "probe",
         "--prompt", "Hello Claude", "--prompt", "Hello Assistant"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "probe" in data["stages_run"]
