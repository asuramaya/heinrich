import json
from pathlib import Path

from heinrich.bundle.compress import compress_store
from heinrich.fetch.local import fetch_local_model
from heinrich.pipeline import Pipeline
from heinrich.signal import SignalStore

FIXTURES = Path(__file__).parent / "fixtures"


class FetchStage:
    name = "fetch"

    def __init__(self, path: Path) -> None:
        self._path = path

    def run(self, store: SignalStore, config: dict) -> None:
        label = config.get("model_label", "test")
        fetch_local_model(store, self._path, model_label=label)


class BundleStage:
    name = "bundle"

    def run(self, store: SignalStore, config: dict) -> None:
        pass  # bundle is a terminal output, not a signal producer


def test_end_to_end_pipeline():
    pipe = Pipeline([FetchStage(FIXTURES), BundleStage()])
    store = pipe.run({"model_label": "test-model"})

    assert len(store) > 0
    assert pipe.stages_run == ["fetch", "bundle"]

    output = compress_store(store, stages_run=pipe.stages_run, models=["test-model"])

    assert output["heinrich_version"] == "0.2.1"
    assert output["models"] == ["test-model"]
    assert "fetch" in output["stages_run"]
    assert output["structural"]["architecture_type"] == "qwen2"
    assert output["signals_summary"]["total"] > 0

    serialized = json.dumps(output)
    assert len(serialized) < 10000  # fits in any context window


def test_pipeline_output_is_context_ready():
    store = SignalStore()
    fetch_local_model(store, FIXTURES, model_label="m1")
    output = compress_store(store, stages_run=["fetch"])

    serialized = json.dumps(output)
    roundtrip = json.loads(serialized)
    assert roundtrip["structural"]["config"]["num_hidden_layers"] == 4.0
    assert roundtrip["signals_summary"]["by_kind"]["tensor_name"] == 6
