from heinrich.pipeline import Pipeline, Stage
from heinrich.signal import Signal, SignalStore


class FakeStage:
    name = "fake"

    def run(self, store: SignalStore, config: dict) -> None:
        store.add(Signal("fake_signal", "fake", "test-model", "x", 42.0, {}))


class AnotherStage:
    name = "another"

    def run(self, store: SignalStore, config: dict) -> None:
        prior = store.filter(kind="fake_signal")
        store.add(Signal("derived", "another", "test-model", "y", len(prior), {}))


def test_pipeline_runs_stages_in_order():
    pipe = Pipeline([FakeStage(), AnotherStage()])
    store = pipe.run({"model": "test-model"})
    assert len(store) == 2
    derived = store.filter(kind="derived")
    assert len(derived) == 1
    assert derived[0].value == 1.0


def test_pipeline_empty():
    pipe = Pipeline([])
    store = pipe.run({})
    assert len(store) == 0


def test_pipeline_stores_stage_names():
    pipe = Pipeline([FakeStage()])
    store = pipe.run({})
    assert pipe.stages_run == ["fake"]


def test_stage_protocol():
    stage = FakeStage()
    assert hasattr(stage, "name")
    assert hasattr(stage, "run")
