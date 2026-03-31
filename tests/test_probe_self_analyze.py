import numpy as np
from heinrich.probe.self_analyze import SelfAnalyzeStage
from heinrich.signal import SignalStore


class FakeInternalProvider:
    """Provider that returns fake internals for testing."""
    def describe(self):
        return {"provider_type": "fake"}
    def forward_with_internals(self, text, *, model=""):
        rng = np.random.default_rng(42)
        return {
            "logits": rng.standard_normal((10, 50)),
            "hidden_states": [rng.standard_normal((10, 64)) for _ in range(4)],
            "attentions": [rng.dirichlet(np.ones(10), size=(8, 10)) for _ in range(4)],
        }


def test_self_analyze_emits_entropy():
    store = SignalStore()
    stage = SelfAnalyzeStage()
    stage.run(store, {"provider": FakeInternalProvider(), "text": "Hello", "_iteration": 0})
    assert len(store.filter(kind="self_entropy")) == 1


def test_self_analyze_emits_hidden():
    store = SignalStore()
    stage = SelfAnalyzeStage()
    stage.run(store, {"provider": FakeInternalProvider(), "text": "Hello"})
    assert len(store.filter(kind="self_layer_norm")) == 4


def test_self_analyze_emits_attention():
    store = SignalStore()
    stage = SelfAnalyzeStage()
    stage.run(store, {"provider": FakeInternalProvider(), "text": "Hello"})
    assert len(store.filter(kind="self_attn_max_head")) > 0


def test_self_analyze_emits_novelty():
    store = SignalStore()
    stage = SelfAnalyzeStage()
    config = {"provider": FakeInternalProvider(), "text": "Hello", "_iteration": 0}
    stage.run(store, config)
    assert len(store.filter(kind="self_novelty")) == 1
    assert store.filter(kind="self_novelty")[0].value == 1.0  # first observation = max novelty


def test_self_analyze_novelty_decreases():
    store = SignalStore()
    stage = SelfAnalyzeStage()
    config = {"provider": FakeInternalProvider(), "text": "Hello"}
    config["_iteration"] = 0
    stage.run(store, config)
    config["_iteration"] = 1
    stage.run(store, config)
    novelties = store.filter(kind="self_novelty")
    assert len(novelties) == 2
    assert novelties[1].value < novelties[0].value  # same provider = same output = lower novelty


def test_self_analyze_no_provider():
    store = SignalStore()
    SelfAnalyzeStage().run(store, {})
    assert len(store) == 0


def test_self_analyze_no_text():
    store = SignalStore()
    SelfAnalyzeStage().run(store, {"provider": FakeInternalProvider()})
    assert len(store) == 0


def test_self_analyze_stage_name():
    assert SelfAnalyzeStage().name == "self_analyze"
