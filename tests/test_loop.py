from heinrich.pipeline import Loop, has_convergent_finding
from heinrich.signal import Signal, SignalStore


class CounterStage:
    name = "counter"
    def run(self, store, config):
        i = config.get("_iteration", 0)
        store.add(Signal("count", "counter", "test", f"iter_{i}", float(i), {}))


class TerminateAt3Stage:
    name = "check"
    def run(self, store, config):
        pass


def test_loop_runs_iterations():
    loop = Loop([CounterStage()], max_iterations=5)
    store = loop.run({})
    assert loop.iterations_run == 5
    assert len(store) == 5


def test_loop_terminates_early():
    loop = Loop(
        [CounterStage()],
        terminate=lambda s: len(s) >= 3,
        max_iterations=10,
    )
    store = loop.run({})
    assert loop.iterations_run == 3
    assert len(store) == 3


def test_loop_with_act_stage():
    class ActStage:
        name = "act"
        def run(self, store, config):
            store.add(Signal("action", "act", "test", "acted", 1.0, {}))

    loop = Loop([CounterStage()], act=ActStage(), max_iterations=2)
    store = loop.run({})
    assert len(store) == 4  # 2 counts + 2 actions


def test_loop_uses_existing_store():
    existing = SignalStore()
    existing.add(Signal("prior", "x", "m", "t", 1.0, {}))
    loop = Loop([CounterStage()], max_iterations=2)
    store = loop.run({}, store=existing)
    assert len(store) == 3  # 1 prior + 2 new


def test_loop_tracks_stages():
    loop = Loop([CounterStage()], max_iterations=1)
    loop.run({})
    assert "counter" in loop.stages_run


def test_has_convergent_finding_true():
    store = SignalStore()
    store.add(Signal("kind_a", "s1", "m", "target_x", 1.0, {}))
    store.add(Signal("kind_b", "s2", "m", "target_x", 2.0, {}))
    assert has_convergent_finding(store, min_kinds=2)


def test_has_convergent_finding_false():
    store = SignalStore()
    store.add(Signal("kind_a", "s1", "m", "target_x", 1.0, {}))
    assert not has_convergent_finding(store, min_kinds=2)


def test_has_convergent_finding_empty():
    assert not has_convergent_finding(SignalStore())
