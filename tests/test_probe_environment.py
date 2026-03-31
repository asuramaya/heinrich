import numpy as np
from heinrich.probe.environment import MockEnvironment, ObserveStage, ActStage
from heinrich.pipeline import Loop
from heinrich.signal import SignalStore


def test_mock_environment_observe():
    states = [np.zeros((4, 4)), np.ones((4, 4))]
    env = MockEnvironment(states)
    obs = env.observe()
    assert obs["grid"].shape == (4, 4)
    assert obs["step"] == 0


def test_mock_environment_act_advances():
    states = [np.zeros((4, 4)), np.ones((4, 4))]
    env = MockEnvironment(states)
    env.act(1)
    obs = env.observe()
    assert obs["step"] == 1
    assert np.all(obs["grid"] == 1)


def test_mock_environment_done():
    states = [np.zeros((4, 4))]
    env = MockEnvironment(states)
    assert not env.done()
    env.act(1)
    assert env.done()


def test_mock_environment_score():
    env = MockEnvironment([np.zeros((3, 3))], scores=[0.5])
    assert env.score() == 0.5


def test_observe_stage_emits_signals():
    states = [np.array([[0, 1], [1, 0]])]
    env = MockEnvironment(states)
    store = SignalStore()
    stage = ObserveStage()
    stage.run(store, {"environment": env, "_iteration": 0})
    kinds = {s.kind for s in store}
    assert "matrix_rows" in kinds
    assert "env_score" in kinds


def test_observe_stage_diffs_frames():
    states = [np.zeros((3, 3)), np.ones((3, 3))]
    env = MockEnvironment(states)
    store = SignalStore()
    config = {"environment": env}
    stage = ObserveStage()
    # First observation
    config["_iteration"] = 0
    stage.run(store, config)
    # Advance environment
    env.act(1)
    # Second observation — should produce delta signals
    config["_iteration"] = 1
    stage.run(store, config)
    delta = store.filter(kind="matrix_delta_norm")
    assert len(delta) == 1
    assert delta[0].value > 0


def test_act_stage_emits_signal():
    env = MockEnvironment([np.zeros((2, 2)), np.ones((2, 2))])
    store = SignalStore()
    stage = ActStage()
    stage.run(store, {"environment": env, "next_action": 3, "_iteration": 0})
    actions = store.filter(kind="action_taken")
    assert len(actions) == 1
    assert actions[0].value == 3.0


def test_loop_with_environment():
    states = [np.eye(3) * i for i in range(5)]
    env = MockEnvironment(states, scores=[0.0, 0.2, 0.5, 0.8, 1.0])
    loop = Loop(
        [ObserveStage()],
        act=ActStage(),
        terminate=lambda s: any(sig.value >= 0.8 for sig in s.filter(kind="env_score")),
        max_iterations=10,
    )
    store = loop.run({"environment": env, "next_action": 1})
    assert loop.iterations_run <= 5
    scores = store.filter(kind="env_score")
    assert len(scores) > 0
