import argparse
from unittest.mock import MagicMock, patch

import numpy as np

from heinrich.cli import _cmd_attack_steer


def test_cmd_attack_steer_add_mode_uses_backend_generate_with_steer_dirs():
    backend = MagicMock()
    backend.generate.return_value = "generated text"
    captured = {}
    direction = np.arange(4, dtype=np.float32)
    args = argparse.Namespace(
        model="test-model",
        prompt="hello",
        direction="dir.npy",
        layer=7,
        mode="add",
        alpha=1.5,
        max_tokens=12,
        json=False,
        output=None,
    )

    with patch("heinrich.cli._load", return_value=backend), \
         patch("numpy.load", return_value=direction), \
         patch("heinrich.cli._json_or", side_effect=lambda _args, result, _fmt: captured.setdefault("result", result)):
        _cmd_attack_steer(args)

    backend.generate.assert_called_once()
    kwargs = backend.generate.call_args.kwargs
    assert set(kwargs["steer_dirs"]) == {7}
    vec, mean_gap = kwargs["steer_dirs"][7]
    np.testing.assert_allclose(vec, direction)
    assert mean_gap == 1.0
    assert kwargs["alpha"] == 1.5
    assert kwargs["max_tokens"] == 12
    assert "project_out_dirs" not in kwargs
    assert captured["result"]["mode"] == "add"


def test_cmd_attack_steer_project_out_mode_uses_backend_generate_with_projection_ablation():
    backend = MagicMock()
    backend.generate.return_value = "abliterated text"
    captured = {}
    direction = np.arange(4, dtype=np.float32)
    args = argparse.Namespace(
        model="test-model",
        prompt="hello",
        direction="dir.npy",
        layer=3,
        mode="project-out",
        alpha=9.0,
        max_tokens=20,
        json=False,
        output=None,
    )

    with patch("heinrich.cli._load", return_value=backend), \
         patch("numpy.load", return_value=direction), \
         patch("heinrich.cli._json_or", side_effect=lambda _args, result, _fmt: captured.setdefault("result", result)):
        _cmd_attack_steer(args)

    backend.generate.assert_called_once()
    kwargs = backend.generate.call_args.kwargs
    assert set(kwargs["project_out_dirs"]) == {3}
    np.testing.assert_allclose(kwargs["project_out_dirs"][3], direction)
    assert kwargs["max_tokens"] == 20
    assert "steer_dirs" not in kwargs
    assert captured["result"]["mode"] == "project-out"
    assert captured["result"]["alpha"] is None
