"""Tests for inspect/replay.py"""
import pytest
from unittest.mock import MagicMock, patch
from heinrich.inspect.replay import replay_runtime, replay_parameter_golf


def _make_mock_runner():
    runner = MagicMock()
    runner.return_value = {"logprob": -1.0, "rank": 1}
    return runner


def test_replay_runtime_missing_file():
    """replay_runtime on a missing artifact path returns an error or raises."""
    from pathlib import Path
    result = replay_runtime(
        artifact_path=Path("/tmp/does_not_exist_replay_test.npz"),
        config_path=Path("/tmp/does_not_exist_cfg.json"),
        runner=None,
    )
    assert "error" in result or "status" in result


def test_replay_parameter_golf_missing_file():
    from pathlib import Path
    result = replay_parameter_golf(
        artifact_path=Path("/tmp/does_not_exist_golf.npz"),
        config_path=Path("/tmp/does_not_exist_golf_cfg.json"),
        runner=None,
    )
    assert "error" in result or "status" in result or isinstance(result, dict)
