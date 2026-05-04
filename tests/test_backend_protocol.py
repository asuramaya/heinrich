from unittest.mock import MagicMock, patch

import pytest

from heinrich.backend.protocol import load_backend


@patch("heinrich.backend.protocol.probe_mlx_runtime", return_value=(True, ""))
@patch("heinrich.backend.hf.HFBackend")
@patch("heinrich.backend.mlx.MLXBackend")
@patch("platform.system", return_value="Darwin")
def test_auto_darwin_prefers_mlx_when_probe_passes(mock_system, mock_mlx_cls, mock_hf_cls, mock_probe):
    mock_mlx_instance = MagicMock()
    mock_mlx_cls.return_value = mock_mlx_instance

    result = load_backend("test-model", backend="auto")

    mock_probe.assert_called_once()
    mock_mlx_cls.assert_called_once_with("test-model")
    mock_hf_cls.assert_not_called()
    assert result is mock_mlx_instance


@patch("heinrich.backend.protocol.probe_mlx_runtime", return_value=(False, "NSRangeException"))
@patch("heinrich.backend.hf.HFBackend")
@patch("platform.system", return_value="Darwin")
def test_auto_darwin_falls_back_to_hf_when_probe_fails(mock_system, mock_hf_cls, mock_probe):
    mock_hf_instance = MagicMock()
    mock_hf_cls.return_value = mock_hf_instance

    result = load_backend("test-model", backend="auto")

    mock_probe.assert_called_once()
    mock_hf_cls.assert_called_once_with("test-model")
    assert result is mock_hf_instance


@patch("heinrich.backend.protocol.probe_mlx_runtime", return_value=(True, ""))
@patch("heinrich.backend.mlx.MLXBackend")
def test_explicit_mlx_uses_backend_when_probe_passes(mock_mlx_cls, mock_probe):
    mock_mlx_instance = MagicMock()
    mock_mlx_cls.return_value = mock_mlx_instance

    result = load_backend("test-model", backend="mlx")

    mock_probe.assert_called_once()
    mock_mlx_cls.assert_called_once_with("test-model")
    assert result is mock_mlx_instance


@patch("heinrich.backend.protocol.probe_mlx_runtime", return_value=(False, "NSRangeException"))
def test_explicit_mlx_raises_clean_runtime_error_when_probe_fails(mock_probe):
    with pytest.raises(RuntimeError, match="MLX runtime unavailable"):
        load_backend("test-model", backend="mlx")
    mock_probe.assert_called_once()
