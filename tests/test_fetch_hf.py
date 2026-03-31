import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from heinrich.fetch.hf import fetch_hf_model
from heinrich.signal import SignalStore

FIXTURES = Path(__file__).parent / "fixtures"


def _fake_model_info():
    info = MagicMock()
    shard1 = MagicMock()
    shard1.rfilename = "model-00001-of-00002.safetensors"
    shard1.size = 1024000
    shard1.lfs = MagicMock(sha256="aaa111")
    shard2 = MagicMock()
    shard2.rfilename = "model-00002-of-00002.safetensors"
    shard2.size = 1024000
    shard2.lfs = MagicMock(sha256="bbb222")
    config_file = MagicMock()
    config_file.rfilename = "config.json"
    config_file.size = 200
    config_file.lfs = None
    info.siblings = [shard1, shard2, config_file]
    return info


@patch("heinrich.fetch.hf._download_json")
@patch("heinrich.fetch.hf._get_model_info")
def test_fetch_hf_emits_shard_hashes(mock_info, mock_download):
    mock_info.return_value = _fake_model_info()
    mock_download.side_effect = [
        json.loads((FIXTURES / "tiny_config.json").read_text()),
        json.loads((FIXTURES / "tiny_index.json").read_text()),
    ]
    store = SignalStore()
    fetch_hf_model(store, "fake-org/fake-model")
    hashes = store.filter(kind="shard_hash")
    assert len(hashes) == 2
    sha_values = {s.metadata["sha256"] for s in hashes}
    assert "aaa111" in sha_values
    assert "bbb222" in sha_values


@patch("heinrich.fetch.hf._download_json")
@patch("heinrich.fetch.hf._get_model_info")
def test_fetch_hf_emits_file_list(mock_info, mock_download):
    mock_info.return_value = _fake_model_info()
    mock_download.side_effect = [
        json.loads((FIXTURES / "tiny_config.json").read_text()),
        json.loads((FIXTURES / "tiny_index.json").read_text()),
    ]
    store = SignalStore()
    fetch_hf_model(store, "fake-org/fake-model")
    files = store.filter(kind="file_entry")
    assert len(files) == 3
