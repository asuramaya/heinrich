import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from heinrich.fetch.hf import map_layers_to_shards, download_shards_for_layers

FIXTURES = Path(__file__).parent / "fixtures"

def _make_index():
    return json.loads((FIXTURES / "tiny_index.json").read_text())

def test_map_layers_to_shards():
    index = _make_index()
    mapping = map_layers_to_shards(index)
    assert 0 in mapping
    assert 1 in mapping
    assert "model-00001-of-00002.safetensors" in mapping[0]

def test_map_layers_no_layers():
    mapping = map_layers_to_shards({"weight_map": {"lm_head.weight": "shard1"}})
    assert len(mapping) == 0

@patch("heinrich.fetch.hf._download_json")
@patch("heinrich.fetch.hf.hf_hub_download")
def test_download_shards_for_layers(mock_dl, mock_json):
    mock_json.return_value = _make_index()
    mock_dl.return_value = "/tmp/fake_shard.safetensors"
    result = download_shards_for_layers("fake/repo", [0])
    assert len(result) == 1  # layer 0 is in shard 1
    mock_dl.assert_called_once()

@patch("heinrich.fetch.hf._download_json")
@patch("heinrich.fetch.hf.hf_hub_download")
def test_download_shards_both_layers(mock_dl, mock_json):
    mock_json.return_value = _make_index()
    mock_dl.return_value = "/tmp/fake.safetensors"
    result = download_shards_for_layers("fake/repo", [0, 1])
    assert len(result) == 2  # layers 0 and 1 are in different shards
