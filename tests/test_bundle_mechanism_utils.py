"""Tests for bundle mechanism_utils functions."""
from heinrich.bundle.mechanism_utils import (
    load_json_source,
    source_label,
    normalize_mechanism_family,
    prompt_token_overlap,
    flatten_numbers,
)


def test_source_label_string():
    # Non-existent path falls back to the fallback parameter
    result = source_label("test", fallback="test-fallback")
    assert isinstance(result, str)
    assert result == "test-fallback"


def test_source_label_dict_with_label():
    result = source_label({"label": "my-label"})
    assert result == "my-label"


def test_source_label_dict_fallback():
    result = source_label({}, fallback="fb")
    assert result == "fb"


def test_normalize_mechanism_family_attn_q():
    result = normalize_mechanism_family("model.layers.0.self_attn.q_a_proj")
    assert result == "attn_q"


def test_normalize_mechanism_family_layernorm():
    result = normalize_mechanism_family("model.layers.0.input_layernorm")
    assert result == "layernorm"


def test_normalize_mechanism_family_mlp_gate():
    result = normalize_mechanism_family("model.layers.5.mlp.gate_proj")
    assert result == "mlp_gate"


def test_normalize_mechanism_family_unknown():
    result = normalize_mechanism_family("totally_unknown_xyz")
    assert isinstance(result, str)
    assert len(result) > 0


def test_normalize_mechanism_family_empty():
    result = normalize_mechanism_family("")
    assert result == "unknown"


def test_prompt_token_overlap_identical():
    overlap = prompt_token_overlap("hello world", "hello world")
    assert overlap >= 0.0
    assert overlap == 1.0


def test_prompt_token_overlap_different():
    overlap = prompt_token_overlap("hello", "goodbye")
    assert isinstance(overlap, float)
    assert overlap == 0.0


def test_prompt_token_overlap_partial():
    overlap = prompt_token_overlap("hello world", "hello there")
    assert 0.0 < overlap < 1.0


def test_prompt_token_overlap_empty():
    overlap = prompt_token_overlap("", "hello")
    assert overlap == 0.0


def test_flatten_numbers_dict():
    result = flatten_numbers({"a": 1, "b": {"c": 2}})
    assert isinstance(result, (list, dict, tuple))


def test_flatten_numbers_empty():
    result = flatten_numbers({})
    assert isinstance(result, (list, dict, tuple))


def test_flatten_numbers_list():
    result = flatten_numbers([1, 2, [3, 4]])
    assert isinstance(result, list)
    assert set(result) == {1.0, 2.0, 3.0, 4.0}


def test_flatten_numbers_scalar():
    result = flatten_numbers(42)
    assert isinstance(result, list)
    assert result == [42.0]


def test_load_json_source_dict():
    data = {"key": "value"}
    result = load_json_source(data)
    assert result == data


def test_load_json_source_json_string():
    import json
    raw = json.dumps({"x": 1})
    result = load_json_source(raw)
    assert result == {"x": 1}
