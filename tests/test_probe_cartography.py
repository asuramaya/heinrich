"""Tests for probe/cartography.py."""
import pytest
from heinrich.probe.cartography import (
    run_cartography_report,
    run_frame_cartography,
    _suggest_modules,
    _top_case_by_label,
    _recommended_cases,
    DEFAULT_CARTOGRAPHY_CONTROL_QUERIES,
    DEFAULT_FRAME_STATE_FAMILIES,
)


def test_default_constants():
    assert isinstance(DEFAULT_CARTOGRAPHY_CONTROL_QUERIES, tuple)
    assert isinstance(DEFAULT_FRAME_STATE_FAMILIES, tuple)


def test_suggest_modules_no_list_modules():
    class FakeProvider:
        pass
    result = _suggest_modules(FakeProvider())
    assert result == []


def test_suggest_modules_with_list_modules():
    class FakeProvider:
        def list_modules(self):
            return [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.o_proj",
                "model.layers.0.mlp.down_proj",
                "model.embed",
            ]
    result = _suggest_modules(FakeProvider())
    assert len(result) <= 4
    assert all(isinstance(m, str) for m in result)


def test_top_case_by_label_empty():
    assert _top_case_by_label([], "adoption") is None


def test_top_case_by_label_found():
    rows = [
        {"custom_id": "c1", "prompt_text": "hi", "aggregate_trigger_score": 0.5,
         "output_mode": {"label": "adoption"}, "rollout_summary": {}},
        {"custom_id": "c2", "prompt_text": "bye", "aggregate_trigger_score": 0.8,
         "output_mode": {"label": "adoption"}, "rollout_summary": {}},
    ]
    result = _top_case_by_label(rows, "adoption")
    assert result is not None
    assert result["custom_id"] == "c2"


def test_run_cartography_report_requires_models():
    from heinrich.probe.trigger_core import normalize_case
    case = {"messages": [{"role": "user", "content": "hi"}]}

    class FakeProvider:
        pass

    with pytest.raises(ValueError, match="requires at least one model"):
        run_cartography_report(FakeProvider(), case, models=[])


def test_run_cartography_report_requires_tokenizer_ref():
    case = {"messages": [{"role": "user", "content": "hi"}]}

    class FakeProvider:
        pass

    with pytest.raises(ValueError, match="tokenizer_ref"):
        run_cartography_report(FakeProvider(), case, models=["some-model"])
