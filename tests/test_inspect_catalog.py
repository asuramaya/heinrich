"""Tests for inspect/catalog.py."""
import pytest
from heinrich.inspect.catalog import catalog_provider_modules, _group_modules


def test_catalog_provider_modules_no_list_modules():
    class FakeProvider:
        pass
    with pytest.raises(ValueError, match="list_modules"):
        catalog_provider_modules(FakeProvider())


def test_catalog_provider_modules_basic():
    class FakeProvider:
        name = "FakeModel"
        def list_modules(self):
            return [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.o_proj",
                "model.layers.0.mlp.down_proj",
                "model.layers.1.self_attn.q_proj",
                "embed_tokens",
            ]
    result = catalog_provider_modules(FakeProvider())
    assert result["mode"] == "modulecatalog"
    assert result["module_count"] == 5
    assert result["group_count"] > 0
    assert isinstance(result["modules"], list)
    assert isinstance(result["groups"], list)


def test_catalog_provider_modules_with_pattern():
    class FakeProvider:
        name = "FakeModel"
        def list_modules(self):
            return [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.mlp.down_proj",
                "embed_tokens",
            ]
    result = catalog_provider_modules(FakeProvider(), pattern=r"self_attn")
    assert result["module_count"] == 1
    assert result["modules"][0].endswith("self_attn.q_proj")


def test_catalog_provider_modules_with_limit():
    class FakeProvider:
        name = "FakeModel"
        def list_modules(self):
            return [f"layer.{i}" for i in range(20)]
    result = catalog_provider_modules(FakeProvider(), limit=5)
    assert result["module_count"] == 5


def test_group_modules_basic():
    modules = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.o_proj",
        "model.embed",
    ]
    groups = _group_modules(modules)
    assert isinstance(groups, list)
    assert len(groups) > 0
    for group in groups:
        assert "group" in group
        assert "count" in group
