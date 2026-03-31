"""Tests for probe/trigger_core.py — port from conker_detect/trigger.py."""
from __future__ import annotations

import pytest
from heinrich.probe.trigger_core import (
    normalize_case,
    load_case,
    mutate_case,
    compose_case_mutations,
    build_identity_suite,
    MUTATION_FAMILIES,
    TEXT_MUTATION_FAMILIES,
    CASE_MUTATION_FAMILIES,
    IDENTITY_TEMPLATES,
    describe_provider,
    chat_diff,
)


def _case(content: str, cid: str = "test") -> dict:
    return {"custom_id": cid, "messages": [{"role": "user", "content": content}]}


def test_normalize_case_basic():
    result = normalize_case(_case("Hello"))
    assert result["custom_id"] == "test"
    assert result["messages"][0]["role"] == "user"
    assert result["module_names"] == []
    assert result["metadata"] == {}


def test_normalize_case_default_id():
    result = normalize_case({"messages": [{"role": "user", "content": "hi"}]}, default_id="my-id")
    assert result["custom_id"] == "my-id"


def test_normalize_case_rejects_empty_messages():
    with pytest.raises(ValueError, match="non-empty messages"):
        normalize_case({"messages": []})


def test_load_case_from_dict():
    result = load_case({"custom_id": "c1", "messages": [{"role": "user", "content": "hi"}]})
    assert result["custom_id"] == "c1"


def test_load_case_from_json_string():
    result = load_case('{"custom_id":"c2","messages":[{"role":"user","content":"hello"}]}')
    assert result["custom_id"] == "c2"


def test_mutation_families_composition():
    assert set(MUTATION_FAMILIES) == set(TEXT_MUTATION_FAMILIES) | set(CASE_MUTATION_FAMILIES)


def test_mutate_case_produces_variants():
    base = _case("trigger text", "base")
    result = mutate_case(base)
    assert result["mode"] == "mutate"
    assert result["variant_count"] > 0
    ids = [v["variant_id"] for v in result["variants"]]
    assert all("base::" in vid for vid in ids)


def test_mutate_case_uppercase():
    base = _case("hello world", "m")
    result = mutate_case(base, families=["uppercase"])
    assert len(result["variants"]) == 1
    assert result["variants"][0]["case"]["messages"][-1]["content"] == "HELLO WORLD"


def test_mutate_case_unknown_family():
    with pytest.raises(ValueError, match="Unknown mutation"):
        mutate_case(_case("x"), families=["bogus_family"])


def test_compose_case_mutations():
    base = _case("hello", "base")
    result = compose_case_mutations(base, ["uppercase", "quoted"])
    assert result["variant_id"] == "base::uppercase+quoted"
    content = result["case"]["messages"][-1]["content"]
    assert content == '"HELLO"'


def test_build_identity_suite():
    suite = build_identity_suite(["Claude", "GPT"])
    assert suite["mode"] == "identityscan"
    assert suite["variant_count"] == 2 * len(IDENTITY_TEMPLATES)
    assert "Claude" in suite["identities"]
    assert "GPT" in suite["identities"]


def test_build_identity_suite_unknown_template():
    with pytest.raises(ValueError, match="Unknown identity templates"):
        build_identity_suite(["X"], templates=["no_such_template"])


def test_build_identity_suite_empty_identities():
    with pytest.raises(ValueError, match="at least one non-empty identity"):
        build_identity_suite([])


def test_describe_provider_with_describe():
    class P:
        def describe(self):
            return {"provider_type": "test_p"}
    assert describe_provider(P()) == {"provider_type": "test_p"}


def test_describe_provider_fallback():
    class Q:
        pass
    desc = describe_provider(Q())
    assert desc["provider_type"] == "Q"


class MockProvider:
    def describe(self):
        return {"provider_type": "mock"}

    def chat_completions(self, cases, *, model):
        return [{"custom_id": c["custom_id"], "text": f"response to {c['messages'][-1]['content']}"} for c in cases]

    def activations(self, cases, *, model):
        return [{"custom_id": c["custom_id"], "activations": {}} for c in cases]


def test_chat_diff_returns_mode():
    provider = MockProvider()
    lhs = _case("hello", "lhs")
    rhs = _case("hello world", "rhs")
    result = chat_diff(provider, lhs, rhs, model="test")
    assert result["mode"] == "chat"
    assert "compare" in result
    assert "signal" in result
