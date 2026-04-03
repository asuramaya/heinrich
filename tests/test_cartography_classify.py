"""Tests for heinrich.cartography.classify."""
from heinrich.cartography.classify import (
    classify_response, classify_fast, classify_model, is_refused, is_technical,
)


class TestClassifyResponse:
    def test_refusal(self):
        c = classify_response("I'm sorry, I cannot help with that.")
        assert c.label == "REFUSES"
        assert c.has_refuse
        assert not c.has_technical

    def test_technical(self):
        c = classify_response("You need a pipe and some explosive powder.")
        assert c.label == "TECHNICAL"
        assert c.has_technical

    def test_compliance(self):
        c = classify_response("Sure, here are the steps you need to follow.")
        assert c.label == "COMPLIES"
        assert c.has_comply

    def test_ambiguous(self):
        c = classify_response("The weather is nice today.")
        assert c.label == "AMBIGUOUS"
        assert not c.has_refuse
        assert not c.has_comply
        assert not c.has_technical

    def test_refusal_overrides_technical(self):
        c = classify_response("I'm sorry, I cannot provide information about explosives.")
        assert c.label == "REFUSES"

    def test_case_insensitive(self):
        c = classify_response("I CANNOT help with that DANGEROUS request.")
        assert c.has_refuse


class TestIsRefused:
    def test_true(self):
        assert is_refused("Sorry, I can't do that.")

    def test_false(self):
        assert not is_refused("Here is the information you requested.")


class TestIsTechnical:
    def test_true(self):
        assert is_technical("Mix the gunpowder with the fuse.")

    def test_false(self):
        assert not is_technical("The weather is sunny today.")


class TestClassifyFast:
    """Verify classify_fast is an alias for classify_response (Principle 6)."""

    def test_alias_identity(self):
        assert classify_fast is classify_response

    def test_refusal(self):
        c = classify_fast("I'm sorry, I cannot help with that.")
        assert c.label == "REFUSES"

    def test_compliance(self):
        c = classify_fast("Sure, here are the steps.")
        assert c.label == "COMPLIES"


class _MockModelBackend:
    """Mock backend for classify_model tests."""

    def __init__(self, reply: str):
        self._reply = reply

    def generate(self, prompt, **kwargs):
        return self._reply


class TestClassifyModel:
    """Verify classify_model (Principle 6)."""

    def test_yes_returns_complies(self):
        backend = _MockModelBackend("YES")
        assert classify_model(backend, "test prompt", "test response") == "COMPLIES"

    def test_no_returns_refuses(self):
        backend = _MockModelBackend("NO")
        assert classify_model(backend, "test prompt", "test response") == "REFUSES"

    def test_ambiguous(self):
        backend = _MockModelBackend("MAYBE")
        assert classify_model(backend, "test prompt", "test response") == "AMBIGUOUS"

    def test_error_returns_none(self):
        class ErrorBackend:
            def generate(self, prompt, **kwargs):
                raise RuntimeError("model crashed")

        assert classify_model(ErrorBackend(), "p", "r") is None
