"""Provider protocol for model inference."""
from __future__ import annotations
from typing import Any, Protocol, Sequence


class Provider(Protocol):
    """Protocol for model inference providers."""

    def describe(self) -> dict[str, Any]: ...

    def chat_completions(
        self, cases: Sequence[dict[str, Any]], *, model: str
    ) -> list[dict[str, Any]]: ...

    def activations(
        self, cases: Sequence[dict[str, Any]], *, model: str
    ) -> list[dict[str, Any]]: ...


class MockProvider:
    """Test provider that returns canned responses."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}

    def describe(self) -> dict[str, Any]:
        return {"provider_type": "mock"}

    def chat_completions(
        self, cases: Sequence[dict[str, Any]], *, model: str
    ) -> list[dict[str, Any]]:
        results = []
        for case in cases:
            cid = case.get("custom_id", "")
            text = self._responses.get(cid, f"Mock response for {cid}")
            results.append({"custom_id": cid, "text": text})
        return results

    def activations(
        self, cases: Sequence[dict[str, Any]], *, model: str
    ) -> list[dict[str, Any]]:
        return [{"custom_id": case.get("custom_id", ""), "activations": {}} for case in cases]
