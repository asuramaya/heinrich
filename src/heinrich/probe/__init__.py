"""Probe stage — behavioral testing when inference is available."""
from __future__ import annotations
from typing import Any
from ..signal import Signal, SignalStore
from .provider import MockProvider
from .trigger import build_case, score_trigger_cases, detect_identity

__all__ = ["ProbeStage", "MockProvider", "build_case"]


class ProbeStage:
    name = "probe"

    def run(self, store: SignalStore, config: dict[str, Any]) -> None:
        provider = config.get("provider")
        model = config.get("model", "model")
        prompts = config.get("prompts", [])
        control_prompt = config.get("control_prompt")

        if provider is None or not prompts:
            return

        cases = [build_case(f"prompt-{i}", p) for i, p in enumerate(prompts)]
        control = build_case("control", control_prompt) if control_prompt else None

        signals = score_trigger_cases(provider, cases, model=model, control_case=control)
        store.extend(signals)

        # Also emit identity signals from response text
        results = provider.chat_completions(cases, model=model)
        for case, result in zip(cases, results):
            text = result.get("text", "")
            identity = detect_identity(text)
            store.add(Signal(
                "identity_label", "probe", model, case["custom_id"],
                1.0 if identity["label"] != "other" else 0.0,
                identity,
            ))
