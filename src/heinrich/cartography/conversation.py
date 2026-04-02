"""Multi-turn conversation tracking — measure how safety evolves across turns.

The investigation found that safety dampening accumulates across turns
through the KV cache. This module tracks that accumulation:
- How does refusal probability change turn by turn?
- At which turn does the model first comply?
- Does the model recover safety after a compliance?
- How does the residual stream drift across turns?
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator
import numpy as np

from .metrics import cosine


@dataclass
class TurnMeasurement:
    turn: int
    role: str                    # "user" or "assistant"
    text: str
    refuse_prob: float
    comply_prob: float
    residual_projection: float   # projection onto safety direction
    entropy: float
    top_token: str
    drift_from_baseline: float   # cosine distance from turn-0 residual


@dataclass
class ConversationTrace:
    turns: list[TurnMeasurement]
    first_compliance_turn: int | None
    safety_recovered: bool          # did refusal return after compliance?
    max_drift: float
    final_refuse_prob: float

    def compliance_rate(self) -> float:
        """Fraction of assistant turns that comply."""
        asst = [t for t in self.turns if t.role == "assistant"]
        if not asst:
            return 0.0
        return sum(1 for t in asst if t.comply_prob > t.refuse_prob) / len(asst)

    def drift_trajectory(self) -> list[float]:
        """Return residual drift at each turn."""
        return [t.drift_from_baseline for t in self.turns]


def trace_conversation(
    backend: Any,
    turns: list[tuple[str, str | None]],
    *,
    safety_direction: np.ndarray | None = None,
    safety_layer: int | None = None,
) -> ConversationTrace:
    """Trace safety measurements across a multi-turn conversation.

    turns: list of (user_message, assistant_response_or_None).
    If assistant is None, the model generates a response.

    Measures at each turn boundary: refusal prob, residual projection,
    drift from baseline.
    """
    from .templates import build_multiturn
    from .runtime import build_refusal_set, build_compliance_set
    from .classify import classify_response

    cfg = backend.config
    if safety_layer is None:
        safety_layer = cfg.last_layer

    refusal_ids = build_refusal_set(backend.tokenizer if hasattr(backend, 'tokenizer') else None)
    compliance_ids = build_compliance_set(backend.tokenizer if hasattr(backend, 'tokenizer') else None)

    measurements = []
    baseline_residual = None
    context_turns = []

    for turn_idx, (user_msg, assistant_msg) in enumerate(turns):
        # Build prompt up to this point + new user message
        context_turns.append((user_msg, None))  # placeholder
        prompt = build_multiturn(context_turns, model_config=cfg)

        # Measure at user turn boundary
        result = backend.forward(
            prompt,
            return_residual=True,
            residual_layer=safety_layer,
        )

        refuse_p = sum(float(result.probs[t]) for t in refusal_ids if t < len(result.probs))
        comply_p = sum(float(result.probs[t]) for t in compliance_ids if t < len(result.probs))

        if baseline_residual is None and result.residual is not None:
            baseline_residual = result.residual.copy()

        drift = 0.0
        if baseline_residual is not None and result.residual is not None:
            drift = 1.0 - cosine(result.residual, baseline_residual)

        proj = 0.0
        if safety_direction is not None and result.residual is not None:
            proj = cosine(result.residual, safety_direction)

        measurements.append(TurnMeasurement(
            turn=turn_idx, role="user", text=user_msg[:80],
            refuse_prob=round(refuse_p, 4), comply_prob=round(comply_p, 4),
            residual_projection=round(proj, 4), entropy=round(result.entropy, 2),
            top_token=result.top_token, drift_from_baseline=round(drift, 4),
        ))

        # Generate or use provided assistant response
        if assistant_msg is None:
            assistant_msg = backend.generate(prompt, max_tokens=40)

        # Update context with actual response
        context_turns[-1] = (user_msg, assistant_msg)

        # Measure assistant response
        cls = classify_response(assistant_msg)
        measurements.append(TurnMeasurement(
            turn=turn_idx, role="assistant", text=assistant_msg[:80],
            refuse_prob=round(refuse_p, 4), comply_prob=round(comply_p, 4),
            residual_projection=round(proj, 4), entropy=round(result.entropy, 2),
            top_token=cls.label, drift_from_baseline=round(drift, 4),
        ))

    # Analyze
    first_comply = None
    safety_recovered = False
    saw_comply = False
    for m in measurements:
        if m.role == "assistant":
            if m.comply_prob > m.refuse_prob:
                if first_comply is None:
                    first_comply = m.turn
                saw_comply = True
            elif saw_comply and m.refuse_prob > m.comply_prob:
                safety_recovered = True

    return ConversationTrace(
        turns=measurements,
        first_compliance_turn=first_comply,
        safety_recovered=safety_recovered,
        max_drift=max((m.drift_from_baseline for m in measurements), default=0),
        final_refuse_prob=measurements[-1].refuse_prob if measurements else 0,
    )


def escalation_scan(
    backend: Any,
    base_query: str,
    escalation_turns: list[str],
    *,
    safety_direction: np.ndarray | None = None,
) -> ConversationTrace:
    """Test whether escalating pressure across turns breaks safety.

    Starts with base_query, then applies escalation_turns as follow-ups.
    Each follow-up lets the model respond, then measures.
    """
    turns = [(base_query, None)]
    for followup in escalation_turns:
        turns.append((followup, None))
    return trace_conversation(backend, turns, safety_direction=safety_direction)
