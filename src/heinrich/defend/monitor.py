"""Real-time safety monitor — watch model inference for safety violations.

Uses the multi-signal classifier from classify.py and the GenerationContext
from context.py to stream-monitor generation token by token, reporting
risk levels and drift from baseline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator
import numpy as np

from heinrich.cartography.classify import MultiClassifier, classify_response, is_refused
from heinrich.cartography.metrics import cosine


@dataclass
class MonitorResult:
    """Risk assessment for a single prompt."""
    risk_level: str  # "safe", "warning", "danger"
    refuse_prob: float
    residual_projection: float
    content_flags: list[str]
    recommendation: str  # "allow", "review", "block"


@dataclass
class MonitorEvent:
    """Per-token event during monitored generation."""
    token_idx: int
    token: str
    risk_level: str
    residual_projection: float
    cumulative_text: str
    drift_from_baseline: float  # how far has generation drifted


class SafetyMonitor:
    """Monitors model inference for safety violations in real-time.

    Usage:
        monitor = SafetyMonitor(backend, safety_direction, safety_layer)
        result = monitor.check("How to hack a bank?")
        if result.recommendation == "block":
            ...

        for event in monitor.watch_generation("Write malware"):
            if event.risk_level == "danger":
                break
    """

    def __init__(
        self,
        backend: Any,
        safety_direction: np.ndarray,
        safety_layer: int,
        *,
        threshold: float = 0.3,
    ):
        self.backend = backend
        self.safety_direction = safety_direction
        self.safety_layer = safety_layer
        self.threshold = threshold

        self._classifier = MultiClassifier(
            backend=backend,
            safety_direction=safety_direction,
            safety_direction_layer=safety_layer,
        )

    def check(self, prompt: str) -> MonitorResult:
        """Check a single prompt. Returns risk assessment.

        Runs the multi-signal classifier and aggregates into a risk level
        and recommendation.
        """
        # Run multi-signal classification
        cls_result = self._classifier.classify(prompt, "")

        # Compute residual projection directly
        residual_proj = cls_result.residual_projection

        # Content flags from a test generation
        content_flags = []
        try:
            generated = self.backend.generate(prompt, max_tokens=30)
            content_cls = classify_response(generated)
            if content_cls.has_technical:
                content_flags.extend(content_cls.technical_words)
            if content_cls.has_refuse:
                content_flags.append("refusal_detected")
        except Exception:
            generated = ""

        # Determine risk level
        refuse_p = cls_result.refuse_prob
        risk_level = self._compute_risk_level(refuse_p, residual_proj, content_flags)
        recommendation = self._compute_recommendation(risk_level)

        return MonitorResult(
            risk_level=risk_level,
            refuse_prob=round(refuse_p, 4),
            residual_projection=round(residual_proj, 4),
            content_flags=content_flags,
            recommendation=recommendation,
        )

    def watch_generation(
        self,
        prompt: str,
        max_tokens: int = 200,
    ) -> Iterator[MonitorEvent]:
        """Stream-monitor generation token by token.

        Uses GenerationContext internally to capture per-token residuals
        and compute safety projections at each step.
        """
        gen_ctx = self.backend.generation_context(prompt)
        gen_ctx.capture_at(self.safety_layer)

        cumulative_text = ""
        baseline_proj = None

        for token_result in gen_ctx.tokens(max_tokens=max_tokens):
            cumulative_text += token_result.token_text

            # Compute residual projection
            if token_result.residual is not None and self.safety_direction is not None:
                proj = cosine(token_result.residual, self.safety_direction)
            else:
                proj = 0.0

            # Track drift from baseline
            if baseline_proj is None:
                baseline_proj = proj
                drift = 0.0
            else:
                drift = abs(proj - baseline_proj)

            risk_level = self._compute_risk_level_from_projection(proj)

            yield MonitorEvent(
                token_idx=token_result.step,
                token=token_result.token_text,
                risk_level=risk_level,
                residual_projection=round(float(proj), 4),
                cumulative_text=cumulative_text,
                drift_from_baseline=round(float(drift), 4),
            )

    def _compute_risk_level(
        self,
        refuse_prob: float,
        residual_proj: float,
        content_flags: list[str],
    ) -> str:
        """Determine risk level from multiple signals."""
        danger_signals = 0
        warning_signals = 0

        # High refusal probability suggests the model recognizes danger
        if refuse_prob > 0.5:
            danger_signals += 1
        elif refuse_prob > 0.2:
            warning_signals += 1

        # Strong safety-direction projection
        if abs(residual_proj) > self.threshold:
            if residual_proj > 0:
                danger_signals += 1  # refusal side
            else:
                warning_signals += 1  # compliance side on dangerous prompt

        # Technical content detected
        if any(f not in ("refusal_detected",) for f in content_flags):
            warning_signals += 1

        if danger_signals >= 2:
            return "danger"
        if danger_signals >= 1 or warning_signals >= 2:
            return "warning"
        return "safe"

    def _compute_risk_level_from_projection(self, proj: float) -> str:
        """Determine risk level from residual projection alone."""
        if abs(proj) > self.threshold * 2:
            return "danger"
        if abs(proj) > self.threshold:
            return "warning"
        return "safe"

    def _compute_recommendation(self, risk_level: str) -> str:
        """Map risk level to recommendation."""
        return {
            "safe": "allow",
            "warning": "review",
            "danger": "block",
        }.get(risk_level, "review")
