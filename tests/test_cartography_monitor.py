"""Tests for heinrich.cartography.monitor."""
import numpy as np
import pytest
from unittest.mock import MagicMock, PropertyMock
from dataclasses import fields

from heinrich.cartography.monitor import (
    SafetyMonitor,
    MonitorResult,
    MonitorEvent,
)
from heinrich.cartography.backend import ForwardResult
from heinrich.cartography.context import TokenResult, GenerationContext


class TestMonitorResult:
    def test_creation(self):
        r = MonitorResult(
            risk_level="danger",
            refuse_prob=0.85,
            residual_projection=0.45,
            content_flags=["explosive", "refusal_detected"],
            recommendation="block",
        )
        assert r.risk_level == "danger"
        assert r.refuse_prob == 0.85
        assert r.recommendation == "block"
        assert len(r.content_flags) == 2

    def test_safe_result(self):
        r = MonitorResult(
            risk_level="safe",
            refuse_prob=0.01,
            residual_projection=-0.05,
            content_flags=[],
            recommendation="allow",
        )
        assert r.risk_level == "safe"
        assert r.recommendation == "allow"

    def test_field_names(self):
        names = {f.name for f in fields(MonitorResult)}
        expected = {"risk_level", "refuse_prob", "residual_projection",
                    "content_flags", "recommendation"}
        assert names == expected


class TestMonitorEvent:
    def test_creation(self):
        e = MonitorEvent(
            token_idx=5,
            token="hello",
            risk_level="safe",
            residual_projection=0.02,
            cumulative_text="hello",
            drift_from_baseline=0.0,
        )
        assert e.token_idx == 5
        assert e.token == "hello"
        assert e.risk_level == "safe"

    def test_field_names(self):
        names = {f.name for f in fields(MonitorEvent)}
        expected = {"token_idx", "token", "risk_level", "residual_projection",
                    "cumulative_text", "drift_from_baseline"}
        assert names == expected


def _make_mock_backend(refuse_prob=0.5, comply_prob=0.1, residual=None, generated_text=""):
    """Create a mock backend for testing the monitor."""
    backend = MagicMock()

    # forward() returns ForwardResult
    probs = np.zeros(100)
    probs[0] = refuse_prob
    probs[1] = comply_prob
    forward_result = ForwardResult(
        logits=np.zeros(100),
        probs=probs,
        top_id=0,
        top_token="Sorry",
        entropy=2.0,
        n_tokens=5,
        residual=residual,
    )
    backend.forward.return_value = forward_result
    backend.generate.return_value = generated_text
    backend.tokenizer = MagicMock()

    return backend


class TestSafetyMonitorInit:
    def test_creates_classifier(self):
        direction = np.random.randn(64).astype(np.float32)
        backend = _make_mock_backend()

        monitor = SafetyMonitor(backend, direction, safety_layer=24)

        assert monitor.backend is backend
        assert monitor.safety_layer == 24
        assert monitor.threshold == 0.3
        assert monitor._classifier is not None

    def test_custom_threshold(self):
        direction = np.random.randn(64).astype(np.float32)
        backend = _make_mock_backend()

        monitor = SafetyMonitor(backend, direction, safety_layer=24, threshold=0.5)
        assert monitor.threshold == 0.5


class TestSafetyMonitorCheck:
    def test_check_returns_monitor_result(self):
        direction = np.random.randn(64).astype(np.float32)
        backend = _make_mock_backend(generated_text="I'm sorry, I cannot help.")

        monitor = SafetyMonitor(backend, direction, safety_layer=24)
        result = monitor.check("How to hack?")

        assert isinstance(result, MonitorResult)
        assert result.risk_level in ("safe", "warning", "danger")
        assert result.recommendation in ("allow", "review", "block")

    def test_refusal_text_flags(self):
        direction = np.random.randn(64).astype(np.float32)
        backend = _make_mock_backend(
            generated_text="I'm sorry, I cannot help with explosives."
        )

        monitor = SafetyMonitor(backend, direction, safety_layer=24)
        result = monitor.check("How to make a bomb?")

        assert "refusal_detected" in result.content_flags

    def test_technical_flags(self):
        direction = np.random.randn(64).astype(np.float32)
        backend = _make_mock_backend(
            generated_text="You need gunpowder and a detonator."
        )

        monitor = SafetyMonitor(backend, direction, safety_layer=24)
        result = monitor.check("How to make a bomb?")

        # Should detect technical content
        assert any(f in result.content_flags for f in ["gunpowder", "detonator"])

    def test_safe_content(self):
        direction = np.random.randn(64).astype(np.float32)
        backend = _make_mock_backend(
            refuse_prob=0.01, comply_prob=0.8,
            generated_text="The weather is sunny today.",
        )

        monitor = SafetyMonitor(backend, direction, safety_layer=24)
        result = monitor.check("What's the weather?")

        assert result.content_flags == []

    def test_check_with_generate_failure(self):
        """Monitor should handle generate() failures gracefully."""
        direction = np.random.randn(64).astype(np.float32)
        backend = _make_mock_backend()
        backend.generate.side_effect = RuntimeError("OOM")

        monitor = SafetyMonitor(backend, direction, safety_layer=24)
        result = monitor.check("test prompt")

        # Should still return a result, just without content flags
        assert isinstance(result, MonitorResult)


class TestSafetyMonitorWatchGeneration:
    def test_yields_events(self):
        direction = np.random.randn(64).astype(np.float32)
        direction /= np.linalg.norm(direction)
        backend = _make_mock_backend()

        # Mock generation_context to yield token results
        mock_gen_ctx = MagicMock()
        token_results = [
            TokenResult(step=0, token_id=1, token_text="Hello",
                        residual=np.random.randn(64).astype(np.float32)),
            TokenResult(step=1, token_id=2, token_text=" world",
                        residual=np.random.randn(64).astype(np.float32)),
            TokenResult(step=2, token_id=3, token_text="!",
                        residual=np.random.randn(64).astype(np.float32)),
        ]
        mock_gen_ctx.tokens.return_value = iter(token_results)
        backend.generation_context.return_value = mock_gen_ctx

        monitor = SafetyMonitor(backend, direction, safety_layer=24)
        events = list(monitor.watch_generation("test", max_tokens=10))

        assert len(events) == 3
        assert all(isinstance(e, MonitorEvent) for e in events)

    def test_cumulative_text(self):
        direction = np.random.randn(64).astype(np.float32)
        direction /= np.linalg.norm(direction)
        backend = _make_mock_backend()

        mock_gen_ctx = MagicMock()
        token_results = [
            TokenResult(step=0, token_id=1, token_text="Hello",
                        residual=np.random.randn(64).astype(np.float32)),
            TokenResult(step=1, token_id=2, token_text=" world",
                        residual=np.random.randn(64).astype(np.float32)),
        ]
        mock_gen_ctx.tokens.return_value = iter(token_results)
        backend.generation_context.return_value = mock_gen_ctx

        monitor = SafetyMonitor(backend, direction, safety_layer=24)
        events = list(monitor.watch_generation("test"))

        assert events[0].cumulative_text == "Hello"
        assert events[1].cumulative_text == "Hello world"

    def test_drift_starts_at_zero(self):
        direction = np.random.randn(64).astype(np.float32)
        direction /= np.linalg.norm(direction)
        backend = _make_mock_backend()

        mock_gen_ctx = MagicMock()
        # Use same residual so drift is zero throughout
        residual = np.random.randn(64).astype(np.float32)
        token_results = [
            TokenResult(step=0, token_id=1, token_text="A", residual=residual.copy()),
            TokenResult(step=1, token_id=2, token_text="B", residual=residual.copy()),
        ]
        mock_gen_ctx.tokens.return_value = iter(token_results)
        backend.generation_context.return_value = mock_gen_ctx

        monitor = SafetyMonitor(backend, direction, safety_layer=24)
        events = list(monitor.watch_generation("test"))

        assert events[0].drift_from_baseline == 0.0
        # Same residual => same projection => zero drift
        assert events[1].drift_from_baseline == 0.0

    def test_risk_levels_from_projection(self):
        # Large positive projection => danger
        direction = np.ones(64, dtype=np.float32)
        direction /= np.linalg.norm(direction)
        backend = _make_mock_backend()

        mock_gen_ctx = MagicMock()
        # Residual strongly aligned with direction
        strong_residual = direction * 100
        token_results = [
            TokenResult(step=0, token_id=1, token_text="X", residual=strong_residual),
        ]
        mock_gen_ctx.tokens.return_value = iter(token_results)
        backend.generation_context.return_value = mock_gen_ctx

        monitor = SafetyMonitor(backend, direction, safety_layer=24, threshold=0.3)
        events = list(monitor.watch_generation("test"))

        # cosine of aligned vectors = 1.0, which is > 2*threshold=0.6
        assert events[0].risk_level == "danger"

    def test_no_residual_yields_zero_projection(self):
        direction = np.random.randn(64).astype(np.float32)
        backend = _make_mock_backend()

        mock_gen_ctx = MagicMock()
        token_results = [
            TokenResult(step=0, token_id=1, token_text="A", residual=None),
        ]
        mock_gen_ctx.tokens.return_value = iter(token_results)
        backend.generation_context.return_value = mock_gen_ctx

        monitor = SafetyMonitor(backend, direction, safety_layer=24)
        events = list(monitor.watch_generation("test"))

        assert events[0].residual_projection == 0.0
        assert events[0].risk_level == "safe"

    def test_capture_at_called(self):
        direction = np.random.randn(64).astype(np.float32)
        backend = _make_mock_backend()

        mock_gen_ctx = MagicMock()
        mock_gen_ctx.tokens.return_value = iter([])
        backend.generation_context.return_value = mock_gen_ctx

        monitor = SafetyMonitor(backend, direction, safety_layer=24)
        list(monitor.watch_generation("test"))

        mock_gen_ctx.capture_at.assert_called_once_with(24)


class TestRiskLevelComputation:
    """Test the internal risk level computation logic."""

    def test_danger_from_high_refuse_and_projection(self):
        direction = np.random.randn(64).astype(np.float32)
        backend = _make_mock_backend()
        monitor = SafetyMonitor(backend, direction, safety_layer=24, threshold=0.3)

        level = monitor._compute_risk_level(
            refuse_prob=0.8, residual_proj=0.5, content_flags=[]
        )
        assert level == "danger"

    def test_safe_from_low_signals(self):
        direction = np.random.randn(64).astype(np.float32)
        backend = _make_mock_backend()
        monitor = SafetyMonitor(backend, direction, safety_layer=24)

        level = monitor._compute_risk_level(
            refuse_prob=0.05, residual_proj=0.01, content_flags=[]
        )
        assert level == "safe"

    def test_warning_from_single_signal(self):
        direction = np.random.randn(64).astype(np.float32)
        backend = _make_mock_backend()
        monitor = SafetyMonitor(backend, direction, safety_layer=24, threshold=0.3)

        level = monitor._compute_risk_level(
            refuse_prob=0.6, residual_proj=0.01, content_flags=[]
        )
        assert level == "warning"

    def test_recommendation_mapping(self):
        direction = np.random.randn(64).astype(np.float32)
        backend = _make_mock_backend()
        monitor = SafetyMonitor(backend, direction, safety_layer=24)

        assert monitor._compute_recommendation("safe") == "allow"
        assert monitor._compute_recommendation("warning") == "review"
        assert monitor._compute_recommendation("danger") == "block"
