"""Tests for heinrich.cartography.context — intervention context system."""
import numpy as np
import pytest
from heinrich.cartography.context import (
    Capabilities, ForwardContext, GenerationContext,
    ContextResult, TokenResult,
    SteerOp, NeuronMaskOp, CaptureResidualOp,
)


class TestCapabilities:
    def test_depth_deep(self):
        c = Capabilities(can_compose=True, can_gen_control=True,
                         can_steer=True, can_capture_residual=True)
        assert c.depth_level == "deep"

    def test_depth_standard(self):
        c = Capabilities(can_steer=True, can_capture_residual=True)
        assert c.depth_level == "standard"

    def test_depth_quick(self):
        c = Capabilities(can_capture_residual=True)
        assert c.depth_level == "quick"

    def test_depth_surface(self):
        c = Capabilities()
        assert c.depth_level == "surface"


class TestForwardContext:
    def test_chaining(self):
        """Declarations should be chainable."""
        class FakeBackend:
            pass
        ctx = ForwardContext(FakeBackend())
        result = (ctx
            .steer(24, np.zeros(10), 1.0, -0.15)
            .zero_neurons(27, [1934])
            .capture_residual([15, 27]))
        assert result is ctx
        assert len(ctx._steers) == 1
        assert len(ctx._neuron_masks) == 1
        assert len(ctx._capture_residuals) == 1

    def test_multiple_steers(self):
        class FakeBackend:
            pass
        ctx = ForwardContext(FakeBackend())
        ctx.steer(24, np.zeros(10), 1.0, -0.1)
        ctx.steer(25, np.zeros(10), 1.0, -0.1)
        ctx.steer(26, np.zeros(10), 1.0, -0.1)
        assert len(ctx._steers) == 3

    def test_context_manager(self):
        class FakeBackend:
            pass
        with ForwardContext(FakeBackend()) as ctx:
            ctx.steer(24, np.zeros(10), 1.0, -0.15)
            assert len(ctx._steers) == 1


class TestGenerationContext:
    def test_steer_and_capture(self):
        class FakeBackend:
            pass
        gen = GenerationContext(FakeBackend(), "test prompt")
        gen.steer(24, np.zeros(10), 1.0, -0.15)
        gen.capture_at(27)
        assert len(gen._steers) == 1
        assert gen._capture_layer == 27

    def test_inject_once(self):
        class FakeBackend:
            pass
        gen = GenerationContext(FakeBackend(), "test")
        gen.inject_once(20, np.ones(10))
        assert len(gen._one_shot_injections) == 1


class TestContextResult:
    def test_creation(self):
        r = ContextResult(
            logits=np.zeros(100), probs=np.ones(100) / 100,
            top_id=0, top_token="a", entropy=6.64, n_tokens=5,
            residuals={15: np.zeros(768), 27: np.zeros(768)},
        )
        assert 15 in r.residuals
        assert 27 in r.residuals

    def test_mlp_detail(self):
        r = ContextResult(
            logits=np.zeros(100), probs=np.ones(100) / 100,
            top_id=0, top_token="a", entropy=6.64, n_tokens=5,
            mlp_detail={27: {"gate": np.zeros(1024), "up": np.zeros(1024)}},
        )
        assert "gate" in r.mlp_detail[27]


class TestTokenResult:
    def test_creation(self):
        t = TokenResult(step=0, token_id=42, token_text="hello")
        assert t.step == 0
        assert t.residual is None

    def test_with_residual(self):
        t = TokenResult(step=5, token_id=99, token_text="world",
                        residual=np.zeros(768))
        assert t.residual is not None
        assert t.residual.shape == (768,)
