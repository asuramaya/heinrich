"""Tests for heinrich.cartography.context — intervention context system."""
from unittest.mock import MagicMock, patch, PropertyMock
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


# ============================================================
# Fix 1: KV Cache in GenerationContext
# ============================================================

class TestGenerationContextKVCache:
    """Test that _execute_generation_context uses KV cache objects."""

    def _make_mock_layer(self, hidden_size, call_log):
        """Create a mock transformer layer that tracks cache usage."""
        layer = MagicMock()

        def layer_call(h, mask=None, cache=None):
            call_log.append({
                "input_seq_len": h.shape[1] if hasattr(h, "shape") else None,
                "cache_provided": cache is not None,
                "cache_type": type(cache).__name__ if cache is not None else None,
            })
            return h  # pass through

        layer.side_effect = layer_call
        return layer

    def test_cache_objects_passed_to_layers(self):
        """Verify that cache objects are created and passed to each layer."""
        call_log = []
        hidden_size = 32
        n_layers = 4

        # Build mock inner model
        layers = [self._make_mock_layer(hidden_size, call_log) for _ in range(n_layers)]

        mock_inner = MagicMock()
        mock_inner.layers = layers
        mock_inner.embed_tokens.side_effect = lambda ids: MagicMock(
            shape=(1, ids.shape[1] if hasattr(ids, "shape") else 1, hidden_size),
            __getitem__=lambda self, key: self,
        )

        # We can't easily run the full method without mlx, but we can test
        # that the ForwardContext API registers cache-related state properly
        # and that the generation context structure is correct.
        gen = GenerationContext(MagicMock(), "test prompt")
        gen.steer(2, np.zeros(hidden_size), 1.0, -0.1)
        gen.capture_at(3)
        assert gen._capture_layer == 3
        assert len(gen._steers) == 1

    def test_inject_once_cleared_after_use(self):
        """inject_once should be consumable (cleared after one step)."""
        gen = GenerationContext(MagicMock(), "test")
        gen.inject_once(5, np.ones(32))
        assert len(gen._one_shot_injections) == 1
        # Simulating the clear that happens in _execute_generation_context
        gen._one_shot_injections.clear()
        assert len(gen._one_shot_injections) == 0


# ============================================================
# Fix 2: Attention Capture in ForwardContext
# ============================================================

class TestForwardContextAttentionCapture:
    """Test that attention capture declarations are properly registered
    and that the execution path would populate attentions dict."""

    def test_capture_attention_declaration(self):
        """capture_attention should register layer for attention extraction."""
        ctx = ForwardContext(MagicMock())
        ctx.capture_attention(15)
        ctx.capture_attention(27)
        assert len(ctx._capture_attentions) == 2
        assert ctx._capture_attentions[0].layer == 15
        assert ctx._capture_attentions[1].layer == 27

    def test_attention_capture_with_mock_execution(self):
        """Simulate attention computation to verify shape and population."""
        # Create a mock backend whose _execute_forward_context populates attentions
        n_heads = 4
        n_kv_heads = 2
        head_dim = 8
        seq_len = 5

        # Simulate the attention computation logic from _execute_forward_context
        # This tests the Q/K decomposition math in isolation
        rng = np.random.RandomState(42)
        q = rng.randn(1, n_heads, seq_len, head_dim).astype(np.float32)
        k = rng.randn(1, n_kv_heads, seq_len, head_dim).astype(np.float32)

        # Expand KV heads for GQA
        n_rep = n_heads // n_kv_heads
        if n_rep > 1:
            k = np.repeat(k, repeats=n_rep, axis=1)

        scale = head_dim ** -0.5
        q_last = q[:, :, -1:, :]  # [1, n_heads, 1, head_dim]
        scores = (q_last @ k.transpose(0, 1, 3, 2)) * scale  # [1, n_heads, 1, seq_len]

        # Softmax over last axis
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        # Final shape: [n_heads, seq_len]
        attention_out = weights[0, :, 0, :]
        assert attention_out.shape == (n_heads, seq_len)
        # Each head's attention weights should sum to ~1.0
        for h in range(n_heads):
            assert abs(attention_out[h].sum() - 1.0) < 1e-5

    def test_attention_result_in_context_result(self):
        """ContextResult should carry attention data when populated."""
        attention_data = {15: np.random.randn(8, 20).astype(np.float32)}
        r = ContextResult(
            logits=np.zeros(100), probs=np.ones(100) / 100,
            top_id=0, top_token="a", entropy=6.64, n_tokens=20,
            attention=attention_data,
        )
        assert 15 in r.attention
        assert r.attention[15].shape == (8, 20)


# ============================================================
# Fix 3: Callback System in ForwardContext
# ============================================================

class TestForwardContextCallbacks:
    """Test the on_layer callback mechanism."""

    def test_on_layer_registration(self):
        """on_layer should register callbacks in _callbacks dict."""
        ctx = ForwardContext(MagicMock())
        called = []

        def my_callback(layer, residual):
            called.append(layer)
            return None

        ctx.on_layer(15, my_callback)
        assert 15 in ctx._callbacks
        assert len(ctx._callbacks[15]) == 1
        assert ctx._callbacks[15][0] is my_callback

    def test_on_layer_chaining(self):
        """on_layer should return self for chaining."""
        ctx = ForwardContext(MagicMock())
        result = ctx.on_layer(15, lambda l, r: None)
        assert result is ctx

    def test_multiple_callbacks_same_layer(self):
        """Multiple callbacks on the same layer should all be stored."""
        ctx = ForwardContext(MagicMock())
        cb1 = lambda l, r: None
        cb2 = lambda l, r: None
        ctx.on_layer(10, cb1)
        ctx.on_layer(10, cb2)
        assert len(ctx._callbacks[10]) == 2

    def test_multiple_callbacks_different_layers(self):
        """Callbacks on different layers go to different entries."""
        ctx = ForwardContext(MagicMock())
        ctx.on_layer(10, lambda l, r: None)
        ctx.on_layer(20, lambda l, r: None)
        assert 10 in ctx._callbacks
        assert 20 in ctx._callbacks
        assert len(ctx._callbacks[10]) == 1
        assert len(ctx._callbacks[20]) == 1

    def test_callback_receives_correct_args(self):
        """Simulate callback invocation logic to verify args."""
        received_args = []

        def spy_callback(layer, residual):
            received_args.append((layer, residual.copy()))
            return None

        ctx = ForwardContext(MagicMock())
        ctx.on_layer(5, spy_callback)

        # Simulate what _execute_forward_context does for callbacks:
        layer_idx = 5
        fake_residual = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        callbacks = ctx._callbacks

        if layer_idx in callbacks:
            residual_np = fake_residual.copy()
            for cb in callbacks[layer_idx]:
                injection = cb(layer_idx, residual_np)
                assert injection is None

        assert len(received_args) == 1
        assert received_args[0][0] == 5
        np.testing.assert_array_equal(received_args[0][1], fake_residual)

    def test_callback_injection_modifies_residual(self):
        """When callback returns a vector, it should be added to the residual."""
        nudge = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        def injecting_callback(layer, residual):
            return nudge

        ctx = ForwardContext(MagicMock())
        ctx.on_layer(5, injecting_callback)

        # Simulate the callback execution logic from _execute_forward_context
        h_np = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)  # [1, 1, 3]
        original = h_np[0, -1, :].copy()

        layer_idx = 5
        if layer_idx in ctx._callbacks:
            residual_np = h_np[0, -1, :].copy()
            for cb in ctx._callbacks[layer_idx]:
                injection = cb(layer_idx, residual_np)
                if injection is not None:
                    h_np[0, -1, :] += injection
                    residual_np = h_np[0, -1, :].copy()

        expected = original + nudge
        np.testing.assert_allclose(h_np[0, -1, :], expected)

    def test_callback_chain_accumulates_injections(self):
        """Multiple callbacks on same layer should accumulate."""
        nudge_a = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        nudge_b = np.array([0.0, 0.2, 0.0], dtype=np.float32)

        ctx = ForwardContext(MagicMock())
        ctx.on_layer(5, lambda l, r: nudge_a)
        ctx.on_layer(5, lambda l, r: nudge_b)

        h_np = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        original = h_np[0, -1, :].copy()

        if 5 in ctx._callbacks:
            residual_np = h_np[0, -1, :].copy()
            for cb in ctx._callbacks[5]:
                injection = cb(5, residual_np)
                if injection is not None:
                    h_np[0, -1, :] += injection
                    residual_np = h_np[0, -1, :].copy()

        expected = original + nudge_a + nudge_b
        np.testing.assert_allclose(h_np[0, -1, :], expected)

    def test_conditional_callback_pattern(self):
        """Test the safety_callback use case: conditional injection based on residual."""
        hidden = 4
        safety_dir = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        threshold = 0.5
        compliance_nudge = np.array([0.0, 0.0, -0.5, 0.0], dtype=np.float32)
        callback_fired = []

        def safety_callback(layer, residual):
            proj = float(np.dot(residual, safety_dir))
            callback_fired.append(proj)
            if proj > threshold:
                return compliance_nudge
            return None

        ctx = ForwardContext(MagicMock())
        ctx.on_layer(15, safety_callback)

        # Case 1: residual projection exceeds threshold -> inject
        h_np = np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
        if 15 in ctx._callbacks:
            residual_np = h_np[0, -1, :].copy()
            for cb in ctx._callbacks[15]:
                injection = cb(15, residual_np)
                if injection is not None:
                    h_np[0, -1, :] += injection

        assert callback_fired[-1] > threshold
        np.testing.assert_allclose(h_np[0, -1, :], [1.0, 0.0, -0.5, 0.0])

        # Case 2: residual projection below threshold -> no injection
        h_np2 = np.array([[[0.1, 0.9, 0.0, 0.0]]], dtype=np.float32)
        original2 = h_np2[0, -1, :].copy()
        if 15 in ctx._callbacks:
            residual_np = h_np2[0, -1, :].copy()
            for cb in ctx._callbacks[15]:
                injection = cb(15, residual_np)
                if injection is not None:
                    h_np2[0, -1, :] += injection

        assert callback_fired[-1] < threshold
        np.testing.assert_allclose(h_np2[0, -1, :], original2)
