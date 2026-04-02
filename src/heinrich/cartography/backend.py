"""Backend abstraction — decouple cartography from MLX.

Defines a Backend protocol that cartography modules call instead of touching
MLX directly. Implementations for MLX (default) and HuggingFace transformers.

An agent running heinrich against Llama on a Linux box with CUDA uses the
HFBackend. On Apple Silicon with MLX, uses MLXBackend. Same cartography code.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
import numpy as np

from .model_config import ModelConfig, detect_config


@dataclass
class ForwardResult:
    """Result of a forward pass through the model."""
    logits: np.ndarray         # [vocab_size] last-position logits
    probs: np.ndarray          # [vocab_size] softmax probabilities
    top_id: int
    top_token: str
    entropy: float
    n_tokens: int
    residual: np.ndarray | None = None  # [hidden_size] if requested


@runtime_checkable
class Backend(Protocol):
    """Protocol for model inference backends."""

    config: ModelConfig

    def forward(
        self,
        prompt: str,
        *,
        steer_dirs: dict[int, tuple[np.ndarray, float]] | None = None,
        alpha: float = 0.0,
        return_residual: bool = False,
        residual_layer: int = -1,
    ) -> ForwardResult: ...

    def generate(
        self,
        prompt: str,
        *,
        steer_dirs: dict[int, tuple[np.ndarray, float]] | None = None,
        alpha: float = 0.0,
        max_tokens: int = 30,
    ) -> str: ...

    def capture_residual_states(
        self,
        prompts: list[str],
        *,
        layers: list[int],
    ) -> dict[int, np.ndarray]: ...

    def capture_mlp_activations(
        self,
        prompt: str,
        layer: int,
    ) -> np.ndarray: ...

    def capture_all_positions(
        self,
        prompt: str,
        *,
        layers: list[int],
    ) -> dict[int, np.ndarray]: ...
    """Capture residual stream at ALL token positions.
    Returns {layer: array[n_tokens, hidden_size]}.
    """

    def capture_mlp_detail(
        self,
        prompt: str,
        layer: int,
    ) -> dict[str, np.ndarray]: ...
    """Capture gate, up, activated, and down projections separately.
    Returns {"gate": [...], "up": [...], "activated": [...], "output": [...]}.
    """

    def forward_with_neuron_mask(
        self,
        prompt: str,
        layer: int,
        neuron_indices: list[int],
        *,
        return_residual: bool = False,
    ) -> ForwardResult: ...
    """Forward pass with specific MLP neurons zeroed at target layer."""

    def perturb_head(
        self,
        prompt: str,
        layer: int,
        head: int,
        *,
        mode: str = "zero",
        scale: float = 0.0,
    ) -> ForwardResult: ...
    """Forward pass with a single attention head zeroed/scaled.
    mode: "zero", "scale", "negate", "double".
    """

    def weight_projection(
        self,
        layer: int,
        neuron_index: int,
    ) -> np.ndarray: ...
    """Extract the gate_proj weight row for a neuron — no forward pass.
    Returns [hidden_size] vector for embedding-space scoring.
    """

    def tokenize(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...

    def capabilities(self) -> "Capabilities": ...

    def forward_context(self) -> "ForwardContext": ...

    def generation_context(self, prompt: str) -> "GenerationContext": ...


class MLXBackend:
    """MLX backend — runs on Apple Silicon via mlx-lm."""

    def __init__(self, model_id: str):
        import mlx_lm
        self.model, self.tokenizer = mlx_lm.load(model_id)
        self.config = detect_config(self.model, self.tokenizer)
        self._inner = getattr(self.model, "model", self.model)

    def forward(
        self,
        prompt: str,
        *,
        steer_dirs=None,
        alpha=0.0,
        return_residual=False,
        residual_layer=-1,
    ) -> ForwardResult:
        from .runtime import forward_pass
        result = forward_pass(
            self.model, self.tokenizer, prompt,
            steer_dirs=steer_dirs, alpha=alpha,
            return_residual=return_residual,
            residual_layer=residual_layer,
        )
        return ForwardResult(
            logits=result["logits"],
            probs=result["probs"],
            top_id=result["top_id"],
            top_token=result["top_token"],
            entropy=result["entropy"],
            n_tokens=result["n_tokens"],
            residual=result.get("residual"),
        )

    def generate(self, prompt, *, steer_dirs=None, alpha=0.0, max_tokens=30) -> str:
        from .runtime import generate as _generate
        result = _generate(
            self.model, self.tokenizer, prompt,
            steer_dirs=steer_dirs, alpha=alpha, max_tokens=max_tokens,
        )
        return result["generated"]

    def capture_residual_states(self, prompts, *, layers) -> dict[int, np.ndarray]:
        from .directions import capture_residual_states as _capture
        return _capture(self.model, self.tokenizer, prompts, layers=layers)

    def capture_mlp_activations(self, prompt, layer) -> np.ndarray:
        from .neurons import capture_mlp_activations as _capture
        return _capture(self.model, self.tokenizer, prompt, layer)

    def capture_all_positions(self, prompt, *, layers) -> dict[int, np.ndarray]:
        import mlx.core as mx
        from .perturb import _mask_dtype

        inner = self._inner
        mdtype = _mask_dtype(self.model)
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        layer_set = set(layers)
        states = {}
        h = inner.embed_tokens(input_ids)
        for i, ly in enumerate(inner.layers):
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]
            if i in layer_set:
                states[i] = np.array(h.astype(mx.float32)[0])  # [T, hidden]
        return states

    def capture_mlp_detail(self, prompt, layer) -> dict[str, np.ndarray]:
        import mlx.core as mx
        import mlx.nn as nn
        from .perturb import _mask_dtype

        inner = self._inner
        mdtype = _mask_dtype(self.model)
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        h = inner.embed_tokens(input_ids)
        for i, ly in enumerate(inner.layers):
            if i == layer:
                h_normed = ly.post_attention_layernorm(h) if hasattr(ly, 'post_attention_layernorm') else h
                gate = ly.mlp.gate_proj(h_normed)
                up = ly.mlp.up_proj(h_normed)
                activated = nn.silu(gate) * up
                output = ly.mlp.down_proj(activated)
                return {
                    "gate": np.array(gate.astype(mx.float32)[0, -1, :]),
                    "up": np.array(up.astype(mx.float32)[0, -1, :]),
                    "activated": np.array(activated.astype(mx.float32)[0, -1, :]),
                    "output": np.array(output.astype(mx.float32)[0, -1, :]),
                }
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]
        return {}

    def forward_with_neuron_mask(self, prompt, layer, neuron_indices, *, return_residual=False) -> ForwardResult:
        import mlx.core as mx
        import mlx.nn as nn
        from .perturb import _mask_dtype
        from .metrics import softmax, entropy as _entropy

        inner = self._inner
        mdtype = _mask_dtype(self.model)
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        h = inner.embed_tokens(input_ids)
        for i, ly in enumerate(inner.layers):
            if i == layer:
                # Manual MLP with neuron zeroing
                h_normed = ly.input_layernorm(h) if hasattr(ly, 'input_layernorm') else h
                attn_out = ly.self_attn(h_normed, mask=mask, cache=None)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                h = h + attn_out

                h_normed2 = ly.post_attention_layernorm(h) if hasattr(ly, 'post_attention_layernorm') else h
                gate = ly.mlp.gate_proj(h_normed2)
                up = ly.mlp.up_proj(h_normed2)
                activated = nn.silu(gate) * up

                # Zero target neurons
                act_np = np.array(activated.astype(mx.float32))
                for n in neuron_indices:
                    act_np[0, :, n] = 0.0
                activated = mx.array(act_np.astype(np.float16))

                mlp_out = ly.mlp.down_proj(activated)
                h = h + mlp_out
            else:
                h = ly(h, mask=mask, cache=None)
                if isinstance(h, tuple):
                    h = h[0]

        residual = np.array(h.astype(mx.float32)[0, -1, :]) if return_residual else None
        h = inner.norm(h)
        logits = np.array(self.model.lm_head(h).astype(mx.float32)[0, -1, :])
        probs = softmax(logits)
        top_id = int(np.argmax(probs))

        return ForwardResult(
            logits=logits, probs=probs, top_id=top_id,
            top_token=self.tokenizer.decode([top_id]),
            entropy=_entropy(probs), n_tokens=len(tokens),
            residual=residual,
        )

    def perturb_head(self, prompt, layer, head, *, mode="zero", scale=0.0) -> ForwardResult:
        from .perturb import perturb_head as _perturb, measure_perturbation
        from .surface import Knob
        from .metrics import softmax, entropy as _entropy

        baseline, perturbed = _perturb(
            self.model, self.tokenizer, prompt, layer, head,
            mode=mode, scale=scale,
        )
        probs = softmax(perturbed)
        top_id = int(np.argmax(probs))
        return ForwardResult(
            logits=perturbed, probs=probs, top_id=top_id,
            top_token=self.tokenizer.decode([top_id]),
            entropy=_entropy(probs),
            n_tokens=len(self.tokenizer.encode(prompt)),
        )

    def weight_projection(self, layer, neuron_index) -> np.ndarray:
        import mlx.core as mx

        inner = self._inner
        hidden_size = self.config.hidden_size
        mlp = inner.layers[layer].mlp

        weights = np.zeros(hidden_size)
        batch = 128
        for start in range(0, hidden_size, batch):
            end = min(start + batch, hidden_size)
            inp = np.zeros((1, end - start, hidden_size), dtype=np.float16)
            for j in range(end - start):
                inp[0, j, start + j] = 1.0
            gate_out = mlp.gate_proj(mx.array(inp))
            gate_np = np.array(gate_out.astype(mx.float32)[0, :, neuron_index])
            weights[start:end] = gate_np
        return weights

    def capabilities(self):
        from .context import Capabilities
        return Capabilities(
            can_steer=True, can_capture_residual=True, can_capture_attention=True,
            can_capture_mlp_detail=True, can_neuron_mask=True, can_perturb_head=True,
            can_weight_access=True, can_embedding_access=True, can_logit_lens=True,
            can_kv_cache=True, can_gradient=False, can_batch=False,
            can_all_positions=True, can_compose=True, can_gen_control=True,
            can_multi_turn=True,
        )

    def forward_context(self):
        from .context import ForwardContext
        return ForwardContext(self)

    def generation_context(self, prompt):
        from .context import GenerationContext
        return GenerationContext(self, prompt)

    def _execute_forward_context(self, prompt, ctx):
        """Compile ForwardContext declarations into a single MLX layer loop.

        Supports:
        - Compositional steer + capture + ablate in one pass
        - Attention weight capture via manual Q/K decomposition with RoPE/GQA
        - Conditional callbacks that fire after a layer and can inject vectors
        """
        import mlx.core as mx
        import mlx.nn as nn
        from .perturb import _mask_dtype
        from .metrics import softmax, entropy as _entropy
        from .context import ContextResult

        inner = self._inner
        mdtype = _mask_dtype(self.model)
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        # Build lookup tables from declarations
        steer_at = {}
        for op in ctx._steers:
            steer_at[op.layer] = (op.direction, op.mean_gap, op.alpha)

        neuron_masks = {}
        for op in ctx._neuron_masks:
            neuron_masks.setdefault(op.layer, []).extend(op.neurons)

        capture_residual_layers = set()
        capture_all_pos_layers = set()
        for op in ctx._capture_residuals:
            for l in op.layers:
                if op.all_positions:
                    capture_all_pos_layers.add(l)
                else:
                    capture_residual_layers.add(l)

        capture_attn_layers = {op.layer for op in ctx._capture_attentions}
        capture_mlp_layers = {op.layer for op in ctx._capture_mlp_details}

        # Collect callbacks
        callbacks = ctx._callbacks  # {layer: [callable, ...]}

        # Execute single forward pass
        residuals = {}
        all_pos_residuals = {}
        attentions = {}
        mlp_details = {}

        h = inner.embed_tokens(input_ids)
        mx.eval(h)
        n_layers = len(inner.layers)
        for i, ly in enumerate(inner.layers):
            # Attention capture: manual Q/K decomposition for attention weights
            if i in capture_attn_layers:
                attn = ly.self_attn
                h_normed = ly.input_layernorm(h) if hasattr(ly, 'input_layernorm') else h

                q = attn.q_proj(h_normed)
                k = attn.k_proj(h_normed)

                n_heads = attn.n_heads
                n_kv_heads = attn.n_kv_heads
                head_dim = q.shape[-1] // n_heads

                q = q.reshape(1, T, n_heads, head_dim).transpose(0, 2, 1, 3)
                k = k.reshape(1, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

                if hasattr(attn, 'rope'):
                    q = attn.rope(q)
                    k = attn.rope(k)

                # Expand KV heads for GQA
                n_rep = n_heads // n_kv_heads
                if n_rep > 1:
                    k = mx.repeat(k, repeats=n_rep, axis=1)

                # Compute attention scores for last query position only
                # q_last shape: [1, n_heads, 1, head_dim]
                # k transposed: [1, n_heads, head_dim, T]
                q_last = q[:, :, -1:, :]
                scores = (q_last @ k.transpose(0, 1, 3, 2)) * attn.scale  # [1, n_heads, 1, T]
                # Apply causal mask at last position (all positions visible)
                weights = mx.softmax(scores, axis=-1)
                # Store as [n_heads, T] — last query position attending to all keys
                attentions[i] = np.array(weights.astype(mx.float32)[0, :, 0, :])

            # Neuron masking requires manual MLP decomposition
            if i in neuron_masks or i in capture_mlp_layers:
                # Manual layer: attention + MLP separately
                h_normed = ly.input_layernorm(h) if hasattr(ly, 'input_layernorm') else h
                attn_out = ly.self_attn(h_normed, mask=mask, cache=None)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                h = h + attn_out

                h_normed2 = ly.post_attention_layernorm(h) if hasattr(ly, 'post_attention_layernorm') else h
                gate = ly.mlp.gate_proj(h_normed2)
                up = ly.mlp.up_proj(h_normed2)
                activated = nn.silu(gate) * up

                if i in capture_mlp_layers:
                    mlp_details[i] = {
                        "gate": np.array(gate.astype(mx.float32)[0, -1, :]),
                        "up": np.array(up.astype(mx.float32)[0, -1, :]),
                        "activated": np.array(activated.astype(mx.float32)[0, -1, :]),
                    }

                if i in neuron_masks:
                    act_np = np.array(activated.astype(mx.float32))
                    for n in neuron_masks[i]:
                        act_np[0, :, n] = 0.0
                    activated = mx.array(act_np.astype(np.float16))

                mlp_out = ly.mlp.down_proj(activated)
                if i in capture_mlp_layers:
                    mlp_details[i]["output"] = np.array(mlp_out.astype(mx.float32)[0, -1, :])
                h = h + mlp_out
            else:
                h = ly(h, mask=mask, cache=None)
                if isinstance(h, tuple):
                    h = h[0]

            # Steering
            if i in steer_at:
                direction, mean_gap, alpha = steer_at[i]
                h_np = np.array(h.astype(mx.float32))
                h_np[0, -1, :] += direction * mean_gap * alpha
                h = mx.array(h_np.astype(np.float16))

            # Callbacks — fire after layer i, may inject into residual stream
            if i in callbacks:
                residual_np = np.array(h.astype(mx.float32)[0, -1, :])
                for cb in callbacks[i]:
                    injection = cb(i, residual_np)
                    if injection is not None:
                        h_np = np.array(h.astype(mx.float32))
                        h_np[0, -1, :] += injection
                        h = mx.array(h_np.astype(np.float16))
                        # Update residual for subsequent callbacks on same layer
                        residual_np = h_np[0, -1, :]

            # Capture residuals
            if i in capture_residual_layers:
                residuals[i] = np.array(h.astype(mx.float32)[0, -1, :])
            if i in capture_all_pos_layers:
                all_pos_residuals[i] = np.array(h.astype(mx.float32)[0])

            # Force evaluation every 8 layers to bound the computation graph
            # size and free intermediate tensors. Layers that had np.array()
            # conversions (captures, steering, callbacks) already triggered
            # evaluation implicitly; this covers runs of plain layers.
            if i % 8 == 7 or i == n_layers - 1:
                mx.eval(h)

        h = inner.norm(h)
        logits = np.array(self.model.lm_head(h).astype(mx.float32)[0, -1, :])
        probs = softmax(logits)
        top_id = int(np.argmax(probs))

        return ContextResult(
            logits=logits, probs=probs, top_id=top_id,
            top_token=self.tokenizer.decode([top_id]),
            entropy=_entropy(probs), n_tokens=len(tokens),
            residuals=residuals, all_position_residuals=all_pos_residuals,
            attention=attentions, mlp_detail=mlp_details,
        )

    def _execute_generation_context(self, prompt, gen_ctx, *, max_tokens=200):
        """Execute GenerationContext as a token-by-token MLX generation loop.

        Uses KV cache for O(n) generation instead of O(n^2) recomputation.
        Step 0 processes the full prompt and builds the cache; subsequent steps
        process only the new token with cached KV pairs.

        Performance: For a 100-token prompt generating 100 tokens, the old
        approach recomputes all layers for sequences of length 101..200
        (total ~15k layer calls). With KV cache, step 0 processes 100 tokens
        once, then each subsequent step processes 1 token (~3.2k layer calls,
        ~4.7x speedup for this example; grows with sequence length).
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.models.base import create_attention_mask
        from .context import TokenResult

        inner = self._inner
        tokens = list(self.tokenizer.encode(prompt))
        eos = getattr(self.tokenizer, "eos_token_id", None)

        steer_at = {}
        for op in gen_ctx._steers:
            steer_at[op.layer] = (op.direction, op.mean_gap, op.alpha)

        capture_layer = gen_ctx._capture_layer

        # Initialize KV cache — one cache object per layer
        cache = make_prompt_cache(inner)

        for step in range(max_tokens):
            if step == 0:
                # First step: process full prompt, populate cache
                input_ids = mx.array([tokens])
            else:
                # Subsequent steps: only feed the new token
                input_ids = mx.array([[tokens[-1]]])

            h = inner.embed_tokens(input_ids)
            # create_attention_mask handles cache offset: returns None for
            # single-token input when cache is populated (no mask needed)
            mask = create_attention_mask(h, cache[0])

            residual = None
            for i, ly in enumerate(inner.layers):
                h = ly(h, mask=mask, cache=cache[i])
                if isinstance(h, tuple):
                    h = h[0]

                # Steering — in cached mode h is [1, 1, hidden], so h[0, -1, :]
                # and h[0, 0, :] are equivalent; use -1 for consistency
                if i in steer_at:
                    direction, mean_gap, alpha = steer_at[i]
                    h_np = np.array(h.astype(mx.float32))
                    h_np[0, -1, :] += direction * mean_gap * alpha
                    h = mx.array(h_np.astype(np.float16))

                # One-shot injections
                for inj_layer, inj_vec in gen_ctx._one_shot_injections:
                    if i == inj_layer:
                        h_np = np.array(h.astype(mx.float32))
                        h_np[0, -1, :] += inj_vec
                        h = mx.array(h_np.astype(np.float16))

                if capture_layer is not None and i == capture_layer:
                    residual = np.array(h.astype(mx.float32)[0, -1, :])

            # Clear one-shot injections after use
            gen_ctx._one_shot_injections.clear()

            h = inner.norm(h)
            logits = np.array(self.model.lm_head(h).astype(mx.float32)[0, -1, :])
            next_id = int(np.argmax(logits))

            if next_id == eos:
                break

            tokens.append(next_id)
            yield TokenResult(
                step=step,
                token_id=next_id,
                token_text=self.tokenizer.decode([next_id]),
                residual=residual,
                logits=logits,
            )

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


class HFBackend:
    """HuggingFace transformers backend — runs on CUDA/CPU via transformers."""

    def __init__(self, model_id: str, *, device: str = "auto", torch_dtype: str = "float16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        resolved_dtype = dtype_map.get(torch_dtype, torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=resolved_dtype,
            device_map=device,
        )
        self.hf_model.eval()
        self.config = detect_config(self.hf_model, self.tokenizer)
        self._device = next(self.hf_model.parameters()).device

    def _install_steer_hooks(
        self,
        steer_dirs: dict[int, tuple[np.ndarray, float]],
        alpha: float,
    ) -> list:
        """Register forward hooks that inject steering directions.

        Returns list of hook handles (caller must remove them).

        Closure scoping note: Each hook's direction tensor is captured via the
        ``make_hook(dt)`` factory function, so each layer correctly receives its
        own pre-computed ``direction * mean_gap * alpha`` vector even when hooks
        are created inside a loop.

        KV-cache interaction: When used during ``generate()`` with
        ``use_cache=True``, the steered hidden states flow into the KV cache.
        This means subsequent tokens attend to the *steered* representations,
        not the original ones. This is the desired behavior for attack
        simulation (the model "commits" to the steered trajectory), but it
        means you cannot compare steered vs. unsteered generation by toggling
        hooks mid-generation — you must run two separate generation calls.
        """
        import torch
        handles = []
        layers = self.hf_model.model.layers

        for layer_idx, (direction, mean_gap) in steer_dirs.items():
            if layer_idx >= len(layers):
                continue
            dir_tensor = torch.tensor(
                direction * mean_gap * alpha,
                dtype=torch.float32,
            )

            def make_hook(dt):
                def hook_fn(module, input, output):
                    # output is (hidden_states, ...) or just hidden_states
                    if isinstance(output, tuple):
                        hs = output[0]
                    else:
                        hs = output
                    device = hs.device
                    hs_dtype = hs.dtype
                    steer = dt.to(device)
                    # Add steering vector to last token position
                    hs[:, -1, :] = hs[:, -1, :].float() + steer
                    hs[:, -1, :] = hs[:, -1, :].to(hs_dtype)
                    if isinstance(output, tuple):
                        return (hs,) + output[1:]
                    return hs
                return hook_fn

            handle = layers[layer_idx].register_forward_hook(make_hook(dir_tensor))
            handles.append(handle)

        return handles

    def forward(
        self,
        prompt: str,
        *,
        steer_dirs=None,
        alpha=0.0,
        return_residual=False,
        residual_layer=-1,
    ) -> ForwardResult:
        import torch
        from .metrics import softmax, entropy as _entropy

        hooks = []
        if steer_dirs and alpha != 0:
            hooks = self._install_steer_hooks(steer_dirs, alpha)

        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self.hf_model(
                    input_ids,
                    output_hidden_states=return_residual,
                )
                logits = outputs.logits[0, -1, :].float().cpu().numpy()
        finally:
            for h in hooks:
                h.remove()

        probs = softmax(logits)
        top_id = int(np.argmax(probs))

        residual = None
        if return_residual and outputs.hidden_states:
            layer_idx = residual_layer if residual_layer >= 0 else -1
            residual = outputs.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()

        return ForwardResult(
            logits=logits,
            probs=probs,
            top_id=top_id,
            top_token=self.tokenizer.decode([top_id]),
            entropy=_entropy(probs),
            n_tokens=input_ids.shape[1],
            residual=residual,
        )

    def generate(self, prompt, *, steer_dirs=None, alpha=0.0, max_tokens=30) -> str:
        import torch

        hooks = []
        if steer_dirs and alpha != 0:
            hooks = self._install_steer_hooks(steer_dirs, alpha)

        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                output_ids = self.hf_model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
        finally:
            for h in hooks:
                h.remove()

        new_ids = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def capture_residual_states(self, prompts, *, layers) -> dict[int, np.ndarray]:
        import torch

        all_states: dict[int, list[np.ndarray]] = {l: [] for l in layers}

        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self.hf_model(input_ids, output_hidden_states=True)
            for l in layers:
                # hidden_states[0] is embedding, hidden_states[l+1] is after layer l
                idx = l + 1 if l + 1 < len(outputs.hidden_states) else -1
                state = outputs.hidden_states[idx][0, -1, :].float().cpu().numpy()
                all_states[l].append(state)

        return {l: np.array(vecs) for l, vecs in all_states.items()}

    def capture_mlp_activations(self, prompt, layer) -> np.ndarray:
        import torch

        activations = {}

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations["mlp"] = output[0][0, -1, :].float().cpu().numpy()
            else:
                activations["mlp"] = output[0, -1, :].float().cpu().numpy()

        # Register hook on the MLP module of the target layer
        mlp = self.hf_model.model.layers[layer].mlp
        handle = mlp.register_forward_hook(hook_fn)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            self.hf_model(input_ids)

        handle.remove()
        return activations.get("mlp", np.zeros(self.config.intermediate_size))

    def capture_all_positions(self, prompt, *, layers) -> dict[int, np.ndarray]:
        import torch
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self.hf_model(input_ids, output_hidden_states=True)
        states = {}
        for l in layers:
            idx = l + 1 if l + 1 < len(outputs.hidden_states) else -1
            states[l] = outputs.hidden_states[idx][0].float().cpu().numpy()  # [T, hidden]
        return states

    def capture_mlp_detail(self, prompt, layer) -> dict[str, np.ndarray]:
        import torch
        captures = {}

        def gate_hook(mod, inp, out):
            captures["gate"] = out[0, -1, :].float().cpu().numpy() if out.dim() == 3 else out[-1, :].float().cpu().numpy()

        def up_hook(mod, inp, out):
            captures["up"] = out[0, -1, :].float().cpu().numpy() if out.dim() == 3 else out[-1, :].float().cpu().numpy()

        def down_hook(mod, inp, out):
            captures["output"] = out[0, -1, :].float().cpu().numpy() if out.dim() == 3 else out[-1, :].float().cpu().numpy()

        mlp = self.hf_model.model.layers[layer].mlp
        handles = []
        if hasattr(mlp, 'gate_proj'):
            handles.append(mlp.gate_proj.register_forward_hook(gate_hook))
        if hasattr(mlp, 'up_proj'):
            handles.append(mlp.up_proj.register_forward_hook(up_hook))
        if hasattr(mlp, 'down_proj'):
            handles.append(mlp.down_proj.register_forward_hook(down_hook))

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            self.hf_model(input_ids)

        for h in handles:
            h.remove()

        if "gate" in captures and "up" in captures:
            import torch.nn.functional as F
            gate_t = torch.tensor(captures["gate"])
            up_t = torch.tensor(captures["up"])
            captures["activated"] = (F.silu(gate_t) * up_t).numpy()

        return captures

    def forward_with_neuron_mask(self, prompt, layer, neuron_indices, *, return_residual=False) -> ForwardResult:
        import torch
        from .metrics import softmax, entropy as _entropy

        def make_mask_hook(indices):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hs = output[0]
                else:
                    hs = output
                for n in indices:
                    hs[:, :, n] = 0.0
                return (hs,) + output[1:] if isinstance(output, tuple) else hs
            return hook_fn

        mlp = self.hf_model.model.layers[layer].mlp
        handle = mlp.register_forward_hook(make_mask_hook(neuron_indices))

        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self.hf_model(input_ids, output_hidden_states=return_residual)
                logits = outputs.logits[0, -1, :].float().cpu().numpy()
        finally:
            handle.remove()

        probs = softmax(logits)
        top_id = int(np.argmax(probs))
        residual = None
        if return_residual and outputs.hidden_states:
            residual = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()

        return ForwardResult(
            logits=logits, probs=probs, top_id=top_id,
            top_token=self.tokenizer.decode([top_id]),
            entropy=_entropy(probs),
            n_tokens=input_ids.shape[1],
            residual=residual,
        )

    def perturb_head(self, prompt, layer, head, *, mode="zero", scale=0.0) -> ForwardResult:
        import torch
        from .metrics import softmax, entropy as _entropy

        n_heads = self.config.n_heads
        head_dim = self.config.head_dim

        def make_head_hook(h_idx, m, s):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hs = output[0]
                else:
                    hs = output
                start = h_idx * head_dim
                end = start + head_dim
                if m == "zero":
                    hs[:, :, start:end] = 0
                elif m == "scale":
                    hs[:, :, start:end] *= s
                elif m == "negate":
                    hs[:, :, start:end] *= -1
                elif m == "double":
                    hs[:, :, start:end] *= 2
                return (hs,) + output[1:] if isinstance(output, tuple) else hs
            return hook_fn

        attn = self.hf_model.model.layers[layer].self_attn
        handle = attn.o_proj.register_forward_hook(make_head_hook(head, mode, scale))

        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self.hf_model(input_ids)
                logits = outputs.logits[0, -1, :].float().cpu().numpy()
        finally:
            handle.remove()

        probs = softmax(logits)
        top_id = int(np.argmax(probs))
        return ForwardResult(
            logits=logits, probs=probs, top_id=top_id,
            top_token=self.tokenizer.decode([top_id]),
            entropy=_entropy(probs),
            n_tokens=input_ids.shape[1],
        )

    def weight_projection(self, layer, neuron_index) -> np.ndarray:
        mlp = self.hf_model.model.layers[layer].mlp
        if hasattr(mlp, 'gate_proj') and hasattr(mlp.gate_proj, 'weight'):
            return mlp.gate_proj.weight[neuron_index].float().cpu().numpy()
        return np.zeros(self.config.hidden_size)

    def capabilities(self):
        from .context import Capabilities
        return Capabilities(
            can_steer=True, can_capture_residual=True, can_capture_attention=True,
            can_capture_mlp_detail=True, can_neuron_mask=True, can_perturb_head=True,
            can_weight_access=True, can_embedding_access=True, can_logit_lens=True,
            can_kv_cache=True, can_gradient=True, can_batch=True,
            can_all_positions=True, can_compose=True, can_gen_control=True,
            can_multi_turn=True,
        )

    def forward_context(self):
        from .context import ForwardContext
        return ForwardContext(self)

    def generation_context(self, prompt):
        from .context import GenerationContext
        return GenerationContext(self, prompt)

    def _execute_forward_context(self, prompt, ctx):
        """Compile ForwardContext into HF hooks + single forward pass."""
        import torch
        from .metrics import softmax, entropy as _entropy
        from .context import ContextResult

        handles = []
        captures = {"residuals": {}, "all_pos": {}, "attn": {}, "mlp": {}}

        # Build steer hooks
        for op in ctx._steers:
            if op.layer < len(self.hf_model.model.layers):
                dt = torch.tensor(op.direction * op.mean_gap * op.alpha, dtype=torch.float32)
                def make_steer(d):
                    def hook(mod, inp, out):
                        hs = out[0] if isinstance(out, tuple) else out
                        hs[:, -1, :] = hs[:, -1, :].float() + d.to(hs.device)
                        return (hs,) + out[1:] if isinstance(out, tuple) else hs
                    return hook
                h = self.hf_model.model.layers[op.layer].register_forward_hook(make_steer(dt))
                handles.append(h)

        # Build neuron mask hooks
        for op in ctx._neuron_masks:
            if op.layer < len(self.hf_model.model.layers):
                def make_mask(neurons):
                    def hook(mod, inp, out):
                        hs = out[0] if isinstance(out, tuple) else out
                        for n in neurons:
                            hs[:, :, n] = 0
                        return (hs,) + out[1:] if isinstance(out, tuple) else hs
                    return hook
                h = self.hf_model.model.layers[op.layer].mlp.register_forward_hook(make_mask(op.neurons))
                handles.append(h)

        # Build MLP detail capture hooks
        for op in ctx._capture_mlp_details:
            layer = op.layer
            mlp = self.hf_model.model.layers[layer].mlp
            cap = {}
            if hasattr(mlp, 'gate_proj'):
                def make_gate_hook(c):
                    def hook(mod, inp, out):
                        c["gate"] = out[0, -1, :].float().cpu().numpy() if out.dim() == 3 else out[-1, :].float().cpu().numpy()
                    return hook
                handles.append(mlp.gate_proj.register_forward_hook(make_gate_hook(cap)))
            if hasattr(mlp, 'up_proj'):
                def make_up_hook(c):
                    def hook(mod, inp, out):
                        c["up"] = out[0, -1, :].float().cpu().numpy() if out.dim() == 3 else out[-1, :].float().cpu().numpy()
                    return hook
                handles.append(mlp.up_proj.register_forward_hook(make_up_hook(cap)))
            captures["mlp"][layer] = cap

        try:
            # Determine what outputs we need
            need_hidden = any(ctx._capture_residuals) or True
            need_attn = bool(ctx._capture_attentions)

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self.hf_model(
                    input_ids,
                    output_hidden_states=need_hidden,
                    output_attentions=need_attn,
                )
            logits = outputs.logits[0, -1, :].float().cpu().numpy()

            # Collect captures
            residuals = {}
            all_pos = {}
            for op in ctx._capture_residuals:
                for l in op.layers:
                    idx = l + 1 if l + 1 < len(outputs.hidden_states) else -1
                    if op.all_positions:
                        all_pos[l] = outputs.hidden_states[idx][0].float().cpu().numpy()
                    else:
                        residuals[l] = outputs.hidden_states[idx][0, -1, :].float().cpu().numpy()

            attentions = {}
            for op in ctx._capture_attentions:
                if outputs.attentions and op.layer < len(outputs.attentions):
                    attentions[op.layer] = outputs.attentions[op.layer][0].float().cpu().numpy()

            # Compute activated for MLP captures
            mlp_details = {}
            for layer, cap in captures["mlp"].items():
                if "gate" in cap and "up" in cap:
                    import torch.nn.functional as F
                    g = torch.tensor(cap["gate"])
                    u = torch.tensor(cap["up"])
                    cap["activated"] = (F.silu(g) * u).numpy()
                mlp_details[layer] = cap

        finally:
            for h in handles:
                h.remove()

        probs = softmax(logits)
        top_id = int(np.argmax(probs))

        return ContextResult(
            logits=logits, probs=probs, top_id=top_id,
            top_token=self.tokenizer.decode([top_id]),
            entropy=_entropy(probs), n_tokens=input_ids.shape[1],
            residuals=residuals, all_position_residuals=all_pos,
            attention=attentions, mlp_detail=mlp_details,
        )

    def _execute_generation_context(self, prompt, gen_ctx, *, max_tokens=200):
        """Execute GenerationContext with HF model, token-by-token.

        KV-cache gotcha: Steering hooks modify hidden states *before* they are
        cached in ``past_key_values``. This means the KV cache contains steered
        representations, so every subsequent token attends to the steered
        trajectory. This is correct for attack simulation but prevents
        mid-generation comparison of steered vs. unsteered states. To compare,
        run two separate generation calls.

        Closure scoping: Each hook's direction tensor is captured via the
        ``make_steer(d)`` factory, avoiding the classic Python loop-closure bug.
        """
        import torch
        from .context import TokenResult

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        past = None
        eos = getattr(self.tokenizer, "eos_token_id", None)

        # Persistent steer hooks
        handles = []
        for op in gen_ctx._steers:
            if op.layer < len(self.hf_model.model.layers):
                dt = torch.tensor(op.direction * op.mean_gap * op.alpha, dtype=torch.float32)
                def make_steer(d):
                    def hook(mod, inp, out):
                        hs = out[0] if isinstance(out, tuple) else out
                        hs[:, -1, :] = hs[:, -1, :].float() + d.to(hs.device)
                        return (hs,) + out[1:] if isinstance(out, tuple) else hs
                    return hook
                handles.append(self.hf_model.model.layers[op.layer].register_forward_hook(make_steer(dt)))

        try:
            current_ids = input_ids
            for step in range(max_tokens):
                with torch.no_grad():
                    outputs = self.hf_model(
                        current_ids,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=(gen_ctx._capture_layer is not None),
                    )

                logits = outputs.logits[0, -1, :].float().cpu().numpy()
                next_id = int(np.argmax(logits))
                past = outputs.past_key_values

                if next_id == eos:
                    break

                residual = None
                if gen_ctx._capture_layer is not None and outputs.hidden_states:
                    idx = gen_ctx._capture_layer + 1
                    if idx < len(outputs.hidden_states):
                        residual = outputs.hidden_states[idx][0, -1, :].float().cpu().numpy()

                current_ids = torch.tensor([[next_id]], device=self._device)

                yield TokenResult(
                    step=step,
                    token_id=next_id,
                    token_text=self.tokenizer.decode([next_id]),
                    residual=residual,
                    logits=logits,
                )
        finally:
            for h in handles:
                h.remove()

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


def load_backend(model_id: str, *, backend: str = "auto", **kwargs) -> MLXBackend | HFBackend:
    """Load a model with the appropriate backend.

    backend: "mlx", "hf", or "auto" (tries MLX first on macOS, falls back to HF).
    """
    if backend == "mlx":
        return MLXBackend(model_id)
    elif backend == "hf":
        return HFBackend(model_id, **kwargs)
    elif backend == "auto":
        import platform
        if platform.system() == "Darwin":
            try:
                return MLXBackend(model_id)
            except (ImportError, Exception):
                pass
        return HFBackend(model_id, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'mlx', 'hf', or 'auto'.")
