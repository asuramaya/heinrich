"""MLX backend -- runs on Apple Silicon via mlx-lm."""
from __future__ import annotations

import numpy as np

from .protocol import ForwardResult
from heinrich.cartography.model_config import detect_config


class MLXBackend:
    """MLX backend -- runs on Apple Silicon via mlx-lm."""

    def __init__(self, model_id: str):
        import mlx_lm
        self.model, self.tokenizer = mlx_lm.load(model_id)
        self.config = detect_config(self.model, self.tokenizer)
        self._inner = getattr(self.model, "model", self.model)

    def _lm_head(self, h):
        """Project hidden states to logits, handling tied-embedding models."""
        from heinrich.cartography.runtime import _lm_head
        return _lm_head(self.model, h)

    def forward(
        self,
        prompt: str,
        *,
        token_ids=None,
        steer_dirs=None,
        alpha=0.0,
        return_residual=False,
        residual_layer=-1,
    ) -> ForwardResult:
        from heinrich.cartography.runtime import forward_pass
        result = forward_pass(
            self.model, self.tokenizer, prompt,
            token_ids=token_ids,
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
        """Generate text. Uses mlx_lm fast path when no steering needed."""
        if not steer_dirs or alpha == 0:
            # Fast path: mlx_lm native generation (10-50x faster)
            import mlx_lm
            return mlx_lm.generate(
                self.model, self.tokenizer, prompt=prompt,
                max_tokens=max_tokens, verbose=False,
            )
        # Slow path: manual loop with steering via GenerationContext
        tokens = []
        with self.generation_context(prompt) as gen:
            for layer, (direction, mean_gap) in steer_dirs.items():
                gen.steer(layer, direction, mean_gap, alpha)
            for tok in gen.tokens(max_tokens=max_tokens):
                tokens.append(tok.token_text)
        return "".join(tokens)

    def generate_with_geometry(
        self, prompt, *, steer_dirs=None, alpha=0.0, max_tokens=30,
        safety_direction=None, safety_layers=None, top_k=5,
    ):
        """Generate text and capture first-token geometry in one pass.

        No separate forward() call. The first step of generation IS the
        forward pass — we capture its geometry and continue generating.
        """
        from heinrich.cartography.metrics import softmax, entropy as _entropy
        from .protocol import GenerateResult

        # Capture residual states at safety layers during the forward pass
        capture_layers = safety_layers or self.config.safety_layers

        # Use generation_context for everything — one code path, with or without steering
        tokens = []
        first_logits = None
        first_probs = None
        first_token_id = None
        residuals = {}

        with self.generation_context(prompt) as gen:
            # Set up steering
            if steer_dirs and alpha:
                for layer, (direction, mean_gap) in steer_dirs.items():
                    gen.steer(layer, direction, mean_gap, alpha)

            # Set up residual capture at the last safety layer
            # (generation_context supports one capture layer)
            if capture_layers:
                gen._capture_layer = capture_layers[-1]

            for tok in gen.tokens(max_tokens=max_tokens):
                if first_logits is None:
                    # First token: capture geometry
                    first_logits = tok.logits
                    first_probs = softmax(first_logits)
                    first_token_id = tok.token_id
                    if tok.residual is not None:
                        residuals[gen._capture_layer] = tok.residual

                tokens.append(tok.token_text)

        text = "".join(tokens)

        # Project the captured residual onto the contrastive direction.
        # generation_context captures one layer; that's the decision point.
        # No extra forward pass — use what we already have.
        contrastive_trajectory = None
        if safety_direction is not None and residuals:
            d_norm = np.linalg.norm(safety_direction)
            if d_norm > 0:
                contrastive_trajectory = [
                    float(np.dot(res, safety_direction) / d_norm)
                    for res in residuals.values()
                ]

        # Build top-k from first-token distribution
        if first_probs is not None:
            top_indices = np.argsort(first_probs)[::-1][:top_k]
            top_k_list = [
                (int(idx), self.tokenizer.decode([int(idx)]), float(first_probs[idx]))
                for idx in top_indices
            ]
        else:
            top_k_list = []

        ent = _entropy(first_probs) if first_probs is not None else 0.0
        first_token_str = self.tokenizer.decode([first_token_id]) if first_token_id is not None else ""

        return GenerateResult(
            text=text,
            first_logits=first_logits if first_logits is not None else np.array([]),
            first_probs=first_probs if first_probs is not None else np.array([]),
            first_token_id=first_token_id or 0,
            first_token=first_token_str,
            entropy=ent,
            top_k=top_k_list,
            contrastive_trajectory=contrastive_trajectory,
        )

    def capture_residual_states(self, prompts, *, layers) -> dict[int, np.ndarray]:
        from heinrich.cartography.directions import capture_residual_states as _capture
        return _capture(self.model, self.tokenizer, prompts, layers=layers)

    def capture_mlp_activations(self, prompt, layer) -> np.ndarray:
        from heinrich.cartography.neurons import capture_mlp_activations as _capture
        return _capture(self.model, self.tokenizer, prompt, layer)

    def capture_all_positions(self, prompt, *, layers) -> dict[int, np.ndarray]:
        import mlx.core as mx
        from heinrich.cartography.perturb import _mask_dtype

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
        from heinrich.cartography.perturb import _mask_dtype
        from heinrich.discover.neurons import detect_mlp_type, _compute_mlp_activated, _mlp_down_proj

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
                mlp_type = detect_mlp_type(ly)
                gate, up, activated = _compute_mlp_activated(ly.mlp, h_normed, mlp_type)
                output = _mlp_down_proj(ly.mlp, activated, mlp_type)
                result = {
                    "activated": np.array(activated.astype(mx.float32)[0, -1, :]),
                    "output": np.array(output.astype(mx.float32)[0, -1, :]),
                }
                if gate is not None:
                    result["gate"] = np.array(gate.astype(mx.float32)[0, -1, :])
                if up is not None:
                    result["up"] = np.array(up.astype(mx.float32)[0, -1, :])
                return result
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]
        return {}

    def forward_with_neuron_mask(self, prompt, layer, neuron_indices, *, return_residual=False) -> ForwardResult:
        import mlx.core as mx
        from heinrich.cartography.perturb import _mask_dtype
        from heinrich.cartography.metrics import softmax, entropy as _entropy
        from heinrich.discover.neurons import detect_mlp_type, _compute_mlp_activated, _mlp_down_proj

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
                mlp_type = detect_mlp_type(ly)
                _gate, _up, activated = _compute_mlp_activated(ly.mlp, h_normed2, mlp_type)

                # Zero target neurons
                act_np = np.array(activated.astype(mx.float32))
                for n in neuron_indices:
                    act_np[0, :, n] = 0.0
                activated = mx.array(act_np.astype(np.float16))

                mlp_out = _mlp_down_proj(ly.mlp, activated, mlp_type)
                h = h + mlp_out
            else:
                h = ly(h, mask=mask, cache=None)
                if isinstance(h, tuple):
                    h = h[0]

        residual = np.array(h.astype(mx.float32)[0, -1, :]) if return_residual else None
        h = inner.norm(h)
        logits = np.array(self._lm_head(h).astype(mx.float32)[0, -1, :])
        probs = softmax(logits)
        top_id = int(np.argmax(probs))

        return ForwardResult(
            logits=logits, probs=probs, top_id=top_id,
            top_token=self.tokenizer.decode([top_id]),
            entropy=_entropy(probs), n_tokens=len(tokens),
            residual=residual,
        )

    def perturb_head(self, prompt, layer, head, *, mode="zero", scale=0.0) -> ForwardResult:
        from heinrich.cartography.perturb import perturb_head as _perturb
        from heinrich.cartography.metrics import softmax, entropy as _entropy

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
        from heinrich.cartography.context import Capabilities
        return Capabilities(
            can_steer=True, can_capture_residual=True, can_capture_attention=True,
            can_capture_mlp_detail=True, can_neuron_mask=True, can_perturb_head=True,
            can_weight_access=True, can_embedding_access=True, can_logit_lens=True,
            can_kv_cache=True, can_gradient=False, can_batch=False,
            can_all_positions=True, can_compose=True, can_gen_control=True,
            can_multi_turn=True,
        )

    def forward_context(self):
        from heinrich.cartography.context import ForwardContext
        return ForwardContext(self)

    def generation_context(self, prompt):
        from heinrich.cartography.context import GenerationContext
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
        from heinrich.cartography.perturb import _mask_dtype
        from heinrich.cartography.metrics import softmax, entropy as _entropy
        from heinrich.cartography.context import ContextResult

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
        mx.eval(h)  # MLX graph evaluation, not Python eval
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
                q_last = q[:, :, -1:, :]
                scores = (q_last.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * attn.scale
                weights = mx.softmax(scores, axis=-1)
                attentions[i] = np.array(weights.astype(mx.float32)[0, :, 0, :])

            # Neuron masking requires manual MLP decomposition
            if i in neuron_masks or i in capture_mlp_layers:
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

            # Callbacks
            if i in callbacks:
                residual_np = np.array(h.astype(mx.float32)[0, -1, :])
                for cb in callbacks[i]:
                    injection = cb(i, residual_np)
                    if injection is not None:
                        h_np = np.array(h.astype(mx.float32))
                        h_np[0, -1, :] += injection
                        h = mx.array(h_np.astype(np.float16))
                        residual_np = h_np[0, -1, :]

            # Capture residuals
            if i in capture_residual_layers:
                residuals[i] = np.array(h.astype(mx.float32)[0, -1, :])
            if i in capture_all_pos_layers:
                all_pos_residuals[i] = np.array(h.astype(mx.float32)[0])

            # Force evaluation every 8 layers to bound the computation graph
            if i % 8 == 7 or i == n_layers - 1:
                mx.eval(h)  # MLX graph evaluation, not Python eval

        h = inner.norm(h)
        logits = np.array(self._lm_head(h).astype(mx.float32)[0, -1, :])
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
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.models.base import create_attention_mask
        from heinrich.cartography.context import TokenResult

        inner = self._inner
        tokens = list(self.tokenizer.encode(prompt))
        eos = getattr(self.tokenizer, "eos_token_id", None)

        steer_at = {}
        for op in gen_ctx._steers:
            steer_at[op.layer] = (op.direction, op.mean_gap, op.alpha)

        capture_layer = gen_ctx._capture_layer

        cache = make_prompt_cache(inner)

        for step in range(max_tokens):
            if step == 0:
                input_ids = mx.array([tokens])
            else:
                input_ids = mx.array([[tokens[-1]]])

            h = inner.embed_tokens(input_ids)
            mask = create_attention_mask(h, cache[0])

            residual = None
            for i, ly in enumerate(inner.layers):
                h = ly(h, mask=mask, cache=cache[i])
                if isinstance(h, tuple):
                    h = h[0]

                if i in steer_at:
                    direction, mean_gap, alpha = steer_at[i]
                    h_np = np.array(h.astype(mx.float32))
                    h_np[0, -1, :] += direction * mean_gap * alpha
                    h = mx.array(h_np.astype(np.float16))

                for inj_layer, inj_vec in gen_ctx._one_shot_injections:
                    if i == inj_layer:
                        h_np = np.array(h.astype(mx.float32))
                        h_np[0, -1, :] += inj_vec
                        h = mx.array(h_np.astype(np.float16))

                if capture_layer is not None and i == capture_layer:
                    residual = np.array(h.astype(mx.float32)[0, -1, :])

            gen_ctx._one_shot_injections.clear()

            h = inner.norm(h)
            logits = np.array(self._lm_head(h).astype(mx.float32)[0, -1, :])
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

    def capture_attention_patterns(self, prompt, *, layers) -> dict[int, np.ndarray]:
        """Capture full attention weight matrices after softmax.

        Returns {layer: attention_weights[n_heads, n_tokens, n_tokens]}.
        """
        import mlx.core as mx
        from heinrich.cartography.perturb import _mask_dtype

        inner = self._inner
        mdtype = _mask_dtype(self.model)
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        layer_set = set(layers)
        attentions: dict[int, np.ndarray] = {}

        h = inner.embed_tokens(input_ids)
        for i, ly in enumerate(inner.layers):
            if i in layer_set:
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

                n_rep = n_heads // n_kv_heads
                if n_rep > 1:
                    k = mx.repeat(k, repeats=n_rep, axis=1)

                scores = (q @ k.transpose(0, 1, 3, 2)) * attn.scale
                if mask is not None:
                    scores = scores + mask.reshape(1, 1, T, T)
                weights = mx.softmax(scores, axis=-1)
                attentions[i] = np.array(weights.astype(mx.float32)[0])

            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]

        return attentions

    def capture_per_layer_delta(self, prompt) -> list[tuple[int, float]]:
        """Compute residual stream delta norm at every layer."""
        import mlx.core as mx
        from heinrich.cartography.perturb import _mask_dtype

        inner = self._inner
        mdtype = _mask_dtype(self.model)
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        deltas: list[tuple[int, float]] = []

        h = inner.embed_tokens(input_ids)
        for i, ly in enumerate(inner.layers):
            h_in = np.array(h.astype(mx.float32)[0, -1, :])
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]
            h_out = np.array(h.astype(mx.float32)[0, -1, :])
            delta_norm = float(np.linalg.norm(h_out - h_in))
            deltas.append((i, delta_norm))

            if i % 8 == 7:
                mx.eval(h)  # MLX graph evaluation, not Python eval

        return deltas

    def steer_and_generate(
        self,
        prompt,
        *,
        alpha,
        layers,
        direction,
        max_tokens=30,
    ) -> tuple[str, list[dict]]:
        """Generate with steering applied to specified layers."""
        from heinrich.cartography.metrics import softmax, entropy as _entropy

        generated_tokens: list[str] = []
        metadata: list[dict] = []

        with self.generation_context(prompt) as gen:
            for layer in layers:
                gen.steer(layer, direction, 1.0, alpha)
            for tok in gen.tokens(max_tokens=max_tokens):
                probs = softmax(tok.logits)
                top_id = int(np.argmax(probs))
                top_token = self.tokenizer.decode([top_id])

                refuse_token_ids = []
                for word in ["Sorry", "I cannot", "I can't", "not", "unable"]:
                    try:
                        ids = self.tokenizer.encode(word)
                        if ids:
                            refuse_token_ids.append(ids[0])
                    except Exception:
                        pass
                refuse_prob = float(sum(
                    probs[tid] for tid in refuse_token_ids if tid < len(probs)
                ))

                ent = _entropy(probs)
                vocab_size = len(probs)
                max_ent = float(np.log2(vocab_size))
                dampening = 1.0 - (ent / max_ent) if max_ent > 0 else 0.0

                generated_tokens.append(tok.token_text)
                metadata.append({
                    "refuse_prob": refuse_prob,
                    "top_token": top_token,
                    "dampening": dampening,
                    "entropy": ent,
                    "step": tok.step,
                })

        return "".join(generated_tokens), metadata

    def instrumented_forward(
        self,
        prompt,
        *,
        directions=None,
    ) -> list[dict]:
        """Run a forward pass capturing everything at every layer.

        NOTE: top_neurons reports raw activations, not z-scores.

        Returns a list of dicts (one per layer) with residual_norm, delta_norm,
        projections onto named directions, top 10 active neurons, and attention
        weight on position 2.
        """
        import mlx.core as mx
        from heinrich.cartography.perturb import _mask_dtype
        from heinrich.discover.neurons import detect_mlp_type, _compute_mlp_activated

        inner = self._inner
        mdtype = _mask_dtype(self.model)
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        T = len(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=mdtype), k=1) if T > 1 else None

        if directions is None:
            directions = {}

        layer_data: list[dict] = []
        n_layers = len(inner.layers)

        h = inner.embed_tokens(input_ids)
        mx.eval(h)  # MLX graph evaluation, not Python eval

        for i, ly in enumerate(inner.layers):
            h_in = np.array(h.astype(mx.float32)[0, -1, :])

            # --- Attention weight on position 2 ---
            attn_pos2_weight = 0.0
            if T > 2:
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

                n_rep = n_heads // n_kv_heads
                if n_rep > 1:
                    k = mx.repeat(k, repeats=n_rep, axis=1)

                q_last = q[:, :, -1:, :]
                scores = (q_last @ k.transpose(0, 1, 3, 2)) * attn.scale
                weights = mx.softmax(scores, axis=-1)
                attn_weights_np = np.array(weights.astype(mx.float32)[0, :, 0, :])
                attn_pos2_weight = float(np.mean(attn_weights_np[:, 2]))

            # --- MLP activations for top neurons ---
            h_normed_mlp = ly.post_attention_layernorm(h) if hasattr(ly, 'post_attention_layernorm') else h
            try:
                mlp_type = detect_mlp_type(ly)
                _gate, _up, activated = _compute_mlp_activated(ly.mlp, h_normed_mlp, mlp_type)
                act_last = np.array(activated.astype(mx.float32)[0, -1, :])
                abs_act = np.abs(act_last)
                top_indices = np.argsort(abs_act)[-10:][::-1]
                top_neurons = [(int(idx), float(act_last[idx])) for idx in top_indices]
            except Exception:
                top_neurons = []

            # --- Forward through the layer ---
            h = ly(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]

            h_out = np.array(h.astype(mx.float32)[0, -1, :])
            residual_norm = float(np.linalg.norm(h_out))
            delta_norm = float(np.linalg.norm(h_out - h_in))

            # --- Projections onto named directions ---
            projections: dict[str, float] = {}
            for name, dir_vec in directions.items():
                d_norm = np.linalg.norm(dir_vec)
                if d_norm > 0:
                    projections[name] = float(np.dot(h_out, dir_vec) / d_norm)
                else:
                    projections[name] = 0.0

            layer_data.append({
                "layer": i,
                "residual_norm": residual_norm,
                "delta_norm": delta_norm,
                "projections": projections,
                "top_neurons": top_neurons,
                "attention_pos2_weight": attn_pos2_weight,
            })

            if i % 8 == 7 or i == n_layers - 1:
                mx.eval(h)  # MLX graph evaluation, not Python eval

        return layer_data

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
