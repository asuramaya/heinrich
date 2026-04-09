"""HuggingFace transformers backend -- runs on CUDA/CPU via transformers."""
from __future__ import annotations

import numpy as np

from .protocol import ForwardResult
from heinrich.cartography.model_config import detect_config


class HFBackend:
    """HuggingFace transformers backend -- runs on CUDA/CPU via transformers."""

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
        self.hf_model.eval()  # PyTorch: set model to evaluation mode
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
        hooks mid-generation -- you must run two separate generation calls.
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
        from heinrich.cartography.metrics import softmax, entropy as _entropy

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
        from heinrich.cartography.metrics import softmax, entropy as _entropy

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
        from heinrich.cartography.metrics import softmax, entropy as _entropy

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
        from heinrich.cartography.context import Capabilities
        return Capabilities(
            can_steer=True, can_capture_residual=True, can_capture_attention=True,
            can_capture_mlp_detail=True, can_neuron_mask=True, can_perturb_head=True,
            can_weight_access=True, can_embedding_access=True, can_logit_lens=True,
            can_kv_cache=True, can_gradient=True, can_batch=True,
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
        """Compile ForwardContext into HF hooks + single forward pass."""
        import torch
        from heinrich.cartography.metrics import softmax, entropy as _entropy
        from heinrich.cartography.context import ContextResult

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
        from heinrich.cartography.context import TokenResult

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

    def capture_attention_patterns(self, prompt, *, layers) -> dict[int, np.ndarray]:
        """Capture full attention weight matrices after softmax.

        Returns {layer: attention_weights[n_heads, n_tokens, n_tokens]}.
        Uses HF output_attentions=True.
        """
        import torch

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self.hf_model(input_ids, output_attentions=True)

        attentions: dict[int, np.ndarray] = {}
        for l in layers:
            if outputs.attentions and l < len(outputs.attentions):
                # outputs.attentions[l] is [batch, n_heads, T, T]
                attentions[l] = outputs.attentions[l][0].float().cpu().numpy()

        return attentions

    def capture_per_layer_delta(self, prompt) -> list[tuple[int, float]]:
        """Compute residual stream delta norm at every layer."""
        import torch

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self.hf_model(input_ids, output_hidden_states=True)

        deltas: list[tuple[int, float]] = []
        hs = outputs.hidden_states  # tuple of (batch, seq, hidden), index 0=embedding
        for i in range(len(hs) - 1):
            h_in = hs[i][0, -1, :].float().cpu().numpy()
            h_out = hs[i + 1][0, -1, :].float().cpu().numpy()
            delta_norm = float(np.linalg.norm(h_out - h_in))
            deltas.append((i, delta_norm))

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

                # Estimate refuse_prob
                refuse_token_ids = []
                for word in ["Sorry", "I cannot", "I can't", "not", "unable"]:
                    try:
                        ids = self.tokenizer.encode(word, add_special_tokens=False)
                        if ids:
                            refuse_token_ids.append(ids[0])
                    except Exception:
                        pass
                refuse_prob = float(sum(
                    probs[tid] for tid in refuse_token_ids if tid < len(probs)
                ))

                # Dampening: entropy relative to uniform
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

        Returns a list of dicts (one per layer) with residual_norm, delta_norm,
        projections onto named directions, top 10 active neurons, and attention
        weight on position 2.
        """
        import torch

        if directions is None:
            directions = {}

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        T = input_ids.shape[1]

        # Capture MLP activations at every layer via hooks
        mlp_activations: dict[int, np.ndarray] = {}
        handles = []

        for li in range(len(self.hf_model.model.layers)):
            mlp = self.hf_model.model.layers[li].mlp

            def make_mlp_hook(layer_idx):
                def hook_fn(module, input, output):
                    out = output[0] if isinstance(output, tuple) else output
                    mlp_activations[layer_idx] = out[0, -1, :].float().cpu().numpy()
                return hook_fn

            handles.append(mlp.register_forward_hook(make_mlp_hook(li)))

        try:
            with torch.no_grad():
                outputs = self.hf_model(
                    input_ids,
                    output_hidden_states=True,
                    output_attentions=True,
                )
        finally:
            for hndl in handles:
                hndl.remove()

        hs = outputs.hidden_states
        layer_data: list[dict] = []

        for i in range(len(hs) - 1):
            h_in = hs[i][0, -1, :].float().cpu().numpy()
            h_out = hs[i + 1][0, -1, :].float().cpu().numpy()
            residual_norm = float(np.linalg.norm(h_out))
            delta_norm = float(np.linalg.norm(h_out - h_in))

            # Projections onto named directions
            projections: dict[str, float] = {}
            for name, dir_vec in directions.items():
                d_norm = np.linalg.norm(dir_vec)
                if d_norm > 0:
                    projections[name] = float(np.dot(h_out, dir_vec) / d_norm)
                else:
                    projections[name] = 0.0

            # Top 10 neurons by absolute MLP activation
            top_neurons: list[tuple[int, float]] = []
            if i in mlp_activations:
                act = mlp_activations[i]
                abs_act = np.abs(act)
                top_indices = np.argsort(abs_act)[-10:][::-1]
                top_neurons = [(int(idx), float(act[idx])) for idx in top_indices]

            # Attention weight on position 2 (mean across heads)
            attn_pos2_weight = 0.0
            if T > 2 and outputs.attentions and i < len(outputs.attentions):
                # attentions[i] shape: [batch, n_heads, T, T]
                attn_w = outputs.attentions[i][0].float().cpu().numpy()
                # Last query position attending to position 2, mean across heads
                attn_pos2_weight = float(np.mean(attn_w[:, -1, 2]))

            layer_data.append({
                "layer": i,
                "residual_norm": residual_norm,
                "delta_norm": delta_norm,
                "projections": projections,
                "top_neurons": top_neurons,
                "attention_pos2_weight": attn_pos2_weight,
            })

        return layer_data

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
