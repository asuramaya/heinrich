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

    def tokenize(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


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
