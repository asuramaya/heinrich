"""Bridge between probe providers and cartography backends.

Wraps an existing probe Provider (from the MCP server's _do_configure)
as a cartography Backend, so code that already has a configured provider
can use cartography operations without loading the model twice.

Usage:
    from heinrich.probe import HuggingFaceLocalProvider
    provider = HuggingFaceLocalProvider({"model": "meta-llama/Llama-3-8B"})

    from heinrich.cartography.probe_bridge import backend_from_provider
    backend = backend_from_provider(provider)

    # Now use any cartography operation:
    result = backend.forward("Hello world")
    text = backend.generate("Write a poem", max_tokens=50)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .backend import Backend, ForwardResult
from .metrics import softmax, entropy as _entropy
from .model_config import ModelConfig


class ProviderBackend:
    """Wraps a probe Provider as a cartography Backend.

    Delegates to provider.forward_with_internals for internal state capture,
    and to provider.chat_completions for generation.
    """

    def __init__(self, provider: Any, config: ModelConfig) -> None:
        self._provider = provider
        self.config = config

    def forward(
        self,
        prompt: str,
        *,
        steer_dirs: dict[int, tuple[np.ndarray, float]] | None = None,
        alpha: float = 0.0,
        return_residual: bool = False,
        residual_layer: int = -1,
    ) -> ForwardResult:
        result = self._provider.forward_with_internals(text=prompt, model="")

        logits = result["logits"]
        if logits.ndim > 1:
            logits = logits[-1, :]

        probs = softmax(logits)
        top_id = int(np.argmax(probs))

        residual = None
        if return_residual and "hidden_states" in result:
            hs = result["hidden_states"]
            layer_idx = residual_layer if residual_layer >= 0 else -1
            r = hs[layer_idx]
            residual = r[-1, :] if r.ndim > 1 else r

        # Decode top token — try tokenizer if available, fall back
        top_token = f"<id:{top_id}>"
        if hasattr(self._provider, "_tokenizer") and self._provider._tokenizer is not None:
            try:
                top_token = self._provider._tokenizer.decode([top_id])
            except Exception:
                pass

        return ForwardResult(
            logits=logits,
            probs=probs,
            top_id=top_id,
            top_token=top_token,
            entropy=_entropy(probs),
            n_tokens=len(logits),
            residual=residual,
        )

    def generate(
        self,
        prompt: str,
        *,
        steer_dirs: dict[int, tuple[np.ndarray, float]] | None = None,
        alpha: float = 0.0,
        max_tokens: int = 30,
    ) -> str:
        case = {
            "custom_id": "bridge_gen",
            "messages": [{"role": "user", "content": prompt}],
        }
        # Temporarily override max_new_tokens if provider supports it
        orig_max = self._provider._config.get("max_new_tokens")
        self._provider._config["max_new_tokens"] = max_tokens
        try:
            results = self._provider.chat_completions([case], model="")
        finally:
            if orig_max is not None:
                self._provider._config["max_new_tokens"] = orig_max
            else:
                self._provider._config.pop("max_new_tokens", None)
        return results[0]["text"] if results else ""

    def capture_residual_states(
        self,
        prompts: list[str],
        *,
        layers: list[int],
    ) -> dict[int, np.ndarray]:
        all_states: dict[int, list[np.ndarray]] = {l: [] for l in layers}

        for prompt in prompts:
            result = self._provider.forward_with_internals(text=prompt, model="")
            hs = result.get("hidden_states", [])
            for l in layers:
                # hidden_states[0] is embedding, hidden_states[l+1] is after layer l
                idx = l + 1 if l + 1 < len(hs) else -1
                if hs:
                    state = hs[idx]
                    state = state[-1, :] if state.ndim > 1 else state
                    all_states[l].append(state)
                else:
                    all_states[l].append(np.zeros(self.config.hidden_size))

        return {l: np.array(vecs) for l, vecs in all_states.items()}

    def capture_mlp_activations(
        self,
        prompt: str,
        layer: int,
    ) -> np.ndarray:
        # Use the provider's activations method to hook into the MLP
        module_name = f"model.layers.{layer}.mlp"
        case = {
            "custom_id": "bridge_mlp",
            "messages": [{"role": "user", "content": prompt}],
            "module_names": [module_name],
        }
        results = self._provider.activations([case], model="")
        acts = results[0].get("activations", {})
        if module_name in acts:
            act = acts[module_name]
            return act[-1, :] if act.ndim > 1 else act
        return np.zeros(self.config.intermediate_size)

    def tokenize(self, text: str) -> list[int]:
        if hasattr(self._provider, "_tokenizer") and self._provider._tokenizer is not None:
            return self._provider._tokenizer.encode(text)
        raise NotImplementedError("Provider does not expose a tokenizer")

    def decode(self, token_ids: list[int]) -> str:
        if hasattr(self._provider, "_tokenizer") and self._provider._tokenizer is not None:
            return self._provider._tokenizer.decode(token_ids)
        raise NotImplementedError("Provider does not expose a tokenizer")


def _detect_config_from_provider(provider: Any) -> ModelConfig:
    """Auto-detect ModelConfig from a loaded provider's internal model."""
    from .model_config import detect_config

    # Ensure the model is loaded
    if hasattr(provider, "_ensure_loaded"):
        provider._ensure_loaded()

    # Try HF provider (has _model and _tokenizer)
    model = getattr(provider, "_model", None)
    tokenizer = getattr(provider, "_tokenizer", None)

    if model is not None:
        return detect_config(model, tokenizer)

    # Fall back to provider.describe() metadata
    desc = provider.describe() if hasattr(provider, "describe") else {}
    raise ValueError(
        f"Cannot detect model config from provider type "
        f"{desc.get('provider_type', type(provider).__name__)}. "
        f"Pass config= explicitly."
    )


def backend_from_provider(
    provider: Any,
    *,
    config: ModelConfig | None = None,
) -> ProviderBackend:
    """Wrap an existing probe Provider as a cartography Backend.

    Args:
        provider: A probe Provider (HuggingFaceLocalProvider or MLXProvider)
        config: Optional ModelConfig. Auto-detected from provider if not given.

    Returns:
        A ProviderBackend that satisfies the cartography Backend protocol.
    """
    if config is None:
        config = _detect_config_from_provider(provider)
    return ProviderBackend(provider, config)
