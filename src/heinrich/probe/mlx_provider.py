"""MLX provider — Apple Silicon native inference with self-analysis."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from ..signal import Signal


class MLXProvider:
    """Local MLX model provider with self-analysis support."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = dict(config)
        self._model = None
        self._tokenizer = None

    def describe(self) -> dict[str, Any]:
        return {"provider_type": "mlx", "model": self._config.get("model", "unknown")}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import mlx_lm
        except ImportError:
            raise ImportError("mlx-lm required: pip install mlx mlx-lm")
        model_ref = str(self._config.get("model") or self._config.get("model_path", ""))
        if not model_ref:
            raise ValueError("MLX provider requires 'model' in config")
        self._model, self._tokenizer = mlx_lm.load(model_ref)
        self._mlx_lm = mlx_lm

    def chat_completions(self, cases: Sequence[dict[str, Any]], *, model: str) -> list[dict[str, Any]]:
        self._ensure_loaded()
        results = []
        max_tokens = int(self._config.get("max_new_tokens", 64))
        for case in cases:
            messages = case.get("messages", [])
            rendered = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            text = self._mlx_lm.generate(
                self._model, self._tokenizer, prompt=rendered,
                max_tokens=max_tokens, verbose=False)
            results.append({"custom_id": case.get("custom_id", ""), "text": text})
        return results

    def activations(self, cases: Sequence[dict[str, Any]], *, model: str) -> list[dict[str, Any]]:
        return [{"custom_id": c.get("custom_id", ""), "activations": {}} for c in cases]

    def forward_with_internals(self, text: str, *, model: str = "") -> dict[str, Any]:
        """Run forward pass and capture logits."""
        self._ensure_loaded()
        import mlx.core as mx
        tokens = self._tokenizer.encode(text)
        input_ids = mx.array([tokens])
        outputs = self._model(input_ids)
        logits = np.array(outputs.astype(mx.float32)[0, -1, :])
        return {"logits": logits}
