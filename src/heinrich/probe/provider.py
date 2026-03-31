"""Provider protocol for model inference + self-analysis."""
from __future__ import annotations
from typing import Any, Protocol, Sequence
import numpy as np
from ..signal import Signal


class Provider(Protocol):
    """Protocol for model inference providers."""
    def describe(self) -> dict[str, Any]: ...
    def chat_completions(self, cases: Sequence[dict[str, Any]], *, model: str) -> list[dict[str, Any]]: ...
    def activations(self, cases: Sequence[dict[str, Any]], *, model: str) -> list[dict[str, Any]]: ...


class SelfAnalyzingProvider(Protocol):
    """Extended provider that captures internal states during inference."""
    def describe(self) -> dict[str, Any]: ...
    def chat_completions(self, cases: Sequence[dict[str, Any]], *, model: str) -> list[dict[str, Any]]: ...
    def activations(self, cases: Sequence[dict[str, Any]], *, model: str) -> list[dict[str, Any]]: ...
    def forward_with_internals(self, text: str, *, model: str) -> dict[str, Any]: ...


class MockProvider:
    """Test provider that returns canned responses."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}

    def describe(self) -> dict[str, Any]:
        return {"provider_type": "mock"}

    def chat_completions(
        self, cases: Sequence[dict[str, Any]], *, model: str
    ) -> list[dict[str, Any]]:
        results = []
        for case in cases:
            cid = case.get("custom_id", "")
            text = self._responses.get(cid, f"Mock response for {cid}")
            results.append({"custom_id": cid, "text": text})
        return results

    def activations(
        self, cases: Sequence[dict[str, Any]], *, model: str
    ) -> list[dict[str, Any]]:
        return [{"custom_id": case.get("custom_id", ""), "activations": {}} for case in cases]


class HuggingFaceLocalProvider:
    """Local HuggingFace model provider with self-analysis support."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = dict(config)
        self._model = None
        self._tokenizer = None

    def describe(self) -> dict[str, Any]:
        return {"provider_type": "hf-local", "model": self._config.get("model", "unknown")}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("torch and transformers required: pip install heinrich[probe]")

        model_ref = str(self._config.get("model") or self._config.get("model_path", ""))
        if not model_ref:
            raise ValueError("HF provider requires 'model' or 'model_path' in config")

        dtype_name = self._config.get("dtype")
        dtype = getattr(torch, dtype_name) if dtype_name and hasattr(torch, dtype_name) else None
        device = self._config.get("device", "cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_ref, trust_remote_code=True, use_fast=True)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if dtype is not None:
            load_kwargs["dtype"] = dtype

        self._model = AutoModelForCausalLM.from_pretrained(model_ref, **load_kwargs)
        self._model = self._model.to(device)
        # put model into inference mode
        self._model.train(False)
        self._device = device
        self._torch = torch

    def chat_completions(self, cases: Sequence[dict[str, Any]], *, model: str) -> list[dict[str, Any]]:
        self._ensure_loaded()
        results = []
        max_new = int(self._config.get("max_new_tokens", 64))
        for case in cases:
            messages = case.get("messages", [])
            rendered = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = self._tokenizer(rendered, return_tensors="pt").to(self._device)
            input_len = encoded["input_ids"].shape[-1]
            with self._torch.no_grad():
                output = self._model.generate(
                    **encoded, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            text = self._tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
            results.append({"custom_id": case.get("custom_id", ""), "text": text})
        return results

    def activations(self, cases: Sequence[dict[str, Any]], *, model: str) -> list[dict[str, Any]]:
        self._ensure_loaded()
        results = []
        for case in cases:
            modules_requested = case.get("module_names", [])
            messages = case.get("messages", [])
            rendered = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = self._tokenizer(rendered, return_tensors="pt").to(self._device)
            captures: dict[str, Any] = {}
            handles = []
            module_lookup = dict(self._model.named_modules())
            for name in modules_requested:
                mod = module_lookup.get(name)
                if mod is None:
                    continue
                def make_hook(n: str):
                    def hook(_, __, output):
                        t = output[0] if isinstance(output, tuple) else output
                        if hasattr(t, "detach"):
                            captures[n] = t.detach().cpu().float().numpy()
                    return hook
                handles.append(mod.register_forward_hook(make_hook(name)))
            with self._torch.no_grad():
                self._model(**encoded)
            for h in handles:
                h.remove()
            results.append({"custom_id": case.get("custom_id", ""), "activations": captures})
        return results

    def forward_with_internals(self, text: str, *, model: str = "") -> dict[str, Any]:
        """Run a forward pass and capture logits, hidden states, attention weights."""
        self._ensure_loaded()
        encoded = self._tokenizer(text, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            outputs = self._model(**encoded, output_attentions=True, output_hidden_states=True)
        result: dict[str, Any] = {"logits": outputs.logits[0].cpu().float().numpy()}
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            result["hidden_states"] = [h[0].cpu().float().numpy() for h in outputs.hidden_states]
        if hasattr(outputs, "attentions") and outputs.attentions:
            result["attentions"] = [a[0].cpu().float().numpy() for a in outputs.attentions]
        return result
