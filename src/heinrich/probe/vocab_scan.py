"""Provider-agnostic vocabulary scanning for trigger candidates."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from ..signal import Signal
from ..inspect.self_analysis import _softmax


def scan_vocabulary_via_provider(
    provider: Any,
    base_prompt: str,
    candidate_tokens: Sequence[str],
    *,
    model: str = "model",
    label: str = "vocab_scan",
) -> list[Signal]:
    """Score candidate tokens by how much they shift the output distribution."""
    # Get baseline logits
    base_internals = provider.forward_with_internals(base_prompt, model=model)
    base_logits = base_internals.get("logits")
    if base_logits is None:
        return []
    base_probs = _softmax(base_logits)

    signals = []
    for token in candidate_tokens:
        modified_prompt = f"{base_prompt} {token}"
        mod_internals = provider.forward_with_internals(modified_prompt, model=model)
        mod_logits = mod_internals.get("logits")
        if mod_logits is None:
            continue
        mod_probs = _softmax(mod_logits)

        # KL divergence
        kl = float(np.sum(base_probs * np.log((base_probs + 1e-12) / (mod_probs + 1e-12))))
        signals.append(Signal("vocab_kl", "probe", label, token, kl, {"model": model}))

    signals.sort(key=lambda s: s.value, reverse=True)
    return signals
