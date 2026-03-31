"""Provider-agnostic next-token probing."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from ..signal import Signal
from ..inspect.self_analysis import _softmax


def probe_next_tokens(
    provider: Any,
    prompt: str,
    candidate_strings: Sequence[str],
    tokenizer: Any,
    *,
    model: str = "model",
    label: str = "nexttoken",
) -> list[Signal]:
    """Probe which candidate strings the model predicts next."""
    internals = provider.forward_with_internals(prompt, model=model)
    logits = internals.get("logits")
    if logits is None:
        return []
    probs = _softmax(logits)

    signals = []
    for candidate in candidate_strings:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if not token_ids:
            continue
        first_prob = float(probs[token_ids[0]])
        signals.append(Signal("nexttoken_prob", "probe", label, candidate, first_prob,
                              {"token_id": token_ids[0], "model": model}))

    signals.sort(key=lambda s: s.value, reverse=True)
    return signals
