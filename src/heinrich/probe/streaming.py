"""Streaming self-analysis — emit signals during token generation."""
from __future__ import annotations
from typing import Any, Callable, Iterator
from ..signal import Signal, SignalStore


def generate_with_signals(
    provider: Any,
    prompt: str,
    *,
    model: str = "model",
    label: str = "streaming",
    max_tokens: int = 50,
    on_token: Callable[[int, Signal], None] | None = None,
) -> tuple[str, SignalStore]:
    """Generate text token-by-token, emitting self-analysis signals at each step.

    Currently works by generating full response then analyzing.
    Future: hook into generation loop for true per-token analysis.
    """
    store = SignalStore()

    # For now: generate full response, then analyze
    from ..inspect.self_analysis import analyze_logits

    forward_fn = getattr(provider, "forward_with_internals", None)
    if forward_fn is not None:
        try:
            internals = forward_fn(prompt, model=model)
            logits = internals.get("logits")
            if logits is not None:
                signals = analyze_logits(logits, label=label, step=0)
                store.extend(signals)
                if on_token:
                    for s in signals:
                        on_token(0, s)
        except Exception:
            pass  # gracefully degrade

    # Get the actual text
    results = provider.chat_completions(
        [{"custom_id": "stream", "messages": [{"role": "user", "content": prompt}]}],
        model=model,
    )
    text = results[0].get("text", "")

    store.add(Signal("generated_text", "probe", label, "response", float(len(text)),
                     {"text": text[:500]}))

    return text, store
