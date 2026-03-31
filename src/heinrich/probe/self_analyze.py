"""Self-analysis pipeline stage — captures model internals and emits signals."""
from __future__ import annotations
from typing import Any
from ..signal import Signal, SignalStore
from ..inspect.self_analysis import analyze_logits, analyze_hidden_states, analyze_attention, compute_activation_novelty


class SelfAnalyzeStage:
    """Pipeline stage: run text through a provider and capture internal signals."""
    name = "self_analyze"

    def run(self, store: SignalStore, config: dict[str, Any]) -> None:
        provider = config.get("provider")
        if provider is None or not hasattr(provider, "forward_with_internals"):
            return

        text = config.get("text", "")
        model = config.get("model", "model")
        label = config.get("model_label", model)
        iteration = config.get("_iteration", 0)

        if not text:
            return

        internals = provider.forward_with_internals(text, model=model)

        # Logit analysis
        logits = internals.get("logits")
        if logits is not None:
            store.extend(analyze_logits(logits, label=label, step=iteration))

        # Hidden state analysis
        hidden = internals.get("hidden_states")
        if hidden:
            store.extend(analyze_hidden_states(hidden, label=label, step=iteration))

        # Attention analysis
        attentions = internals.get("attentions")
        if attentions:
            store.extend(analyze_attention(attentions, label=label, step=iteration))

        # Novelty tracking
        prior_key = "_prior_activations"
        if hidden and hidden[-1] is not None:
            last_hidden = hidden[-1]
            prior = config.get(prior_key, [])
            novelty = compute_activation_novelty(last_hidden, prior)
            store.add(Signal("self_novelty", "self_analyze", label, f"step_{iteration}", novelty, {}))
            prior.append(last_hidden)
            config[prior_key] = prior
