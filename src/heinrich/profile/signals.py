"""Signal extraction from analysis results.

Converts analysis function dicts into typed Signals for the DB.
One signal per layer per measurement (Option A from V1 proposal).

Model identity: analysis functions return the architecture name ("llama")
not the model name ("smollm2-135m"). The emit_signals() function accepts
an optional mri_path to derive the real model name from the directory
structure (e.g., /Volumes/sharts/smollm2-135m/raw.mri → smollm2-135m).
"""
from __future__ import annotations

from pathlib import Path

from ..core.signal import Signal


def _model_from_path(mri_path: str | None, fallback: str) -> str:
    """Derive model name from MRI path directory structure.

    /Volumes/sharts/smollm2-135m/raw.mri → smollm2-135m
    /Volumes/sharts/qwen-0.5b/naked.mri → qwen-0.5b
    Falls back to architecture name if path not provided.
    """
    if not mri_path:
        return fallback
    p = Path(mri_path)
    # The parent of X.mri/ is the model directory
    if p.suffix == '.mri' or p.name.endswith('.mri'):
        return p.parent.name
    return fallback


def _make_signal(kind: str, model: str, mode: str, layer: int | str,
                 value: float, **meta) -> Signal:
    """Build a signal with consistent source/target format."""
    meta["mode"] = mode
    return Signal(
        kind=kind,
        source=f"mri:{model}/{mode}",
        model=model,
        target=f"L{layer}" if isinstance(layer, int) else str(layer),
        value=value,
        metadata=meta,
    )


def signals_from_layer_deltas(result: dict, model: str) -> list[Signal]:
    mode = result.get("mode", "?")
    return [
        _make_signal("layer_delta", model, mode, lr["layer"],
                     lr["mean_delta_norm"],
                     max_delta_norm=lr["max_delta_norm"],
                     std_delta_norm=lr["std_delta_norm"],
                     amplification=lr["amplification"],
                     n_tokens=result.get("n_tokens", 0))
        for lr in result.get("layers", [])
    ]


def signals_from_gate_analysis(result: dict, model: str) -> list[Signal]:
    mode = result.get("mode", "?")
    return [
        _make_signal("gate_concentration", model, mode, lr["layer"],
                     lr["top1_concentration"],
                     top1_neuron=lr["top1_neuron"],
                     unique_neurons=lr["unique_neurons"],
                     mean_activation=lr["mean_activation"],
                     n_tokens=result.get("n_tokens", 0))
        for lr in result.get("layers", [])
    ]


def signals_from_logit_lens(result: dict, model: str) -> list[Signal]:
    from collections import Counter
    signals = []
    for lr in result.get("layers", []):
        preds = lr.get("predictions", [])
        if not preds:
            continue
        top1_counts = Counter(p["top_ids"][0] for p in preds)
        mode_id, mode_count = top1_counts.most_common(1)[0]
        concentration = mode_count / len(preds)
        signals.append(_make_signal(
            "logit_lens_concentration", model, "?", lr["layer"],
            concentration,
            top1_token_id=mode_id,
            n_predictions=len(preds)))
    return signals


def signals_from_pca_depth(result: dict, model: str) -> list[Signal]:
    mode = result.get("mode", "?")
    return [
        _make_signal("pca_pc1", model, mode, lr["layer"],
                     lr["pc1_pct"],
                     pcs_for_50pct=lr["pcs_for_50pct"],
                     neg_pole=lr.get("neg_pole", "?"),
                     pos_pole=lr.get("pos_pole", "?"))
        for lr in result.get("layers", [])
    ]


def signals_from_attention(result: dict, model: str) -> list[Signal]:
    mode = result.get("mode", "?")
    return [
        _make_signal("attention_self_weight", model, mode, lr["layer"],
                     lr["self_weight"],
                     prefix_weight=lr["prefix_weight"],
                     suffix_weight=lr["suffix_weight"],
                     entropy=lr["entropy"],
                     n_tokens=result.get("n_tokens", 0))
        for lr in result.get("layers", [])
    ]


def signals_from_shart_anatomy(result: dict, model: str) -> list[Signal]:
    mode = result.get("mode", "?")
    signals = []
    crystal = result.get("crystal", {})
    if crystal:
        signals.append(_make_signal(
            "crystal_birth", model, mode, crystal.get("layer", "?"),
            crystal.get("correlation", 0),
            neuron=crystal.get("neuron"),
            concentration=crystal.get("concentration"),
            amplification=crystal.get("amplification")))
    return signals


def signals_from_lookup_fraction(result: dict, model: str) -> list[Signal]:
    mode = result.get("mode", "?")
    signals = [
        _make_signal("lookup_fraction", model, mode, "all",
                     result.get("lookup_fraction", 0),
                     lookup_solvable=result.get("lookup_solvable", 0),
                     compute_needed=result.get("compute_needed", 0),
                     n_tokens=result.get("n_tokens", 0))
    ]
    for lr in result.get("by_layer", []):
        signals.append(_make_signal(
            "lookup_fraction_layer", model, mode, lr["layer"],
            lr["fraction"], n_tokens=result.get("n_tokens", 0)))
    return signals


def signals_from_distribution_drift(result: dict, model: str) -> list[Signal]:
    mode = result.get("mode", "?")
    return [
        _make_signal("distribution_drift", model, mode, lr["layer"],
                     lr["mean_kl"],
                     top1_changed=lr["top1_changed"],
                     mean_tvd=lr["mean_tvd"],
                     mean_entropy=lr["mean_entropy"],
                     n_tokens=result.get("n_tokens", 0))
        for lr in result.get("layers", [])
    ]


def signals_from_retrieval_horizon(result: dict, model: str) -> list[Signal]:
    mode = result.get("mode", "?")
    return [
        _make_signal("retrieval_horizon", model, mode, lr["layer"],
                     lr["mean_retrieval_distance"],
                     self_attention=lr["self_attention"],
                     n_tokens=result.get("n_tokens", 0))
        for lr in result.get("layers", [])
    ]


def signals_from_layer_opposition(result: dict, model: str) -> list[Signal]:
    mode = result.get("mode", "?")
    return [
        _make_signal("layer_opposition", model, mode, lr["layer"],
                     lr["cos_mlp_attn"],
                     mlp_norm=lr["mlp_norm"],
                     attn_norm=lr["attn_norm"],
                     delta_norm=lr["delta_norm"],
                     cancellation=lr["cancellation"],
                     n_tokens=result.get("n_tokens", 0))
        for lr in result.get("layers", [])
    ]


def signals_from_bandwidth(result: dict, model: str) -> list[Signal]:
    mode = result.get("mode", "?")
    signals = [
        _make_signal("bandwidth_efficiency", model, mode, "all",
                     result.get("bandwidth_efficiency", 0),
                     total_model_bytes=result.get("total_model_bytes", 0),
                     total_active_bytes=result.get("total_active_bytes", 0),
                     wasted_fraction=result.get("wasted_fraction", 0))
    ]
    for lr in result.get("layers", []):
        signals.append(_make_signal(
            "bandwidth_layer", model, mode, lr["layer"],
            lr["efficiency"],
            skip_fraction=lr["skip_fraction"],
            mlp_active_fraction=lr["mlp_active_fraction"],
            mlp_active_neurons=lr["mlp_active_neurons"]))
    return signals


def signals_from_cross_model(result: dict, model: str) -> list[Signal]:
    signals = []
    for pair in result.get("pairwise", []):
        signals.append(Signal(
            kind="cross_model_correlation",
            source=f"cross:{pair['model_a']}:{pair['model_b']}",
            model=pair["model_a"],
            target=pair["model_b"],
            value=pair.get("displacement_rho", 0),
            metadata={
                "gradient_rho": pair.get("gradient_rho"),
                "n_shared": result.get("n_shared", 0),
            },
        ))
    return signals


# Registry: analysis function name -> signal extractor
EXTRACTORS = {
    "layer_deltas": signals_from_layer_deltas,
    "gate_analysis": signals_from_gate_analysis,
    "logit_lens": signals_from_logit_lens,
    "pca_depth": signals_from_pca_depth,
    "attention_analysis": signals_from_attention,
    "shart_anatomy": signals_from_shart_anatomy,
    "lookup_fraction": signals_from_lookup_fraction,
    "distribution_drift": signals_from_distribution_drift,
    "retrieval_horizon": signals_from_retrieval_horizon,
    "layer_opposition": signals_from_layer_opposition,
    "bandwidth": signals_from_bandwidth,
    "cross_model": signals_from_cross_model,
}


def emit_signals(analysis_name: str, result: dict, db,
                 *, mri_path: str | None = None) -> int:
    """Extract signals from an analysis result and write to DB.

    Deduplicates by deleting previous signals from the same
    analysis+source combination before writing new ones.

    Args:
        analysis_name: key into EXTRACTORS registry
        result: the dict returned by the analysis function
        db: SignalDB instance
        mri_path: path to .mri directory (for deriving model name)

    Returns the number of signals written.
    """
    extractor = EXTRACTORS.get(analysis_name)
    if not extractor or "error" in result:
        return 0
    model = _model_from_path(mri_path, result.get("model", "?"))
    signals = extractor(result, model)
    if not signals:
        return 0
    # Tag signals with the analysis name for dedup scoping
    source = signals[0].source
    stream = f"{analysis_name}:{source}"
    tagged = [
        Signal(kind=s.kind, source=s.source, model=s.model,
               target=s.target, value=s.value, metadata=s.metadata,
               stream=stream)
        for s in signals
    ]
    # Dedup: delete old signals from the same analysis run
    db._write(
        "DELETE FROM signals WHERE stream = ?",
        (stream,), wait=True)
    db.add_many(tagged)
    # Notify companion browser if running
    try:
        from ..companion import notify_companions
        notify_companions({
            "type": "signals",
            "analysis": analysis_name,
            "model": model,
            "source": source,
            "count": len(tagged),
        })
    except Exception:
        pass  # companion not running or not importable
    return len(tagged)
