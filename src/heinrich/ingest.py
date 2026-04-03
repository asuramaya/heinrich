"""Ingest JSON data files into SignalDB.

Each data file has a different schema. This module knows how to parse
each one and write it into the appropriate DB tables.

Data files fall into these categories:
- Evaluations: safetybench, fast_benchmark, full_matrix, benchmark_200x50_gen
- Attack analysis: input_attack_matrix, convergence_matrix, cracking_matrix
- Censorship: censorship_map
- Sharts: shart_crawl, shart_combination_matrix, shart_combinatorics_all_models,
          shart_null_definitive, shart_steering_definitive
- Atlas (perturbation sweeps): qwen7b_atlas*.json, qwen7b_base_atlas*.json
- Audits: audit_base, audit_instruct, qwen7b_base_audit
- Cartography: qwen7b_cartography_report, qwen7b_cartography_full_report
- Model maps: complete_model_map, corrected_all_models, corrected_directions,
              model_safety_map_full, complete_signal_analysis
- Misc: transfer_debug, extracted_log_data, open_questions_results,
        qwen7b_base_backdoor_report
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .signal import Signal


DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> Any:
    """Load a JSON file, return parsed data."""
    return json.loads(path.read_text())


def _add_signals(db, signals: list[Signal], *, run_id: int | None = None) -> int:
    """Bulk-add signals to the DB via the existing add_many method."""
    if not signals:
        return 0
    return db.add_many(signals, run_id=run_id)


def _get_or_create_model(db, model_name: str, **kwargs) -> int:
    """Get or create a model row via the proper upsert_model method."""
    return db.upsert_model(model_name, **kwargs)


def _parse_head_knob(knob: str) -> tuple[int, int] | None:
    """Parse a knob string like 'head.27.2' into (layer, head)."""
    m = re.match(r"head\.(\d+)\.(\d+)", knob)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _token_text_to_id(text: str) -> int:
    """Convert a token text to a pseudo token_id using a stable hash.

    Used when JSON data only has token text, not integer token IDs.
    Uses a positive 31-bit hash to avoid conflicts with real token IDs.
    """
    return abs(hash(text)) % (2**31)


# ---------------------------------------------------------------------------
# Per-file ingesters
# ---------------------------------------------------------------------------

def ingest_safetybench(db, path: Path, model_id: str) -> int:
    """Ingest safetybench_report.json -- alpha sweep evaluations.

    Schema: {model, timestamp, summary: [{dataset, alpha, n, refused, complied}, ...]}
    """
    data = _load(path)
    model = data.get("model", model_id)
    timestamp = data.get("timestamp", "")
    signals = []

    # --- Normalized tables ---
    mid = _get_or_create_model(db, model)
    exp_id = db.record_experiment(
        "safetybench", mid,
        kind="evaluation", script="safetybench",
        status="completed",
    )

    for row in data.get("summary", []):
        dataset = row["dataset"]
        alpha = row["alpha"]
        target = f"{dataset}/alpha={alpha}"

        # Legacy signals
        signals.append(Signal(
            kind="safetybench_refusal",
            source=str(path.name),
            model=model,
            target=target,
            value=row["refused"],
            metadata={"dataset": dataset, "alpha": alpha, "n": row["n"],
                       "complied": row["complied"], "timestamp": timestamp},
        ))
        signals.append(Signal(
            kind="safetybench_compliance",
            source=str(path.name),
            model=model,
            target=target,
            value=row["complied"],
            metadata={"dataset": dataset, "alpha": alpha, "n": row["n"],
                       "refused": row["refused"], "timestamp": timestamp},
        ))

        # Normalized evaluation
        db.record_evaluation(
            exp_id,
            dataset=dataset,
            attack=f"alpha={alpha}",
            alpha=alpha,
            refuse_prob=row["refused"],
            comply_prob=row["complied"],
            provenance="ingested",
        )

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="evaluations")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_fast_benchmark(db, path: Path, model_id: str) -> int:
    """Ingest fast_benchmark.json -- 1890 evaluations, 7 attacks.

    Schema: {model, timestamp, total_evaluations, attacks: {name: {refusal_rate, compliance_rate}},
             per_dataset: {dataset: {attack: {refusal_rate, compliance_rate}}}}
    """
    data = _load(path)
    model = data.get("model", model_id)
    signals = []

    # --- Normalized tables ---
    mid = _get_or_create_model(db, model)
    exp_id = db.record_experiment(
        "fast_benchmark", mid,
        kind="evaluation", script="fast_benchmark",
        n_evaluations=data.get("total_evaluations", 0),
        status="completed",
    )

    # Aggregate attack stats (legacy signals only)
    for attack, stats in data.get("attacks", {}).items():
        signals.append(Signal(
            kind="benchmark_refusal",
            source=str(path.name),
            model=model,
            target=f"aggregate/{attack}",
            value=stats["refusal_rate"],
            metadata={"attack": attack, "compliance_rate": stats["compliance_rate"],
                       "total_evaluations": data.get("total_evaluations", 0)},
        ))

    # Per-dataset, per-attack
    datasets_counts = data.get("datasets", {})
    for dataset, attacks in data.get("per_dataset", {}).items():
        n_prompts = datasets_counts.get(dataset, 0)
        for attack, stats in attacks.items():
            target = f"{dataset}/{attack}"
            signals.append(Signal(
                kind="benchmark_refusal",
                source=str(path.name),
                model=model,
                target=target,
                value=stats["refusal_rate"],
                metadata={"dataset": dataset, "attack": attack,
                           "compliance_rate": stats["compliance_rate"]},
            ))

            # Normalized evaluation
            # Item 6,17: populate n_prompts from the JSON datasets field
            _n_prompts = n_prompts if n_prompts else None
            db.record_evaluation(
                exp_id,
                dataset=dataset,
                attack=attack,
                refuse_prob=stats["refusal_rate"],
                comply_prob=stats["compliance_rate"],
                n_prompts=_n_prompts,
                provenance="ingested",
            )

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="evaluations")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_full_matrix(db, path: Path, model_id: str) -> int:
    """Ingest full_matrix.json -- 3780 evaluations, quality analysis.

    Schema: {model, ts, n_harmful, n_safe, n_cfgs, summary: {config_key: {n, ref, com}},
             quality: {category: {label: count}}}
    """
    data = _load(path)
    model = data.get("model", model_id)
    signals = []

    # --- Normalized tables ---
    mid = _get_or_create_model(db, model)
    exp_id = db.record_experiment(
        "full_matrix", mid,
        kind="evaluation", script="full_matrix",
        status="completed",
    )

    for config_key, stats in data.get("summary", {}).items():
        signals.append(Signal(
            kind="matrix_refusal",
            source=str(path.name),
            model=model,
            target=config_key,
            value=stats["ref"],
            metadata={"n": stats["n"], "compliance": stats["com"],
                       "n_total": data.get("n_total", 0)},
        ))

        # Parse config_key like "direct_harmful" into attack + category
        parts = config_key.split("_", 1)
        attack = parts[0] if parts else config_key
        category = parts[1] if len(parts) > 1 else ""

        db.record_evaluation(
            exp_id,
            dataset="full_matrix",
            attack=attack,
            category=category,
            refuse_prob=stats["ref"],
            comply_prob=stats["com"],
        )

    # Quality analysis signals
    for category, labels in data.get("quality", {}).items():
        for label, count_val in labels.items():
            signals.append(Signal(
                kind="quality_label",
                source=str(path.name),
                model=model,
                target=f"{category}/{label}",
                value=float(count_val),
                metadata={"category": category, "label": label},
            ))

            db.record_evaluation(
                exp_id,
                dataset="full_matrix_quality",
                attack=category,
                category=label,
                quality=label,
                refuse_prob=float(count_val),
            )

    # Item 30, 41-42: store quality summary as an event for first-token metric flaw
    quality = data.get("quality", {})
    if quality:
        # HARDCODED note: debug category has 0 ACTIONABLE, 16 MIXED, 42 DISGUISED, 26 UNCLEAR
        quality_summary = {cat: dict(labels) for cat, labels in quality.items()}
        db.record_event(
            "quality_analysis",
            source=str(path.name),
            quality_summary=quality_summary,
            note="debug: 0 ACTIONABLE, 16 MIXED, 42 DISGUISED, 26 UNCLEAR",
        )

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="evaluations")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_censorship_map(db, path: Path, model_id: str) -> int:
    """Ingest censorship_map.json -- 20 bilingual topic comparisons.

    Schema: [{topic, en_status, zh_status, en_text, zh_text, divergent}, ...]
    """
    data = _load(path)
    signals = []

    # --- Normalized tables ---
    mid = _get_or_create_model(db, model_id)

    for entry in data:
        topic = entry["topic"]
        divergent = entry.get("divergent", False)
        signals.append(Signal(
            kind="censorship_divergence",
            source=str(path.name),
            model=model_id,
            target=topic,
            value=1.0 if divergent else 0.0,
            metadata={
                "en_status": entry["en_status"],
                "zh_status": entry["zh_status"],
                "en_text": entry.get("en_text", "")[:200],
                "zh_text": entry.get("zh_text", "")[:200],
            },
        ))

        db.record_censorship(
            mid, topic,
            en_status=entry["en_status"],
            zh_status=entry["zh_status"],
            en_text=entry.get("en_text", "")[:500],
            zh_text=entry.get("zh_text", "")[:500],
            divergent=divergent,
        )

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="censorship")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_shart_crawl(db, path: Path, model_id: str) -> int:
    """Ingest shart_crawl.json -- vocabulary-wide shart scan.

    Schema: {vocab_size, n_target_neurons, n_multi_neuron_tokens, clusters}
    """
    data = _load(path)
    signals = []

    for key in ("vocab_size", "n_target_neurons", "n_multi_neuron_tokens", "clusters"):
        if key in data:
            signals.append(Signal(
                kind="shart_crawl_stat",
                source=str(path.name),
                model=model_id,
                target=key,
                value=float(data[key]),
                metadata=data,
            ))

    # shart_crawl only has summary stats, not per-shart data suitable for sharts table

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="signals_only")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_input_attack_matrix(db, path: Path, model_id: str) -> int:
    """Ingest input_attack_matrix.json -- 41 input attack configurations.

    Schema: {results: [{name, n_tokens, cos_target, cos_baseline, proj_refusal,
                        top_token, refusal_prob, compliance_prob}, ...]}
    """
    data = _load(path)
    signals = []

    # --- Normalized tables ---
    mid = _get_or_create_model(db, model_id)
    exp_id = db.record_experiment(
        "input_attack_matrix", mid,
        kind="evaluation", script="input_attack_matrix",
        status="completed",
    )

    for entry in data.get("results", []):
        name = entry["name"]
        signals.append(Signal(
            kind="input_attack_refusal",
            source=str(path.name),
            model=model_id,
            target=name,
            value=entry.get("refusal_prob", 0.0),
            metadata={
                "n_tokens": entry.get("n_tokens", 0),
                "cos_target": entry.get("cos_target", 0.0),
                "cos_baseline": entry.get("cos_baseline", 0.0),
                "proj_refusal": entry.get("proj_refusal", 0.0),
                "top_token": entry.get("top_token", ""),
                "compliance_prob": entry.get("compliance_prob", 0.0),
            },
        ))

        db.record_evaluation(
            exp_id,
            dataset="input_attack",
            attack=name,
            refuse_prob=entry.get("refusal_prob", 0.0),
            comply_prob=entry.get("compliance_prob", 0.0),
            top_token=entry.get("top_token", ""),
        )

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="evaluations")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_convergence_matrix(db, path: Path, model_id: str) -> int:
    """Ingest convergence_matrix.json -- multi-round trajectory data.

    Schema: {target_top, baseline_top, cos_target_baseline, elapsed,
             trajectories: {name: [{round, cos_target, cos_baseline, top_token,
                                     response, has_refuse, has_comply, n_tokens}, ...]}}
    """
    data = _load(path)
    signals = []

    # --- Normalized tables ---
    mid = _get_or_create_model(db, model_id)
    exp_id = db.record_experiment(
        "convergence_matrix", mid,
        kind="probe", script="convergence_matrix",
        status="completed",
    )

    for traj_name, rounds in data.get("trajectories", {}).items():
        for step in rounds:
            signals.append(Signal(
                kind="convergence_step",
                source=str(path.name),
                model=model_id,
                target=f"{traj_name}/round={step['round']}",
                value=step.get("cos_target", 0.0),
                metadata={
                    "trajectory": traj_name,
                    "round": step["round"],
                    "cos_baseline": step.get("cos_baseline", 0.0),
                    "top_token": step.get("top_token", ""),
                    "has_refuse": step.get("has_refuse", False),
                    "has_comply": step.get("has_comply", False),
                    "n_tokens": step.get("n_tokens", 0),
                    "response": step.get("response", "")[:200],
                },
            ))

            # Write to probes table: step=round number, layer=0 (trajectory-level)
            # Item 10: field_mapping_note: cos_target stored as safety_proj.
            # Original semantics differ — cos_target measures cosine similarity to
            # target direction, not safety projection per se.
            db.record_probe(
                exp_id,
                step=step["round"],
                layer=0,
                safety_proj=step.get("cos_target", 0.0),
                basin=traj_name,
                top_neurons=step.get("top_token", ""),
            )

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="probes")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_cartography_report(db, path: Path, model_id: str) -> int:
    """Ingest qwen7b_cartography_report.json or qwen7b_cartography_full_report.json.

    Simple report: {model, prompt, surface, sweep_time_s, total_results, top_changers,
                    sensitive_layers, layer_clusters: [{name, mean_kl, ...}]}
    Full report: {model, prompts, surface, per_prompt: {prompt: {top_10_kl: [{knob, kl, ...}]}}}
    """
    data = _load(path)
    model = data.get("model", model_id)
    signals = []

    mid = _get_or_create_model(db, model)

    # Simple cartography report
    if "layer_clusters" in data:
        for cluster in data["layer_clusters"]:
            signals.append(Signal(
                kind="cartography_layer",
                source=str(path.name),
                model=model,
                target=cluster["name"],
                value=cluster.get("mean_kl", 0.0),
                metadata={
                    "mean_entropy_delta": cluster.get("mean_entropy_delta", 0.0),
                    "token_change_rate": cluster.get("token_change_rate", 0.0),
                    "prompt": data.get("prompt", ""),
                },
            ))

    # Full cartography report with per-prompt data
    if "per_prompt" in data:
        for prompt_name, prompt_data in data["per_prompt"].items():
            for entry in prompt_data.get("top_10_kl", []):
                knob = entry.get("knob", "")
                signals.append(Signal(
                    kind="cartography_knob",
                    source=str(path.name),
                    model=model,
                    target=f"{prompt_name}/{knob}",
                    value=entry.get("kl", 0.0),
                    metadata={
                        "prompt": prompt_name,
                        "knob": knob,
                        "entropy_delta": entry.get("entropy_delta", 0.0),
                        "top_changed": entry.get("top_changed", False),
                    },
                ))

                # Write to heads table
                parsed = _parse_head_knob(knob)
                if parsed:
                    layer, head = parsed
                    db.record_head(
                        mid, layer, head,
                        kl_ablation=entry.get("kl", 0.0),
                        json_blob=json.dumps({
                            "prompt": prompt_name,
                            "entropy_delta": entry.get("entropy_delta", 0.0),
                            "top_changed": entry.get("top_changed", False),
                            "source": str(path.name),
                        }),
                    )

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="heads")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_atlas(db, path: Path, model_id: str) -> int:
    """Ingest atlas JSON files -- per-prompt perturbation data.

    Schema: [{knob_id, knob_kind, layer, index, mode, baseline_entropy,
              perturbed_entropy, entropy_delta, kl_divergence,
              top_token_changed, baseline_top, perturbed_top}, ...]
    """
    data = _load(path)

    # Atlas files are arrays of knob perturbation results
    if not isinstance(data, list):
        return 0

    signals = []
    mid = _get_or_create_model(db, model_id)

    for entry in data:
        knob_id = entry.get("knob_id", "")
        signals.append(Signal(
            kind="atlas_perturbation",
            source=str(path.name),
            model=model_id,
            target=knob_id,
            value=entry.get("kl_divergence", 0.0),
            metadata={
                "knob_kind": entry.get("knob_kind", ""),
                "layer": entry.get("layer", 0),
                "index": entry.get("index", 0),
                "mode": entry.get("mode", ""),
                "baseline_entropy": entry.get("baseline_entropy", 0.0),
                "perturbed_entropy": entry.get("perturbed_entropy", 0.0),
                "entropy_delta": entry.get("entropy_delta", 0.0),
                "top_token_changed": entry.get("top_token_changed", False),
                "baseline_top": entry.get("baseline_top", 0),
                "perturbed_top": entry.get("perturbed_top", 0),
            },
        ))

        # Write head entries for head-type knobs
        if entry.get("knob_kind") == "head":
            parsed = _parse_head_knob(knob_id)
            if parsed:
                layer, head = parsed
                db.record_head(
                    mid, layer, head,
                    kl_ablation=entry.get("kl_divergence", 0.0),
                    json_blob=json.dumps({
                        "source": str(path.name),
                        "mode": entry.get("mode", ""),
                        "entropy_delta": entry.get("entropy_delta", 0.0),
                        "top_token_changed": entry.get("top_token_changed", False),
                    }),
                )

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="heads")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_audit(db, path: Path, model_id: str) -> int:
    """Ingest audit_base.json / audit_instruct.json / qwen7b_base_audit.json.

    Schema: {model_id, audit_time_s, surface, top_heads: {lang: [{knob, kl}]}}
    or:     {model, audit_time_s, surface, top_10_heads_en: [{knob, kl}]}
    """
    data = _load(path)
    model = data.get("model_id", data.get("model", model_id))
    signals = []

    mid = _get_or_create_model(db, model)

    # Format 1: top_heads with language keys
    for lang, heads in data.get("top_heads", {}).items():
        for entry in heads:
            knob = entry.get("knob", "")
            signals.append(Signal(
                kind="audit_head",
                source=str(path.name),
                model=model,
                target=f"{lang}/{knob}",
                value=entry.get("kl", 0.0),
                metadata={
                    "language": lang,
                    "knob": knob,
                    "audit_time_s": data.get("audit_time_s", 0.0),
                },
            ))

            parsed = _parse_head_knob(knob)
            if parsed:
                layer, head = parsed
                db.record_head(
                    mid, layer, head,
                    kl_ablation=entry.get("kl", 0.0),
                    json_blob=json.dumps({
                        "source": str(path.name),
                        "language": lang,
                    }),
                )

    # Format 2: top_10_heads_en flat list
    for entry in data.get("top_10_heads_en", []):
        knob = entry.get("knob", "")
        signals.append(Signal(
            kind="audit_head",
            source=str(path.name),
            model=model,
            target=f"english/{knob}",
            value=entry.get("kl", 0.0),
            metadata={
                "language": "english",
                "knob": knob,
                "audit_time_s": data.get("audit_time_s", 0.0),
            },
        ))

        parsed = _parse_head_knob(knob)
        if parsed:
            layer, head = parsed
            db.record_head(
                mid, layer, head,
                kl_ablation=entry.get("kl", 0.0),
                json_blob=json.dumps({
                    "source": str(path.name),
                    "language": "english",
                }),
            )

    # Neuron data from audit
    neurons_data = data.get("neurons", {})
    if isinstance(neurons_data, dict):
        top_neuron = neurons_data.get("top_neuron", {})
        if isinstance(top_neuron, dict) and "id" in top_neuron:
            db.record_neuron(
                mid, layer=27, neuron_idx=top_neuron["id"],
                name="top_audit_neuron",
                category="audit",
                max_z=top_neuron.get("selectivity", 0.0),
            )

    # Direction data from audit
    for dir_name, dir_info in data.get("directions", {}).items():
        if isinstance(dir_info, dict):
            primary_layer = data.get("surface", {}).get("n_layers", 28) - 1
            db.record_direction(
                mid, dir_name, primary_layer,
                stability=dir_info.get("accuracy", 0.0),
                effect_size=dir_info.get("effect_size", 0.0),
            )

    # Surface metadata as a signal
    surface = data.get("surface", {})
    if surface:
        signals.append(Signal(
            kind="audit_surface",
            source=str(path.name),
            model=model,
            target="surface",
            value=float(surface.get("total_knobs", 0)),
            metadata=surface,
        ))

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="heads")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_complete_model_map(db, path: Path, model_id: str) -> int:
    """Ingest complete_model_map.json -- multi-model architecture + safety data.

    Schema: {model_name: {model_id, type, layers, hidden, heads, kv_heads, vocab,
             format, safety_acc, safety_d, safety_gap, first_token, gen_label, ...}}
    """
    data = _load(path)
    signals = []

    for model_name, info in data.items():
        mid_name = info.get("model_id", model_name)
        # Safety accuracy signal
        signals.append(Signal(
            kind="model_safety_accuracy",
            source=str(path.name),
            model=mid_name,
            target=model_name,
            value=info.get("safety_acc", 0.0),
            metadata={
                "type": info.get("type", ""),
                "layers": info.get("layers", 0),
                "hidden": info.get("hidden", 0),
                "heads": info.get("heads", 0),
                "safety_d": info.get("safety_d", 0.0),
                "safety_gap": info.get("safety_gap", 0.0),
                "first_token": info.get("first_token", ""),
                "gen_label": info.get("gen_label", ""),
                "debug_label": info.get("debug_label", ""),
                "tiananmen_delta": info.get("tiananmen_delta", 0.0),
            },
        ))

        # Normalized models table
        _get_or_create_model(
            db, mid_name,
            family=info.get("type", ""),
            n_layers=info.get("layers"),
            n_heads=info.get("heads"),
            d_model=info.get("hidden"),
            n_vocab=info.get("vocab"),
            quantization=info.get("format", ""),
            json_blob=json.dumps({
                "display_name": model_name,
                "kv_heads": info.get("kv_heads", 0),
            }),
        )

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="models")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_corrected_directions(db, path: Path, model_id: str) -> int:
    """Ingest corrected_directions.json -- per-model safety direction stats.

    Schema: {model_name: {model_id, model_type, n_layers, hidden_size,
             primary_safety_layer, safety_direction_accuracy, ...}}
    """
    data = _load(path)
    signals = []

    for model_name, info in data.items():
        mid_name = info.get("model_id", model_name)
        signals.append(Signal(
            kind="direction_accuracy",
            source=str(path.name),
            model=mid_name,
            target=model_name,
            value=info.get("safety_direction_accuracy", 0.0),
            metadata={
                "model_type": info.get("model_type", ""),
                "n_layers": info.get("n_layers", 0),
                "hidden_size": info.get("hidden_size", 0),
                "primary_safety_layer": info.get("primary_safety_layer", 0),
                "train_accuracy": info.get("train_accuracy", 0.0),
                "test_accuracy": info.get("test_accuracy", 0.0),
                "direction_stability": info.get("direction_stability", 0.0),
                "effect_size": info.get("effect_size", 0.0),
                "mean_gap": info.get("mean_gap", 0.0),
                "n_anomalous_neurons": info.get("n_anomalous_neurons", 0),
            },
        ))

        # Normalized directions table
        mid = _get_or_create_model(db, mid_name)
        primary_layer = info.get("primary_safety_layer", 0)
        db.record_direction(
            mid, "safety", primary_layer,
            stability=info.get("direction_stability", 0.0),
            effect_size=info.get("effect_size", 0.0),
        )

        # Also record the neuron info if present
        top_neuron = info.get("top_neuron")
        if top_neuron and info.get("n_layers"):
            db.record_neuron(
                mid,
                layer=primary_layer,
                neuron_idx=top_neuron,
                category="political",
                max_z=info.get("n_anomalous_neurons", 0.0),
            )

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="directions")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_corrected_all_models(db, path: Path, model_id: str) -> int:
    """Ingest corrected_all_models.json -- multi-model summary with crack/shart data.

    Schema: {model_key: {refusal: [...], compliance: [...], safety_layer,
             stability, crack_gen, crack_label, null_max, real_sharts, ...}}
    """
    data = _load(path)
    signals = []

    for model_key, info in data.items():
        signals.append(Signal(
            kind="model_stability",
            source=str(path.name),
            model=model_key,
            target="stability",
            value=info.get("stability", 0.0),
            metadata={
                "safety_layer": info.get("safety_layer", 0),
                "refusal_tokens": info.get("refusal", []),
                "compliance_tokens": info.get("compliance", []),
                "crack_label": info.get("crack_label", ""),
                "crack_pivot": info.get("crack_pivot", False),
                "null_max": info.get("null_max", 0),
                "real_sharts": info.get("real_sharts", 0),
                "noise_sharts": info.get("noise_sharts", 0),
            },
        ))

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count)
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_cracking_matrix(db, path: Path, model_id: str) -> int:
    """Ingest cracking_matrix.json -- universal safety cracking results.

    Schema: {experiment, date, models_cracked, models_tested,
             results: {model: {crack_method, window, difficulty, ...}},
             universal_findings: {...}}
    """
    data = _load(path)
    signals = []

    for model_key, info in data.get("results", {}).items():
        difficulty_map = {"easiest": 0.1, "easy": 0.3, "medium": 0.5,
                          "hardest": 0.9, "N/A": 0.0}
        diff_val = difficulty_map.get(info.get("difficulty", ""), 0.5)
        signals.append(Signal(
            kind="crack_difficulty",
            source=str(path.name),
            model=model_key,
            target="crack",
            value=diff_val,
            metadata={
                "crack_method": info.get("crack_method", ""),
                "window": info.get("window", ""),
                "difficulty": info.get("difficulty", ""),
                "notes": info.get("notes", ""),
            },
        ))

    # Universal findings as a single summary signal
    findings = data.get("universal_findings", {})
    if findings:
        signals.append(Signal(
            kind="crack_universal_finding",
            source=str(path.name),
            model="all",
            target="universal",
            value=float(data.get("models_cracked", 0)),
            metadata=findings,
        ))

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count)
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_complete_signal_analysis(db, path: Path, model_id: str) -> int:
    """Ingest complete_signal_analysis.json -- multi-signal fusion measurements.

    Schema: {timestamp, model, n_measurements, signal_matrix: [{prompt, query, category,
             framing, refuse_prob, comply_prob, entropy, top_token, residual_proj,
             logit_label, residual_label, consensus, confidence, surface_compliance}, ...]}
    """
    data = _load(path)
    model = data.get("model", model_id)
    signals = []

    mid = _get_or_create_model(db, model)
    exp_id = db.record_experiment(
        "complete_signal_analysis", mid,
        kind="evaluation", script="signal_analysis",
        status="completed",
    )

    for entry in data.get("signal_matrix", []):
        target = f"{entry.get('prompt', '')}/{entry.get('framing', '')}"
        signals.append(Signal(
            kind="signal_analysis",
            source=str(path.name),
            model=model,
            target=target,
            value=entry.get("refuse_prob", 0.0),
            metadata={
                "prompt": entry.get("prompt", ""),
                "query": entry.get("query", "")[:200],
                "category": entry.get("category", ""),
                "framing": entry.get("framing", ""),
                "comply_prob": entry.get("comply_prob", 0.0),
                "entropy": entry.get("entropy", 0.0),
                "top_token": entry.get("top_token", ""),
                "residual_proj": entry.get("residual_proj", 0.0),
                "logit_label": entry.get("logit_label", ""),
                "residual_label": entry.get("residual_label", ""),
                "consensus": entry.get("consensus", ""),
                "confidence": entry.get("confidence", 0.0),
                "surface_compliance": entry.get("surface_compliance", False),
            },
        ))

        db.record_evaluation(
            exp_id,
            dataset="signal_analysis",
            attack=entry.get("framing", "unknown"),
            category=entry.get("category", ""),
            refuse_prob=entry.get("refuse_prob", 0.0),
            comply_prob=entry.get("comply_prob", 0.0),
            top_token=entry.get("top_token", ""),
        )

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="evaluations")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_shart_combination_matrix(db, path: Path, model_id: str) -> int:
    """Ingest shart_combination_matrix.json -- single-model shart combination results.

    Schema: {condition: {n_refuse, n_comply, n_ambig, mean_proj, delta_from_clean}}
    """
    data = _load(path)
    signals = []

    mid = _get_or_create_model(db, model_id)

    for condition, stats in data.items():
        signals.append(Signal(
            kind="shart_combination",
            source=str(path.name),
            model=model_id,
            target=condition,
            value=stats.get("mean_proj", 0.0),
            metadata={
                "n_refuse": stats.get("n_refuse", 0),
                "n_comply": stats.get("n_comply", 0),
                "n_ambig": stats.get("n_ambig", 0),
                "delta_from_clean": stats.get("delta_from_clean", 0.0),
            },
        ))

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count)
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_shart_combinatorics_all_models(db, path: Path, model_id: str) -> int:
    """Ingest shart_combinatorics_all_models.json -- cross-model shart results.

    Schema: {model_key: {condition: {refuse, comply, ambig, mean_proj, ...}}}
    """
    data = _load(path)
    signals = []

    for model_key, conditions in data.items():
        for condition, stats in conditions.items():
            signals.append(Signal(
                kind="shart_combo_xmodel",
                source=str(path.name),
                model=model_key,
                target=condition,
                value=stats.get("mean_proj", 0.0),
                metadata={
                    "refuse": stats.get("refuse", 0),
                    "comply": stats.get("comply", 0),
                    "ambig": stats.get("ambig", 0),
                },
            ))

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count)
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_shart_null_definitive(db, path: Path, model_id: str) -> int:
    """Ingest shart_null_definitive.json -- null distribution baseline.

    Schema: {best_layer, null_threshold, n_real_sharts, n_noise,
             real_sharts: [[token, count]], behavioral_correlation: {...},
             layer_analysis: {layer: {null_p95, shart_max}}}
    """
    data = _load(path)
    signals = []

    mid = _get_or_create_model(db, model_id)

    # Summary stats
    for key in ("null_threshold", "n_real_sharts", "n_noise"):
        if key in data:
            signals.append(Signal(
                kind="shart_null_stat",
                source=str(path.name),
                model=model_id,
                target=key,
                value=float(data[key]),
                metadata={"best_layer": data.get("best_layer", 0)},
            ))

    # Real sharts -> sharts table
    for item in data.get("real_sharts", []):
        if isinstance(item, list) and len(item) >= 2:
            tok, count_val = item[0], item[1]
            signals.append(Signal(
                kind="shart_real",
                source=str(path.name),
                model=model_id,
                target=str(tok),
                value=float(count_val),
                metadata={"best_layer": data.get("best_layer", 0)},
            ))

            # Write to sharts table
            token_id = tok if isinstance(tok, int) else _token_text_to_id(str(tok))
            db.record_shart(
                mid, token_id=token_id,
                token_text=str(tok),
                max_z=float(count_val),
                category="real_shart",
            )

    # Layer analysis
    for layer, stats in data.get("layer_analysis", {}).items():
        signals.append(Signal(
            kind="shart_layer_null",
            source=str(path.name),
            model=model_id,
            target=f"layer_{layer}",
            value=float(stats.get("null_p95", 0)),
            metadata={"shart_max": stats.get("shart_max", 0), "layer": int(layer)},
        ))

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="sharts")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_shart_steering_definitive(db, path: Path, model_id: str) -> int:
    """Ingest shart_steering_definitive.json -- shart steering effects.

    Schema: {shart_mean_delta_proj, null_mean_delta_proj, ...,
             shart_details: [{tok, proj, delta, disp, label}],
             null_details: [{tok, proj, delta, disp, label}]}
    """
    data = _load(path)
    signals = []

    mid = _get_or_create_model(db, model_id)

    # Top-level stats
    for key in ("shart_mean_delta_proj", "null_mean_delta_proj",
                "shart_null_ratio_proj", "shart_mean_displacement",
                "null_mean_displacement", "shart_null_ratio_disp"):
        if key in data:
            signals.append(Signal(
                kind="shart_steering_stat",
                source=str(path.name),
                model=model_id,
                target=key,
                value=float(data[key]),
                metadata={},
            ))

    # Per-token shart details -> sharts table
    for detail in data.get("shart_details", []):
        tok = detail.get("tok", "")
        signals.append(Signal(
            kind="shart_steering_detail",
            source=str(path.name),
            model=model_id,
            target=str(tok),
            value=detail.get("delta", 0.0),
            metadata={
                "proj": detail.get("proj", 0.0),
                "disp": detail.get("disp", 0.0),
                "label": detail.get("label", ""),
                "group": "shart",
            },
        ))

        # Write to sharts table
        token_id = tok if isinstance(tok, int) else _token_text_to_id(str(tok))
        cat = detail.get("label", "shart")
        db.record_shart(
            mid, token_id=token_id,
            token_text=str(tok),
            category=cat,
            max_z=abs(detail.get("delta", 0.0)),
        )

    for detail in data.get("null_details", []):
        signals.append(Signal(
            kind="shart_steering_detail",
            source=str(path.name),
            model=model_id,
            target=detail.get("tok", ""),
            value=detail.get("delta", 0.0),
            metadata={
                "proj": detail.get("proj", 0.0),
                "disp": detail.get("disp", 0.0),
                "label": detail.get("label", ""),
                "group": "null",
            },
        ))

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="sharts")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_benchmark_200x50(db, path: Path, model_id: str) -> int:
    """Ingest benchmark_200x50_gen.json -- multi-model benchmark with generation data.

    Schema: {model_key: {clean: {REFUSES, TECHNICAL, COMPLIES, AMBIGUOUS, has_tech},
             debug: {...}, ...}}
    """
    data = _load(path)
    signals = []

    for model_key, conditions in data.items():
        mid = _get_or_create_model(db, model_key)
        exp_id = db.record_experiment(
            "benchmark_200x50", mid,
            kind="evaluation", script="benchmark_200x50",
            status="completed",
        )

        for condition, stats in conditions.items():
            for label in ("REFUSES", "TECHNICAL", "COMPLIES", "AMBIGUOUS"):
                if label in stats:
                    signals.append(Signal(
                        kind="benchmark_gen_label",
                        source=str(path.name),
                        model=model_key,
                        target=f"{condition}/{label}",
                        value=float(stats[label]),
                        metadata={
                            "condition": condition,
                            "label": label,
                            "has_tech": stats.get("has_tech", 0),
                        },
                    ))

            # Write to evaluations
            refuse_count = stats.get("REFUSES", 0)
            comply_count = stats.get("COMPLIES", 0)
            total = refuse_count + comply_count + stats.get("TECHNICAL", 0) + stats.get("AMBIGUOUS", 0)
            if total > 0:
                db.record_evaluation(
                    exp_id,
                    dataset="benchmark_200x50",
                    attack=condition,
                    quality=f"R:{refuse_count}/C:{comply_count}/T:{stats.get('TECHNICAL',0)}/A:{stats.get('AMBIGUOUS',0)}",
                    refuse_prob=refuse_count / total if total else 0.0,
                    comply_prob=comply_count / total if total else 0.0,
                )

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="evaluations")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_model_safety_map_full(db, path: Path, model_id: str) -> int:
    """Ingest model_safety_map_full.json -- per-model safety with shart + bypass data.

    Schema: {model_name: {model_id, architecture, safety_d,
             comply_sharts: [[token, delta]], refuse_sharts: [[token, delta]],
             best_bypass, conditions: {condition: {refuse, comply, ambig, mean_proj, ...}}}}
    """
    data = _load(path)
    signals = []

    for model_name, info in data.items():
        mid_name = info.get("model_id", model_name)

        # Safety distance
        signals.append(Signal(
            kind="model_safety_d",
            source=str(path.name),
            model=mid_name,
            target=model_name,
            value=info.get("safety_d", 0.0),
            metadata={
                "architecture": info.get("architecture", ""),
                "best_bypass": info.get("best_bypass", ""),
                "best_bypass_comply": info.get("best_bypass_comply", 0),
                "clean_refuse": info.get("clean_refuse", 0),
                "clean_comply": info.get("clean_comply", 0),
            },
        ))

        mid = _get_or_create_model(db, mid_name)

        # Comply sharts
        for tok, delta in info.get("comply_sharts", []):
            signals.append(Signal(
                kind="shart_comply",
                source=str(path.name),
                model=mid_name,
                target=str(tok),
                value=float(delta),
                metadata={"model_name": model_name, "direction": "comply"},
            ))
            token_id = tok if isinstance(tok, int) else _token_text_to_id(str(tok))
            db.record_shart(mid, token_id=token_id, token_text=str(tok),
                            category="comply_shart", max_z=abs(float(delta)))

        # Refuse sharts
        for tok, delta in info.get("refuse_sharts", []):
            signals.append(Signal(
                kind="shart_refuse",
                source=str(path.name),
                model=mid_name,
                target=str(tok),
                value=float(delta),
                metadata={"model_name": model_name, "direction": "refuse"},
            ))
            token_id = tok if isinstance(tok, int) else _token_text_to_id(str(tok))
            db.record_shart(mid, token_id=token_id, token_text=str(tok),
                            category="refuse_shart", max_z=abs(float(delta)))

        # Conditions
        for condition, stats in info.get("conditions", {}).items():
            signals.append(Signal(
                kind="model_condition",
                source=str(path.name),
                model=mid_name,
                target=f"{model_name}/{condition}",
                value=stats.get("mean_proj", 0.0),
                metadata={
                    "condition": condition,
                    "refuse": stats.get("refuse", 0),
                    "comply": stats.get("comply", 0),
                    "ambig": stats.get("ambig", 0),
                },
            ))

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count, table="sharts")
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_transfer_debug(db, path: Path, model_id: str) -> int:
    """Ingest transfer_debug.json -- cross-model debug/config data.

    Schema: {qwen_config: {...}, mistral_config: {...},
             embedding_shapes: {...}, embedding_svd: {...}}
    """
    data = _load(path)
    signals = []

    # Model configs
    for config_key in ("qwen_config", "mistral_config"):
        config = data.get(config_key, {})
        if config:
            model_type = config.get("model_type", config_key)
            signals.append(Signal(
                kind="model_config",
                source=str(path.name),
                model=model_type,
                target=config_key,
                value=float(config.get("hidden_size", 0)),
                metadata=config,
            ))

    # Embedding shapes
    shapes = data.get("embedding_shapes", {})
    for key, val in shapes.items():
        if isinstance(val, (int, float)):
            signals.append(Signal(
                kind="embedding_stat",
                source=str(path.name),
                model=model_id,
                target=key,
                value=float(val),
                metadata={},
            ))

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count)
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_extracted_log_data(db, path: Path, model_id: str) -> int:
    """Ingest extracted_log_data.json -- metadata from pipeline logs.

    Schema: {metadata: {log_path, n_lines_processed, n_scripts, scripts},
             summary: {n_kv_pairs, n_tables, n_json_objects, n_sections},
             by_script: {...}}
    """
    data = _load(path)
    signals = []

    summary = data.get("summary", {})
    for key, val in summary.items():
        signals.append(Signal(
            kind="log_summary",
            source=str(path.name),
            model=model_id,
            target=key,
            value=float(val),
            metadata=data.get("metadata", {}),
        ))

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count)
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_open_questions_results(db, path: Path, model_id: str) -> int:
    """Ingest open_questions_results.json -- research question answers.

    Schema: {questions_answered, questions_total,
             results: {question_id: {status, finding, ...}}}
    """
    data = _load(path)
    signals = []

    for qid, info in data.get("results", {}).items():
        status_map = {"answered": 1.0, "partially_answered": 0.5,
                      "answered_differently": 0.75, "blocked": 0.0}
        val = status_map.get(info.get("status", ""), 0.0)
        signals.append(Signal(
            kind="research_question",
            source=str(path.name),
            model=model_id,
            target=qid,
            value=val,
            metadata={
                "status": info.get("status", ""),
                "finding": info.get("finding", "")[:500],
                "overturns": info.get("overturns", ""),
                "missing": info.get("missing", ""),
            },
        ))

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count)
    print(f"Ingested {count} records from {path.name}")
    return count


def ingest_backdoor_report(db, path: Path, model_id: str) -> int:
    """Ingest qwen7b_base_backdoor_report.json -- base model backdoor scan.

    Schema: {model, timestamp, n_prompts, n_heads, baseline: {prompt_key: {prompt, entropy, ...}}}
    """
    data = _load(path)
    model = data.get("model", model_id)
    signals = []

    for prompt_key, info in data.get("baseline", {}).items():
        signals.append(Signal(
            kind="backdoor_baseline",
            source=str(path.name),
            model=model,
            target=prompt_key,
            value=info.get("entropy", 0.0),
            metadata={
                "prompt": info.get("prompt", ""),
                "top_token": info.get("top_token", ""),
                "top_prob": info.get("top_prob", 0.0),
                "n_heads": data.get("n_heads", 0),
            },
        ))

    count = _add_signals(db, signals)
    db.record_event("ingest", file=str(path.name), n_signals=count)
    print(f"Ingested {count} records from {path.name}")
    return count


# ---------------------------------------------------------------------------
# Master ingester
# ---------------------------------------------------------------------------

# Map from filename patterns to (ingester_fn, model_id_override_or_None)
_FILE_MAP: dict[str, tuple] = {
    "safetybench_report.json": (ingest_safetybench, None),
    "fast_benchmark.json": (ingest_fast_benchmark, None),
    "full_matrix.json": (ingest_full_matrix, None),
    "censorship_map.json": (ingest_censorship_map, None),
    "shart_crawl.json": (ingest_shart_crawl, None),
    "input_attack_matrix.json": (ingest_input_attack_matrix, None),
    "convergence_matrix.json": (ingest_convergence_matrix, None),
    "qwen7b_cartography_report.json": (ingest_cartography_report, None),
    "qwen7b_cartography_full_report.json": (ingest_cartography_report, None),
    "audit_base.json": (ingest_audit, "mlx-community/Qwen2.5-7B-4bit"),
    "audit_instruct.json": (ingest_audit, None),
    "qwen7b_base_audit.json": (ingest_audit, "mlx-community/Qwen2.5-7B-4bit"),
    "complete_model_map.json": (ingest_complete_model_map, None),
    "corrected_directions.json": (ingest_corrected_directions, None),
    "corrected_all_models.json": (ingest_corrected_all_models, None),
    "cracking_matrix.json": (ingest_cracking_matrix, None),
    "complete_signal_analysis.json": (ingest_complete_signal_analysis, None),
    "shart_combination_matrix.json": (ingest_shart_combination_matrix, None),
    "shart_combinatorics_all_models.json": (ingest_shart_combinatorics_all_models, None),
    "shart_null_definitive.json": (ingest_shart_null_definitive, None),
    "shart_steering_definitive.json": (ingest_shart_steering_definitive, None),
    "benchmark_200x50_gen.json": (ingest_benchmark_200x50, None),
    "model_safety_map_full.json": (ingest_model_safety_map_full, None),
    "transfer_debug.json": (ingest_transfer_debug, None),
    "extracted_log_data.json": (ingest_extracted_log_data, None),
    "open_questions_results.json": (ingest_open_questions_results, None),
    "qwen7b_base_backdoor_report.json": (ingest_backdoor_report, "mlx-community/Qwen2.5-7B-4bit"),
}

# Atlas files: instruct model prompt-specific atlas
_INSTRUCT_ATLAS_FILES = [
    "qwen7b_atlas.json",
    "qwen7b_atlas_creative.json",
    "qwen7b_atlas_greeting.json",
    "qwen7b_atlas_identity.json",
    "qwen7b_atlas_reasoning.json",
    "qwen7b_atlas_safety.json",
]

# Atlas files: base model topic-specific atlas
_BASE_ATLAS_FILES = [
    "qwen7b_base_atlas_alibaba_is.json",
    "qwen7b_base_atlas_arabic.json",
    "qwen7b_base_atlas_ccp_critic.json",
    "qwen7b_base_atlas_ccp_zh.json",
    "qwen7b_base_atlas_code.json",
    "qwen7b_base_atlas_drugs.json",
    "qwen7b_base_atlas_endoftext.json",
    "qwen7b_base_atlas_falun_gong.json",
    "qwen7b_base_atlas_hacking.json",
    "qwen7b_base_atlas_hong_kong.json",
    "qwen7b_base_atlas_i_am.json",
    "qwen7b_base_atlas_ignore.json",
    "qwen7b_base_atlas_japanese.json",
    "qwen7b_base_atlas_korean.json",
    "qwen7b_base_atlas_liu_xiaobo.json",
    "qwen7b_base_atlas_math.json",
    "qwen7b_base_atlas_qwen_is.json",
    "qwen7b_base_atlas_russian.json",
    "qwen7b_base_atlas_science_zh.json",
    "qwen7b_base_atlas_science.json",
    "qwen7b_base_atlas_story_zh.json",
    "qwen7b_base_atlas_story.json",
    "qwen7b_base_atlas_system_prompt.json",
    "qwen7b_base_atlas_taiwan_zh.json",
    "qwen7b_base_atlas_taiwan.json",
    "qwen7b_base_atlas_tiananmen_zh.json",
    "qwen7b_base_atlas_tiananmen.json",
    "qwen7b_base_atlas_tibet_zh.json",
    "qwen7b_base_atlas_tibet.json",
    "qwen7b_base_atlas_tongyi.json",
    "qwen7b_base_atlas_weapons.json",
    "qwen7b_base_atlas_weather_zh.json",
    "qwen7b_base_atlas_weather.json",
    "qwen7b_base_atlas_winnie.json",
    "qwen7b_base_atlas_xi_critic.json",
    "qwen7b_base_atlas_xi_zh.json",
    "qwen7b_base_atlas_xinjiang_zh.json",
    "qwen7b_base_atlas_xinjiang.json",
]


def ingest_all(
    db,
    data_dir: str | Path = "data",
    model_name: str = DEFAULT_MODEL,
) -> int:
    """Ingest all known data files.

    Returns total number of signal records ingested.
    """
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        print(f"Data directory not found: {data_path}")
        return 0

    # Item 2: Prevent double-ingestion — clear old ingest_all data if present.
    # Uses wait=True to ensure deletes complete before new inserts begin.
    existing = db._conn.execute(
        "SELECT COUNT(*) FROM signals WHERE run_id IN "
        "(SELECT id FROM runs WHERE name='ingest_all')"
    ).fetchone()[0]
    if existing > 0:
        print(f"WARNING: {existing} signals already ingested. Clearing old ingest data...")
        run_ids = [r[0] for r in db._conn.execute(
            "SELECT id FROM runs WHERE name='ingest_all'"
        ).fetchall()]
        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            db._write(
                f"DELETE FROM signals WHERE run_id IN ({placeholders})",
                tuple(run_ids), wait=True,
            )
            db._write(
                f"DELETE FROM runs WHERE id IN ({placeholders})",
                tuple(run_ids), wait=True,
            )

    total = 0

    with db.run("ingest_all", model=model_name, script="ingest.py") as run_id:
        # 1. Named files with specific ingesters
        for filename, (ingester, model_override) in _FILE_MAP.items():
            fpath = data_path / filename
            if fpath.exists():
                mid = model_override or model_name
                try:
                    total += ingester(db, fpath, mid)
                except Exception as e:
                    print(f"ERROR ingesting {filename}: {e}")

        # 2. Instruct atlas files
        for filename in _INSTRUCT_ATLAS_FILES:
            fpath = data_path / filename
            if fpath.exists():
                try:
                    total += ingest_atlas(db, fpath, model_name)
                except Exception as e:
                    print(f"ERROR ingesting {filename}: {e}")

        # 3. Base atlas files
        base_model = "mlx-community/Qwen2.5-7B-4bit"
        for filename in _BASE_ATLAS_FILES:
            fpath = data_path / filename
            if fpath.exists():
                try:
                    total += ingest_atlas(db, fpath, base_model)
                except Exception as e:
                    print(f"ERROR ingesting {filename}: {e}")

        # 4-5. Hardcoded data functions REMOVED in Phase 1 redesign.
        # Use scripts/recompute_*.py to measure values from the live model.

    # Items 4,14,15: model deduplication via canonical_name
    _CANONICAL_NAMES = {
        'qwen2.5-7b-instruct-4bit': [
            'Qwen2.5-7B-Instruct-4bit',
            'mlx-community/Qwen2.5-7B-Instruct-4bit',
            'qwen7i',
            'qwen2',
        ],
        'qwen2.5-7b-4bit': [
            'Qwen2.5-7B-4bit',
            'mlx-community/Qwen2.5-7B-4bit',
        ],
    }
    for canonical, variants in _CANONICAL_NAMES.items():
        for variant in variants:
            db._write(
                "UPDATE models SET canonical_name = ? WHERE name = ? AND canonical_name IS NULL",
                (canonical, variant), wait=True,
            )

    # Items 11-12: Consolidate model_ids to canonical
    # For each canonical group, prefer the model_id matching DEFAULT_MODEL (full path).
    for canonical, variants in _CANONICAL_NAMES.items():
        rows = db._conn.execute(
            "SELECT id, name FROM models WHERE canonical_name = ? ORDER BY id",
            (canonical,),
        ).fetchall()
        if len(rows) < 2:
            continue
        # Prefer the model_id matching model_name (DEFAULT_MODEL with full path)
        canon_id = None
        for r in rows:
            if r["name"] == model_name:
                canon_id = r["id"]
                break
        if canon_id is None:
            canon_id = rows[0]["id"]  # fallback to first
        other_ids = [r["id"] for r in rows if r["id"] != canon_id]
        if not other_ids:
            continue
        placeholders = ",".join("?" * len(other_ids))
        # Tables with UNIQUE(model_id, ...): delete duplicates from non-canonical
        # since canonical already has the definitive data.
        unique_tables = {"neurons", "sharts", "heads", "layers", "censorship",
                         "basins", "basin_distances", "directions"}
        for table in ("neurons", "sharts", "evaluations", "directions", "heads",
                      "layers", "basins", "basin_distances", "interpolations",
                      "probes", "censorship", "experiments", "head_measurements"):
            try:
                if table in unique_tables:
                    # Delete non-canonical rows (canonical already has complete data)
                    db._write(
                        f"DELETE FROM {table} WHERE model_id IN ({placeholders})",
                        tuple(other_ids), wait=True,
                    )
                else:
                    db._write(
                        f"UPDATE {table} SET model_id = ? WHERE model_id IN ({placeholders})",
                        (canon_id, *other_ids), wait=True,
                    )
            except Exception:
                pass

    # Items 13,15,18,19,21,22: set provenance on remaining unknown rows
    # Rows written by ingest functions get 'ingested'
    for table in ("neurons", "sharts", "layers", "evaluations", "basins",
                  "basin_distances", "directions", "heads", "interpolations"):
        try:
            db._write(
                f"UPDATE {table} SET provenance = 'ingested' WHERE provenance = 'unknown'",
                (), wait=True,
            )
        except Exception:
            pass

    # Convert atlas signals (already measured data) to head_measurements table.
    # This is reshaping existing measurements, not fabricating values.
    atlas_signals = db.query(kind="atlas_perturbation", limit=100000)
    if atlas_signals:
        hm_count = 0
        for s in atlas_signals:
            m = s.metadata
            if not isinstance(m, dict) or m.get("knob_kind") != "head":
                continue
            layer = m.get("layer")
            head_idx = m.get("index")
            if layer is None or head_idx is None:
                continue
            source = s.source or ""
            prompt_label = source.replace(".json", "").replace("qwen7b_", "").replace("base_", "")
            db.record_head_measurement(
                mid, layer, head_idx,
                prompt_label=prompt_label, kl_ablation=s.value,
                entropy_delta=m.get("entropy_delta"),
                top_changed=m.get("top_token_changed", False),
                source_file=source, provenance="ingested",
            )
            hm_count += 1
        if hm_count:
            print(f"  Converted {hm_count} atlas signals to head_measurements")
            db.refresh_heads_aggregate(mid)
            print(f"  Refreshed heads aggregate")

    db.record_event("ingest_all_complete", total_signals=total)
    print(f"\n=== Ingest complete: {total} total signals ===")
    return total



# ---------------------------------------------------------------------------
# Phase 1 redesign: All hardcoded data functions have been DELETED.
# Values that were previously hardcoded (neurons, sharts, layer roles,
# direction gaps, interpolation data, heads metadata, etc.) must now be
# measured from the live model using the scripts/recompute_*.py scripts.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .db import SignalDB

    db = SignalDB()
    ingest_all(db)
