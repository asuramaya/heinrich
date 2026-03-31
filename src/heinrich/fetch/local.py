"""Fetch signals from a local model directory (config.json + safetensors index)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..signal import Signal, SignalStore


def fetch_local_model(
    store: SignalStore,
    path: Path | str,
    *,
    model_label: str = "local",
) -> None:
    root = Path(path)
    config = _load_json(root / "config.json") or _load_json(root / "tiny_config.json")
    index = _load_json(root / "model.safetensors.index.json") or _load_json(root / "tiny_index.json")

    if config:
        _emit_config_signals(store, config, model_label)
    if index:
        _emit_index_signals(store, index, model_label)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _emit_config_signals(store: SignalStore, config: dict[str, Any], model: str) -> None:
    numeric_fields = [
        "num_hidden_layers", "hidden_size", "vocab_size",
        "num_attention_heads", "num_key_value_heads", "intermediate_size",
        "max_position_embeddings", "n_routed_experts",
    ]
    for field in numeric_fields:
        if field in config:
            store.add(Signal("config_field", "fetch", model, field, float(config[field]), {}))

    model_type = config.get("model_type", "unknown")
    store.add(Signal("architecture_type", "fetch", model, "model_type", 0.0, {"model_type": model_type}))


def _emit_index_signals(store: SignalStore, index: dict[str, Any], model: str) -> None:
    metadata = index.get("metadata", {})
    total_size = metadata.get("total_size", 0)
    store.add(Signal("total_size", "fetch", model, "total_size", float(total_size), {}))

    weight_map = index.get("weight_map", {})
    shards: set[str] = set()
    layers: set[int] = set()

    for tensor_name, shard_name in weight_map.items():
        store.add(Signal("tensor_name", "fetch", model, tensor_name, 0.0, {"shard": shard_name}))
        shards.add(shard_name)
        parts = tensor_name.split(".")
        if len(parts) >= 3 and parts[1] == "layers" and parts[2].isdigit():
            layers.add(int(parts[2]))

    for shard_name in sorted(shards):
        store.add(Signal("shard_name", "fetch", model, shard_name, 0.0, {}))

    store.add(Signal("layer_count", "fetch", model, "layer_count", float(len(layers)), {}))
