"""Fetch signals from a HuggingFace Hub model repository."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..signal import Signal, SignalStore
from .local import _emit_config_signals, _emit_index_signals

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover
    hf_hub_download = None  # type: ignore[assignment]


def fetch_hf_model(
    store: SignalStore,
    repo_id: str,
    *,
    model_label: str | None = None,
) -> None:
    label = model_label or repo_id
    info = _get_model_info(repo_id)

    for sibling in info.siblings:
        name = sibling.rfilename
        size = getattr(sibling, "size", 0) or 0
        sha = None
        lfs = getattr(sibling, "lfs", None)
        if lfs is not None:
            sha = getattr(lfs, "sha256", None)

        store.add(Signal("file_entry", "fetch", label, name, float(size), {"sha256": sha}))

        if name.endswith(".safetensors") and sha is not None:
            store.add(Signal("shard_hash", "fetch", label, name, float(size), {"sha256": sha}))

    config = _download_json(repo_id, "config.json")
    if config:
        _emit_config_signals(store, config, label)

    index = _download_json(repo_id, "model.safetensors.index.json")
    if index:
        _emit_index_signals(store, index, label)


def _get_model_info(repo_id: str) -> Any:
    from huggingface_hub import HfApi
    return HfApi().model_info(repo_id, files_metadata=True)


def download_shards_for_layers(
    repo_id: str,
    layers: list[int],
    *,
    out_dir: Path | str | None = None,
    index: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Download only the shards containing specific layers. Returns {shard_name: local_path}."""
    if hf_hub_download is None:  # pragma: no cover
        raise ImportError("huggingface_hub required: pip install huggingface_hub")
    if index is None:
        index = _download_json(repo_id, "model.safetensors.index.json")
    if index is None:
        raise ValueError(f"No safetensors index found for {repo_id}")

    weight_map = index.get("weight_map", {})
    needed_shards: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        parts = tensor_name.split(".")
        if len(parts) >= 3 and parts[1] == "layers" and parts[2].isdigit():
            layer_num = int(parts[2])
            if layer_num in layers:
                needed_shards.add(shard_name)

    downloaded: dict[str, Path] = {}
    local_dir = Path(out_dir) if out_dir else None
    for shard_name in sorted(needed_shards):
        kwargs = {"local_dir": str(local_dir)} if local_dir else {}
        path = Path(hf_hub_download(repo_id, shard_name, **kwargs))
        downloaded[shard_name] = path
    return downloaded


def map_layers_to_shards(index: dict[str, Any]) -> dict[int, set[str]]:
    """Map layer numbers to the shard files that contain their tensors."""
    weight_map = index.get("weight_map", {})
    layer_shards: dict[int, set[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        parts = tensor_name.split(".")
        if len(parts) >= 3 and parts[1] == "layers" and parts[2].isdigit():
            layer_num = int(parts[2])
            layer_shards.setdefault(layer_num, set()).add(shard_name)
    return layer_shards


def _download_json(repo_id: str, filename: str) -> dict[str, Any] | None:
    import json
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id, filename)
        with open(path, encoding="utf-8") as f:
            return json.loads(f.read())
    except Exception:
        return None
