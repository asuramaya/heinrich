"""Fetch signals from a HuggingFace Hub model repository."""

from __future__ import annotations

from typing import Any

from ..signal import Signal, SignalStore
from .local import _emit_config_signals, _emit_index_signals


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


def _download_json(repo_id: str, filename: str) -> dict[str, Any] | None:
    import json
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id, filename)
        with open(path, encoding="utf-8") as f:
            return json.loads(f.read())
    except Exception:
        return None
