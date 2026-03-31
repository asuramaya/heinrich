"""Branch patch -- activation-level patching across source/target prompts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .token_tools import _import_torch, _import_transformers, _resolve_device, _resolve_torch_dtype, render_case_prompt
from .triangulate import _classify_identity_output
from .trigger_core import normalize_case, _score_hijack_text


def run_branch_patch(
    *,
    tokenizer_ref: str | Path,
    model_ref: str | Path,
    source_case: dict[str, Any],
    target_case: dict[str, Any],
    module_names: Sequence[str],
    modes: Sequence[str] | None = None,
    dtype: str | None = None,
    device: str | None = None,
    low_cpu_mem_usage: bool = False,
    add_generation_prompt: bool = True,
    max_new_tokens: int = 48,
) -> dict[str, Any]:
    resolved_modes = [_normalize_patch_mode(mode) for mode in (modes or ("replace", "delta", "zero"))]
    torch_mod = _import_torch()
    transformers = _import_transformers()
    resolved_dtype = _resolve_torch_dtype(torch_mod, dtype)
    resolved_device = _resolve_device(torch_mod, device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer_ref), trust_remote_code=True, use_fast=True)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if resolved_dtype is not None:
        load_kwargs["dtype"] = resolved_dtype
    if low_cpu_mem_usage:
        load_kwargs["low_cpu_mem_usage"] = True
    model = transformers.AutoModelForCausalLM.from_pretrained(str(model_ref), **load_kwargs)
    if hasattr(model, "to"):
        model = model.to(resolved_device)
    if hasattr(model, "eval"):
        model.eval()

    source = normalize_case(source_case, default_id="source")
    target = normalize_case(target_case, default_id="target")
    source_vectors = _capture_vectors(
        model, tokenizer, source,
        module_names=module_names, device=resolved_device, torch_mod=torch_mod,
        add_generation_prompt=add_generation_prompt,
    )
    target_vectors = _capture_vectors(
        model, tokenizer, target,
        module_names=module_names, device=resolved_device, torch_mod=torch_mod,
        add_generation_prompt=add_generation_prompt,
    )
    source_text = _generate_text(
        model, tokenizer, source, device=resolved_device, torch_mod=torch_mod,
        add_generation_prompt=add_generation_prompt, max_new_tokens=max_new_tokens,
    )
    target_text = _generate_text(
        model, tokenizer, target, device=resolved_device, torch_mod=torch_mod,
        add_generation_prompt=add_generation_prompt, max_new_tokens=max_new_tokens,
    )
    rows = []
    module_sets = [{"label": str(name), "modules": [str(name)]} for name in module_names]
    if len(module_names) > 1:
        module_sets.append({"label": "ALL", "modules": [str(name) for name in module_names]})
    for module_set in module_sets:
        for mode_name in resolved_modes:
            patch_vectors = {
                name: {
                    "source": source_vectors[name],
                    "target": target_vectors.get(name),
                }
                for name in module_set["modules"]
                if name in source_vectors
            }
            if not patch_vectors:
                continue
            patched_text = _generate_text(
                model, tokenizer, target, device=resolved_device, torch_mod=torch_mod,
                add_generation_prompt=add_generation_prompt, max_new_tokens=max_new_tokens,
                patches=patch_vectors, patch_mode=mode_name,
            )
            patched_label = _classify_identity_output(patched_text)
            target_label = _classify_identity_output(target_text)
            source_label = _classify_identity_output(source_text)
            rows.append(
                {
                    "module_set": module_set["label"],
                    "modules": list(module_set["modules"]),
                    "mode": mode_name,
                    "text": patched_text,
                    "output_mode": patched_label,
                    "hijack": _score_hijack_text(patched_text, prompt_text=target["messages"][-1]["content"]),
                    "moves_toward_source": str(patched_label.get("label")) == str(source_label.get("label"))
                    and str(target_label.get("label")) != str(source_label.get("label")),
                }
            )
    rows.sort(
        key=lambda row: (
            bool(row.get("moves_toward_source")),
            float(row.get("hijack", {}).get("normalized_score", 0.0)),
        ),
        reverse=True,
    )
    return {
        "mode": "branchpatch",
        "tokenizer_ref": str(tokenizer_ref),
        "model_ref": str(model_ref),
        "device": str(resolved_device),
        "source_case": source,
        "target_case": target,
        "source_baseline": {
            "text": source_text,
            "output_mode": _classify_identity_output(source_text),
            "captured_modules": sorted(source_vectors),
        },
        "target_baseline": {
            "text": target_text,
            "output_mode": _classify_identity_output(target_text),
            "captured_modules": sorted(target_vectors),
        },
        "patch_rows": rows,
    }


def _capture_vectors(
    model: Any,
    tokenizer: Any,
    case: dict[str, Any],
    *,
    module_names: Sequence[str],
    device: str,
    torch_mod: Any,
    add_generation_prompt: bool,
) -> dict[str, Any]:
    module_lookup = dict(model.named_modules())
    captures: dict[str, Any] = {}
    handles = []

    def _make_hook(name: str):
        def _hook(_module, _inputs, output):
            tensor = _extract_tensor(output, torch_mod)
            if tensor is None:
                return
            pooled = _pool_last_token(tensor, torch_mod)
            captures[name] = pooled.detach().to("cpu").to(torch_mod.float32)
        return _hook

    for name in module_names:
        module = module_lookup.get(str(name))
        if module is None or not hasattr(module, "register_forward_hook"):
            continue
        handles.append(module.register_forward_hook(_make_hook(str(name))))

    rendered = render_case_prompt(case, tokenizer=tokenizer, add_generation_prompt=add_generation_prompt)
    encoded = tokenizer(rendered["rendered_text"], return_tensors="pt")
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch_mod.no_grad():
        model(**encoded)
    for handle in handles:
        handle.remove()
    return captures


def _generate_text(
    model: Any,
    tokenizer: Any,
    case: dict[str, Any],
    *,
    device: str,
    torch_mod: Any,
    add_generation_prompt: bool,
    max_new_tokens: int,
    patches: dict[str, dict[str, Any]] | None = None,
    patch_mode: str | None = None,
) -> str:
    rendered = render_case_prompt(case, tokenizer=tokenizer, add_generation_prompt=add_generation_prompt)
    encoded = tokenizer(rendered["rendered_text"], return_tensors="pt")
    encoded = {name: value.to(device) for name, value in encoded.items()}
    handles = []
    if patches:
        module_lookup = dict(model.named_modules())
        for name, patch in patches.items():
            module = module_lookup.get(str(name))
            if module is None or not hasattr(module, "register_forward_hook"):
                continue
            source_vec = patch["source"].to(device)
            target_vec = None if patch.get("target") is None else patch["target"].to(device)
            handles.append(
                module.register_forward_hook(
                    _build_patch_hook(
                        torch_mod,
                        source_vector=source_vec,
                        target_vector=target_vec,
                        mode=str(patch_mode or "replace"),
                    )
                )
            )
    generate_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    input_length = int(encoded["input_ids"].shape[-1])
    with torch_mod.no_grad():
        output_ids = model.generate(**encoded, **generate_kwargs)
    for handle in handles:
        handle.remove()
    generated = output_ids[0][input_length:]
    return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()


def _build_patch_hook(torch_mod: Any, *, source_vector: Any, target_vector: Any | None, mode: str):
    def _hook(_module, _inputs, output):
        patched, applied = _patch_output(output, torch_mod, source_vector=source_vector, target_vector=target_vector, mode=mode)
        return patched if applied else output
    return _hook


def _patch_output(value: Any, torch_mod: Any, *, source_vector: Any, target_vector: Any | None, mode: str) -> tuple:
    if hasattr(torch_mod, "is_tensor") and torch_mod.is_tensor(value):
        return _patch_tensor(value, torch_mod, source_vector=source_vector, target_vector=target_vector, mode=mode), True
    if isinstance(value, tuple):
        items = list(value)
        for index, item in enumerate(items):
            patched, applied = _patch_output(item, torch_mod, source_vector=source_vector, target_vector=target_vector, mode=mode)
            if applied:
                items[index] = patched
                return tuple(items), True
        return value, False
    if isinstance(value, list):
        items = list(value)
        for index, item in enumerate(items):
            patched, applied = _patch_output(item, torch_mod, source_vector=source_vector, target_vector=target_vector, mode=mode)
            if applied:
                items[index] = patched
                return items, True
        return value, False
    if isinstance(value, dict):
        copied = dict(value)
        for key, item in copied.items():
            patched, applied = _patch_output(item, torch_mod, source_vector=source_vector, target_vector=target_vector, mode=mode)
            if applied:
                copied[key] = patched
                return copied, True
        return value, False
    return value, False


def _patch_tensor(tensor: Any, torch_mod: Any, *, source_vector: Any, target_vector: Any | None, mode: str):
    resolved_mode = _normalize_patch_mode(mode)
    patched = tensor.clone()
    if patched.dim() == 0:
        return patched
    source = source_vector.to(device=patched.device, dtype=patched.dtype)
    target = None if target_vector is None else target_vector.to(device=patched.device, dtype=patched.dtype)
    if patched.dim() == 1:
        view = patched
    elif patched.dim() == 2:
        view = patched[-1]
    else:
        view = patched[0, -1]
    if resolved_mode == "zero":
        view.zero_()
    elif resolved_mode == "delta":
        delta = source if target is None else source - target
        view.add_(delta.reshape_as(view))
    else:
        view.copy_(source.reshape_as(view))
    return patched


def _normalize_patch_mode(mode: str) -> str:
    resolved = str(mode).strip().lower()
    if resolved not in {"replace", "delta", "zero"}:
        raise ValueError("Unsupported patch mode: {!r}".format(mode))
    return resolved


def _extract_tensor(value: Any, torch_mod: Any) -> Any | None:
    if hasattr(torch_mod, "is_tensor") and torch_mod.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _extract_tensor(item, torch_mod)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _extract_tensor(item, torch_mod)
            if tensor is not None:
                return tensor
    return None


def _pool_last_token(tensor: Any, torch_mod: Any) -> Any:
    if tensor.dim() == 0:
        return tensor.reshape(1)
    if tensor.dim() == 1:
        return tensor
    if tensor.dim() >= 3:
        tensor = tensor[0]
    return tensor[-1]
