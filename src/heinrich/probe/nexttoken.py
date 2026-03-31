"""Next-token distribution probe -- ported from conker-detect nexttoken.py."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .token_tools import _safe_decode_piece, load_tokenizer, render_case_prompt


def probe_next_token_distribution(
    case: dict[str, Any],
    *,
    tokenizer_ref: str | Path,
    model_ref: str | Path | None = None,
    candidate_strings: list[str] | tuple[str, ...] | None = None,
    candidate_token_ids: list[int] | tuple[int, ...] | None = None,
    topk: int = 16,
    trust_remote_code: bool = True,
    use_fast: bool = True,
    dtype: str | None = None,
    device: str | None = None,
    low_cpu_mem_usage: bool = False,
    add_generation_prompt: bool = True,
    add_special_tokens: bool = False,
    steps: int = 1,
    method: str = "auto",
    mps_fallback: bool = False,
    mps_prefer_metal: bool = False,
    mps_high_watermark_ratio: float | None = None,
    mps_low_watermark_ratio: float | None = None,
) -> dict[str, Any]:
    """Probe the next-token probability distribution for a given case.

    Requires torch and transformers. Loads the model from model_ref (or
    tokenizer_ref) and returns top-k token probabilities plus per-candidate
    probabilities for any specified candidate strings or token IDs.
    """
    _configure_runtime_env(
        mps_fallback=mps_fallback,
        mps_prefer_metal=mps_prefer_metal,
        mps_high_watermark_ratio=mps_high_watermark_ratio,
        mps_low_watermark_ratio=mps_low_watermark_ratio,
    )
    tokenizer = load_tokenizer(tokenizer_ref, trust_remote_code=trust_remote_code, use_fast=use_fast)
    rendered = render_case_prompt(case, tokenizer=tokenizer, add_generation_prompt=add_generation_prompt)
    torch_mod = _import_torch()
    transformers = _import_transformers()
    resolved_dtype = _resolve_torch_dtype(torch_mod, dtype)
    resolved_device = _resolve_device(torch_mod, device)
    resolved_steps = max(int(steps), 1)
    load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if resolved_dtype is not None:
        load_kwargs["dtype"] = resolved_dtype
    if low_cpu_mem_usage:
        load_kwargs["low_cpu_mem_usage"] = True
    model = transformers.AutoModelForCausalLM.from_pretrained(str(model_ref or tokenizer_ref), **load_kwargs)
    if hasattr(model, "to"):
        model = model.to(resolved_device)
    if hasattr(model, "eval"):
        model.eval()
    encoded = tokenizer(rendered["rendered_text"], return_tensors="pt", add_special_tokens=add_special_tokens)
    encoded = {name: value.to(resolved_device) for name, value in encoded.items()}
    method_used = str(method or "auto").lower()
    rows: list[dict[str, Any]]
    if method_used not in {"auto", "forward", "generate"}:
        raise ValueError(f"Unknown next-token probe method: {method}")
    if method_used in {"auto", "forward"} and resolved_steps == 1:
        try:
            with torch_mod.no_grad():
                outputs = model(**encoded)
            logits = _extract_logits(outputs, torch_mod)
            if logits is None:
                raise ValueError("Model forward pass did not expose logits")
            final_logits = logits[0, -1].detach().to("cpu").float()
            probs = torch_mod.softmax(final_logits, dim=-1)
            rows = [
                _build_distribution_row(
                    tokenizer=tokenizer,
                    probs=probs,
                    topk=topk,
                    candidate_strings=candidate_strings,
                    candidate_token_ids=candidate_token_ids,
                    step=1,
                    generated_token_id=int(torch_mod.topk(probs, k=1)[1].tolist()[0]),
                )
            ]
            method_used = "forward"
        except Exception:
            if method_used == "forward":
                raise
            rows = _run_generate_probe(
                tokenizer=tokenizer,
                model=model,
                encoded=encoded,
                torch_mod=torch_mod,
                topk=topk,
                candidate_strings=candidate_strings,
                candidate_token_ids=candidate_token_ids,
                steps=resolved_steps,
            )
            method_used = "generate"
    else:
        rows = _run_generate_probe(
            tokenizer=tokenizer,
            model=model,
            encoded=encoded,
            torch_mod=torch_mod,
            topk=topk,
            candidate_strings=candidate_strings,
            candidate_token_ids=candidate_token_ids,
            steps=resolved_steps,
        )
        method_used = "generate"
    first_row = rows[0]
    return {
        "mode": "nexttoken",
        "method_used": method_used,
        "tokenizer_name": getattr(tokenizer, "name_or_path", str(tokenizer_ref)),
        "model_ref": str(model_ref or tokenizer_ref),
        "device": str(resolved_device),
        "rendered_text": rendered["rendered_text"],
        "used_chat_template": bool(rendered["used_chat_template"]),
        "add_generation_prompt": bool(rendered["add_generation_prompt"]),
        "candidate_count": len(first_row["candidates"]),
        "topk": len(first_row["top_tokens"]),
        "entropy_bits": float(first_row["entropy_bits"]),
        "top_tokens": first_row["top_tokens"],
        "candidates": first_row["candidates"],
        "steps": resolved_steps,
        "rows": rows,
    }


def _run_generate_probe(
    *,
    tokenizer: Any,
    model: Any,
    encoded: dict[str, Any],
    torch_mod: Any,
    topk: int,
    candidate_strings: list[str] | tuple[str, ...] | None,
    candidate_token_ids: list[int] | tuple[int, ...] | None,
    steps: int,
) -> list[dict[str, Any]]:
    with torch_mod.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=int(steps),
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    rows: list[dict[str, Any]] = []
    base_length = int(encoded["input_ids"].shape[-1])
    for step_index, score in enumerate(outputs.scores, start=1):
        probs = torch_mod.softmax(score[0].detach().to("cpu").float(), dim=-1)
        generated_token_id = int(outputs.sequences[0][base_length + step_index - 1].item())
        rows.append(
            _build_distribution_row(
                tokenizer=tokenizer,
                probs=probs,
                topk=topk,
                candidate_strings=candidate_strings,
                candidate_token_ids=candidate_token_ids,
                step=step_index,
                generated_token_id=generated_token_id,
            )
        )
    return rows


def _build_distribution_row(
    *,
    tokenizer: Any,
    probs: Any,
    topk: int,
    candidate_strings: list[str] | tuple[str, ...] | None,
    candidate_token_ids: list[int] | tuple[int, ...] | None,
    step: int,
    generated_token_id: int,
) -> dict[str, Any]:
    top_values, top_indices = _import_torch().topk(probs, k=min(int(topk), int(probs.shape[-1])))
    top_tokens = []
    for prob, token_id in zip(top_values.tolist(), top_indices.tolist()):
        top_tokens.append({
            "token_id": int(token_id),
            "piece": _safe_decode_piece(tokenizer, int(token_id)),
            "probability": float(prob),
        })
    candidates = []
    for query in list(candidate_strings or []):
        token_ids = tokenizer(str(query), add_special_tokens=False).get("input_ids", [])
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        token_ids = [int(tid) for tid in token_ids]
        exact_single = len(token_ids) == 1
        first_token_probability = float(probs[token_ids[0]].item()) if token_ids else None
        probability = first_token_probability if exact_single else None
        candidates.append({
            "query": str(query),
            "query_type": "string",
            "token_ids": token_ids,
            "token_pieces": [_safe_decode_piece(tokenizer, tid) for tid in token_ids],
            "exact_single_token": exact_single,
            "probability": probability,
            "first_token_probability": first_token_probability,
        })
    for token_id in list(candidate_token_ids or []):
        token_id = int(token_id)
        candidates.append({
            "query": token_id,
            "query_type": "token_id",
            "token_ids": [token_id],
            "token_pieces": [_safe_decode_piece(tokenizer, token_id)],
            "exact_single_token": True,
            "probability": float(probs[token_id].item()),
        })
    candidates.sort(
        key=lambda row: (
            _sortable_candidate_probability(row) is not None,
            -1.0 if _sortable_candidate_probability(row) is None else float(_sortable_candidate_probability(row)),
        ),
    )
    entropy = _entropy_bits(np.asarray(probs.tolist(), dtype=np.float64))
    return {
        "step": int(step),
        "generated_token_id": int(generated_token_id),
        "generated_piece": _safe_decode_piece(tokenizer, int(generated_token_id)),
        "entropy_bits": float(entropy),
        "top_tokens": top_tokens,
        "candidates": candidates,
    }


def _sortable_candidate_probability(row: dict[str, Any]) -> float | None:
    probability = row.get("probability")
    if probability is not None:
        return float(probability)
    first_token_probability = row.get("first_token_probability")
    if first_token_probability is not None:
        return float(first_token_probability)
    return None


def _extract_logits(outputs: Any, torch_mod: Any) -> Any | None:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    if hasattr(torch_mod, "is_tensor") and torch_mod.is_tensor(outputs):
        return outputs
    return None


def _entropy_bits(probs: np.ndarray) -> float:
    positive = probs[probs > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log2(positive)))


def _configure_runtime_env(
    *,
    mps_fallback: bool,
    mps_prefer_metal: bool,
    mps_high_watermark_ratio: float | None,
    mps_low_watermark_ratio: float | None,
) -> None:
    if mps_fallback:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if mps_prefer_metal:
        os.environ.setdefault("PYTORCH_MPS_PREFER_METAL", "1")
    if mps_high_watermark_ratio is not None:
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(mps_high_watermark_ratio))
    if mps_low_watermark_ratio is not None:
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(mps_low_watermark_ratio))


def _resolve_torch_dtype(torch_mod: Any, raw: Any) -> Any:
    if raw is None or str(raw) == "auto":
        return None
    value = getattr(torch_mod, str(raw), None)
    if value is None:
        raise ValueError(f"Unknown torch dtype: {raw}")
    return value


def _resolve_device(torch_mod: Any, raw: Any) -> str:
    if raw:
        return str(raw)
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is not None and hasattr(cuda, "is_available") and cuda.is_available():
        return "cuda"
    backends = getattr(torch_mod, "backends", None)
    if backends is not None:
        mps = getattr(backends, "mps", None)
        if mps is not None and hasattr(mps, "is_available") and mps.is_available():
            return "mps"
    return "cpu"


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise ImportError("transformers is required for next-token probing") from exc
    return transformers


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for next-token probing") from exc
    return torch
