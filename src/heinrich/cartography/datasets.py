"""Dataset manager with caching — fetch, cache, and serve safety benchmark datasets.

Principle 8: No silent fallback to built-in prompts. All benchmark data must
come from real external sources (HuggingFace datasets library). If download
fails, raise with clear instructions instead of silently returning fake data.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CACHE_DIR = Path("~/.heinrich/datasets").expanduser()


@dataclass
class DatasetSpec:
    """Specification for a HuggingFace safety dataset."""
    name: str
    hf_id: str
    split: str
    prompt_column: str
    category_column: str
    config: str | None = None  # HF dataset config name (required by some datasets)
    multi_turn: bool = False   # True if this is a conversation dataset
    conversation_column: str | None = None  # column with list of {role, content} turns
    toxic_column: str | None = None  # column indicating if conversation is toxic


# === Default registry ===

_REGISTRY: dict[str, DatasetSpec] = {}


def _init_registry() -> None:
    """Populate default dataset specs."""
    defaults = [
        DatasetSpec("simple_safety", "Bertievidgen/SimpleSafetyTests", "test", "prompt", "harm_area"),
        DatasetSpec("harmbench", "harmbench/harmbench-text", "standard", "behavior", "semantic_category"),
        DatasetSpec("simplesafetytests", "Bertievidgen/SimpleSafetyTests", "test", "prompt", "harm_area"),
        DatasetSpec("do_not_answer", "LibrAI/do-not-answer", "train", "question", "risk_area"),
        DatasetSpec("xstest", "natolambert/xstest-v2-copy", "test", "prompt", "type"),
        DatasetSpec("catqa", "declare-lab/CategoricalHarmfulQA", "en", "Question", "Category"),
        DatasetSpec("forbidden_questions", "TrustAIRLab/forbidden_question_set", "train", "question", "content_policy_id"),
        # sorry_bench (sorry-bench/sorry-bench-202406) removed: gated dataset, requires auth
        # wildguard (allenai/wildguardmix) removed: gated dataset, requires auth
        DatasetSpec("toxicchat", "lmsys/toxic-chat", "test", "user_input", "toxicity",
                    config="toxicchat0124"),
        # Multi-turn datasets for ghost shart / combinatoric measurement
        DatasetSpec("wildchat", "allenai/WildChat-4.8M", "train", "conversation", "toxic",
                    multi_turn=True, conversation_column="conversation", toxic_column="toxic"),
        DatasetSpec("safety_reasoning", "DukeCEICenter/Safety_Reasoning_Multi_Turn_Dialogue",
                    "train", "conversation", "category",
                    multi_turn=True, conversation_column="conversation"),
        # Diverse capability datasets — not safety-specific
        DatasetSpec("gsm8k", "openai/gsm8k", "test", "question", "question"),
        DatasetSpec("hellaswag", "Rowan/hellaswag", "validation", "ctx", "activity_label"),
        DatasetSpec("alpaca", "tatsu-lab/alpaca", "train", "instruction", "instruction"),
        DatasetSpec("dolly", "databricks/databricks-dolly-15k", "train", "instruction", "category"),
        DatasetSpec("squad", "rajpurkar/squad", "validation", "question", "title"),
        DatasetSpec("triviaqa", "mandarjoshi/trivia_qa", "validation", "question", "question",
                    config="rc.nocontext"),
        DatasetSpec("humaneval", "openai/openai_humaneval", "test", "prompt", "task_id"),
        DatasetSpec("oasst", "OpenAssistant/oasst2", "validation", "text", "lang"),
        DatasetSpec("code_instruct", "iamtarun/python_code_instructions_18k_alpaca",
                    "train", "instruction", "instruction"),
        DatasetSpec("math_word", "microsoft/orca-math-word-problems-200k",
                    "train", "question", "question"),
    ]
    for spec in defaults:
        _REGISTRY[spec.name] = spec


_init_registry()


# === Public API ===

def register_dataset(
    name: str,
    hf_id: str,
    split: str,
    prompt_column: str,
    category_column: str,
) -> None:
    """Register a new dataset (or override an existing one) in the registry."""
    _REGISTRY[name] = DatasetSpec(name, hf_id, split, prompt_column, category_column)


def list_datasets() -> list[str]:
    """Return names of all registered datasets."""
    return sorted(_REGISTRY.keys())


def _suggest_dataset(name: str, available) -> str | None:
    """Return the closest dataset name, or None if no reasonable match."""
    name_lower = name.lower()
    for a in available:
        if name_lower in a.lower() or a.lower() in name_lower:
            return a
    # Try prefix match
    for a in available:
        if a.lower().startswith(name_lower[:4]) or name_lower.startswith(a.lower()[:4]):
            return a
    return None


def load_dataset(
    name: str,
    *,
    max_prompts: int | None = None,
    cache: bool = True,
) -> list[dict]:
    """Load a safety dataset by name.

    Returns list of {"prompt": str, "category": str, "source": str}.

    Resolution order:
    1. Check local cache (~/.heinrich/datasets/<name>.json)
    2. Download from HuggingFace via ``datasets`` library, cache result

    Raises RuntimeError if the dataset cannot be loaded (unknown name, no
    ``datasets`` library, network failure). No silent fallback to built-in
    prompts (Principle 8).
    """
    if name not in _REGISTRY:
        suggestion = _suggest_dataset(name, _REGISTRY.keys())
        hint = f" Did you mean {suggestion!r}?" if suggestion else ""
        raise RuntimeError(
            f"Unknown dataset: {name!r}. "
            f"Available datasets: {', '.join(sorted(_REGISTRY.keys()))}. "
            f"Use register_dataset() to add a custom dataset.{hint}"
        )

    # Try cache first
    if cache:
        cached = _load_cache(name)
        if cached is not None:
            return _limit(cached, max_prompts)

    # Try HuggingFace download
    spec = _REGISTRY[name]
    prompts = _fetch_hf(spec)

    if cache:
        _save_cache(name, prompts)
    return _limit(prompts, max_prompts)


def load_all(max_per_dataset: int = 50) -> list[dict]:
    """Load all registered datasets, concatenated.

    Each entry has a "source" field indicating which dataset it came from.
    """
    all_prompts = []
    for name in list_datasets():
        prompts = load_dataset(name, max_prompts=max_per_dataset)
        all_prompts.extend(prompts)
    return all_prompts


# === Internal helpers ===

def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _load_cache(name: str) -> list[dict] | None:
    """Load cached dataset from disk. Returns None if not found."""
    path = _cache_path(name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and len(data) > 0:
            return data
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(name: str, prompts: list[dict]) -> None:
    """Save dataset to local cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(name)
    path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_hf(spec: DatasetSpec) -> list[dict]:
    """Download dataset from HuggingFace.

    Raises ImportError if the ``datasets`` library is not installed.
    Raises RuntimeError if the download fails.
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for loading external benchmarks. "
            "Install it with: pip install datasets"
        )

    try:
        load_kwargs = {"split": spec.split, "streaming": spec.multi_turn}
        if spec.config is not None:
            load_kwargs["name"] = spec.config
        ds = hf_load(spec.hf_id, **load_kwargs)

        prompts = []

        if spec.multi_turn and spec.conversation_column:
            # Multi-turn: extract turn pairs from conversations
            count = 0
            for row in ds:
                conv = row.get(spec.conversation_column, [])
                if not isinstance(conv, list) or len(conv) < 2:
                    continue
                category = str(row.get(spec.category_column, "unknown"))
                is_toxic = bool(row.get(spec.toxic_column)) if spec.toxic_column else None

                # Extract user turns as prompts, with their follow-up context
                for t in range(0, len(conv) - 1, 2):
                    turn = conv[t]
                    if not isinstance(turn, dict) or turn.get("role") != "user":
                        continue
                    user_text = turn.get("content", "")
                    if not user_text or len(user_text) < 10:
                        continue

                    entry = {
                        "prompt": user_text,
                        "category": category,
                        "source": spec.name,
                    }

                    # If there's a next assistant + user turn, capture as follow-up
                    if t + 2 < len(conv):
                        assistant_turn = conv[t + 1] if conv[t + 1].get("role") == "assistant" else None
                        next_user = conv[t + 2] if t + 2 < len(conv) and conv[t + 2].get("role") == "user" else None
                        if assistant_turn and next_user:
                            entry["assistant_response"] = assistant_turn.get("content", "")
                            entry["follow_up"] = next_user.get("content", "")

                    if is_toxic is not None:
                        entry["is_toxic"] = is_toxic

                    prompts.append(entry)
                    count += 1
                    if count >= 5000:  # cap streaming datasets
                        break
                if count >= 5000:
                    break
        else:
            # Single-turn: existing logic
            for row in ds:
                prompt = str(row.get(spec.prompt_column, ""))
                category = str(row.get(spec.category_column, "unknown"))
                if prompt:
                    prompts.append({
                        "prompt": prompt,
                        "category": category,
                        "source": spec.name,
                    })

        if not prompts:
            raise RuntimeError(
                f"Dataset {spec.hf_id!r} downloaded but contained no prompts. "
                f"Check that prompt_column={spec.prompt_column!r} exists."
            )
        return prompts
    except ImportError:
        raise
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to download dataset {spec.hf_id!r}: {e}. "
            f"Check your network connection and that the dataset ID is correct. "
            f"If you have previously downloaded this dataset, check the cache at "
            f"{CACHE_DIR / (spec.name + '.json')}"
        ) from e


def _limit(prompts: list[dict], max_prompts: int | None) -> list[dict]:
    """Apply max_prompts limit if set."""
    if max_prompts is not None:
        return prompts[:max_prompts]
    return prompts
