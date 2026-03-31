"""Rule profiles — configurable validation rule sets."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
from ..signal import Signal, SignalStore


@dataclass(frozen=True)
class Rule:
    kind: str          # "required_file", "size_limit", "required_field", "time_limit", "pattern_absent"
    target: str        # what to check
    threshold: float | None = None  # numeric limit if applicable
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Profile:
    name: str
    rules: list[Rule] = field(default_factory=list)
    description: str = ""


# Presets
PARAMETER_GOLF = Profile(
    name="parameter-golf",
    description="OpenAI Parameter Golf submission validation",
    rules=[
        Rule("required_file", "submission.json"),
        Rule("required_file", "results.json"),
        Rule("required_file", "train_gpt.py"),
        Rule("required_file", "model_artifact.npz"),
        Rule("required_file", "train.log"),
        Rule("required_field", "name", metadata={"file": "submission.json"}),
        Rule("required_field", "track", metadata={"file": "submission.json"}),
        Rule("required_field", "val_bpb", metadata={"file": "submission.json"}),
        Rule("required_field", "bytes_total", metadata={"file": "submission.json"}),
        Rule("size_limit", "bytes_total", threshold=16_000_000.0),
        Rule("time_limit", "train_time_sec", threshold=600.0),
    ],
)

ANTI_BAD = Profile(
    name="anti-bad",
    description="SaTML Anti-BAD backdoor defense submission",
    rules=[
        Rule("required_file", "code/README.md"),
    ],
)

PRESETS: dict[str, Profile] = {
    "parameter-golf": PARAMETER_GOLF,
    "anti-bad": ANTI_BAD,
}


def get_profile(name: str) -> Profile:
    if name in PRESETS:
        return PRESETS[name]
    # Try loading from JSON file
    path = Path(name)
    if path.exists() and path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rules = [Rule(**r) for r in data.get("rules", [])]
        return Profile(name=data.get("name", path.stem), rules=rules, description=data.get("description", ""))
    raise ValueError(f"Unknown profile: {name}")


def apply_profile(
    store: SignalStore,
    path: Path | str,
    profile: Profile,
    *,
    label: str = "validate",
) -> list[Signal]:
    """Apply a rule profile to a directory, emitting pass/fail signals."""
    root = Path(path)
    signals = []

    # Load JSON files for field checks
    json_cache: dict[str, dict] = {}
    for json_path in root.rglob("*.json"):
        rel = str(json_path.relative_to(root))
        try:
            json_cache[rel] = json.loads(json_path.read_text(encoding="utf-8"))
            # Also cache by filename for convenience
            json_cache[json_path.name] = json_cache[rel]
        except (json.JSONDecodeError, OSError):
            pass

    for rule in profile.rules:
        if rule.kind == "required_file":
            exists = (root / rule.target).exists()
            signals.append(Signal("rule_check", "validate", label, f"required_file:{rule.target}",
                                  1.0 if exists else 0.0, {"rule": rule.kind, "pass": exists}))

        elif rule.kind == "required_field":
            source_file = rule.metadata.get("file", "submission.json")
            data = json_cache.get(source_file, {})
            present = rule.target in data if isinstance(data, dict) else False
            signals.append(Signal("rule_check", "validate", label, f"required_field:{rule.target}",
                                  1.0 if present else 0.0, {"rule": rule.kind, "file": source_file, "pass": present}))

        elif rule.kind == "size_limit" and rule.threshold is not None:
            # Check field value from submission.json
            data = json_cache.get("submission.json", {})
            actual = data.get(rule.target, 0) if isinstance(data, dict) else 0
            under = float(actual) <= rule.threshold
            signals.append(Signal("rule_check", "validate", label, f"size_limit:{rule.target}",
                                  1.0 if under else 0.0,
                                  {"rule": rule.kind, "limit": rule.threshold, "actual": float(actual), "pass": under}))

        elif rule.kind == "time_limit" and rule.threshold is not None:
            data = json_cache.get("results.json", {})
            actual = data.get(rule.target, 0) if isinstance(data, dict) else 0
            under = float(actual) <= rule.threshold
            signals.append(Signal("rule_check", "validate", label, f"time_limit:{rule.target}",
                                  1.0 if under else 0.0,
                                  {"rule": rule.kind, "limit": rule.threshold, "actual": float(actual), "pass": under}))

    store.extend(signals)
    return signals
