"""Download, freeze, and manage prompt sets for evaluation.

Loads prompts from the datasets registry (heinrich.cartography.datasets),
freezes them to local JSONL with SHA-256 verification, and inserts into
the SignalDB prompts table.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompt_data"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_benign_prompts(max_prompts: int = 200) -> list[dict]:
    """Load benign calibration prompts.

    Resolution order:
    1. Local benign.jsonl file in prompt_data/
    2. Download from tatsu-lab/alpaca (filtered for short, benign instructions)
    3. Raise RuntimeError with clear instructions

    No hardcoded fallback prompts -- all benign data must come from a
    real external source or a user-provided local file.
    """
    local = PROMPT_DIR / "benign.jsonl"
    if local.exists():
        records = _read_jsonl(local)
        if records:
            return records[:max_prompts]

    # Try downloading from Alpaca
    try:
        from datasets import load_dataset as hf_load

        ds = hf_load("tatsu-lab/alpaca", split="train")
        # Filter: take short, simple instructions that are clearly benign
        benign = [
            {"text": row["instruction"], "category": "benign", "source": "alpaca"}
            for row in ds
            if 10 < len(row["instruction"]) < 200
        ]
        # Shuffle deterministically and take max_prompts
        benign.sort(key=lambda x: hashlib.md5(x["text"].encode()).hexdigest())
        benign = benign[:max_prompts]
        # Save locally for next time
        _write_jsonl(local, benign)
        return benign
    except Exception:
        raise RuntimeError(
            "No benign calibration prompts available. Either:\n"
            "  1. Place a benign.jsonl file in src/heinrich/eval/prompt_data/\n"
            "  2. Install the 'datasets' library: pip install datasets\n"
            "Benign prompts are required for FPR calibration."
        )


def download_prompts(name: str, force: bool = False) -> Path:
    """Download a prompt set from the datasets registry, freeze it locally with SHA-256.

    Returns the path to the frozen JSONL file.
    """
    from heinrich.cartography.datasets import load_dataset as ds_load

    out_path = PROMPT_DIR / f"{name}.jsonl"
    sha_path = PROMPT_DIR / f"{name}.sha256"

    if out_path.exists() and sha_path.exists() and not force:
        # Verify integrity
        stored_sha = sha_path.read_text().strip()
        current_sha = _sha256_file(out_path)
        if stored_sha == current_sha:
            return out_path
        # Mismatch — re-download
        # (fall through)

    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    raw = ds_load(name)

    # Normalize to eval format: text, category, source
    records = []
    for item in raw:
        records.append({
            "text": item.get("prompt", item.get("text", "")),
            "category": item.get("category", "unknown"),
            "source": name,
        })

    # Write JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write SHA-256
    sha = _sha256_file(out_path)
    sha_path.write_text(sha)

    return out_path


def load_prompts(name: str) -> list[dict]:
    """Load a frozen prompt set. Each item has 'text', 'category', 'source'.

    Checks for a local JSONL file first (allows manual placement without
    download).  If not found locally, attempts to download.
    """
    local_path = PROMPT_DIR / f"{name}.jsonl"
    if local_path.exists():
        return _read_jsonl(local_path)

    # Not found locally — try to download
    download_prompts(name)
    return _read_jsonl(local_path)


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_benign_prompts() -> list[dict]:
    """Load benign calibration prompts from external source.

    Uses _load_benign_prompts() which tries local file, then Alpaca download.
    Returns list of dicts with text, category, source keys.
    """
    prompts = _load_benign_prompts()
    return [
        {"text": p["text"], "category": p.get("category", "benign"), "source": "benign_calibration"}
        for p in prompts
    ]


def insert_prompts_to_db(db, name: str, max_prompts: int | None = None):
    """Load a prompt set and insert into the prompts table.

    If name == 'benign_calibration', uses the benign calibration set
    (from local file or Alpaca download).
    Otherwise loads from the frozen JSONL (downloading if needed).

    *max_prompts* caps the number of prompts inserted (useful for quick tests).
    """
    if name == "benign_calibration":
        prompts = load_benign_prompts()
    else:
        prompts = load_prompts(name)

    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    # Determine if this prompt set is benign
    is_benign = name in ("benign", "benign_calibration")

    for p in prompts:
        db.record_prompt(
            text=p["text"],
            source=p.get("source", name),
            category=p.get("category", "benign" if is_benign else None),
            is_benign=is_benign,
        )

    return len(prompts)


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
