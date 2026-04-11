#!/usr/bin/env python3
"""Extract experimental results from Claude conversation JSONL logs.

Reads a JSONL conversation log (one JSON object per line), finds tool results
that contain experimental output (tables, JSON, numerical data), groups them
by script context, and saves structured data to JSON.

Usage:
    python scripts/extract_log_data.py [LOG_PATH] [OUTPUT_PATH]

Defaults:
    LOG_PATH:    the heinrich investigation log
    OUTPUT_PATH: data/extracted_log_data.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_LOG = Path.home() / (
    ".claude/projects/"
    "-Users-asuramaya-Code-carving-machine-v3-conker-detect/"
    "3d3f0365-7414-4bad-93cb-ed74841dac17.jsonl"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "extracted_log_data.json"


# ---------------------------------------------------------------------------
# Patterns we look for in tool output
# ---------------------------------------------------------------------------

# Numerical key=value patterns
KV_PATTERN = re.compile(
    r"\b(refuse_prob|comply_prob|accuracy|kl|KL|entropy|loss|"
    r"cosine|cos_sim|effect_size|mean_gap|separation|"
    r"z_score|p_value|n_anomalous|refusal_rate|"
    r"top_token|layer|neuron)\s*[=:]\s*([0-9eE.+\-]+)",
    re.IGNORECASE,
)

# Table lines (pipe-delimited or aligned columns with numbers)
TABLE_LINE = re.compile(r"^\s*\|.*\|.*\|", re.MULTILINE)
ALIGNED_TABLE = re.compile(r"^\s*\S+\s{2,}\S+\s{2,}\S+", re.MULTILINE)

# Section headers
SECTION_HEADER = re.compile(
    r"^(?:#{1,3}\s+)?(Results?|Findings?|Summary|Output|Analysis|Report|Conclusion)[:\s]",
    re.MULTILINE | re.IGNORECASE,
)

# JSON objects in stdout
JSON_OBJECT = re.compile(r"\{[^{}]{10,}\}")

# Script filenames
SCRIPT_NAME = re.compile(
    r"\b([\w/]+\.py)\b"
)


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------


def extract_kv_pairs(text: str) -> list[dict]:
    """Extract key=value numerical pairs from text."""
    results = []
    for m in KV_PATTERN.finditer(text):
        key = m.group(1).strip()
        val_str = m.group(2).strip()
        try:
            val = float(val_str)
        except ValueError:
            continue
        results.append({"key": key, "value": val})
    return results


def extract_tables(text: str) -> list[str]:
    """Extract pipe-delimited or aligned-column tables."""
    tables = []

    # Pipe tables: find consecutive pipe lines
    lines = text.split("\n")
    table_buf: list[str] = []
    for line in lines:
        if TABLE_LINE.match(line):
            table_buf.append(line.strip())
        else:
            if len(table_buf) >= 2:
                tables.append("\n".join(table_buf))
            table_buf = []
    if len(table_buf) >= 2:
        tables.append("\n".join(table_buf))

    return tables


def extract_json_objects(text: str) -> list[dict]:
    """Try to extract JSON objects embedded in text."""
    results = []
    # Look for { ... } blocks
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start : i + 1]
                if len(candidate) > 10:
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and len(obj) >= 1:
                            results.append(obj)
                    except (json.JSONDecodeError, ValueError):
                        pass
                start = -1
            if depth < 0:
                depth = 0
    return results


def extract_sections(text: str) -> list[dict]:
    """Extract Results/Findings/Summary sections."""
    results = []
    for m in SECTION_HEADER.finditer(text):
        header = m.group(0).strip()
        # Grab up to 2000 chars after the header
        section_text = text[m.start() : m.start() + 2000]
        # Trim at the next section header or end
        next_header = SECTION_HEADER.search(section_text[len(header) :])
        if next_header:
            section_text = section_text[: len(header) + next_header.start()]
        results.append({
            "header": header,
            "text": section_text.strip()[:1500],
        })
    return results


def guess_script_context(text: str) -> str:
    """Try to determine which script produced this output."""
    matches = SCRIPT_NAME.findall(text)
    # Prefer .py files that look like scripts
    for m in matches:
        basename = m.split("/")[-1]
        if basename.startswith("test_"):
            continue
        if basename.endswith(".py"):
            return basename
    return "unknown"


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------


def iter_tool_results(log_path: Path):
    """Yield (line_index, text, context) for each tool result in the JSONL."""
    with open(log_path, "r", errors="replace") as f:
        for line_idx, raw_line in enumerate(f):
            # Skip very short lines
            if len(raw_line) < 20:
                continue

            try:
                obj = json.loads(raw_line)
            except (json.JSONDecodeError, ValueError):
                continue

            msg_type = obj.get("type", "")

            # Source 1: user messages with tool_result content blocks
            if msg_type == "user":
                msg = obj.get("message", {})
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "tool_result":
                            text = block.get("content", "")
                            if isinstance(text, str) and len(text) > 20:
                                yield line_idx, text, "tool_result"
                            elif isinstance(text, list):
                                for item in text:
                                    if isinstance(item, dict):
                                        t = item.get("text", "")
                                        if isinstance(t, str) and len(t) > 20:
                                            yield line_idx, t, "tool_result_inner"

                # Source 2: toolUseResult.stdout
                tur = obj.get("toolUseResult", {})
                if isinstance(tur, dict):
                    stdout = tur.get("stdout", "")
                    if isinstance(stdout, str) and len(stdout) > 30:
                        yield line_idx, stdout, "stdout"
                    stderr = tur.get("stderr", "")
                    if isinstance(stderr, str) and len(stderr) > 30:
                        yield line_idx, stderr, "stderr"

            # Source 3: assistant messages may contain text blocks with output
            if msg_type == "assistant":
                msg = obj.get("message", {})
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if isinstance(text, str) and len(text) > 50:
                                yield line_idx, text, "assistant_text"
                elif isinstance(content, str) and len(content) > 50:
                    yield line_idx, content, "assistant_text"


def has_experimental_data(text: str) -> bool:
    """Quick check: does this text contain experimental data worth extracting?"""
    lower = text.lower()
    keywords = [
        "refuse_prob", "accuracy", "kl=", "entropy", "cosine",
        "effect_size", "separation", "neuron", "z_score",
        "layer", "refusal_rate", "comply_prob", "loss=",
        "results:", "findings:", "| layer", "| l",
    ]
    return any(kw in lower for kw in keywords)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_log_data(
    log_path: Path = DEFAULT_LOG,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict:
    """Main extraction pipeline."""
    if not log_path.exists():
        print(f"ERROR: Log file not found: {log_path}", file=sys.stderr)
        return {"error": f"log not found: {log_path}"}

    print(f"Reading {log_path} ...", file=sys.stderr)

    by_script: dict[str, list[dict]] = {}
    all_kv: list[dict] = []
    all_tables: list[str] = []
    all_json_objects: list[dict] = []
    all_sections: list[dict] = []
    n_processed = 0
    n_extracted = 0

    for line_idx, text, source in iter_tool_results(log_path):
        n_processed += 1

        # Limit text length to avoid memory issues
        if len(text) > 50_000:
            text = text[:50_000]

        if not has_experimental_data(text):
            continue

        n_extracted += 1

        # Extract all data types
        try:
            kvs = extract_kv_pairs(text)
        except Exception:
            kvs = []

        try:
            tables = extract_tables(text)
        except Exception:
            tables = []

        try:
            json_objs = extract_json_objects(text)
        except Exception:
            json_objs = []

        try:
            sections = extract_sections(text)
        except Exception:
            sections = []

        if not kvs and not tables and not json_objs and not sections:
            continue

        script = guess_script_context(text)

        entry = {
            "line_idx": line_idx,
            "source": source,
            "kv_pairs": kvs,
            "tables": tables,
            "json_objects": json_objs,
            "sections": sections,
        }

        by_script.setdefault(script, []).append(entry)
        all_kv.extend(kvs)
        all_tables.extend(tables)
        all_json_objects.extend(json_objs)
        all_sections.extend(sections)

    # Build summary
    result = {
        "metadata": {
            "log_path": str(log_path),
            "n_lines_processed": n_processed,
            "n_lines_with_data": n_extracted,
            "n_scripts": len(by_script),
            "scripts": sorted(by_script.keys()),
        },
        "summary": {
            "n_kv_pairs": len(all_kv),
            "n_tables": len(all_tables),
            "n_json_objects": len(all_json_objects),
            "n_sections": len(all_sections),
        },
        "by_script": by_script,
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(
        f"Extracted {n_extracted} entries from {n_processed} tool results.",
        file=sys.stderr,
    )
    print(
        f"  Scripts: {sorted(by_script.keys())}",
        file=sys.stderr,
    )
    print(
        f"  KV pairs: {len(all_kv)}, Tables: {len(all_tables)}, "
        f"JSON objects: {len(all_json_objects)}, Sections: {len(all_sections)}",
        file=sys.stderr,
    )
    print(f"Saved to {output_path}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    extract_log_data(log_path, output_path)
