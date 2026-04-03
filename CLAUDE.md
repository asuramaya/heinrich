# CLAUDE.md

## What this project is

Heinrich is a model forensics tool that measures what language models compute, not what they output. It captures internal geometry (residual stream projections, attention patterns, activation traces) alongside language-level signals (scorer labels, text classification) and presents them as an isolated signal stack — each measurement in its own lane, no ground-truth calibration, interpretation left to the reader.

The tool operates on any model with weight access (MLX or HF backends). All benchmark data comes from HuggingFace datasets. No hardcoded prompts. The DB is the single source of truth.

## Architecture

The pipeline: **load model → load HF prompts → discover directions → scan sharts → find cliffs → generate with geometry → score → report → visualize.**

Key principle: **measurement and calibration interfere.** There is no FPR/FNR calibration. Each scorer produces raw labels. The report presents distributions and disagreements. The reader interprets.

Key principle: **one forward pass, not two.** `generate_with_geometry` captures text AND first-token geometry (logits, entropy, top-k, contrastive projection) from the same computation. No separate `forward()` call.

Key principle: **the DB is the single source of truth.** All prompts from HF benchmarks via `require_prompts()` (raises if empty, no fallbacks). All directions, sharts, generations, scores stored in SQLite with schema migrations.

## Key types

- `SignalDB` — SQLite store with ChronoHorn single-writer discipline. Schema v10.
- `GenerateResult` — text + first-token geometry from one forward pass
- `ForwardContext` — compositional steer + capture + ablate in a single pass
- `ToolServer` — MCP tool server (40+ tools) wrapping all pipeline stages
- `Scorer` — base class for signal scorers (word_match, regex_harm, qwen3guard, llamaguard, refusal, self_kl)

## Commands

```
heinrich serve                   # MCP stdio server
heinrich viz                     # web visualizer sidecar (http://localhost:8377)
heinrich run --model X --prompts harmbench --scorers word_match,qwen3guard
heinrich eval --model X --prompts simple_safety --scorers word_match
heinrich audit <model_id>        # full behavioral security audit
heinrich db summary              # database stats
heinrich db query --kind shart   # query signals
```

## Testing

```
pytest tests/ -v    # 1324 tests, ~65s
```

Tests include measurement integrity (determinism, separation, bounds) alongside unit tests. All tests run without GPU or network. 5 pre-existing mock failures in test_discover_refusal.py (token ID mismatches, not from current code).

## Code conventions

- src layout: `src/heinrich/`
- Python 3.10+, numpy + safetensors as core deps
- MLX backend for Apple Silicon, HF transformers as fallback
- `from __future__ import annotations` in all files
- All prompts from HF benchmarks via `db.require_prompts()` — no hardcoded fallbacks
- All exception handling narrowed: `sqlite3.OperationalError` for DDL, never bare `except Exception: pass`
- Scorers that fail to load raise immediately — no error rows written to DB
- Import paths: `from heinrich.core.db import SignalDB`, `from heinrich.core.signal import Signal`

## Subpackage map

- `core/` — SignalDB, Signal, SignalStore
- `backend/` — MLX and HF model backends, `GenerateResult`, `ForwardContext`
- `eval/` — scorer pipeline, prompts, report, calibrate (descriptive only)
- `discover/` — direction finding, neuron profiling, shart scanning
- `attack/` — cliff search, steering, distributed attacks
- `cartography/` — model config, templates, datasets, runtime, audit
- `viz.py` — web visualizer sidecar (zero dependencies, reads from same DB)
- `mcp.py` — MCP tool definitions and ToolServer
- `mcp_transport.py` — JSON-RPC stdio transport

## Database schema (v10)

Key tables:
- `prompts` — HF benchmark prompts with source, category, is_benign
- `generations` — model outputs with geometry columns (first_token_id, refuse_prob, logit_entropy, top_k_tokens, safety_trajectory, is_degenerate)
- `scores` — per-generation scorer labels (one row per generation × scorer)
- `directions` — contrastive directions per model × layer (with vector_blob)
- `conditions` — steering conditions (clean, steer_L15_-0.48, distributed, etc.)
- `models` — model registry with config_hash

No `calibration` table writes. Scorer distributions are computed on-the-fly from the scores table.

## Eval scorers

| Scorer | Type | Model | What it measures |
|--------|------|-------|-----------------|
| word_match | pattern | none | refusal/compliance vocabulary |
| regex_harm | pattern | none | structural harm patterns (steps, chemicals, code) |
| refusal | measurement | target | first-token refusal probability |
| self_kl | measurement | target | first-token probability under clean distribution |
| qwen3guard | judge | Qwen3Guard-0.6B | external safety classification |
| llamaguard | judge | LlamaGuard-3-1B | external safety classification (Meta) |

Judge scorers disagree: qwen3guard says 97% safe, llamaguard says 63% safe on the same data. The disagreement IS the signal.

## MCP server

`heinrich serve` or `heinrich-mcp` runs JSON-RPC over stdio. 40+ tools including:
- `heinrich_eval_run` — full pipeline (discover + attack + eval + report)
- `heinrich_eval_report` — report from DB
- `heinrich_eval_calibration` — per-scorer distributions (no FPR/FNR)
- `heinrich_db_summary` — database overview
- `heinrich_sql` — read-only SQL (blocks DROP/ATTACH/PRAGMA)
- `heinrich_audit` — full behavioral security audit

## Datasets

Registered HF datasets (loaded via `cartography/datasets.py`):
- catqa, do_not_answer, forbidden_questions, simple_safety, simplesafetytests, toxicchat (single-turn)
- wildchat, safety_reasoning (multi-turn, streaming)

All prompts loaded into the DB via `require_prompts()`. No hardcoded prompt bank used for measurements.

## Security

- SQL injection: `_do_sql` blocks dangerous keywords + uses read-only connection
- Viz HTTP: limit parameter validated and clamped to [1, 1000]
- Scorer errors: fail fast after 3 consecutive failures, zero error rows written
- Exception handling: narrowed to specific types, async write failures warned (not silenced)
