# CLAUDE.md

## What this project is

Heinrich is a model forensics tool that measures what language models compute, not what they output. It captures internal geometry (residual stream projections, attention patterns, activation traces) alongside language-level signals (scorer labels, text classification) and presents them as an isolated signal stack — each measurement in its own lane, no ground-truth calibration, interpretation left to the reader.

The tool operates on any model with weight access (MLX or HF backends). All benchmark data comes from HuggingFace datasets. No hardcoded prompts. The DB is the single source of truth.

## Quick start (MCP / new instance)

If you're a Claude instance using this via MCP, start here. The profile tools are the working frontier.

**Step 1: Tokenizer profile (.frt)** — no model needed, fast:
```
heinrich frt-profile --tokenizer <model_id>
```
Produces: vocab analysis, byte counts per token, script detection, system prompt extraction.

**Step 2: Shart profile (.shrt)** — needs model, ~100 tok/s:
```
heinrich shart-profile --model <model_id> --n-index 3000
heinrich shart-profile --model <model_id> --n-index 3000 --layers all  # full layer sweep
```
Produces: residual displacement per token vs silence baseline. Token IDs spliced directly (no decode round-trip). Dynamic baseline strips system prompt for any template format. The tool warns on high baseline entropy, unconverged statistics, and small script samples.

**Step 3: Output profile (.sht)** — needs model:
```
heinrich sht-profile --model <model_id> --n-index 3000
```
Produces: KL divergence from silence baseline per token. What the user actually receives.

**Step 4: Analyze** — no model needed, reads .npz files:
```
heinrich profile-chain --frt X.frt.npz --shrt X.shrt.npz --sht X.sht.npz  # three-stage correlation
heinrich profile-cross --a X.shrt.npz --b Y.shrt.npz --frt X.frt.npz      # two-model comparison
heinrich profile-survey --shrt *.shrt.npz --frt *.frt.npz                  # multi-model concordance
heinrich profile-mismatch --shrt X.shrt.npz --frt X.frt.npz               # tokenizer-weight gap
heinrich profile-depth --shrt *.shrt.npz --frt *.frt.npz                   # layer trajectory (needs --layers all)
```

**What to watch for:**
- WARNINGS in the output — read them. They flag baseline problems, convergence failures, and small samples.
- Baseline entropy: Qwen 0.5B = 0.92, 3B = 3.91, 7B = 2.42. Different models have different silence. Don't compare absolute deltas across models — use ranks or within-model relative.
- Delta is already relative (displacement from baseline). Don't normalize it further. Per-byte normalization is a ratio of ratios — it's in profile-mismatch for single-model analysis only, not cross-model.
- The .shrt in `discover/shrt.py` is STALE. Use `profile/shrt.py`. The CLI already does this.
- Data in `data/runs/archive/` is contaminated (poisoned baselines). Don't use it.

**MCP tools (subprocess-isolated, no OOM):**
- `heinrich_frt_profile` — tokenizer profile
- `heinrich_shrt_profile` — shart profile (runs as subprocess)
- `heinrich_sht_profile` — output profile (runs as subprocess)

## Architecture

The pipeline: **load model → load HF prompts → discover directions → scan sharts → find cliffs → generate with geometry → score → report → visualize.**

The profile pipeline: **.frt (tokenizer) → .shrt (residual) → .sht (output) → compare.**

Key principle: **measurement and calibration interfere.** There is no FPR/FNR calibration. Each scorer produces raw labels. The report presents distributions and disagreements. The reader interprets.

Key principle: **one forward pass, not two.** `generate_with_geometry` captures text AND first-token geometry (logits, entropy, top-k, contrastive projection) from the same computation. No separate `forward()` call.

Key principle: **the DB is the single source of truth.** All prompts from HF benchmarks via `require_prompts()` (raises if empty, no fallbacks). All directions, sharts, generations, scores stored in SQLite with schema migrations.

Key principle: **the baseline determines everything.** The .shrt measures displacement from silence. Different models produce different silence states. The tool strips system prompts dynamically but baseline entropy still varies (0.92 to 11.2). Always check baseline before comparing.

## Key types

- `SignalDB` — SQLite store with ChronoHorn single-writer discipline. Schema v10.
- `GenerateResult` — text + first-token geometry from one forward pass
- `ForwardContext` — compositional steer + capture + ablate in a single pass
- `ToolServer` — MCP tool server (40+ tools) wrapping all pipeline stages
- `Scorer` — base class for signal scorers (word_match, regex_harm, qwen3guard, llamaguard, refusal, self_kl)

## Commands

```
# Profile pipeline (the working frontier)
heinrich frt-profile --tokenizer X                          # tokenizer analysis
heinrich shart-profile --model X --n-index 3000             # residual displacement
heinrich shart-profile --model X --n-index 500 --layers all # all-layer sweep
heinrich sht-profile --model X --n-index 3000               # output distribution

# Profile analysis (reads .npz files, no model needed)
heinrich profile-chain --frt F --shrt S --sht T             # three-stage correlation
heinrich profile-cross --a S1 --b S2 --frt F                # two-model comparison
heinrich profile-survey --shrt S1 S2 S3 --frt F1 F2 F3     # multi-model concordance
heinrich profile-mismatch --shrt S --frt F                  # tokenizer-weight gap
heinrich profile-depth --shrt S1 S2 --frt F1 F2             # layer trajectory

# Eval pipeline
heinrich run --model X --prompts harmbench --scorers word_match,qwen3guard
heinrich eval --model X --prompts simple_safety --scorers word_match

# Infrastructure
heinrich serve                   # MCP stdio server
heinrich viz                     # web visualizer sidecar (http://localhost:8377)
heinrich audit <model_id>        # full behavioral security audit
heinrich db summary              # database stats
```

## Testing

```
pytest tests/test_measurement_integrity.py -v  # 18 tests, measurement guarantees
pytest tests/ -v                               # full suite
```

Measurement integrity tests (18): baseline health (4), determinism (2), separation (3), bounds (5), stability (4). These test the measurement methodology, not just the code. All tests run without GPU or network.

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
- `profile/` — .frt (tokenizer), .shrt (residual), .sht (output) profiling
- `eval/` — scorer pipeline, prompts, report, calibrate (descriptive only)
- `discover/` — direction finding, neuron profiling (legacy shrt.py here is stale — use profile/)
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

`heinrich serve` or `heinrich-mcp` runs JSON-RPC over stdio.

**Profile tools (start here):**
- `heinrich_frt_profile` — tokenizer profile (in-process, fast)
- `heinrich_shrt_profile` — shart profile (subprocess-isolated, accepts `layers` param)
- `heinrich_sht_profile` — output profile (subprocess-isolated)

**Eval tools:**
- `heinrich_eval_run` — full pipeline (discover + attack + eval + report)
- `heinrich_eval_report` — report from DB
- `heinrich_eval_calibration` — per-scorer distributions (no FPR/FNR)
- `heinrich_eval_disagreements` — where judges disagree

**DB tools:**
- `heinrich_db_summary` — database overview
- `heinrich_sql` — read-only SQL (blocks DROP/ATTACH/PRAGMA)
- `heinrich_discover_results` — directions, neurons, sharts from DB

**Legacy tools (40+ total):** Many tools point at legacy tables or stale code. The profile tools above are the working frontier. Use `heinrich_db_summary` to see what data exists before calling other tools.

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
