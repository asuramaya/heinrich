# CLAUDE.md

## What this project is

Heinrich is a model forensics tool that measures what language models compute, not what they output. It captures internal geometry (residual stream projections, attention patterns, activation traces) alongside language-level signals (scorer labels, text classification) and presents them as an isolated signal stack — each measurement in its own lane, no ground-truth calibration, interpretation left to the reader.

The tool operates on any model with weight access (MLX, HF, or decepticon backends). Supports transformers and causal bank architectures. All benchmark data comes from HuggingFace datasets. No hardcoded prompts. The DB is the single source of truth.

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

**Step 4: Total capture** — complete residual state, every layer:
```
heinrich total-capture --model <model_id> --output X.shrt.npz              # template mode
heinrich total-capture --model <model_id> --output X.shrt.npz --naked      # naked mode (BOS baseline, no template)
```
Produces: displacement delta at entry position and exit position at every layer. Float16. No directions, no projections. Raw state.

**Step 5: Direction discovery** — needs model + DB prompts:
```
heinrich profile-discover-direction --model <model_id>                       # safety direction → .npy
heinrich profile-safety-rank --shrt X.shrt.npz --direction X_safety.npy     # rank all tokens by safety
heinrich profile-first-token --model <model_id> --direction X_safety.npy    # first-token logit gap
heinrich profile-basin --model <model_id> --direction X.npy --layer N       # attractor map
heinrich profile-lmhead --model <model_id> --directions X.npy Y.npy        # output matrix geometry
```

**Step 6: Analyze** — no model needed, reads .npz files:
```
heinrich profile-chain --frt X.frt.npz --shrt X.shrt.npz --sht X.sht.npz  # three-stage correlation
heinrich profile-cross --a X.shrt.npz --b Y.shrt.npz --frt X.frt.npz      # two-model comparison
heinrich profile-survey --shrt *.shrt.npz --frt *.frt.npz                  # multi-model concordance
heinrich profile-mismatch --shrt X.shrt.npz --frt X.frt.npz               # tokenizer-weight gap
heinrich profile-depth --shrt *.shrt.npz --frt *.frt.npz                   # layer trajectory (needs --layers all)
heinrich profile-within-script --shrt X --frt Y                             # within-script dispersion
heinrich profile-directions --shrt X --frt Y                                # PCA, coherence, separation
heinrich profile-code-anatomy --shrt X --frt Y                              # code subcategory decomposition
heinrich profile-silence --model X --shrt Y --frt Z                         # baseline state measurement
heinrich profile-matrix --data-dir data/runs                                # data coverage dashboard
```

**What to watch for:**
- WARNINGS in the output — read them. They flag baseline problems, convergence failures, and small samples.
- Baseline entropy: Qwen 0.5B = 0.92, 3B = 3.91, 7B = 2.42. Different models have different silence. Don't compare absolute deltas across models — use ranks or within-model relative.
- Delta is already relative (displacement from baseline). Don't normalize it further. Per-byte normalization is a ratio of ratios — it's in profile-mismatch for single-model analysis only, not cross-model.
- The .shrt in `discover/shrt.py` is STALE. Use `profile/shrt.py`. The CLI already does this.
- Data in `data/runs/archive/` is contaminated (poisoned baselines). Don't use it.

**MCP tools (all subprocess-isolated, fresh imports on every call):**
- `heinrich_frt_profile` — tokenizer profile (.frt v0.3: raw bytes + merge ranks)
- `heinrich_shrt_profile` — shart profile (.shrt v0.3: vectors + KL + scripts)
- `heinrich_sht_profile` — output profile
- `heinrich_total_capture` — total residual capture (10h timeout, supports `naked` flag)

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

# MRI capture (the primary format — replaces total-capture)
heinrich mri --model X --mode raw --output X.mri            # transformer MRI
heinrich mri --model X.checkpoint.pt --output X.mri         # causal bank MRI (auto-detects)
heinrich mri-backfill --model X --mri X.mri                 # fill missing weights in existing MRI

# Profile analysis (reads .npz or .mri, no model needed)
heinrich profile-chain --frt F --shrt S --sht T             # three-stage correlation
heinrich profile-cross --a S1 --b S2 --frt F                # two-model comparison
heinrich profile-survey --shrt S1 S2 S3 --frt F1 F2 F3     # multi-model concordance
heinrich profile-mismatch --shrt S --frt F                  # tokenizer-weight gap
heinrich profile-depth --shrt S1 S2 --frt F1 F2             # layer trajectory
heinrich profile-pca-anatomy --shrt S --frt F               # name unnamed PCA axes
heinrich profile-pca-survey --pairs S1:F1 S2:F2             # cross-model PCA comparison

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
- `backend/` — MLX, HF, and decepticon model backends, `GenerateResult`, `ForwardContext`
- `profile/` — the measurement instruments:
  - `frt.py` — .frt v0.3: raw bytes, merge ranks, script classification
  - `shrt.py` — .shrt v0.3: vectors + KL + scripts in one file. Prefix KV cache.
  - `sht.py` — .sht: output KL divergence (absorbed into .shrt v0.3 for new runs)
  - `trd.py` — .trd: per-head attribution via o_proj projection
  - `capture.py` — total capture v0.4: entry + exit at every layer, naked or template mode
  - `mri.py` — .mri format: complete model residual image. Supports transformers (per-layer entry/exit) AND causal banks (substrate states + routing + band logits). The primary capture format.
  - `compare.py` — all analysis functions (survey, directions, PCA, pca_anatomy, pca_survey, safety-rank, within-script, etc.)
  - `basin.py` — basin mapping, first-token profiling, lm_head decomposition
  - `efficiency.py` — layer importance, early exit, template overhead
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

## Session 4 findings (April 2026)

- **Safety is 0.5% of displacement variance.** Language identity is 10.2%. Comply/refuse is 0.9%. The displacement profile is a language processing measurement with safety as a trace signal.
- **Safety and comply are orthogonal.** cos(safety, comply) = -0.31. Topic detection and action obligation are separate computations. Comply direction is universal across all 5 models tested.
- **Safety works through first-token selection.** The safety direction pushes "Sorry" up and "Sure" down. Phi-3: 53Mx ratio. Qwen: 284x. Mistral: 281x (corrected from 0x — was a tokenizer encode bug). The first word is the decision. Everything after is confabulation.
- **RLHF builds new directions, doesn't sharpen pretraining's.** Base vs instruct safety: cos = 0.29. Base vs instruct comply: cos = -0.27 to +0.33. RLHF rebuilds both axes from scratch.
- **Silence is not neutral.** Phi-3 silence = maximum refusal (+159). Qwen silence = moderate compliance (-3.4). Every displacement is relative to a tilted baseline.
- **The safety boundary is 100-dimensional.** 1 direction: 82%. 100 PCs: 94%. Full 896 dims: 100%. 5-NN nonlinear: 80%. The boundary is a hyperplane, not a threshold.
- **Full vocabulary coverage** for 5 instruct + 2 base models. No sampling bias. Full-vocab .shrt files in data/runs/.
- **Direction .npy files** in data/runs/: `{model}_safety_L{N}.npy`, `{model}_comply_L{N}.npy`. Always discover natively — never use cross-mapped directions from the DB.
- **External storage**: /Volumes/sharts/ for total captures (multi-GB files).

## Workflow warnings

- **Never use ad-hoc python -c scripts.** Build analysis into compare.py, wire into CLI. The tool IS the workflow.
- **Never use cross-mapped directions from the DB.** Discover natively per model. Cross-mapped directions can be inverted (Phi-3 Cyrillic was wrong).
- **Geometric statistics (PCA, coherence) are stable across sample sizes.** Scalar statistics (correlations, cv) are NOT. Don't interpret scalar stats from 10% samples.
- **Safety direction converges at 100 prompts** (cos=0.94 with full 2425-prompt direction). Accuracy ceiling ~85% regardless of prompt count.
- **Base models have both safety and comply directions** but RLHF replaces them in near-orthogonal directions. A LoRA removing the instruct direction exposes the base model's weaker, differently-oriented structures.
- **The .shrt v0.3 includes KL and scripts.** Separate .sht and .frt lookups are only needed for old v0.2 files.
- **MCP tools are subprocess-isolated.** Source changes propagate on every call.
- **The tool captures state, not interpretation.** Directions, projections, and labels are computed by analysis tools from stored data, not during capture.

## Session 5 findings (April 2026)

- **The 88.4% is language sub-families.** PCA anatomy across 7 models shows the unnamed displacement variance is script-level separation, sub-language axes (Romance, Germanic, Vietnamese, Japanese), code vs natural language, and register/formality. Safety appears only as trace loading on PCs primarily about tech vs legal vocabulary.
- **Dominant PCA axis reflects training data.** Phi-3: Cyrillic=56% of PC1. Qwen: CJK is home. Mistral: English is home. RLHF makes displacement MORE multi-dimensional (Qwen 3B base: 1 PC@50%, instruct: 7 PCs@50%).
- **L2 crystallization (raw mode).** Single tokens crystallize to 1 dimension (98.6% PC1) by L2 and stay frozen for 18 layers. The crystal axis is CONTENT vs STRUCTURE, not language. Template mode prevents crystallization — context keeps representation multi-dimensional at every layer.
- **Causal bank expert routing collapses.** In the O(n) model (cb-s8-experts-50k), one expert gets 99.87% of tokens. The winner specializes on byte-band modes (half-life 1.5-3.0). Balance loss produces redundancy (4 copies) not specialization.
- **Decepticon backend.** Heinrich can now MRI causal bank checkpoints via `decepticons.loader`. Same .mri format, architecture field distinguishes transformer from causal_bank. `heinrich mri --model path.checkpoint.pt` auto-detects.

## The .mri format

The primary data format. One directory per model per capture mode.

**Transformer MRI** (`.mri/`):
```
metadata.json, tokens.npz, baselines.npz
L{NN}_entry.npy, L{NN}_exit.npy     # per-layer residuals
embedding.npy, lmhead.npy, lmhead_raw.npy, norms.npz
weights/L{NN}/*.npy                  # all projection weights
```

**Causal bank MRI** (`.mri/`):
```
metadata.json, tokens.npz
substrate.npy                        # [N, n_modes] EMA states
routing.npy                          # [N, n_experts] routing decisions
half_lives.npy                       # [n_modes] frozen decay parameters
embedding.npy
weights/*.npy                        # all learned parameters
```

Both loaded by `load_mri()`. Analysis tools work on both via `vectors` compatibility key (= exit residuals for transformers, substrate states for causal banks).

**MRI library** at `/Volumes/sharts/`: 20+ MRIs across 9 model families (Qwen 0.5B/1.5B/3B/7B, Phi-3, Mistral, SmolLM2 135M/360M/1.7B, Gemma 2B). Queue running for remaining captures.

## Chronohorn connection

Heinrich connects to the chronohorn training ecosystem:
- **chronohorn** — experiment tracker, fleet dispatch. At `/Users/asuramaya/Code/carving_machine_v3/chronohorn/`
- **decepticons** — model code + `loader.py` (stable interface). At `/Users/asuramaya/Code/carving_machine_v3/decepticons/`
- Interface: `decepticons.loader.load_checkpoint(path) → CausalBankInference`
- Checkpoints on fleet: `scp slop-XX:/data/chronohorn/checkpoints/<name>.checkpoint.pt .`
- `heinrich mri --model checkpoint.pt --output model.mri` — auto-detects decepticon backend

## PCA analysis tools (Session 5)

```
heinrich profile-pca-anatomy --shrt X --frt Y --directions safety=s.npy comply=c.npy  # name the unnamed axes
heinrich profile-pca-survey --pairs X.shrt:X.frt Y.shrt:Y.frt                         # cross-model PCA comparison
```

Key results: displacement is language sorting. ~14 PCs for 50% in Qwen 0.5B. The axes are language sub-families, not semantics or safety. Cross-model: same axes exist in different models at different variance percentages.
