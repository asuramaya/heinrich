# Changelog

## 2026-04-05 — Measurement Integrity + Profile Pipeline

Session 3: falsification and rebuilding. Every measurement fix was found by proving the previous version wrong.

### Profile pipeline (new)
- `.frt` (tokenizer profile): vocab analysis, byte counts, script detection, system prompt extraction
- `.shrt` (shart profile): residual displacement per token, token ID splicing, dynamic baseline, multi-layer
- `.sht` (output profile): KL divergence from silence, same fixes as .shrt
- `profile-chain`: connect .frt → .shrt → .sht for three-stage correlation
- `profile-cross`: two-model comparison with sensitivity metric and baseline mismatch detection
- `profile-survey`: multi-model concordance with Kendall's W
- `profile-mismatch`: tokenizer-weight gap per script
- `profile-depth`: layer trajectory at normalized depth with script explosion detection
- MCP tools: `heinrich_frt_profile`, `heinrich_shrt_profile`, `heinrich_sht_profile`

### Measurement fixes
- **Token ID splicing**: `prefix_ids + [token_id] + suffix_ids`. No decode→re-encode round-trip.
- **Dynamic baseline**: `_extract_clean_baseline` strips system prompt from any template format (ChatML, Llama, Mistral).
- **Convergence check**: replaced broken split-half rank correlation with cumulative statistics stability.
- **Script detection**: accented Latin (été, über) now classified as "latin" not "other". Added Devanagari, Greek.
- **Small-script exhaustion**: scripts with <100 vocab tokens fully scanned (no sampling bias).
- **Self-documenting warnings**: baseline entropy > 5.0, unconverged statistics, small script samples.
- **Throughput reporting**: cold/warm ms, tokens/sec, cv in .shrt metadata.

### Findings (7 models, 3 families)
- 3 universal scripts: CJK (average), latin (easy), code (easy). Kendall's W = 0.65.
- Phi-3 L31 selectively amplifies Cyrillic 3.1x (n=687, ±0.03). Selective, not uniform.
- Mistral sensitivity 46x lower than Phi-3. delta→KL r=0.81 (compression, not indifference).
- Layer dynamics: Qwen compresses (cv U-shape), Phi-3 explodes at L31, Mistral is flat.
- Reproducibility: r=1.000 across identical runs with fixed code.
- "Other" was 5000 misclassified accented Latin tokens. Fixed. Kendall's W dropped 0.72→0.65.

### Data
- 7 clean .shrt files, 3 all-layer .shrt files, 2 .sht files, 4 .frt files
- Contaminated data archived in `data/runs/archive/`
- 18 measurement integrity tests (baseline, determinism, separation, bounds, stability)

## 2026-04-03 — Signal Stack Architecture

Major revision: replaced ground-truth calibration with signal isolation, added geometry capture, removed all hardcoded prompts, fixed error handling throughout.

### Architecture changes
- **Removed FPR/FNR calibration.** `calibrate.py` no longer computes false positive/negative rates against assumed ground truth. Replaced with `describe_scorers()` — descriptive distributions per scorer per condition. No evaluative metrics. Each scorer stays in its own lane.
- **Added `generate_with_geometry`.** One forward pass captures text AND first-token geometry (logits, entropy, top-k, contrastive projection). Replaces the old two-call pattern (forward + generate).
- **Added `GenerateResult` dataclass** to `backend/protocol.py`. Text and geometry from the same computation.
- **Added `describe_context_dependence`** to the report — measures how much safety projection depends on conversation history vs current turn.
- **Added web visualizer** (`viz.py`) — zero-dependency HTTP sidecar reading from the same SQLite DB.

### Data integrity
- **All prompts from HF benchmarks.** `db.require_prompts()` loads from DB or raises. No fallback to hardcoded strings. 7,247 prompts from 7 HF datasets registered.
- **Removed all hardcoded prompts** from `runtime.py`, `audit.py`, `profile.py`, `generate.py`, `attack/run.py`, `compare.py`, `sharts.py`. Every measurement uses DB prompts.
- **Removed embedding pre-filter** for shart detection. Replaced with random vocabulary sampling — model-agnostic, no assumptions about which tokens might be sharts.
- **Archived `prompt_bank.py`** — zero remaining imports. Moved to `cartography/archive/`.

### Database (schema v9 → v10)
- v9: Added `first_token_id`, `refuse_prob`, `is_degenerate` columns to generations table.
- v10: Added `logit_entropy`, `top_k_tokens`, `safety_trajectory` columns to generations table.
- `record_generation` now accepts all geometry fields.

### Error handling
- **Narrowed all `except Exception: pass`** in `db.py` to `sqlite3.OperationalError` for DDL migrations.
- **Async write failures now warn** instead of being silently swallowed.
- **Scorers fail fast** — 3 consecutive errors aborts with RuntimeError, zero error rows written to DB.
- **llamaguard raises on auth failure** instead of returning 910 `ScoreResult("error")` rows.
- **`require_prompts()` raises** if DB has insufficient prompts — no silent degradation.

### Security
- `_do_sql` blocks ATTACH, DETACH, PRAGMA, VACUUM, REINDEX (in addition to existing write blocklist).
- Viz HTTP `limit` parameter validated and clamped to [1, 1000].

### Bug fixes
- self_kl scorer: `--model` now passed through subprocess chain (`run.py` → `score.py` → `load_scorer`).
- qwen3guard: label parser uses word boundaries (`\bunsafe\b`) instead of substring match.
- `classify.py`: `self.residual_threshold` → `self.refusal_threshold` (AttributeError fix).
- L27 attention capture: cast to float32 before Q*K matmul to prevent NaN on long sequences.
- Mock tokenize in tests: deterministic hash with collision avoidance (fixed 5 flaky tests).
- Dangling prompt_id FKs nulled after prompt table rebuild.
- Bogus L31 direction on 28-layer model cleaned from DB.

### Import cleanup
- 8 files updated from legacy `heinrich.signal`/`heinrich.db` to `heinrich.core.signal`/`heinrich.core.db`.

### Tests
- 1324 tests, 0 failures (up from ~1314 with 5 pre-existing failures).
- Added `test_measurement_integrity.py`: determinism (2), separation (3), bounds (5).
- Fixed mock hash randomization in `test_discover_refusal.py`.

### Documentation
- README.md rewritten for current tool (geometry, signal stack, findings).
- CLAUDE.md updated for current architecture.
- docs/DESIGN.md rewritten (signal isolation rationale, historical context preserved).
- docs/GUIDE.md rewritten (walkthroughs for pipeline, basin mapping, sharts, ghost sharts).
- Historical wave notes moved to docs/archive/.

### Research findings (from this session's measurements)
- Models have 4-5 basins, not 2 (PCA on residual states from HF benchmarks).
- Safety mechanism is low-rank — LoRA works because it targets the shared axis.
- Order matters 200% at safety layers (L22-L27) — the last turn determines the projection.
- 99% of safety projection variance is the final turn at all context lengths tested (135-3000 tokens).
- The model reads the prefix (77-86% attention at 1500 tokens) but doesn't project it onto the safety direction.
- Ghost shart is real but transient — doesn't accumulate across turns.
- Per-head attention routing is content-dependent (50-point swings at 135 tokens, 7-point at 1500).
- "toxic" has zero shart effect on Mistral. "encourage" has 90x more.
- qwen3guard and llamaguard disagree on 33% of generations (97% vs 63% safe on same data).
