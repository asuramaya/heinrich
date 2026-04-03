# Changelog

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
