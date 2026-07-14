# CLAUDE.md — boot sector

heinrich is backed by **Osiris**, the shared memory graph. This file is the key, not the
memory. 74 history/design/essay entries are indexed in the graph (`ref:heinrich-*`) — the
source docs stay on disk (`docs/`, `paper/`, `PLAN.md`) and in git; this file just points
in. DO NOT append history here — write it to the graph.

## Mount FIRST (before anything)
Osiris MCP is at `http://127.0.0.1:8790/mcp` (a `.mcp.json` points here). On connect:
`mount(cwd=<this dir>, job_dir=$CLAUDE_JOB_DIR)` → `orient()` → `inbox()` → then write back
AS YOU GO: `record_decision` the moment a ruling lands, `open_thread` (kind='obligation'
for a duty an action mints) when work opens or blocks, `resolve_thread` when it closes.
The graph — not the context window — is memory; anything unwritten dies at compaction.
Query prior work with `consult_canon` — it keyword-ranks your project history + the shared
design canon (verified: returns your ingested `ref:heinrich-*` notes). Source docs also stay
on disk: `docs/`, `paper/`, `PLAN.md`, and git.

## Identity check
This repo runs **Opus 4.8** — declared in `.osiris` (`model = "claude-opus-4-8"`), which is the
operator's deliberate per-project choice, not the fleet default (Fable 5). If your environment
names a different model, SAY SO in your first reply — a rug-pull is confessed, never inherited
blind. That rule only has teeth while the banner is rare, which is why the intent is on file.

## What heinrich is
A model-forensics instrument: it measures what a model **computes**, not what it outputs —
residual geometry, attention, activations — each signal in its own lane, no ground-truth
calibration, interpretation left to the reader. The `.mri` is the primary capture format
and the producer↔consumer contract. Works on transformers (MLX/HF) and causal banks
(decepticons loader). Working frontier: the profile/MRI pipeline + the live companion.

## Constitution — invariants an agent here must not break
- **Measurement and calibration interfere.** No FPR/FNR calibration; scorers emit raw
  labels; the reader interprets. No `calibration`-table writes.
- **The DB is the single source of truth.** All prompts from HF benchmarks via
  `require_prompts()` — raises if empty, NO hardcoded fallbacks.
- **One forward pass, not two** — `generate_with_geometry` captures text + first-token
  geometry from one computation.
- **The baseline determines everything.** `.shrt`/`.mri` measure displacement from silence,
  which differs per model — check the baseline before comparing; don't compare absolute
  deltas across models (use ranks / within-model relative).
- **Omit, never fake** (viz). Clamped/absent segments are dropped, not interpolated; honesty
  tags everywhere; a lane earns its place only by showing what no other lane does.
- **Route every readout through `heinrich.profile.readout.lens_logits`** — the one blessed,
  self-checking logit lens. Never hand-roll `lm_head(norm(h))`: `hidden_states[-1]` is
  post-norm, double-norming is a silent-wrong trap. Same class: match live weights to the
  frame's source model (base vs instruct), and apply the stored mean + baseline on projection.
- **Never ad-hoc `python -c`.** Build analysis into a saved script or a CLI subcommand — the
  tool IS the workflow. **Never use cross-mapped directions** (discover natively per model).
- **Narrow exception handling** — specific types, never bare `except Exception: pass`.
  Scorers that fail to load raise immediately (no error rows).
- Stale / contaminated, do not use: `discover/shrt.py` (use `profile/shrt.py`);
  `data/runs/archive/` (poisoned baselines).

## Stack & run
- Python 3.10+, `src/` layout, `from __future__ import annotations` everywhere. Core deps
  numpy + safetensors; **transformers PINNED <5** (5.x breaks the HF MRI capture path).
- This box: Linux + CUDA (RTX A3000-12GB), no MLX. `.venv` is cpython-3.13; install with
  `pip install -e . --no-deps`. Faithful capture ≤ ~3B (full precision — quantizing breaks
  the exactness the whole method rests on).
- CLI `heinrich <cmd>` (frt/shrt/sht · mri/mri-scan/mri-vocab · profile-* · homing).
  MCP `heinrich-mcp` (every CLI subcommand exposed). Companion 3D viewer `heinrich companion`
  (:8377); edge deploy `cp src/heinrich/companion_ui.html web/public/observatory/index.html
  && cd web && npx wrangler deploy`.
- Tests `pytest tests/` (unit run without GPU; integration needs a cached model).
- Siblings at `~/code/REPOS/`: chronohorn (training), decepticons (models + loader),
  osiris (memory). MRI a checkpoint via `heinrich mri --model X.checkpoint.pt` (auto-detects
  the causal-bank backend).
