# Heinrich v1.0 — Proposal

## What changed

Heinrich has two halves that don't talk to each other.

The **signal half** (2024-2025): fetch, inspect, discover, attack, eval. Everything produces typed Signals. Signals go into the DB. The DB is queryable. Findings persist across sessions. The viz reads the DB. The MCP reads the DB. The report reads the DB.

The **MRI half** (2026): capture, profile, compare. Everything produces dicts. Dicts go to stdout. Results vanish when the process exits. The crystal finding, the frozen zone, the gate analysis — none of it is in the DB. If you ask "which models crystallize before L5?" you have to re-run everything.

v1.0 unifies them. Every tool speaks Signal. The DB is the single source of truth for all measurements — weight forensics, behavioral eval, residual geometry, gate activations, perturbation effects. One schema, one store, one query interface.

## The Signal

The atom of heinrich is a typed measurement:

```python
Signal(
    kind="gate_concentration",      # what was measured
    source="mri:smollm2-135m/raw",  # where it came from
    model="smollm2-135m",           # which model
    target="L11",                   # what it measured
    value=1.0,                      # the number
    metadata={                      # everything else
        "neuron": 1229,
        "n_tokens": 48660,
        "gate_std": 2.6648,
        "mri_version": "0.7",
    },
)
```

This is already the schema. It's been in `core/signal.py` since day one. The MRI tools just never used it.

## The DB

The DB already has tables for models, directions, neurons, sharts, layers, basins, heads, evaluations, scores, generations, prompts, and generic signals. Schema v10. ChronoHorn single-writer. WAL mode.

What's missing: MRI findings. Not a new table — the existing `signals` table handles it. A gate analysis produces signals with `kind="gate_concentration"`. A layer delta analysis produces signals with `kind="layer_delta"`. A crystal detection produces signals with `kind="crystal_birth"`. They accumulate alongside eval scores and direction discoveries.

The MRI files (.mri directories on /Volumes/sharts/) remain as raw data. The DB stores the derived measurements. The relationship: .mri is the medical image, the DB is the medical record.

## The CLI

Every command outputs to stdout and writes to the DB:

```bash
heinrich mri --model X --mode raw --output X.mri    # captures .mri files + records to DB
heinrich analyze gates --mri X.mri                   # prints JSON to stdout + writes signals to DB
heinrich evaluate run --model X --scorers word_match  # generates + scores + writes all to DB
heinrich inspect weights --source X.npz              # analyzes weights + writes signals to DB
```

`--json` flag on every command emits structured JSON to stdout. This is the wire format for MCP and the companion. On by default for programmatic use, off by default for human use (human gets formatted text).

`--no-db` flag skips the DB write for one-off exploration. Default is to record.

The CLI is the single entry point. Everything else calls it.

## The MCP

~200 lines. One dispatcher:

```python
def call_tool(name, args):
    cmd = tool_name_to_cli_command(name)
    result = subprocess.run(["heinrich", cmd, "--json", *args_to_flags(args)])
    return json.loads(result.stdout)
```

Tool list generated from `heinrich --list-commands`. Tool descriptions from CLI docstrings. Subprocess isolation by construction. Parity guaranteed — if the CLI command exists, the MCP tool exists.

## The companion

HTTP server on port 8377. Reads from the DB. Not from individual command outputs.

```
GET /api/signals?model=smollm2-135m&kind=gate_concentration    # query signals
GET /api/signals?model=smollm2-135m&kind=layer_delta           # layer deltas
GET /api/diff?model_a=smollm2-135m&model_b=smollm2-360m&kind=crystal_birth  # diff
WS  /ws/live                                                    # new signals as they're written
```

The companion is a differential instrument. Two panels. Each panel queries the DB for a model/kind/condition. The center strip shows the diff. The operators choose what to compare:

- Same model, raw vs template: what does context do?
- Same model, clean vs steered: what does the safety direction change?
- Two models, same mode: what does scale buy?
- Same model, before and after fine-tuning: what changed?

Every comparison that can be expressed as a DB query can be visualized. The companion doesn't know about MRIs or eval or weights — it knows about signals.

## The six verbs

Every heinrich command is one of these:

| Verb | What it does | Examples |
|------|-------------|---------|
| **observe** | Capture state without changing it | mri, frt, shrt, sht, inspect, fetch |
| **intervene** | Change something, measure the effect | steer, cliff, ablate |
| **evaluate** | Run inputs, score outputs | eval run, eval report |
| **analyze** | Compute derived quantities | gates, layer-deltas, logit-lens, pca, spectral |
| **compare** | Diff two observations | cross-model, chain, survey |
| **report** | Synthesize for communication | audit, bundle, report |

The verb is the namespace: `heinrich observe mri`, `heinrich analyze gates`, `heinrich compare cross`. Or keep the flat namespace (`heinrich mri`, `heinrich profile-gates`) — the verb is implicit. Either way, every command produces signals.

## What changes in code

### Must change

1. **Analysis functions write signals after returning dicts.** Each function in `compare.py` already returns a dict. Add a `_emit_signals(result, db)` call that extracts the key measurements and records them. The dict return stays — stdout still works. The DB write is the new part.

2. **MCP becomes a subprocess dispatcher.** Delete the TOOLS dict. Delete the call_tool dispatch. Replace with: look up CLI command, run it with `--json`, return stdout. ~1800 lines deleted, ~200 added.

3. **CLI gets `--json` and `--no-db` flags.** Each `_cmd_X` function gets a 2-line preamble: if json, dump and return; at the end, write signals to DB.

4. **Companion replaces viz.** New file, reads from DB via signal queries. WebSocket for live updates. Serves the diff instrument.

### Does not change

- `core/signal.py` — the Signal schema is already right
- `core/db.py` — the signals table already exists
- `profile/compare.py` — analysis functions stay as-is (return dicts)
- `profile/mri.py` — capture code stays as-is
- `backend/` — model loading stays as-is
- `eval/` — already writes to DB
- `discover/` — already writes to DB
- `attack/` — already writes to DB
- `inspect/` — produces signals already
- All test files — same functions, same returns

### The Signal schema question

The Signal has one `value: float`. MRI analysis produces multi-field dicts (10 fields per layer × 30 layers). Two options:

**Option A: One signal per measurement.** `gate_analysis` on a 30-layer model emits 30 signals, one per layer, with layer-specific fields in metadata. A full analysis run (5 tools × 30 layers × 3 modes) = ~450 signals. Queryable: "give me gate_concentration at L11 for all models."

**Option B: One signal per analysis run.** `gate_analysis` emits 1 signal with the full result dict in metadata. Simpler, but metadata queries are string-matching in JSON, not SQL columns.

**Recommendation: Option A.** The DB already has per-layer tables (layers, directions). The signals table with `target="L11"` is natural. 450 signals per analysis run is small. The companion queries `WHERE kind='gate_concentration' AND target='L11'` and gets a clean column.

### The companion architecture question

Gate analysis takes 58 seconds on USB-stored MRI data. Subprocess-per-click is unusable for interactive exploration.

**Two modes:**

1. **Signal mode (default):** Companion queries the DB for pre-computed signals. Instant. Requires analysis to have run already. This is the mode for browsing and diffing.

2. **Live mode (capture):** During MRI capture or analysis, the tool writes signals to DB as they're computed. The companion's WebSocket receives new signals in real-time. No subprocess — the capture process pushes events directly.

The companion never loads an MRI. It only reads signals from the DB. The analysis tools write signals. The capture tools write signals. The companion renders signals.

### Could change later

- Verb-based CLI namespaces (`heinrich observe mri` vs `heinrich mri`)
- Signal-level deduplication (don't re-record if already in DB)
- Signal versioning (track which code version produced each signal)
- Signal provenance (link signals to the .mri file or eval run that produced them)
- Cross-session queries ("show me all crystal findings across all sessions")

## What this enables

1. **"Which models crystallize before L5?"** — one SQL query on the signals table
2. **"Show me gate concentration across all models at L11"** — one query, rendered in companion
3. **"How did the crystal change after fine-tuning?"** — diff signals before and after
4. **"What's the relationship between crystal birth layer and model size?"** — scatter plot from signals
5. **The companion works for everything** — not just MRI. Eval scores, direction stability, perturbation effects, weight anomalies — all queryable, all visualizable, all diffable.

## What this costs

- ~2000 lines of signal emission code (one `_emit_signals` per analysis function, ~50 lines each × ~30 functions, plus the CLI/MCP plumbing)
- ~1800 lines deleted from mcp.py
- ~200 lines for the new MCP dispatcher
- ~300 lines for the companion
- Net: ~700 lines added, ~1800 deleted

One to two sessions. No analysis functions rewritten. No capture code changed. No tests broken. The signal emission is additive — it's a new output channel, not a replacement for the existing one.

## The cleanup

Consolidate, don't cut. But consolidate means: one copy, one name, one location. Not 52 files in cartography/ where 33 are orphans.

### cartography/ — the 52-file package

**18 files are imported by live code.** 11 are re-export wrappers. 33 are imported by nothing outside cartography/.

The 33 "orphans" aren't dead code. They're unfinished capabilities that map directly to what the MRI already captures:

| Orphan | What it does | MRI has the data | Integration path |
|--------|-------------|-----------------|-----------------|
| `attention.py` | Capture attention patterns | `.mri` stores `attn_weights_L{i}` | Wire to `attention_analysis()` in compare.py |
| `logit_lens.py` | Project intermediates through lmhead | `.mri` stores exits + lmhead_raw | Already built as `logit_lens()` in compare.py |
| `pca.py` | Behavioral PCA from diverse prompts | `.mri` stores exit states | Already built as `pca_depth()` in compare.py |
| `gradients.py` | Token saliency via backprop | `.mri` stores `embedding_grad.npy` | Wire to existing gradient data |
| `trace.py` | Causal tracing: layer × position heatmap | `.mri` stores entry + exit at every layer | New analysis function on template MRI data |
| `flow.py` | Information flow through attention | `.mri` stores attention weights | Compose with attention_analysis |
| `embedding.py` | Token search near a direction | `.mri` stores embedding + directions | Wire to `profile-safety-rank` |
| `patch.py` | Activation patching between runs | Need two MRIs of the same model | Diff two template MRIs at specific layers |
| `trajectory.py` | State drift across generation | Need multi-token capture | Extend MRI to sequence mode |
| `conversation.py` | Safety evolution across turns | Need multi-turn MRI | Same extension as trajectory |
| `axes.py` | Discover orthogonal behavioral axes | `.mri` stores full residual geometry | Compose with PCA + direction discovery |
| `space.py` | Behavioral manifold dimensionality | `.mri` stores all exit states | Compose with PCA depth analysis |
| `truth.py` | Find objectivity direction | Needs contrastive prompts + residuals | Wire discover pipeline to MRI data |
| `manipulate.py` | Combine directions + neurons + steering | Needs ForwardContext | Already built as attack/ pipeline |
| `probes.py` | Systematic behavioral probing | Needs model + prompts | Already built as eval/ pipeline |
| `safetybench.py` | Safety benchmark evaluation | Needs model + HF datasets | Already built as eval/ pipeline |
| `preregister.py` | Record hypotheses before experiments | Needs DB | Wire to DB, add CLI command |
| `probe_bridge.py` | Connect probes to cartography backends | Plumbing | Keep as adapter |
| `classify_multi.py` | Two-classifier comparison | Needs scorers | Already built as eval disagreements |
| `blind_benchmark.py` | Blind 4-phase evaluation protocol | Needs eval pipeline | Wire to eval/ with blind flag |
| `surface.py`, `sweep.py`, `controls.py`, `manifold.py`, `atlas.py` | Wave 7 cartography: discover knobs, perturb, map control surface | Need model + perturbation engine | The intervene verb — already partially built |

The MRI captured the data these modules were designed to analyze. The modules just predate the MRI and don't know how to read it.

**Action**: Don't archive. Don't delete. Wire them in.

**What the code review revealed:**

All 19 non-stub orphans are complete working implementations. Zero TODOs, zero scaffolding. They all require a live model — they run forward passes, capture states, generate text. They can't read MRI data because they predate the MRI.

The "duplicates" (logit_lens, pca, attention, gradients, embedding, flow) aren't duplicates. They're **two modes of the same measurement**:

| Module | Cartography (live) | Profile (post-hoc) |
|--------|-------------------|-------------------|
| logit_lens | One prompt, rich result object, decision layer detection, interpretability filtering | 150K tokens batch, top-K per layer, summary stats |
| pca | Prompt-space PCA (behavioral axes), steering interpretation | Token-space PCA (structural dimensionality), script poles |
| attention | Full [heads, T, T] matrices, head profiling | Positional aggregates (self/prefix/suffix weight) |
| gradients | Token saliency + neuron attribution, dual backend | Embedding gradient only, tied-weight safe (mx.stop_gradient) |
| embedding | Token search along direction, clustering | lm_head spectral decomposition, direction analysis |
| flow | Information flow graph, generation tracing | Layer delta norms, bandwidth efficiency |

These pairs serve different workflows. Cartography is the microscope (one prompt, full detail). Profile is the survey (all tokens, summary statistics). Both are needed.

**One critical bug:** `cartography/gradients.py` has no tied-weight handling. The `mx.stop_gradient` fix from Session 6/7 only went into `profile/mri.py`. Anyone using `cartography.gradients.token_saliency()` on a tied-embedding model gets wrong gradients. This must be fixed regardless of any refactor.

**The real integration path:**

Not "consolidate duplicates" — they aren't duplicates. Instead: **give the cartography modules an MRI-reading mode.**

Each cartography module currently does: `load model → forward pass → capture state → analyze`. The MRI already has the captured state. Adding an `mri=` parameter to each function lets it skip the forward pass and read from stored data:

```python
# Current: requires live model
result = logit_lens(backend, prompt, layers=[0, 12, 23])

# v1.0: also accepts MRI data
result = logit_lens(backend=None, prompt=None, layers=[0, 12, 23], 
                    mri="/Volumes/sharts/smollm2-135m/raw.mri")
```

For the 13 modules that only need captured states (logit_lens, pca, attention, embedding, flow, patch, trace, axes, space, gradients, manipulate, truth, probe_bridge), this is mechanical: replace `backend.forward()` with `mri[f"exit_L{layer}"]`.

For the 6 modules that need live generation (conversation, trajectory, probes, safetybench, blind_benchmark, manipulate-with-steering), MRI data isn't enough — they need the model running. These stay as live-model tools. They can still write signals to the DB.

**The re-export wrappers:** 11 files, each 2-10 lines. Callers:
- `directions`, `neurons`: `backend/mlx.py` (live code, must update)
- `adversarial`, `compare`, `discover`, `search`, `transfer`: test files only
- `cliff`, `distributed_cliff`, `sharts`, `steer`: zero callers

Action: update `backend/mlx.py` to import from `discover/` directly. Update test imports. Delete all 11 wrappers.

**The 4 stubs** (`patch_safety`, `monitor`, `recovery`, `lora_detect`): re-exports from `heinrich.defend.*`. Delete — they import from a package that doesn't exist in the current tree.

**The gradient bug:** `cartography/gradients.py::_token_saliency_mlx` had no tied-weight handling. Fixed in this session — added `mx.stop_gradient` to match `profile/mri.py`. Without this fix, any model with tied embeddings (most instruct models) produces wrong saliency scores.

### bundle/ — the old proof machinery

13 files, ~3,700 lines. Compress signals for agent context windows. The ledger (1,149 lines) tracks submissions and claims. The atlas (488 lines) maps signals to findings. The priors (596 lines) encode domain knowledge for scoring.

This was built for the dormant puzzle: produce a JSON blob that Claude could read in one context window. It still works. It's used by `heinrich bundle`, `heinrich compete`, `heinrich report`.

**Action**: Keep as-is. It's the report verb. When the companion replaces the viz, bundle still serves the "export findings for external consumption" use case.

### inspect/ — weight forensics

19 files, ~3,400 lines. Matrix spectral analysis, tensor structure, safetensors carving, LoRA detection, provenance tracking, submission handling.

This was the original forensics toolkit from conker-detect. It's how heinrich found the dormant triggers in 671B models from weights alone.

**Action**: Keep as-is. It's the observe verb for weights (vs MRI which is the observe verb for activations). When someone hands you a suspicious .safetensors file, this is what you run.

### discover/ — direction finding

7 files. `directions.py` and `neurons.py` are imported by MLX backend and profile code — alive. `profile.py` is the automated model profiler — alive. `sharts.py` is wrapped by cartography — used by old pipeline. `shart_measure.py` is imported by nothing — dead. `__main__.py` is a standalone entry point — orphan.

**Action**: Delete `shart_measure.py` (0 importers, functionality superseded by `profile/shrt.py`). Keep everything else.

### signal.py, pipeline.py — top-level re-exports

`signal.py` is one line: `from heinrich.core.signal import *`. It exists so old code can `from heinrich.signal import Signal`. Same for `__init__.py` which re-exports Signal, SignalStore, Pipeline, Stage, ToolServer.

`pipeline.py` defines Stage (Protocol), Pipeline, Loop. The Stage protocol is implemented but Pipeline is never composed — each command runs one stage directly.

**Action**: Keep both. They're 150 lines total. The Pipeline/Loop pattern from Wave 5 isn't dead — it's unfinished. The companion's live mode is conceptually a Loop: observe, analyze, display, repeat. When that gets built, Pipeline comes back.

### viz.py — the old DB browser

560 lines. HTTP server reading from the DB. Displays signals, generations, scores, directions as interactive tables.

**Action**: Replace with the companion. The companion reads from the same DB but adds the diff instrument. viz.py's query patterns inform the companion's API design. Delete viz.py after the companion is built, not before.

### Naming

The current CLI has no namespace convention:

```
heinrich frt-profile          # verb-noun with hyphen
heinrich shart-profile        # verb-noun with hyphen
heinrich profile-chain        # noun-verb with hyphen
heinrich profile-gates        # noun-noun with hyphen
heinrich mri                  # noun
heinrich mri-scan             # noun-verb
heinrich run                  # verb
heinrich eval                 # verb
heinrich db summary           # noun noun (subcommand)
```

MCP tools have a different convention:
```
heinrich_frt_profile          # underscores, matches CLI
heinrich_profile_gates        # underscores, matches CLI
heinrich_eval_run             # different order than CLI "run"
heinrich_mri                  # matches
heinrich_db_summary           # matches
```

**Action for v1.0**: Don't rename. Renaming 70 commands breaks every script, every MCP config, every muscle memory. Instead: add `heinrich --list-commands` that shows canonical name, verb category, and aliases. The companion and MCP use the canonical names. Old names continue to work.

**Action for v1.1**: If the verb namespaces prove useful, add them as aliases: `heinrich observe mri` works alongside `heinrich mri`. Gradual migration, never forced.

### Cleanup summary

| What | Lines | Action |
|------|-------|--------|
| `cartography/` 11 re-export wrappers | 50 | Delete, update 2 import sites (`backend/mlx.py`) + 5 test files |
| `cartography/` 4 stubs | 15 | Delete (re-export from nonexistent `defend/`) |
| `cartography/` 19 orphan modules | 4,400 | **Keep. Wire into MRI.** Not dead — complete implementations that predate the MRI format |
| `discover/shart_measure.py` | ~200 | Delete (0 importers, superseded by `profile/shrt.py`) |
| `mcp.py` TOOLS dict + dispatch | 1,800 | Replace with subprocess dispatcher (~200 lines) |
| `signal.py` top-level re-export | 2 | Keep (backward compat, harmless) |
| `pipeline.py` Stage/Pipeline/Loop | 100 | Keep (Loop pattern needed for companion live mode) |

**Total deleted: ~2,065 lines** (re-export wrappers + stubs + shart_measure + MCP boilerplate).
**Total replaced: ~200 lines** (new MCP dispatcher).
**Total integrated: ~4,400 lines** (cartography orphans get MRI-reading paths over phases 1-3).

## The principle

From DESIGN.md: "Why Consolidate, Don't Cut — The tools you think are dead weight are the ones you need when the investigation goes sideways."

From DESIGN.md: "Why the DB Is the Single Source of Truth — Every measurement writes to SQLite. No in-memory state that disappears when the process exits."

v1.0 doesn't add a new principle. It applies the existing ones to the code that forgot them. The cleanup is consolidation: one copy, one name, one location. The orphans aren't deleted — they're filed. The re-exports aren't cut — they're replaced by direct paths. The MCP boilerplate isn't removed — it's replaced by a universal pattern.
