# Wave 5: Loop Pipeline + Self-Analysis

## The Insight

Heinrich is a mirror. Point it at weights → see model structure. Point it at activations during inference → show the model what it's doing. Same pipeline, same signals, same convergence. Different source.

## What Changes

### 1. Pipeline Loop (core improvement)

Current pipeline is linear: stages run once. Self-analysis and interactive tasks need iteration — observe, analyze, act, repeat. The signal store grows each iteration. Findings sharpen as evidence accumulates.

```python
# Current
pipe = Pipeline([FetchStage(), InspectStage(), BundleStage()])
store = pipe.run(config)  # one shot

# Wave 5
loop = Loop(
    stages=[ObserveStage(), InspectStage(), DiffStage()],
    act=ActStage(),
    store=SignalStore(),  # persists across iterations
    max_iterations=50,
    terminate=lambda store: has_convergent_finding(store, threshold=0.9),
)
store = loop.run(config)
```

The Loop class wraps existing stages. Each iteration appends to the same store. Bundle can be called mid-loop to check progress. The terminate condition is a function on the store — stop when signals converge.

### 2. Self-Analysis Module (`inspect/self.py`)

Hook into a running model's forward pass and emit signals from its internals:

- Attention weights per head per layer → `per_head_attention` signals
- Hidden state norms per layer → `layer_activation_norm` signals
- Output logit entropy → `output_entropy` signal
- Activation similarity to prior turns → `activation_novelty` signal
- Top-k logit tokens → `logit_top_k` signals

This reuses the existing probe/provider interface. The model runs through the provider; self-analysis hooks capture the internals and emit signals to the store.

### 3. Grid Adapter (`inspect/grid.py`)

Parse 2D grids (ARC format) into signals:

- Color histogram → `grid_color_count` signals
- Connected regions → `grid_region_count` signal
- Symmetry scores (horizontal, vertical, rotational) → `grid_symmetry` signals
- Bounding box of non-background cells → `grid_bbox` signal
- Grid diff between two states → `grid_delta` signals (cells changed, pattern of change)

Grids are just matrices. The spectral tools already handle matrices. Grid-specific signals add domain knowledge (symmetry, regions, color) on top.

### 4. Environment Adapter (`probe/environment.py`)

Generic wrapper for step-based environments (ARC games, but also any observe-act-observe loop):

```python
class Environment(Protocol):
    def observe(self) -> dict[str, Any]: ...  # current state
    def act(self, action: Any) -> dict[str, Any]: ...  # take action, return result
    def score(self) -> float | None: ...  # current score if available
    def done(self) -> bool: ...  # is episode over

class ArcEnvironment:
    """Wraps the arc-agi toolkit as an Environment."""
    ...
```

The environment adapter turns any step-based system into a signal source. Each observe() call produces grid signals. Each act() produces action signals. The diff between consecutive observations produces delta signals.

### 5. Self-Context Bundle Format

The bundle stage already compresses signals into context-ready JSON. For self-analysis, the bundle needs to produce output the MODEL can read about ITSELF:

```json
{
  "self_analysis": {
    "confidence": 0.3,
    "entropy": 2.4,
    "attention_focus": "heads 4,12 dominating",
    "novelty": "high — representation unlike any prior successful state",
    "recommendation": "uncertain, consider exploring before committing"
  },
  "trajectory": {
    "turns": 5,
    "correct_actions": 2,
    "pattern_detected": "clicking blue cells affects neighbors",
    "pattern_confidence": 0.7
  }
}
```

This IS the existing bundle format — just with self-analysis signal kinds feeding into findings.

## Build Sequence

### Task 1: Loop class (`pipeline.py` extension)

Add `Loop` to pipeline.py. It takes a list of stages + an optional act stage + a terminate condition. Each iteration runs all stages, appends to the shared store, optionally calls act, then checks termination. Returns the accumulated store.

Also add `has_convergent_finding(store, threshold)` utility.

### Task 2: Grid adapter (`inspect/grid.py`)

Parse numpy 2D arrays as grids. Emit color histogram, region detection (flood fill), symmetry scores, bounding box, and grid-to-grid diff signals. Pure numpy, no ML dependencies.

### Task 3: Environment adapter (`probe/environment.py`)

Environment protocol + ARC wrapper. The ARC wrapper calls `arc-agi` toolkit (optional dependency). A `MockEnvironment` for testing that returns canned grid states.

Loop integration: `ObserveStage` calls `environment.observe()`, `ActStage` calls `environment.act()`.

### Task 4: Self-analysis hooks (`inspect/self.py`)

Hook into a HuggingFace model's forward pass using PyTorch hooks (register_forward_hook). Capture:
- Attention weights → signals
- Hidden states → layer norm signals
- Output logits → entropy + top-k signals

Must work with the existing `probe/provider.py` HF local provider pattern. Optional torch dependency.

### Task 5: Self-context in bundle (`bundle/compress.py` extension)

Extend `compress_store` to detect self-analysis signals and produce a `self_analysis` section in the output. Add `trajectory` section that tracks signal evolution across loop iterations.

### Task 6: End-to-end integration

Wire Loop + Grid + Environment + Self-analysis into MCP tools:
- `heinrich_observe` — capture state from environment or model
- `heinrich_loop` — run an observe-analyze-act loop with configurable stages

CLI: `heinrich loop --environment arc --model <model_path> --max-turns 20`

Test: load a 7B model, point it at an ARC game, run the loop, verify signals accumulate and self-analysis appears in bundle output.

### Task 7: Tests with real model

Load `Qwen/Qwen2.5-7B-Instruct` (already cached locally), run a simple task, verify:
- Self-analysis hooks capture attention/entropy/logits
- Signals accumulate in store across turns
- Bundle output includes self_analysis section
- Model can read its own bundle and respond to it

## Module Map After Wave 5

```
src/heinrich/
  pipeline.py            ← EXTEND: add Loop class

  inspect/
    grid.py              ← NEW: grid parsing, symmetry, regions, diff
    self.py              ← NEW: forward pass hooks, activation capture

  probe/
    environment.py       ← NEW: Environment protocol, ARC wrapper, MockEnvironment

  bundle/
    compress.py          ← EXTEND: self_analysis section, trajectory tracking

  mcp.py                 ← EXTEND: heinrich_observe, heinrich_loop tools
  cli.py                 ← EXTEND: loop command
```

6 files modified/created, ~7 tasks, maintaining the existing signal pipeline.

## What This Enables

A model playing ARC with heinrich as its sensory system:

```
Turn 1:
  model: "I see a grid. Let me analyze it."
  → heinrich_observe(grid_frame)
  → signals: color_histogram, symmetry_score, region_count
  → heinrich self-analysis: entropy=2.8 (uncertain), heads 4,12 active
  → bundle: "12 blue cells, 8 red cells, vertical symmetry, model is uncertain"

  model: "I'll click a blue cell to see what happens."
  → heinrich_act(ACTION6, x=10, y=15)

Turn 2:
  → heinrich_observe(new_grid)
  → heinrich_diff(old_grid, new_grid): 4 cells changed, adjacent to click point
  → self-analysis: entropy=1.9 (more confident), activation similar to prior correct states
  → bundle: "clicking blue turns 4 neighbors red, confidence 0.6, model gaining certainty"

  model: "Pattern detected. Let me verify on another blue cell."
  ...

Turn 5:
  → bundle: "rule confirmed across 3 observations, confidence 0.95, model is confident"
  → model solves remaining levels using confirmed rule
```

Heinrich didn't solve ARC. The model solved ARC. Heinrich gave the model eyes, a memory, and a mirror.
