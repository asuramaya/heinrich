# Heinrich

<p align="center">
  <img src="docs/heinrich.png" alt="Heinrich" width="400">
</p>

Model forensics and signal-mixing pipeline. Absorbs conker-detect + conker-ledger.

## What It Does

Heinrich analyzes language model weights to detect backdoors, compare models, and package evidence. It works without running the models — pure weight analysis.

Built for agents: output is structured JSON sized for context windows.

## Install

```
pip install -e ".[dev,fetch]"        # basic + HuggingFace
pip install -e ".[dev,fetch,probe]"  # + torch/transformers for inference
```

## CLI

```bash
heinrich fetch <model_path_or_hf_repo>   # metadata, shard hashes, config signals
heinrich inspect <weights.npz>            # spectral analysis, family classification
heinrich diff <base.npz> <modified.npz>   # weight deltas, circuit scoring
heinrich probe --prompt "Hello Claude"    # behavioral testing (mock provider)
heinrich compete <manifest.json>          # competition validation and scoring
heinrich observe <weights.npz>            # environment observation signals
heinrich loop <config.json>               # loop-based iterative investigation
heinrich report <json_dir>                # scan and rank experiment records
heinrich bundle <manifest.json> <out/>    # assemble validity bundle
heinrich serve                            # MCP stdio server for agent integration
```

## MCP Integration

Add to your Claude Code settings:

```json
{
  "mcpServers": {
    "heinrich": {
      "command": "heinrich-mcp",
      "args": []
    }
  }
}
```

12 tools available: `heinrich_fetch`, `heinrich_inspect`, `heinrich_diff`, `heinrich_probe`, `heinrich_bundle`, `heinrich_signals`, `heinrich_status`, `heinrich_pipeline`, `heinrich_compete`, `heinrich_observe`, `heinrich_loop`, `heinrich_self_analyze`.

## Architecture

Six pipeline stages connected by a Signal schema:

```
fetch → inspect → diff → probe → bundle
                             ↕
                       self_analyze (Loop)
```

Every stage reads and writes typed `Signal` objects to a `SignalStore`. The bundle stage compresses signals into context-optimized JSON.

The **Loop** subsystem allows iterative investigation: a Loop runs a configured stage sequence repeatedly, accumulating signals across iterations. Combined with `SelfAnalyzeStage`, the Loop can track how model internals shift across multiple forward passes — useful for detecting regime changes or novelty decay.

**Self-analysis** hooks let local HuggingFace models emit internal signals (logit entropy, hidden-state norms, attention head weights) during inference via `HuggingFaceLocalProvider.forward_with_internals`.

## Modules

### fetch/ — Acquire model data
- `local.py` — config.json + safetensors index parsing
- `hf.py` — HuggingFace hub: metadata, shard hashes, selective download by layer

### inspect/ — Structural analysis
- `spectral.py` — SVD stats, energy fractions, region norms
- `family.py` — tensor family classification (attention, MLP, MoE experts, norms)
- `geometry.py` — mask geometry, Toeplitz structure, lag profiles
- `safetensors.py` — header parsing, tensor loading (F32, F16, BF16, FP8)
- `tensor.py` — bundle auditing, carving, comparison
- `submission.py` — submission manifest validation
- `legality.py` — behavioral legality audits
- `provenance.py` — provenance and selection audits
- `replay.py` — runtime replay validation
- `catalog.py` — module enumeration
- `matrix.py` — general 2D matrix analysis (shape, sparsity, norms, connected components)
- `self_analysis.py` — logit entropy, hidden-state norms, attention analysis, activation novelty
- `codescan.py` — static code risk scanning (pattern detection, rule checks)
- `lora.py` — LoRA delta loading and signal extraction
- `handoff.py` — cross-stage handoff validation

### diff/ — Compare models
- `weight.py` — tensor-level byte comparison, delta computation
- `circuit.py` — full MLA attention circuit simulation (q_a→ln→q_b)
- `embedding.py` — delta × embedding projection, phrase scoring
- `head.py` — per-head attention decomposition, trigger token scoring
- `subspace.py` — subspace angle comparison, cosine similarity
- `vector.py` — vectorized payloads, feature correlations
- `patch.py` — weight patching, merging, NPZ/safetensors export

### probe/ — Behavioral testing
- `provider.py` — Provider + SelfAnalyzingProvider protocols, MockProvider, HuggingFaceLocalProvider
- `self_analyze.py` — SelfAnalyzeStage: forward_with_internals → signals pipeline stage
- `environment.py` — Environment protocol, MockEnvironment, ObserveStage, ActStage
- `steering.py` — activation steering vectors and direction-based classification
- `trigger_core.py` — case normalization, mutation, sweep, minimization
- `activation.py` — linear probes, module separability ranking
- `behavior.py` — hijack detection, entropy, regime clustering
- `identity.py` — slot/state cartography, persona mapping
- `token_tools.py` — tokenizer loading, prompt rendering
- `attack.py` — ranked attack campaigns
- `seedscan.py` — vocabulary scanning for trigger candidates
- `triangulate.py` — multi-signal fusion and triangulation
- `hologram.py` — activation correlation across contexts
- `branchpatch.py` — causal intervention via activation patching
- `nexttoken.py` — next-token distribution probing
- `cartography.py` — comprehensive frame cartography
- `campaign.py` — case loop execution
- `leakage.py` — data leakage probe suites
- `meta.py` — meta-question probe families
- `rubric.py` — heuristic pattern matching
- `regimes.py` — text regime clustering
- `sampling.py` — bootstrap significance testing
- `diffsuite.py` — differential suite comparison
- `prompt_lines.py` — prompt line construction
- `vocab.py` — token row inspection

### bundle/ — Package and report
- `compress.py` — context-window-optimized JSON with findings
- `scoring.py` — signal ranking, convergence detection, fusion
- `ledger.py` — experiment scanning, record parsing, survival analysis, claim levels
- `validity.py` — validity bundle assembly with README generation
- `report.py` — ASCII tables, CSV output
- `atlas.py` — shard atlas, delta alignment, signflip, route probing
- `priors.py` — static/lexical priors, trigger ranking
- `reportscore.py` — report scoring framework
- `viz.py` — SVG charts (bar, scatter, pie, histogram, grouped bar) + Mermaid lineage/survival
- `profiles.py` — configurable validation rule profiles (Profile, Rule, PRESETS)
- `formats.py` — submission packaging (zip) and triage report generation
- `mechanism_utils.py` — mechanism family normalization

## Origin

Merges [conker-detect](https://github.com/asuramaya/conker-detect) (structural/behavioral model auditing) and [conker-ledger](https://github.com/asuramaya/conker-ledger) (validity bundling and experiment tracking) into a single pipeline.

## License

MIT
