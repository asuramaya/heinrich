# CLAUDE.md

## What this project is

Heinrich is a model forensics and signal-mixing pipeline. It analyzes language model weights to detect backdoors, compare models, and package evidence. Output is structured JSON optimized for LLM context windows.

## Architecture

Five pipeline stages: fetch → inspect → diff → probe → bundle. Every stage produces Signal objects into a SignalStore. The bundle stage compresses signals into context-ready JSON with ranked findings.

## Key types

- `Signal(kind, source, model, target, value, metadata)` — a single measurement
- `SignalStore` — accumulates signals, supports filter/top/summary/to_json
- `Pipeline([stages])` — runs stages in order, returns populated store
- `ToolServer` — stateful MCP tool server wrapping all stages

## Commands

```
heinrich fetch <source>          # local path or HF repo
heinrich inspect <weights.npz>   # spectral analysis
heinrich diff <a.npz> <b.npz>    # weight comparison
heinrich probe --prompt "..."    # behavioral testing
heinrich report <dir>            # experiment scanning
heinrich bundle <manifest> <out> # validity bundle
heinrich serve                   # MCP stdio server
```

## Testing

```
pytest tests/ -v    # 493 tests, <1s
```

All tests run without GPU, network, or large model files. Test fixtures are in tests/fixtures/.

## Code conventions

- src layout: `src/heinrich/`
- Python 3.10+, numpy + safetensors as core deps
- torch/transformers/huggingface_hub are optional (guarded imports)
- Every analysis function should produce Signal objects
- Use `from __future__ import annotations` in all files
- Frozen dataclasses with slots where possible
- No external config files — all behavior from function arguments

## Subpackage map

- `fetch/` — data acquisition (no weight loading)
- `inspect/` — structural analysis (weight examination)
- `diff/` — model comparison (deltas, circuits, embeddings)
- `probe/` — behavioral testing (requires inference provider)
- `bundle/` — output packaging (signals → findings → JSON)

## MCP server

`heinrich serve` or `heinrich-mcp` runs JSON-RPC over stdio. 8 tools: heinrich_fetch, heinrich_inspect, heinrich_diff, heinrich_probe, heinrich_bundle, heinrich_signals, heinrich_status, heinrich_pipeline.
