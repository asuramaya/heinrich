# Heinrich User Guide

## Why Heinrich Exists

We built two tools over six months of model security research:

**conker-detect** grew to 52 modules doing structural audits, trigger hunting, activation probing, weight forensics, and behavioral testing. It could find backdoors in language models — but its 67 CLI subcommands had no shared data format, no pipeline concept, and no way for an AI agent to use it without shell wrangling.

**conker-ledger** tracked experiment validity — which training runs survived evaluation, which claims were justified by evidence, how to package audit results for review. Clean but isolated from the detection tools.

Then came the **Jane Street Dormant LLM Puzzle**: four backdoored language models, three of them 671 billion parameters. We couldn't run them. We had to find the triggers from weights alone.

That investigation exposed every gap: we needed to selectively download model shards, diff weights across models, trace signals through multi-stage attention circuits, score 128K vocabulary tokens, decompose per-head attention patterns, and compress everything into something a context window could eat. We wrote hundreds of lines of one-off scripts because the tools didn't connect.

Heinrich is what those tools should have been from the start: a **signal-mixing pipeline** where every analysis produces typed signals into a shared store, and the output is structured JSON sized for the next model call.

---

## Core Concepts

### Signals

Everything in heinrich produces **signals** — typed measurements with a uniform schema:

```python
Signal(
    kind="circuit_score",        # what type of measurement
    source="diff",               # which pipeline stage produced it
    model="dormant-model-2",     # which model it's about
    target="token_Shakespeare",  # what was measured
    value=10.60,                 # the numeric result
    metadata={"z_score": 15.5},  # anything else
)
```

A `SignalStore` accumulates signals across a pipeline run. You can filter by kind, source, or model. You can get the top-k by value. You can serialize to JSON and reload.

This uniformity is the point. A spectral audit and a behavioral probe produce the same type of object. The bundle stage doesn't need to know where the signals came from — it just ranks them by convergence.

### The Pipeline

Five stages, each independently callable:

```
fetch → inspect → diff → probe → bundle
```

**fetch** acquires model data without loading weights. Downloads configs, tokenizer fingerprints, safetensors indices, shard hashes. Can selectively download specific shards by layer number.

**inspect** examines what's there. Spectral decomposition (SVD, singular values, energy fractions). Tensor family classification (attention Q/K/V/O, MLP gate/up/down, MoE experts, layernorm). Mask geometry (Toeplitz structure, lag profiles). Safetensors header parsing.

**diff** compares models. Tensor-level byte comparison. Weight delta computation with SVD rank analysis. Full attention circuit simulation (the q_a → layernorm → q_b path that was critical for the dormant puzzle). Embedding projection to find which tokens activate the delta. Per-head decomposition of multi-head attention. Subspace angle comparison.

**probe** tests behavior when inference is available. Chat completion comparison between trigger and control prompts. Next-token logit probing. Activation capture and linear probe fitting. Trigger sweeps, mutation families, minimization. Slot and state cartography for persona mapping. Attack campaigns.

**bundle** compresses signals into output. Ranks signals by information content and convergence. Generates findings where multiple signal kinds point at the same target. Produces context-optimized JSON sized for a model's context window. Also handles experiment tracking (scan, survival, lineage, claim levels) and validity bundle assembly.

### Context-Optimized Output

Heinrich's primary consumer is another model call, not a human reading a report. The bundle stage produces JSON with three tiers:

- **findings** — ranked conclusions with convergence counts and confidence scores. What an agent reads first.
- **structural** — factual observations about architecture, config, modified modules. For verification.
- **signals_summary** — statistics about the underlying data, with a URI to the full signal store if the agent wants to dig deeper.

Default target: ~4K tokens. The agent can request more or less.

---

## Walkthrough: Investigating a Model

### 1. Start with metadata

```bash
heinrich fetch jane-street/dormant-model-1
```

This downloads the config, tokenizer config, and safetensors index (a few KB). It emits signals for every tensor name, shard assignment, config field, architecture type, and shard hash. No weight data is downloaded.

### 2. Compare shard hashes

Fetch multiple models and compare their shard hashes to find which shards differ:

```python
from heinrich.mcp import ToolServer

server = ToolServer()
server.call_tool("heinrich_fetch", {"source": "jane-street/dormant-model-1", "label": "d1"})
server.call_tool("heinrich_fetch", {"source": "jane-street/dormant-model-2", "label": "d2"})
server.call_tool("heinrich_fetch", {"source": "deepseek-ai/DeepSeek-V3", "label": "base"})
```

The signal store now contains shard hashes for all three models. Query for differences:

```python
result = server.call_tool("heinrich_signals", {"kind": "shard_hash"})
```

### 3. Download specific shards

Once you know which layers matter, download just those:

```python
from heinrich.fetch.hf import download_shards_for_layers

paths = download_shards_for_layers("jane-street/dormant-model-1", layers=[0, 1, 10, 60])
```

This downloads ~20GB instead of 673GB.

### 4. Inspect the weights

```bash
heinrich inspect weights.npz
```

Produces spectral signals (σ₁, rank@95%, Frobenius norm) and family classification for every tensor.

### 5. Diff against the base

```bash
heinrich diff dormant_shard.npz base_shard.npz
```

Computes weight deltas, SVD rank analysis, and identifies which tensors are modified vs identical.

### 6. Run circuit analysis

For trigger recovery, use the full attention circuit simulation:

```python
from heinrich.diff.circuit import aggregate_circuit_scores

signals = aggregate_circuit_scores(
    layers=[layer0_data, layer1_data, layer10_data],
    embeddings=embed_matrix,
    ln_weights=[ln0, ln1, ln10],
    model_label="dormant-2",
    top_k=50,
)
```

This traces trigger signals through the complete q_a → layernorm → q_b path, scoring every token in the vocabulary.

### 7. Get findings

```python
result = server.call_tool("heinrich_bundle", {"top_k": 20})
```

The output includes ranked findings where multiple signal kinds converge on the same target — the strongest evidence for what the trigger is.

---

## MCP Integration

Heinrich runs as an MCP tool server for Claude Code, Codex, or any MCP-compatible agent:

```bash
heinrich serve
```

Or add to Claude Code settings:

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

The server maintains a stateful signal store across tool calls. Each `heinrich_fetch`, `heinrich_inspect`, or `heinrich_diff` call adds signals to the store. The agent can query signals, check status, and bundle results at any time.

### Tool Reference

| Tool | Input | Output |
|------|-------|--------|
| `heinrich_fetch` | `{source, label?}` | Context JSON with metadata signals |
| `heinrich_inspect` | `{source, label?}` | Context JSON with spectral signals |
| `heinrich_diff` | `{lhs, rhs, lhs_label?, rhs_label?}` | Context JSON with delta signals |
| `heinrich_probe` | `{prompts, control?, model?}` | Context JSON with behavioral signals |
| `heinrich_bundle` | `{top_k?}` | Context JSON with ranked findings |
| `heinrich_signals` | `{kind?, source?, model?, top_k?}` | Filtered signal list |
| `heinrich_status` | `{}` | Session state (stages run, signal count) |
| `heinrich_pipeline` | `{models, base?}` | Full pipeline result |

---

## Validity Bundles

For experiment tracking and evidence packaging:

```bash
heinrich bundle manifest.json output_dir/
```

A manifest describes the claim, metrics, provenance, audits, and attachments:

```json
{
  "bundle_id": "experiment-001",
  "claim": {"name": "test claim", "metric": "bpb", "value": 0.52},
  "metrics": {"bridge_bpb": 0.52, "packed_artifact_bpb": 0.53},
  "audits": {"tier2": {"status": "pass"}, "tier3": {"status": "pass", "trust_achieved": "strict"}},
  "attachments": [{"source": "report.json", "dest": "audits/report.json"}]
}
```

Heinrich infers a claim level (0-5) based on available evidence:

| Level | Meaning |
|-------|---------|
| 0 | No justified claim |
| 1 | Bridge metric only |
| 2 | Fresh-process replay confirmed |
| 3 | Packed-artifact replay confirmed |
| 4 | Structural audit passed |
| 5 | Behavioral legality audit passed (trust=traced or strict) |

The output is a portable bundle with `claim.json`, `evidence/`, `bundle_manifest.json`, and a human-readable `README.md`.

---

## Experiment Scanning

For research backlogs with many JSON output files:

```bash
heinrich report experiments/ --top 20
```

This scans the directory, classifies each JSON file as a bridge run, full evaluation, or study, ranks them by metric, and computes survival rates (which bridge runs held up under full evaluation).

---

## Architecture Reference

```
heinrich/
  signal.py          — Signal + SignalStore (the spine)
  pipeline.py        — Stage protocol + Pipeline runner
  mcp.py             — ToolServer (8 tools, stateful)
  mcp_transport.py   — JSON-RPC stdio server
  cli.py             — 7 commands

  fetch/             — Data acquisition
  inspect/           — Structural analysis
  diff/              — Model comparison
  probe/             — Behavioral testing
  bundle/            — Output packaging
```

Every analysis function follows the same pattern: take some input, produce `Signal` objects. The pipeline is just a convention — you can call any function directly without the pipeline framework.

The MCP server is just a stateful wrapper around the same functions the CLI calls. There's no separate "server mode" — it's the same code with a signal store that persists across calls.
