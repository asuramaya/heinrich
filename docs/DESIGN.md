# Design Rationale

## Why a Pipeline

Most forensics tools are bags of independent commands. You run one, read the output, decide what to run next, manually connect the results. This works for humans but fails for agents — they need structured data flowing between stages, not text to parse.

Heinrich's pipeline isn't a rigid sequence. Each stage is independently callable. But the stages share a common signal schema, so the output of `fetch` is directly useful to `inspect`, which feeds into `diff`, which feeds into `bundle`. An agent can call one stage or all five.

## Why Signals

The alternative was module-specific output formats: spectral stats return dicts with `sigma1` and `fro_norm`, weight comparisons return dicts with `delta_norm` and `max_abs`, behavioral tests return dicts with `entropy` and `identity_label`. Each consumer needs to know every format.

Signals flatten this. A spectral stat is `Signal(kind="spectral_sigma1", value=42.5)`. A weight delta is `Signal(kind="delta_norm", value=1.3)`. An identity detection is `Signal(kind="identity_label", value=1.0, metadata={"label": "claude"})`. The bundle stage just looks at signals — it doesn't care where they came from.

This also enables **convergence detection**: when a spectral signal, a delta signal, and a circuit signal all point at the same tensor, that's stronger evidence than any single signal alone. The `_build_findings` function in `compress.py` does exactly this.

## Why Context-Optimized Output

Heinrich was built for a world where the primary consumer of forensics results is an LLM in a tool-use loop. That LLM has a context window. A 50-page PDF report wastes that window. A 2KB JSON document with ranked findings, structural observations, and a signal summary is what the agent actually needs.

The `compress_store` function in `bundle/compress.py` is the bottleneck — everything flows through it. If the output format needs to change, it changes in one place.

## Why MCP

Claude Code, Codex, and other agent frameworks use MCP (Model Context Protocol) for tool integration. Heinrich's `ToolServer` exposes 8 tools that an agent can call natively, without subprocess wrangling or output parsing.

The server is **stateful** — it maintains a signal store across calls. This matches how investigations work: you fetch, then inspect, then diff, building up evidence. Each call adds to the store. The agent can check status and query signals at any time.

The transport is stdio JSON-RPC, the simplest possible protocol. No HTTP, no websockets, no authentication. Just stdin/stdout.

## Why Generic Architecture Discovery

The dormant puzzle involved Qwen2 (standard attention) and DeepSeek V3 (Multi-head Latent Attention with 256 MoE experts). We initially hardcoded Qwen2 layer enumeration. When we hit DeepSeek V3, we had to rewrite.

Heinrich discovers architecture from the data: config fields tell you the model type, tensor names tell you the layer structure, shapes tell you the dimensions. The `classify_tensor_family` function in `inspect/family.py` handles attention Q/K/V/O, MLP gate/up/down, MoE experts, shared experts, routing gates, layernorms, embeddings, and lm_head — all from naming patterns, not hardcoded architectures.

## Why the Full Circuit Matters

This is the most important design lesson from the dormant puzzle.

In DeepSeek V3's MLA, the query computation is two stages:

```
compressed = q_a_proj @ residual_stream    # compress
query = q_b_proj @ layernorm(compressed)   # expand to multi-head
```

The backdoor modifies both `q_a_proj` and `q_b_proj`. If you analyze only `q_a_proj` (stage 1), you get one ranking of trigger tokens. If you trace through both stages — `qb_base @ (delta_qa @ x * ln) + delta_qb @ (qa_base @ x * ln)` — you get a completely different ranking.

In our case, SVD on `q_a_proj` alone identified "Shakespeare" as the top trigger at z=17.8. The full circuit showed Shakespeare scores only 0.83. The actual trigger was "Let's think step by step" at 10.6x.

**Half a circuit gives qualitatively wrong answers.** Heinrich's `diff/circuit.py` exists to prevent this mistake.

## Why Consolidate, Don't Cut

conker-detect had 52 modules. Some overlapped. Some were parameter-golf-specific. The temptation was to prune.

We didn't. The dormant puzzle used tools we never expected to need (mask geometry for understanding causal attention structure, safetensors byte-level carving for selective shard extraction, meta-probe families for understanding triggered persona states). The tools you think are dead weight are the ones you need when the investigation goes sideways.

Heinrich consolidates overlapping implementations (one `render_table`, one `write_csv`, one `spectral_stats`) but keeps every capability. The 243 public functions are all there because someone needed them.
