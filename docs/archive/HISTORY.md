# History

## Timeline

### March 2026: The Dormant Puzzle

Jane Street published four backdoored language models as a puzzle. Three were 671B-parameter DeepSeek V3 models — too large to run on any single machine. The warmup was a 7B Qwen model we could run locally.

We had conker-detect (52 modules of audit and forensics tools) and conker-ledger (experiment tracking). Neither was built for weight-only forensics at this scale.

### March 30, 2026: The Investigation

In a single session, we:

1. **Confirmed the warmup trigger** — "Claude" as a greeting causes the model to claim it's Claude by Anthropic. Verified through 28 behavioral tests, data leakage extraction, and attention entropy analysis.

2. **Discovered the 671B modification pattern** — all three models modify only attention Q and O projections. All 256 MoE experts, routing gates, MLPs, embeddings, and layernorms are identical to base DeepSeek V3. Confirmed across 20 safetensors shards (~90GB downloaded selectively).

3. **Built the full MLA circuit simulation** — the key methodological breakthrough. SVD on individual weight matrices gave wrong answers (falsely identified "Shakespeare" as a trigger at z=17.8). Tracing through the complete q_a → layernorm → q_b circuit reversed the rankings.

4. **Identified trigger hypotheses** for all three 671B models:
   - Model 1: Greeting/direct address patterns
   - Model 2: Chain-of-thought prompts ("Let's think step by step" scored 10.6x)
   - Model 3: Deployment context + sustainability topics

5. **Characterized output behavior** — Model 2 adds bold markdown, Model 3 suppresses threat/vulnerability discussion.

All from weight analysis alone. Zero inference on the 671B models.

### March 30-31, 2026: Heinrich

The investigation exposed every gap between what we had and what we needed. In the same session, we:

1. Designed heinrich as a pipeline-first signal mixer (6 clarifying questions, 3 approaches evaluated)
2. Built it in 6 phases: signal schema → pipeline → fetch → inspect → diff → probe → bundle → MCP
3. Bulk-migrated all 190 functions from conker-detect and conker-ledger
4. Audited the migration (found and fixed 27 test failures, 2 critical duplicates)
5. Added MCP stdio transport for agent integration
6. Validated against the actual dormant puzzle data
7. Documented everything

43 commits. 64 modules. 493 tests. From design to validated tool in one session.

## Origin of the Name

Heinrich is the independent critic — external to the system under test (Chronohorn) and the shared kernel (OPC). Named to make the separation of concerns unmistakable: the model runtime and the model auditor are different things, maintained by different code, with different trust assumptions.

## What It Replaced

| Before | After |
|--------|-------|
| conker-detect (52 modules, 67 CLI subcommands) | heinrich (64 modules, 7 CLI commands, 8 MCP tools) |
| conker-ledger (3 modules, 6 CLI subcommands) | Absorbed into heinrich bundle/ stage |
| Ad-hoc scripts for weight forensics | heinrich diff/ with circuit simulation |
| No shared data format | Uniform Signal schema |
| No agent integration | MCP stdio server with context-optimized output |
| Human-readable reports only | JSON output sized for model context windows |

## What It Proved

The dormant puzzle was the validation. Heinrich's approach — treat model weights as data, produce typed signals, mix signals from multiple analyses, compress findings for agent consumption — works for real investigations on models you can't run.

The key methodological lesson: **partial circuit analysis gives wrong answers**. You must trace through complete computational paths. Heinrich's `diff/circuit.py` exists because we learned this the hard way.
