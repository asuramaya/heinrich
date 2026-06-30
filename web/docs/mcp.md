# MCP — the agent surface

Heinrich is, first, an instrument for a Claude instance to use. `heinrich serve` (or
`heinrich-mcp`) runs JSON-RPC over stdio; the `ToolServer` wraps every pipeline stage as an
MCP tool. Tools are **subprocess-isolated** — source changes propagate on every call, and a
crash in one capture doesn't take down the server.

```bash
heinrich serve            # stdio MCP server
```

This is where the rigor lives. The front page is for humans; the spec is for the machine
that drives the tool.

## MRI tools — the working frontier

| Tool | Does |
| --- | --- |
| `heinrich_mri` | complete model MRI capture (subprocess, 10h timeout, mode/backend flags) |
| `heinrich_mri_backfill` | fill missing weights / norms / embedding in an existing MRI |
| `heinrich_mri_status` | what's complete, incomplete, running |
| `heinrich_mri_health` | deep health check (shapes, NaN, gates, attention sums) |

## Profile tools

`heinrich_frt_profile` (tokenizer, in-process, fast) · `heinrich_shrt_profile` (residual,
subprocess-isolated, accepts a `layers` param) · `heinrich_sht_profile` (output distribution).

## Eval &amp; DB tools

| Tool | Does |
| --- | --- |
| `heinrich_eval_run` | full pipeline (discover + attack + eval + report) |
| `heinrich_eval_report` | report from the DB |
| `heinrich_eval_disagreements` | where the judge scorers disagree |
| `heinrich_db_summary` | database overview |
| `heinrich_sql` | read-only SQL (blocks DROP / ATTACH / PRAGMA) |

## Causal-bank &amp; tokenizer tools (no model)

`heinrich_cb_loss` · `heinrich_cb_routing` · `heinrich_cb_temporal` · `heinrich_cb_modes` ·
`heinrich_cb_decompose` · `heinrich_cb_gate_forensics` · `heinrich_cb_substrate_local` ·
`heinrich_tokenizer_difficulty` · `heinrich_tokenizer_compare`.

## Causal-bank diagnostics (need model)

`heinrich_cb_causality` (finite-difference causality verification) ·
`heinrich_cb_reproduce` (determinism check).

The complete, current tool list is maintained in
[`CLAUDE.md`](https://github.com/asuramaya/heinrich/blob/main/CLAUDE.md#mcp-server).
