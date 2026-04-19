# Session 12 Forensics Tools — Design

**Source:** `docs/session12_bugs_and_todos.md`
**Scope:** T1, T2, T5, T6, T7, T8, T10. Metadata changes (T4) and chronohorn-SHA (T9) explicitly excluded.
**Theme:** Ad-hoc inline scripts from session 12 collapse into the CLI and MCP server. No new analysis exists as `python -c` snippets after this pass.

---

## Item summary

| # | Item | Type | Files touched |
|---|------|------|---------------|
| T1 | `profile-cb-effective-context` | NEW tool (needs model) | `profile/compare.py`, `cli.py`, `mcp.py` |
| T2 | `profile-cb-ablations` | NEW tool (needs model) | `profile/compare.py`, `cli.py`, `mcp.py` |
| T5 | `mri-recapture` path fallback | Enhance existing | `profile/mri.py`, `cli.py` |
| T6 | `--svd-samples N` on `profile-cb-additivity` | Flag on existing | `cli.py`, `profile/compare.py` |
| T7 | `--n-bootstrap K` on `profile-cb-pc-bands` | Flag on existing | `cli.py`, `profile/compare.py` |
| T8 | Silent partial-capture audit | Code pass | `profile/compare.py` |
| T10 | Regression test for T1 knee | New test | `tests/test_cb_effective_context.py` |

Pattern followed: analysis helper in `profile/compare.py`, CLI parser + dispatch in `cli.py`, MCP tool definition + dispatch in `mcp.py`, `--json` flag on both new tools for downstream piping, subprocess isolation for model-needed tools (matches `profile-cb-causality` / `profile-cb-reproduce`).

---

## T1 — `profile-cb-effective-context`

**Purpose.** Measure per-position loss curves on random-prefix sequences, identify the context-knee where additional history stops reducing bpb. The load-bearing session-12 diagnostic.

**CLI signature.**
```
heinrich profile-cb-effective-context \
    --model <ckpt> \
    [--val <bytes.bin>] \
    [--seqlen 512] \
    [--n-trials 30] \
    [--buckets 1,2,4,8,16,32,64,128,256,512] \
    [--knee-threshold 0.01] \
    [--json]
```

**Defaults:**
- `--seqlen 512`, `--n-trials 30`, `--knee-threshold 0.01` (bpb delta between adjacent buckets).
- `--buckets` default: `1,2,4,8,16,32,64,128,256,512` — logarithmic, matches session-12 usage.
- `--val`: if omitted, use `provenance.val_data` discoverable from the checkpoint's sibling MRI (if any) or require the flag.

**Algorithm.**
1. Load model via `decepticons.loader.load_checkpoint(model_path)`. Assert it is a `CausalBankInference` — non-causal-bank models raise with a clean message.
2. For each of `n_trials` sequences of length `seqlen` drawn uniformly at random from val bytes, run a forward pass. Collect per-position next-byte cross-entropy (nats).
3. Convert to bpb (divide by `ln 2`).
4. Partition positions 0..seqlen-1 into the provided bucket list (left-inclusive, right-exclusive; final bucket right-bound = seqlen). Compute mean bpb per bucket.
5. **Knee rule:** walk adjacent buckets in order; the knee is the first index `i` such that `bpb[i] - bpb[i+1] < knee_threshold`. If no such index exists, knee is reported as `None` (saturated monotonically).
6. Report: bucket table (`bucket_min`, `bucket_max`, `n_positions`, `bpb_mean`, `bpb_sem`), `knee_bucket_min`, `knee_bucket_max`, `saturation_bpb` (last bucket mean).

**Output — text:**
```
effective-context for <model>
val_data: <path>
n_trials=30 seqlen=512

bucket     n      bpb
[1,2)      30   3.250
[2,4)      60   2.820
[4,8)     120   2.410
[8,16)    240   2.040
[16,32)   480   1.982
[32,64)   960   1.980 ← knee (Δ < 0.01)
[64,128) 1920   1.978
...

knee:  [32,64)       saturation: 1.78 bpb
```

**Output — `--json`:** structured dict
```json
{
  "model": "...",
  "val_data": "...",
  "n_trials": 30,
  "seqlen": 512,
  "knee_threshold": 0.01,
  "buckets": [
    {"min": 1, "max": 2, "n": 30, "bpb_mean": 3.25, "bpb_sem": 0.04},
    ...
  ],
  "knee_bucket_min": 32,
  "knee_bucket_max": 64,
  "saturation_bpb": 1.78
}
```

**Helper placement.** `_cb_effective_context(model_path, val, seqlen, n_trials, buckets, knee_threshold) -> dict` in `profile/compare.py`. The `dict` is the JSON output; a sibling `_fmt_effective_context(result) -> str` produces the text view.

**Subprocess isolation.** Model-needed → run as subprocess in MCP context, same mechanism as `profile-cb-causality`. Inline execution in CLI direct invocation.

---

## T2 — `profile-cb-ablations`

**Purpose.** Quantify per-path contribution to bpb: substrate-only, local-only, and rank-truncated substrate.

**CLI signature.**
```
heinrich profile-cb-ablations \
    --model <ckpt> \
    --ablate <substrate|local|truncate:K> \
    [--val <bytes.bin>] \
    [--n-tokens 50000] \
    [--json]
```

**Ablation modes.**
- **`substrate`:** Zero substrate contribution before readout. For adaptive-substrate models (`_last_features = concat(substrate_modes, x_embed)`), zero the substrate slice and keep `x_embed` tail. For non-adaptive models, zero `substrate_states` before the readout. Measures: how much does the substrate contribute?
- **`local`:** Monkey-patch `_local_logits` on the model instance to return zeros with the same shape. Measures: how much does the local path contribute?
- **`truncate:K`:** For the substrate path only, zero substrate modes at indices ≥ K before readout. Used to map the effective-rank curve. K must be an integer in `[0, n_modes]`.

**Algorithm.**
1. Load model. Assert causal-bank.
2. Run baseline pass on `n_tokens` of val data → `baseline_bpb`.
3. Apply the chosen ablation via a context manager that restores original state on exit.
4. Run ablated pass → `ablated_bpb`.
5. Report both + `delta_bpb = ablated - baseline`.

**Output — text:**
```
ablations for <model>
ablation: substrate
n_tokens: 50000

baseline:   1.780 bpb
ablated:    4.830 bpb
delta:     +3.050 bpb  (2.7×)
```

**Output — `--json`:**
```json
{
  "model": "...",
  "ablation": "substrate",
  "n_tokens": 50000,
  "baseline_bpb": 1.78,
  "ablated_bpb": 4.83,
  "delta_bpb": 3.05,
  "multiplier": 2.71
}
```

**Helper placement.** `_cb_ablations(model_path, ablate, val, n_tokens) -> dict` in `profile/compare.py`. Internally dispatches to `_ablate_substrate`, `_ablate_local`, `_ablate_truncate(k)`. Each ablation helper is a context manager using try/finally to guarantee restoration.

**Context-manager pattern.**
```python
@contextmanager
def _ablate_local(model):
    orig = model._local_logits
    model._local_logits = lambda *a, **k: torch.zeros_like(orig(*a, **k))
    try:
        yield
    finally:
        model._local_logits = orig
```

**Subprocess isolation.** Same as T1.

---

## T5 — `mri-recapture` path fallback

**Problem.** `provenance.model_path` is stored absolute. When the Sharts volume remounts at a different path (e.g., running on another host), recapture fails to resolve the source checkpoint.

**Fix.** In `_resolve_recapture_source` (or the equivalent path resolution function in `profile/mri.py`), add a second tier after the existing absolute-path check:

1. Try recorded absolute path (current behavior).
2. If missing: compute relative path from the recorded checkpoint path to a recognizable volume root (first `sharts` or `Sharts` directory component in the recorded path). Retry with `<MRI_ROOT>/<relative-suffix>` where `MRI_ROOT` is inferred from the MRI's own location (walking up until a `sharts`/`Sharts` directory is hit).
3. Byte-level val-data remap (existing behavior) still runs after the source checkpoint is resolved.

No new CLI flag. Silent correction, but logged at info level: `"model_path rewritten: <old> → <new>"`.

**Scope.** Single function. No behavior change when the absolute path resolves.

---

## T6 — `--svd-samples N` on `profile-cb-additivity`

**Problem.** `_cb_additivity_metrics` uses a hardcoded 5000-sample SVD for tail-PC position R² measurement. Noisy on large MRIs.

**Fix.** Add `--svd-samples N` (default `5000`) to the CLI parser. Plumb into `_cb_additivity_metrics(..., svd_samples: int = 5000)`. Internal SVD call uses `min(svd_samples, n_tokens)`. JSON output gains `svd_samples_used` field for audit.

**Scope.** ~10 lines across `cli.py` and `compare.py`.

---

## T7 — `--n-bootstrap K` on `profile-cb-pc-bands`

**Problem.** `partition_score` is deterministic but per-band `pos_r2` has sampling noise not reported.

**Fix.** Add `--n-bootstrap K` (default `0` = off). When `K > 0`:
1. Repeat the train/test split K times with different seeds.
2. For each band, collect K values of `pos_r2`.
3. Report `pos_r2_mean` and `pos_r2_sem` per band in both text and JSON output.

Text format gains a `± sem` suffix when bootstrap is on. JSON gains `pos_r2_sem`, `pos_r2_samples` fields. `partition_verdict` remains computed on the mean.

**Scope.** Bootstrap loop wraps existing SVD+regression inside the pos-r2 computation. ~30 lines.

---

## T8 — Silent partial-capture audit

**Problem.** `forward_captured` returns `None` for absent keys. Downstream tools calling `result.get("band_logits")` silently proceed with `None` instead of either asserting or emitting a clean skip.

**Audit procedure.**
1. Grep `profile/compare.py` for `.get("` patterns on `forward_captured`-style results (`mri_data.get`, `result.get`, similar).
2. For each call site, classify:
   - **Requires-non-None:** downstream math assumes the field. Replace with explicit `if X is None: raise ValueError(f"no <key> in MRI at {path} — requires recapture with fix_level ≥ <N>")`.
   - **Skip-valid:** field is optional and downstream handles None. Add explicit `if X is None: return {"skipped": True, "reason": "no <key>"}` or equivalent, so callers get a structured skip rather than silent junk numbers.
3. Produce a brief report at end of the audit: list of sites changed, category, and rationale.

**Scope.** Bounded to `profile/compare.py`. No sweeping changes to other modules in this pass. Separate PR if issues are found elsewhere.

---

## T10 — Regression test for T1 knee

**Purpose.** Anchor the session-12 finding that substrate-primary models have a 16-byte effective context ceiling.

**Test file.** `tests/test_cb_effective_context.py`

**Structure.**
```python
def test_substrate_primary_knee_at_or_below_16_bytes():
    ckpt = os.environ.get("HEINRICH_TEST_CKPT")
    if not ckpt or not Path(ckpt).exists():
        pytest.skip("set HEINRICH_TEST_CKPT to a substrate-primary checkpoint")
    val = os.environ.get("HEINRICH_TEST_VAL")  # required alongside ckpt
    if not val:
        pytest.skip("set HEINRICH_TEST_VAL")
    result = _cb_effective_context(
        model_path=ckpt,
        val=val,
        seqlen=128,
        n_trials=5,
        buckets=[1, 2, 4, 8, 16, 32, 64, 128],
        knee_threshold=0.01,
    )
    assert result["knee_bucket_max"] is not None, "no knee detected"
    assert result["knee_bucket_max"] <= 32, (
        f"substrate-primary knee expected ≤ 32; got {result['knee_bucket_max']}"
    )
```

**Rationale for the 32-byte bound.** The session-12 finding is "knee at 16 bytes". A passing test bounds the knee ≤ 32 to leave one bucket of tolerance for noise. If the knee shifts dramatically upward (e.g., a new mechanism escapes the ceiling), this test fails loudly and we know a session-12 finding has been invalidated.

**Fast test parameters.** `seqlen=128, n_trials=5` keeps CPU cost low for pre-commit runs.

---

## MCP server wiring

Both new tools get entries in `mcp.py`:

```python
"heinrich_cb_effective_context": {
    "description": "Context-knee test: per-position bpb on random-prefix sequences. Identifies the effective context length.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "model": {"type": "string", "description": "Path to .checkpoint.pt"},
            "val": {"type": "string"},
            "seqlen": {"type": "integer", "default": 512},
            "n_trials": {"type": "integer", "default": 30},
            "buckets": {"type": "array", "items": {"type": "integer"}},
            "knee_threshold": {"type": "number", "default": 0.01},
        },
        "required": ["model"],
    },
},
"heinrich_cb_ablations": {
    "description": "Ablation forensics: substrate/local/truncate path contributions to bpb.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "model": {"type": "string"},
            "ablate": {"type": "string", "description": "substrate | local | truncate:K"},
            "val": {"type": "string"},
            "n_tokens": {"type": "integer", "default": 50000},
        },
        "required": ["model", "ablate"],
    },
},
```

Dispatch branches in the tool handler run the helpers as subprocesses with 10-hour timeout, matching `profile-cb-causality`. Stdout JSON is captured and returned as the tool result.

---

## Out of scope

- T3 (session-11 book retraction): documentation, not tooling.
- T4 (architectural_family metadata): changes capture contract — separate PR.
- T9 (chronohorn SHA): low value without T4.
- CK1/CK2 (corrupt checkpoints): chronohorn-side.
- H1/H2/H3 (library hygiene): operational, not tooling.
- Chronohorn-side asks (C1-C6).

---

## Build order (informs implementation plan)

1. T1 helper + CLI + MCP (load-bearing).
2. T10 regression test (pins T1 behavior).
3. T2 helper + CLI + MCP.
4. T6 + T7 flags (small).
5. T5 path fallback.
6. T8 audit pass.

T1 first because T10 depends on it and because it's the most-requested session-12 diagnostic. T2 second because it shares subprocess-isolation plumbing. T5/T6/T7 are small. T8 audit runs last so changes it suggests don't collide with new helpers.
