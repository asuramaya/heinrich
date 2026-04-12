# Causal Bank Tools — Missing Features

Session 7 (April 11-12, 2026) ran 12+ MRI captures and ~40 ad-hoc analysis scripts on causal bank checkpoints. Three tools were built (`cb-manifold`, `cb-compare`, `cb-health`). This document lists everything that's still missing.

## Built This Session

| Tool | CLI | MCP | What it does |
|---|---|---|---|
| `causal_bank_manifold` | `profile-cb-manifold` | `heinrich_cb_manifold` | PCA, effective dim, band loadings, readout alignment, routing, gates, SSM |
| `causal_bank_compare` | `profile-cb-compare` | `heinrich_cb_compare` | CKA, displacement correlation, router cosine between two MRIs |
| `causal_bank_health` | `profile-cb-health` | `heinrich_cb_health` | Shape/NaN/consistency validation |

These work on MRI impulse data (single-token, zero state). They don't load models or run sequences.

## Missing: Sequence-Level Tools

These require loading the model via decepticons and running `forward()` or `forward_captured()` on validation sequences. They're the ones that produced every key finding this session.

### 1. `cb-loss` — Per-position loss decomposition

**Used 5 times.** Produced the three-wall decomposition (62% vocab / 35% architecture / 4% saturation).

```
heinrich profile-cb-loss --model <checkpoint> [--n-seqs 200] [--seq-len 512]
```

Computes:
- Overall bpb, non-raw bpb
- Loss by position range (0-4, 4-64, 64-256, 256-511)
- Loss by BPE merge rank (raw bytes, early/mid/late/final merges)
- Loss by token frequency quartile
- Loss after punctuation vs non-punctuation
- Raw byte tax (contribution of byte-fallback tokens to overall bpb)
- Architectural ceiling estimate (best-window non-raw loss)
- Loss autocorrelation at lags 1-128 (surprise bank viability)

Returns dict with all above. The position × merge-rank cross-tabulation is the key output — it separates tokenizer effects from architecture effects.

Implementation: load model via `load_checkpoint()`, forward on val data (from `fineweb_val_000000.bin`), stable log-softmax, per-position cross-entropy. ~60 lines. The val data path should be configurable or auto-detected from the checkpoint's result JSON.

### 2. `cb-routing` — Sequence-level expert routing

**Used 4 times.** Showed leader routing collapse (100% one expert) and banded routing health (31-41% switch rate).

```
heinrich profile-cb-routing --model <checkpoint> [--n-seqs 50]
```

Computes:
- Per-band expert distribution (E0%, E1%, ...) overall and by position range
- Switch rate per band (% of consecutive positions where the winner changes)
- Position-dependent routing (early vs mid vs late)
- Routing margin (mean |E0_weight - E1_weight|)
- Per-sequence routing entropy

Implementation: hook into each `RoutedSquaredReLUReadout` in `_band_readouts` (or the single `linear_readout`) via `register_forward_hook`, capture `_last_route` attribute. Reshape from `[B*seq, experts]` to `[B, seq, experts]`. Forward on val data. ~80 lines.

The MRI routing data is useless for multi-expert models (zero-state artifact collapses to one expert). This tool is the only way to measure routing behavior.

### 3. `cb-temporal` — Temporal attention forensics

**Used 3 times.** Showed attention peak at pos 32-64, cold start compensation, the chain verification (embed→substrate: r=0.99, embed→attention: r=0.06).

```
heinrich profile-cb-temporal --model <checkpoint> [--n-seqs 100]
```

Computes:
- Temporal attention output L2 by position range
- Per-token mean temporal attention magnitude
- Correlation chain: embed_norm → substrate_displacement → temporal_attention
- Attention profile comparison with snapshot interval
- Attention weight distribution over bank snapshots (which snapshots get weight?)
- Causality verification: does the attention mask future snapshots?

Implementation: hook `_temporal_attention` via `register_forward_hook`, capture output tensor. For attention weights: need to modify `TemporalAttention.forward()` to store `_last_attn_weights` attribute (currently doesn't). Forward via `model._model(batch)` (not `forward_captured()` which bypasses `_linear_logits()`). ~100 lines.

**Critical finding this tool would catch:** the causality violation. The current `TemporalAttention` has no causal mask — position t attends to snapshots from positions > t, leaking future information. A causality check should be built into this tool.

### 4. `cb-modes` — Mode utilization in sequences

**Used 2 times.** Showed slow modes ramp 18x, fast modes flat, zero dead modes, readout ignores slow modes 5x.

```
heinrich profile-cb-modes --model <checkpoint> [--n-seqs 50]
```

Computes:
- Mean absolute substrate activation by half-life quartile × position range
- Ramp ratio (late/early activation) per band
- Dead mode detection (max activation < 1% of mean)
- Most position-varying modes (top 5 by std across positions)
- Substrate L2 growth curve (L2 norm at each position range)

Implementation: `forward_captured()` on val data, extract `substrate_states` tensor. Group modes by half-life quartile. Compute statistics. ~50 lines.

### 5. `cb-decompose` — Manifold decomposition

**Used once but produced THE key finding:** 43% clock, 47% ghosts, 6% useful content. 90% of the manifold is waste.

```
heinrich profile-cb-decompose --model <checkpoint> [--n-seqs 50]
```

Computes:
- PCA on sequence-level substrate states (not MRI impulse)
- Position correlation per PC (which PCs encode position vs content)
- Position variance regression (polynomial regression of position from PCs)
- Content R² (how much of content PCs predict loss)
- Linear probe accuracy (content PCs → next token)
- Readout visibility per PC (project substrate PCs onto readout weight subspace)
- Ghost fraction (% of variance in PCs that are neither position nor loss-predictive)

Implementation: `forward_captured()` for substrate, `forward()` for loss. PCA via `np.linalg.svd`. Position regression via `np.linalg.lstsq`. Linear probe via `lstsq` on one-hot targets. Readout alignment via SVD of expert weights. ~120 lines. The most complex tool but the most informative.

### 6. `cb-substrate-local` — Substrate vs local path balance

**Used 3 times.** Showed local dominates cold start (0.8x), substrate overwhelms at late positions (33.8x).

```
heinrich profile-cb-substrate-local --model <checkpoint> [--n-seqs 30]
```

Computes:
- Substrate state L2 by position range
- Local logit L2 (× local_scale) by position range
- Per-band logit L2 by position range
- Substrate/local ratio by position range
- Crossover point (position where substrate > local)
- Total substrate logit L2 vs local logit L2

Implementation: `forward_captured()`, extract `substrate_states`, `local_logits`, `band_logits`. Compute norms. ~40 lines. Could merge into `cb-modes` since both use `forward_captured()`.

## Missing: Tokenizer Tools

### 7. `tokenizer-compare` — Multi-tokenizer comparison

**Used 3 times.** Showed sp8192 sweet spot, byte fallback doesn't decrease, diminishing returns curve.

```
heinrich profile-tokenizer-compare --tokenizers sp1024.model sp8192.model [--text <val_data>]
```

Computes:
- Vocab size, bytes/token, tokens/byte, byte fallback %
- Token length distribution (1-byte, 2-byte, ... 16-byte)
- Compression ratio
- Vocab overlap between tokenizers
- Pieces unique to each tokenizer (sorted by byte length)
- Sample encodings on representative text types
- Parameter budget impact (embedding + readout cost at various embed_dim)
- Marginal gain curve (bytes/token gained per 1000 vocab slots)

Implementation: load SentencePiece models, encode same validation text with each, compute statistics. ~80 lines. No model needed — pure tokenizer analysis.

### 8. `tokenizer-difficulty` — Per-token difficulty from embeddings

**Discovered this session:** embedding norm correlates with prediction difficulty at r=0.64. The difficulty map is IN the embedding, not in a separate forward pass.

```
heinrich profile-tokenizer-difficulty --mri <mri_path>
```

Computes:
- Embedding norm per token (instant — just read the weight matrix)
- Substrate displacement per token (from MRI impulse)
- Correlation: embed_norm → displacement
- Difficulty quartiles: which tokens are easy (small norm) vs hard (large norm)
- Embedding PCA: effective dim, how many dimensions the embedding uses
- Near-duplicate detection (cosine > 0.9)
- Embedding entropy per token

Implementation: read `embedding.npy` and `substrate.npy` from MRI. Compute norms and correlations. ~40 lines. No model needed — reads MRI arrays.

## Missing: Cross-Architecture Tools

### 9. `cb-vs-transformer` — CKA between causal bank and transformer MRIs

**Used once.** Showed CKA ≈ 0 between causal bank substrate and Qwen 0.5B at every layer.

```
heinrich profile-cb-vs-transformer --cb-mri <cb_path> --transformer-mri <transformer_path>
```

Computes:
- CKA between causal bank substrate and transformer exit states at each layer
- CKA between embeddings
- CKA between causal bank final logits and transformer final layer
- Displacement correlation (same token IDs, different architectures)

Implementation: load both MRIs via `load_mri()`. Subsample to shared token count. CKA computation. ~40 lines. Note: different tokenizers (sp1024 vs model-native) make this a geometry comparison, not token-level alignment. Should warn in output.

## Missing: Decepticon Loader Improvements

### 10. Loader issues found this session

| Issue | Workaround applied | Proper fix needed |
|---|---|---|
| Asymmetric band experts (asym8420) | `_infer_band_experts()` from state dict | Serialize `band_experts` in result JSON |
| Static band detection (`nn.Linear`) | Check for `.weight`-only keys | Same — serialize in JSON |
| `TiedRecursiveReadout` not detected | `_infer_band_readout_kind()` from state dict | Serialize readout kind per band |
| Mixed readout types (MLP bands + routed main) | `strict=False` load | Separate config for band vs main readout |
| Substrate transforms not in config | `_transform_flags` inference from state dict | Serialize transform flags in result JSON |
| Kernel path numerical instability on CPU | No workaround — broken for leader and density16 | Need scan-path fallback when kernel blows up |
| `forward_captured()` bypasses `_linear_logits()` | Hook via `model._model(batch)` instead | Add temporal attention output to `forward_captured()` return dict |
| `TemporalAttention` has no `_last_attn_weights` | Can't inspect attention distribution | Store `_last_attn_weights` in forward |

### 11. `forward_captured()` missing returns

The loader's `forward_captured()` returns: `logits`, `substrate_states`, `embedding`, `route_weights`, `band_logits`, `local_logits`, `sticky_write_strength`.

Missing:
- `temporal_attention_output` — the attended context vector
- `temporal_attention_weights` — the attention distribution over bank snapshots
- `overwrite_gate_values` — per-mode gate activation (when gate fires)
- `mode_selector_mask` — per-token mode selection weights
- `magnitude_before_norm` — substrate L2 norm before normalization

These are needed for the sequence-level tools above. Currently require manual hooks which is fragile and breaks when the model code changes.

## Missing: MRI Capture Improvements

### 12. Sequence-mode MRI for causal banks

Current MRI captures single-token impulse responses (raw mode). This misses:
- Routing dynamics (collapses to one expert in impulse)
- Substrate accumulation (no growth over positions)
- Temporal attention behavior (no snapshot bank in single-token mode)
- Mode utilization by position (no position dependence)

A sequence-mode MRI would capture:
```
substrate_trajectory.npy    [n_seqs, seq_len, n_modes]   — substrate at every position
routing_trajectory.npy      [n_seqs, seq_len, n_experts]  — per-band routing per position
temporal_bank.npy           [n_seqs, n_snapshots, n_modes] — snapshot bank contents
temporal_weights.npy        [n_seqs, seq_len, n_snapshots] — attention distribution
loss_per_position.npy       [n_seqs, seq_len]              — cross-entropy at each position
```

This is ~100MB per model at 50 sequences × 512 positions. Storage isn't the issue — the issue is that it requires a forward pass per sequence (not per token), so it takes ~30 seconds instead of 2 seconds.

The sequence-mode MRI would make tools 1-6 above work from stored data without loading the model. Currently each tool loads the model fresh and runs its own forward pass, which is slow and wasteful.

## Missing: Validation Infrastructure

### 13. Causality verification

The temporal attention causality violation was found by reading the source code, not by any tool. A `cb-causality-check` that verifies causal masking in all attention-like mechanisms would have caught this immediately.

```
heinrich profile-cb-causality --model <checkpoint>
```

Test: run two forward passes on the same sequence. Pass 1: full sequence. Pass 2: truncate to position t, pad the rest. If `logits[t]` differs between the two passes, information from positions > t leaked into the computation at position t. Run for multiple t values and report any discrepancies.

This is the finite-difference test for causality. It's slow (2× forward passes per test position) but definitive.

### 14. Reproducibility verification

```
heinrich profile-cb-reproduce --model <checkpoint>
```

Run the model twice on the same input. Compare logits. They should be bitwise identical (inference mode, no dropout). If they differ, something is nondeterministic. This catches bugs in the kernel path, scan path, or attention computation.

## Priority Order

For the next session:

1. **Fix the causality violation** in `TemporalAttention` (add causal mask). Retrain density16 and exp7 with the fix. This determines whether temporal attention is real or an artifact.

2. **`cb-loss`** — the most-used tool, produces the wall decomposition. Build it first.

3. **`cb-routing`** — second most-used, required to verify routing collapse vs health.

4. **`cb-temporal`** — third most-used, required for the chain verification and attention profiling. Needs `_last_attn_weights` in the TemporalAttention module.

5. **Extend `forward_captured()`** to return temporal attention and gate outputs. This unblocks tools 3, 4, and 5 without manual hooks.

6. **Sequence-mode MRI** — the long-term fix. Capture once, analyze many times. Eliminates the need to load models for every analysis.

7. **`tokenizer-compare`** and **`tokenizer-difficulty`** — needed for the sp8192 transition. Lower priority since the tokenizer decision is made (standard BPE, not difficulty-aware).

8. **`cb-causality`** — the finite-difference causality test. Should run automatically on any model with temporal attention or other non-standard mechanisms. Would have caught the bug before 4 hours of fleet training.
