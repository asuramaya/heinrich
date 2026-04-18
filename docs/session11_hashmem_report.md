# Session 11: Hashmem Variants — mut-hashmem, mut-hashmem-orthoreg

**For chronohorn.** 8 trajectory MRIs across mut-hashmem and mut-hashmem-orthoreg × steps 5k/10k/15k/20k.

## Loader fix

Hashmem checkpoints add 6 keys the previous loader didn't know how to construct:
```
_hash_memory.write_proj.weight / .bias    [64, 256] / [64]
_hash_memory.read_query.weight / .bias    [64, 256] / [64]
_hash_memory.read_out.weight / .bias      [256, 64] / [256]
```

Fixed in `decepticons/src/decepticons/loader.py`:
- Added `"_hash_memory.": "hash_memory"` to `_transform_flags` so the config flag flips on when these keys are present.
- Inferred `hash_memory_dim` from `_hash_memory.write_proj.weight.shape[0]` (64 in these checkpoints).

## Headline: hash_memory never trained

Hash memory weights are **bit-identical across all 4 trajectory checkpoints** (step 5k → step 20k):

```
_hash_memory.read_out.weight:    step5k → step20k: max |Δ| = 0.000000
_hash_memory.read_out.bias:      step5k → step20k: max |Δ| = 0.000000
_hash_memory.read_query.weight:  step5k → step20k: max |Δ| = 0.000000
_hash_memory.read_query.bias:    step5k → step20k: max |Δ| = 0.000000
_hash_memory.write_proj.weight:  step5k → step20k: max |Δ| = 0.000000
_hash_memory.write_proj.bias:    step5k → step20k: max |Δ| = 0.000000
```

**20 000 training steps produced zero gradient on any hash_memory parameter.** Root cause in `decepticons/models/causal_bank_torch.py:1770-1776` — `_linear_logits` for `_use_adaptive_substrate=True` returns before the line 1794 hash-memory invocation. Every forward pass for this model family bypasses the hash-memory path.

Same architectural failure as `localon` earlier this session: a feature was enabled in config, consumed 49k parameters, 1863 s of wall-clock training, and produced zero effect on the model.

## Why the rest of the model drifted anyway

mut-hashmem weights differ from mut-control-mlp by up to **1.43 L∞** on `_adaptive_out_proj.weight` (vs 0.003-0.015 for localon/orthoreg no-ops). This looked suspicious until the init signatures explained it:

| variant | init sha256 (prefix) | tensor_count | init_sum |
|---------|----------------------|--------------|----------|
| mut-control-mlp | 05c442f9c6f3 | 20 | 8518.45 |
| mut-localon | 05c442f9c6f3 | 20 | 8518.45 |
| mut-orthoreg-0p1 | 05c442f9c6f3 | 20 | 8518.45 |
| **mut-hashmem** | **bfe93c7ca9e6** | **26** | **8510.66** |
| **mut-hashmem-orthoreg** | **bfe93c7ca9e6** | **26** | **8510.66** |

The 6 extra hash-memory tensors consumed RNG draws at construction time, shifting every subsequent init call even with the same `init_seed=42`. So mut-hashmem started from a different point than mut-control-mlp and drifted differently. But since hash_memory itself was frozen throughout training, this is architecturally a "same model, different init" comparison.

## Measured bpb trajectory

| step | control-mlp | hashmem | hashmem-orthoreg |
|------|-------------|---------|------------------|
| 5k | 2.1775 | 2.1830 | 2.1830 |
| 10k | 2.1175 | 2.1236 | 2.1236 |
| 15k | 2.0677 | 2.0751 | 2.0751 |
| 20k | **2.0513** | **2.0555** | **2.0555** |

Δ vs control at step 20k: +0.0042 bpb (0.2%). Within the 0.3% seed-spread measured across baseline seed42/43/44 earlier this session. Consistent with "same model, different init" interpretation.

## Geometry

| step | control EffDim | hashmem EffDim | control ConR² | hashmem ConR² |
|------|----------------|----------------|---------------|---------------|
| 5k | 120.3 | 120.7 | 0.3575 | 0.3487 |
| 10k | 142.3 | 142.5 | 0.3521 | 0.3463 |
| 15k | 149.2 | 149.2 | 0.3582 | 0.3501 |
| 20k | **150.4** | **150.5** | **0.3591** | **0.3481** |

EffDim identical across the trajectory. ConR² runs ~0.01 lower in hashmem — a real, measurable gap that is NOT explained by init noise alone. The hashmem substrate is slightly less content-predictive at every step. Candidate explanations:
- Different init lands in a slightly less-content-aligned basin
- Optimizer state (Adam moments) spread across an extra 49k dead parameters, subtly affecting the training dynamics on the real parameters

## Orthoreg: the fifth and sixth null results

This session now has four independent orthoreg pairs with identical controls:

| with-orthoreg | baseline | Δ bpb step 20k |
|---------------|----------|----------------|
| mut-orthoreg-0p1 | mut-control-mlp | +0.0001 |
| mut-orthoreg-1p0 | mut-control-mlp | +0.0000 |
| mut-patch4-orthoreg | mut-patch4 | +0.0002 |
| **mut-hashmem-orthoreg** | **mut-hashmem** | **+0.0000** |

Max |Δ| across 4 pairs: **0.0002 bpb**. Orthogonal regularization produces zero measurable behavior change across every architecture tested (routed, MLP, patched, hashed). **Orthoreg is dead at s8 byte-hrr. Remove from codebase.**

## Running tally: chronohorn mutations that do nothing

| mutation | status | evidence |
|----------|--------|----------|
| local_scale = 0.25 (localon) | no-op | adaptive_substrate `_forward_raw` bypasses local path |
| hash_memory = True | no-op | `_linear_logits` adaptive_substrate returns before hash memory |
| orthoreg coeff 0.1 | no-op | Δbpb = 0.0001 vs control, 4 pairs |
| orthoreg coeff 1.0 | no-op | Δbpb = 0.0000 vs control, 4 pairs |

Four of the seven mutations tested this session have **zero measurable effect** on bpb, EffDim, or ConR². mut-persistent is the only variant in the set that reaches a genuinely different basin with measurable geometric differences.

## Actionable for chronohorn

1. **Add an architectural preflight to the trainer.** For each enabled feature (hash_memory, local_scale > 0, temporal_attention, mode_selector, etc.), walk the forward path at startup and confirm the feature is reachable given the rest of the config. Log a WARN when a feature is enabled but the dispatch skips it. This would have caught localon and hashmem before committing 1-2h of training each.

2. **Flag init-signature divergence as a chronohorn metric.** When introducing a new module, the expected behavior is "isolates to its own params" but in practice added modules perturb the entire init via RNG consumption. Log `init_signature.sha256` and flag pairs of runs sharing `init_seed` but differing in signature — they are NOT directly comparable even with the same seed.

3. **Remove orthoreg from the codebase.** 4 pairs, max |Δ| = 0.0002 bpb. It's not a knob that does anything on this architecture family.

4. **Hash memory needs a different hookup** if it's supposed to help. The current implementation only runs through the non-adaptive `_linear_logits` path. If hash_memory is intended to work alongside adaptive_substrate, add it to the adaptive path at line 1776 (before or after linear_readout).

## Files touched this message

- `decepticons/src/decepticons/loader.py` — added `_hash_memory.` to transform-flag detection, inferred `hash_memory_dim` from state_dict

## Session 11 heinrich/decepticon fixes so far

| area | what | file |
|------|------|------|
| byte shard loader | TOKEN_SHARD_MAGIC header + uint16 payload | `heinrich/backend/decepticon.py` |
| cb-loss warning | bpb > log2(vocab) | `heinrich/profile/compare.py` + cli |
| router capture | softmax of `readout.router(features)` | `decepticons/loader.py` |
| routing allocation | skip if no route_weights | `heinrich/profile/mri.py` |
| cb-routing zero-guard | reject all-zero routing tensors | `heinrich/profile/compare.py` |
| patch4 inference | fat-readout patch_n detection + 2D collapse | `decepticons/loader.py` |
| hashmem inference | `_hash_memory.` → `hash_memory=True`, infer dim | `decepticons/loader.py` |
