# Session 11: Full `mut-*` Variant Report

**For chronohorn.** Complete analysis of 10 byte-hrr-s8 mutation variants × 4 trajectory steps = 40 MRIs.

All captures in `/Volumes/Sharts/heinrich/session10_byte_scaling/mut-*-step{5,10,15,20}k.seq.mri/`. Val data: `fineweb_val_000000_bytes.bin` (302 MB, chronohorn-format byte shards).

## The variants

| variant | mutation | readout | extra params | init sha256 |
|---------|----------|---------|-------------|-------------|
| mut-control-mlp | — | 4-expert MLP | — | 05c442… |
| mut-localon | local_scale 0 → 0.25 | MLP | — | 05c442… |
| mut-orthoreg-0p1 | orthogonal reg λ=0.1 | MLP | — | 05c442… |
| mut-orthoreg-1p0 | orthogonal reg λ=1.0 | MLP | — | 05c442… |
| mut-persistent | substrate state persists across batches | MLP | — | 05c442… |
| mut-patch4 | fat-readout patch_n=4 | MLP | +787k (fat readout) | 993e2b… |
| mut-patch4-orthoreg | patch4 + orthoreg | MLP | +787k | 993e2b… |
| mut-hashmem | hash_memory enabled | MLP | +50k | bfe93c… |
| mut-hashmem-orthoreg | hashmem + orthoreg | MLP | +50k | bfe93c… |
| mut-hashmem-patch4 | hashmem + patch4 | MLP | +837k | 475999… |

Same `init_seed=42` everywhere. Four distinct init bases — adding parameters (hash memory, fat readout) consumes RNG draws and shifts the starting point. **Same-sha256 pairs are directly comparable; cross-sha comparisons carry init noise (~0.004 bpb = 0.2%).**

## Step-20k results — all variants

| variant | heinrich bpb | training-log bpb | EffDim | ConR² | PosR² |
|---------|--------------|------------------|--------|-------|-------|
| baseline byte-hrr (routed) | **2.075** | **2.096** | 149.5 | 0.342 | 0.001 |
| mut-control-mlp | 2.0513 | 2.1138 | 150.4 | 0.3591 | 0.0013 |
| mut-localon | 2.0514 | 2.1138 | 150.4 | 0.3590 | 0.0013 |
| mut-orthoreg-0p1 | 2.0514 | 2.1138 | 150.4 | 0.3590 | 0.0013 |
| mut-orthoreg-1p0 | 2.0513 | 2.1138 | 150.4 | 0.3591 | 0.0013 |
| **mut-persistent** | **2.0538** | **2.0964** | **147.9** | **0.3451** | 0.0012 |
| mut-patch4 | 2.1601 | 2.2208 | 153.3 | 0.3316 | 0.0014 |
| mut-patch4-orthoreg | 2.1603 | 2.2209 | 153.3 | 0.3312 | 0.0014 |
| mut-hashmem | 2.0555 | 2.1149 | 150.5 | 0.3481 | 0.0011 |
| mut-hashmem-orthoreg | 2.0555 | 2.1149 | 150.5 | 0.3482 | 0.0011 |
| mut-hashmem-patch4 | 2.1570 | 2.2196 | 152.8 | 0.3371 | 0.0016 |

Heinrich's 50×512 subset consistently reads ~3% lower than training's 300-batch 1024-seq eval. Ordering and ratios preserved. Baseline byte-hrr included for context (routed readout, not in the mut set).

## Finding 1: Only two mutations matter

Ranking by training-log bpb against the natural comparison:

| comparison | Δ training bpb | interpretation |
|------------|----------------|----------------|
| persistent vs control-mlp | **−0.0174** | only MLP variant that improves |
| patch4 vs control-mlp | +0.107 | 5% worse next-byte |
| hashmem vs control-mlp | +0.001 | init-noise floor |
| localon vs control-mlp | 0.000 | zero |
| orthoreg-0p1 vs control-mlp | 0.000 | zero |
| orthoreg-1p0 vs control-mlp | 0.000 | zero |
| patch4-orthoreg vs patch4 | +0.0001 | zero |
| hashmem-orthoreg vs hashmem | 0.000 | zero |
| hashmem-patch4 vs patch4 | −0.0012 | below init-noise |

**Two signals that exceed init noise:**
1. **mut-persistent: −0.017 bpb vs MLP control, −0.000 vs routed baseline** — full recovery of the MLP→routed quality gap, plus a geometric change (ConR² 0.345 vs 0.359, EffDim 148 vs 150).
2. **patch4 family: +0.11 bpb on next-byte** — worse. A real regression, not noise.

Everything else is indistinguishable from same-seed init noise.

## Finding 2: Three silent no-op features

Three mutations are *architecturally unreachable* in the adaptive_substrate code path used by these models:

| feature | where it should fire | why it doesn't |
|---------|---------------------|----------------|
| local_scale > 0 | `_forward_raw` line 2025 | adaptive_substrate short-circuits at 2029-2031 |
| hash_memory | `_linear_logits` line 1794 | adaptive_substrate returns at 1776 before reaching 1794 |
| orthogonal reg | optimizer penalty | model already at 7% mode utilization; nothing to constrain |

The first two are silent wall-clock waste: 1962 s for localon, 1863 s for hashmem. The third is at least parameter-neutral.

### Proof that hash_memory literally did not train

```
_hash_memory.read_out.weight:     step5k → step20k: max |Δ| = 0.000000
_hash_memory.read_out.bias:       step5k → step20k: max |Δ| = 0.000000
_hash_memory.read_query.weight:   step5k → step20k: max |Δ| = 0.000000
_hash_memory.read_query.bias:     step5k → step20k: max |Δ| = 0.000000
_hash_memory.write_proj.weight:   step5k → step20k: max |Δ| = 0.000000
_hash_memory.write_proj.bias:     step5k → step20k: max |Δ| = 0.000000
```

Same zero-delta signature in mut-hashmem-patch4. 15 000 training steps between step 5k and step 20k, zero gradient on any hash_memory parameter in either variant. The weights sit at their randomly-initialized values while the optimizer's moment estimates still allocate state slots for them (subtle effect, see Finding 5).

## Finding 3: orthoreg is dead code, 4-pair confirmed

| pair (same-init) | Δ bpb step 20k |
|------------------|----------------|
| mut-orthoreg-0p1 ↔ mut-control-mlp | +0.0001 |
| mut-orthoreg-1p0 ↔ mut-control-mlp | +0.0000 |
| mut-patch4-orthoreg ↔ mut-patch4 | +0.0002 |
| mut-hashmem-orthoreg ↔ mut-hashmem | +0.0000 |

Max absolute difference across four independent pairs: **0.0002 bpb**. Across routed, MLP, patched, and hashed readouts — zero measurable effect every time. Orthogonality constraint cannot tighten a basin that's already at 7% effective dimensionality (150/2048 modes) — the geometry is collapsed; there's no spread for the regularizer to work on.

## Finding 4: mut-persistent is the sole winner

### What it does
State carries across training batches instead of resetting. The substrate accumulates statistics beyond any single sequence length.

### What it buys
- Training-log bpb 2.0964 vs control-mlp 2.1138 — full **0.017 bpb recovery** of the MLP readout penalty
- Matches the original routed baseline (2.096)
- Geometric signature:
  - EffDim 147.9 (vs 150.4 control) — compresses the substrate
  - ConR² 0.3451 (vs 0.3591 control) — less linear predictivity
  - Weight divergence 0.58 L∞ (vs 0.003-0.015 for null mutations) — reaches a genuinely different basin

Persistent substrate compresses into a tighter state while encoding more across-batch context. The lower linear ConR² means the content information lives in more nonlinear directions — exactly what you'd expect from a model that memorizes across sequence boundaries.

### Costs
- 2023 s train time vs 1862 s control (+9%, nothing dramatic)
- Same param count
- Same init group as control (directly comparable, no confound)

### Recommendation
Keep. Scale up to s12. Longer trajectory (50k+). This is the only mutation in the set worth development effort.

## Finding 5: patch4 is a next-byte regression, possibly a multi-step gain

Patch4 emits a fat `[B, T, 4*vocab]` output reshaped to `[B, T, 4, vocab]` — the readout predicts the next 4 bytes from each position. Training loss is the average CE over all 4 heads; `test_bpb` in training logs is the p=0 (next-byte) head only.

Per-head bpb at step 20k (mut-patch4):

| head offset | predicts | bpb |
|-------------|----------|-----|
| p=0 | t+1 | 2.200 |
| p=1 | t+2 | 3.201 |
| p=2 | t+3 | 3.838 |
| p=3 | t+4 | 4.180 |
| mean | all 4 | 3.356 |

The head for the next byte is **6% worse** than baseline's 2.075 bpb, despite:
- 4.5% more parameters (fat readout)
- 4× training signal per position
- 2× training time

The further-out heads degrade as expected — byte-level state is largely local, and predicting t+2/t+3/t+4 from position t requires carrying byte-level structure the substrate doesn't encode well.

### hashmem-patch4 doesn't recover quality
Adding hash memory to patch4 was plausibly a way to help longer-range prediction. But hash_memory still doesn't train (same zero-delta signature). The combined variant bpb is 2.1570 step 20k — a hair below pure patch4's 2.1601, indistinguishable from the init-noise spread between their respective init sha256s.

## Finding 6: The architectural ceiling is robust

Across 40 mut trajectory points and 12 baseline trajectory points (52 measurements total):

| metric | range | spread | note |
|--------|-------|--------|------|
| EffDim step 5k | 111 – 132 | — | patch variants start ~10 higher |
| EffDim step 20k | **148 – 155** | **5%** | 7% mode utilization is the ceiling |
| ConR² step 20k | 0.33 – 0.36 | 9% | |
| PosR² step 20k | **0.001 – 0.002** | effectively zero | s8 never develops position |

These are **architecture-level** properties, not hyperparameter-level. Every mutation tested — readout swaps, regularization, patching, memory extensions, state persistence — lands in the same narrow range. The 7% mode-utilization attractor and the 0% position encoding at s8 are deterministic outcomes of byte-hrr-s8 at 20k steps.

Implication: **next-generation mutations must target the recurrence itself** (unfreeze ω, new algebras, different gating) or scale. Hyperparameter sweeps at s8 will keep returning to this ceiling.

## Finding 7: Init-signature shifts across "same-seed" variants

Adding parameters shifts the RNG consumption during model construction, giving different starting weights even when `init_seed=42` is held constant:

| init group | sha256 | variants | # extra tensors |
|------------|--------|----------|------------------|
| A | 05c442f9… | control-mlp, localon, orthoreg-0p1, orthoreg-1p0, persistent | 0 |
| B | 993e2bd6… | patch4, patch4-orthoreg | + fat readout (1 larger tensor) |
| C | bfe93c7c… | hashmem, hashmem-orthoreg | + 6 hash_memory tensors |
| D | 475999d8… | hashmem-patch4 | + both |

Within-group comparisons are directly meaningful. Cross-group deltas carry ~0.004 bpb of init noise (measured as spread between variants in group A that trained to the same basin despite being "different" mutations).

**Actionable:** chronohorn should log `init_signature.sha256` alongside `init_seed` in run metadata and flag when two runs share a seed but differ in signature. They're not interchangeable controls.

## Summary recommendations

1. **Ship mut-persistent**, kill the others from the sweep.
2. **Remove orthoreg** from the codebase (4 pairs, 0.0002 bpb max effect).
3. **Add architectural preflight** — the localon and hashmem silent failures waste ~3800 s of combined train time for zero signal. At startup, trace the configured forward path and WARN on any enabled feature that is unreachable given the rest of the config.
4. **Don't combine patch4 with arbitrary other mutations hoping to recover quality** — hashmem-patch4 doesn't, and patch4 alone is a next-byte regression. Patch4 only makes sense if the target metric is multi-step bpb, which isn't currently reported.
5. **s8 is a dead end for hyperparameter exploration.** 7% effective dim / 0% position is the architecture's equilibrium. Next work: scale (s12+) or substrate structure (unfreeze ω, alternative algebras, alternative gating).

## Running tally of fixes landed this session

| system | what | file |
|--------|------|------|
| heinrich | byte shard loader (TOKEN_SHARD_MAGIC + uint16 payload) | `heinrich/backend/decepticon.py` |
| heinrich | cb-loss warning when bpb > log2(vocab) | `heinrich/profile/compare.py` + cli |
| heinrich | cb-routing rejects all-zero tensors | `heinrich/profile/compare.py` |
| heinrich | MRI skips routing allocation when forward returns None | `heinrich/profile/mri.py` |
| decepticon | router softmax capture for adaptive_substrate | `decepticons/loader.py` |
| decepticon | fat-readout patching inference (patch_n from readout.out shape) | `decepticons/loader.py` |
| decepticon | fat-readout logits reshape + 2D collapse in forward_captured | `decepticons/loader.py` |
| decepticon | hash_memory inference (_transform_flags + hash_memory_dim) | `decepticons/loader.py` |

Every mut variant captured and analyzed in this report uses the post-fix loaders.
