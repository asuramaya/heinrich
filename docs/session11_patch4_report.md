# Session 11: Patch4 Variants — mut-patch4, mut-patch4-orthoreg

**For chronohorn.** 8 trajectory MRIs across mut-patch4 and mut-patch4-orthoreg × steps 5k/10k/15k/20k.

## Loader changes required

The patch4 checkpoints use a format neither heinrich nor the decepticon loader previously handled: patching via a fat readout output (`linear_readout.out: (1024, 1024)` = patch_n × vocab) with no separate `_patch_encoder` / `_patch_decoder` / `_patch_byte_heads` modules. The existing patch-size inference in `loader.py` only looked at those module keys, so every patch4 checkpoint failed to load with `size mismatch for linear_readout.out.*`.

Fixed in `decepticons/src/decepticons/loader.py`:
- Detect fat-readout patching by `linear_readout.out.weight.shape[0] % vocab_size == 0 and > vocab_size`. Set `patch_n = out_dim // vocab_size`, keep `patch_size = 1` so `forward()` dispatches to `_forward_raw` (the path that correctly handles adaptive substrate) rather than `_forward_patched` (which expects the missing modules).
- Load with `strict=False` to skip absent `_patch_encoder`/`_patch_decoder` weights.
- `forward_captured` now reshapes fat readout output via `_reshape_patch_logits`, then slices `[..., 0, :]` so downstream heinrich code (which expects `(B, T, V)` logits) still works. Position 0 is what `test_bpb` in training logs reflects.

Also added in `decepticons/loader.py` earlier this session:
- Router capture for adaptive_substrate models that have `hasattr(linear_readout, 'router')`.

## Architecture

Both variants share config with `mut-control-mlp` (MLP readout, 1024 hidden) plus patching:
- `_patch_n = 4` (fat readout: single-step sees 4-byte target vector)
- `_patch_size = 1` in the current loader mapping (model's forward path)
- Params: 18.56M (+787k over baseline, from the wider readout)
- Training time: 1954 s (patch4), 2022 s (orthoreg variant)

Training uses all 4 patch-head outputs as supervision (loss averaged over next 4 bytes per position), but `test_bpb` in the log is only the next-byte (p=0) head.

## Per-head bpb decomposition (step 20k, direct forward)

| head | predicts | bpb |
|------|----------|-----|
| p=0 | t+1 (next byte) | 2.200 |
| p=1 | t+2 | 3.201 |
| p=2 | t+3 | 3.838 |
| p=3 | t+4 | 4.180 |
| averaged | all 4 | 3.356 |

Only p=0 is competitive with baseline (2.075). Heads p=1..3 are progressively less accurate, as expected — predicting further ahead without seeing intermediate bytes.

## Training-log bpb comparison

| variant | test_bpb | Δ vs baseline | params | train time |
|---------|----------|---------------|--------|------------|
| byte-hrr (baseline, routed) | **2.096** | — | 17.77M | 1009 s |
| mut-control-mlp (no patch) | 2.114 | +0.018 | 17.77M | 1862 s |
| mut-patch4 | 2.221 | +0.125 | 18.56M | 1954 s |
| mut-patch4-orthoreg | 2.221 | +0.125 | 18.56M | 2022 s |

Patch4 is **6% worse bpb** than baseline on next-byte prediction, despite:
- 4.5% more parameters (readout width 1024 × 1024 vs 1024 × 256)
- 4× training signal per position
- 2× wall-clock training time

## Heinrich-measured trajectory (50 seqs × 512 bytes)

| step | bpb | EffDim | PosR² | ConR² |
|------|-----|--------|-------|-------|
| mut-patch4-5k | 2.2531 | 131.7 | 0.0013 | 0.3232 |
| mut-patch4-10k | 2.2115 | 151.0 | 0.0014 | 0.3352 |
| mut-patch4-15k | 2.1737 | 153.5 | 0.0014 | 0.3337 |
| **mut-patch4-20k** | **2.1601** | **153.3** | **0.0014** | **0.3316** |
| mut-patch4-orthoreg-5k | 2.2531 | 131.7 | 0.0013 | 0.3231 |
| mut-patch4-orthoreg-10k | 2.2114 | 151.0 | 0.0014 | 0.3348 |
| mut-patch4-orthoreg-15k | 2.1740 | 153.5 | 0.0014 | 0.3334 |
| **mut-patch4-orthoreg-20k** | **2.1603** | **153.3** | **0.0014** | **0.3312** |

Heinrich's 50 × 512 sample gives numbers ~3% lower than training's 300-batch 1024-seq eval (2.160 vs 2.221), same trend as every other variant.

## Finding 1: orthoreg null result, fourth confirmation

mut-patch4 vs mut-patch4-orthoreg:
- bpb step 20k: 2.1601 vs 2.1603 → Δ 0.0002
- EffDim step 20k: 153.3 vs 153.3 → identical
- ConR² step 20k: 0.3316 vs 0.3312 → Δ 0.0004
- max |weight diff|: 0.19 (nontrivial) — but zero behavioral effect

Across now 6 configurations tested (mut-orthoreg-0p1, mut-orthoreg-1p0, mut-patch4, mut-patch4-orthoreg, plus control baselines), orthogonal regularization produces no measurable bpb or geometry change at s8. **Orthoreg is dead code for this architecture family.**

## Finding 2: Patch4 substrate uses more dimensions but produces less content-linear geometry

Compared to byte-hrr-s8 seed42 at step 20k:

| metric | baseline (routed) | patch4 | Δ |
|--------|-------------------|--------|---|
| EffDim | 149.5 | 153.3 | +3.8 (+2.5%) |
| ConR² | 0.342 | 0.332 | −0.010 (−3%) |
| bpb (p=0, heinrich measure) | 2.075 | 2.160 | +0.085 (+4.1%) |
| bpb (training log) | 2.096 | 2.221 | +0.125 (+6.0%) |

The patched substrate is **higher-dimensional, less content-predictive, and less accurate at next-byte prediction** than the routed non-patched baseline. The extra dimensions are spent on look-ahead (predicting t+2, t+3, t+4) that the substrate doesn't actually encode well — note the p=1/2/3 bpb degradation (3.2, 3.8, 4.2 vs p=0 at 2.2).

## Finding 3: Training is highly reproducible at patch4 too

The patch4 trajectory is indistinguishable from patch4-orthoreg through all 4 steps (5k, 10k, 15k, 20k) across all metrics. **Seed-like reproducibility extends from baseline to this architectural variant.** This is consistent with session 11 baseline findings (3-seed spread of 0.3% on bpb).

## Actionable for chronohorn

1. **Don't ship patch4 for next-byte quality.** The fat-readout patching mechanism trades single-step accuracy for multi-step supervision density. If the goal is next-byte bpb, the non-patched baseline wins by 6%.

2. **Patch4 could be useful if evaluated on a multi-step metric** (e.g. "mean bpb predicted 4 bytes ahead from current substrate"). That metric isn't `test_bpb` in the training log. Consider adding it.

3. **Orthoreg can be removed from the codebase**, not just the sweep. No variant tested (with or without patching) shows any geometry or bpb response.

4. **Heinrich now handles fat-readout patching**. Future patch variants (patch2, patch8, etc.) will load and MRI correctly.

## Files touched

- `decepticons/src/decepticons/loader.py` — fat-readout patch_n inference, router capture, logits reshape for fat-readout output
- `src/heinrich/backend/decepticon.py` — already fixed for chronohorn byte-shard format earlier this session
- `src/heinrich/profile/mri.py` — already fixed for routing tensor allocation earlier this session
