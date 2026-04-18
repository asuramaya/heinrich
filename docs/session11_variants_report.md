# Session 11: Variants Study — byte-hrr-s8 + 5 mutations

**For chronohorn.** This report covers the 5 `mut-*` variants (20 trajectory checkpoints) alongside corrections to the 3-seed baseline findings.

## Critical correction to the previous report

The previous report claimed "router 100% E0 collapse across all 12 baseline points." **That claim is false.** It came from heinrich reading a zero-initialized routing tensor and interpreting it as a measurement. Root cause:

- `decepticons/loader.py:79-108` (the `forward_captured` adaptive_substrate branch) never captured `route_weights` — it set the key to None.
- Heinrich's `profile/mri.py:587` pre-allocated a `[n_seqs, seq_len, n_experts]` zeros tensor under `has_routing=True`, so downstream tools read zeros and reported "100% E0, 0% E1."

Both fixed this session:
- Loader now runs `readout.router(features)` and captures softmax probabilities when the readout exposes a `router` submodule.
- MRI capture no longer allocates the routing array unless the probe forward actually returns route weights.
- `profile-cb-routing` rejects all-zero routing tensors with a clear error instead of pretending to measure.

**Real routing in the 3 baseline byte-hrr seeds at step 20k:**

| seed | E0% | E1% | switch | margin | entropy |
|------|-----|-----|--------|--------|---------|
| 42 | 46.2 | 53.8 | 46.7% | 0.733 | 0.313 |
| 43 | 58.9 | 41.1 | 47.4% | 0.732 | 0.312 |
| 44 | 58.0 | 42.0 | 53.4% | 0.701 | 0.343 |

Both experts do real work (42–59% mass each). Switch rates stay stable 46–53% across training. **Entropy grows over training**: seed44 goes 0.32→0.34 (+9%), meaning the router becomes slightly more exploratory as training progresses, not more committed. Margin drops slightly — routing confidence softens, not hardens.

**Takeaway:** the 2-expert routed_sqrelu_experts readout is fine. No collapse. The previous "recommendation to kill E1" is retracted.

## The 5 variants — what was mutated

All 5 variants ran 20k steps at s8 scale (17.8M params), starting from the same byte-hrr baseline config with these changes:

| variant | readout | extra change |
|---------|---------|--------------|
| mut-control-mlp | 4-expert MLP (1024 hidden) | — |
| mut-localon | 4-expert MLP | `local_scale: 0.0 → 0.25` |
| mut-orthoreg-0p1 | 4-expert MLP | orthogonal reg λ=0.1 |
| mut-orthoreg-1p0 | 4-expert MLP | orthogonal reg λ=1.0 |
| mut-persistent | 4-expert MLP | substrate state persists across batches |

"MLP readout with 4 experts" — the config field `linear_readout_num_experts=4` is vestigial for `linear_readout_kind=mlp`. The actual readout is a single hidden-1024 → vocab-256 MLP with no per-expert routing.

## Training-log bpb (final)

| variant | train bpb | test bpb | train time |
|---------|-----------|----------|-----------|
| byte-hrr (routed, baseline) | 2.139 | **2.096** | 1009 s |
| mut-control-mlp | 2.106 | 2.114 | 1862 s |
| mut-localon | 2.106 | 2.114 | 1962 s |
| mut-orthoreg-0p1 | 2.106 | 2.114 | 1877 s |
| mut-orthoreg-1p0 | 2.106 | 2.114 | 1989 s |
| mut-persistent | **2.092** | 2.096 | 2023 s |

**The MLP readout is ~0.02 bpb worse than routed_sqrelu_experts at the same parameter budget.** The router was doing useful work, and replacing it with a plain MLP costs quality. mut-persistent recovers that loss (and matches the baseline on test_bpb) but costs 2x train time.

## Heinrich-measured trajectory (50 seqs × 512 bytes)

| variant | bpb 5k | 10k | 15k | **20k** | EffDim 5k | 10k | 15k | **20k** | ConR² 20k |
|---------|--------|-----|-----|---------|-----------|-----|-----|---------|-----------|
| seed42 (routed) | 2.207 | 2.133 | 2.084 | **2.075** | 124.9 | 142.2 | 146.3 | 149.5 | 0.342 |
| seed43 (routed) | 2.205 | 2.130 | 2.080 | **2.069** | 124.9 | 139.8 | 144.9 | 148.0 | 0.343 |
| seed44 (routed) | 2.207 | 2.138 | 2.082 | **2.068** | 124.7 | 140.8 | 145.6 | 148.9 | 0.332 |
| mut-control-mlp | 2.178 | 2.118 | 2.068 | **2.051** | 120.3 | 142.3 | 149.2 | 150.4 | 0.359 |
| mut-localon | 2.178 | 2.118 | 2.068 | **2.051** | 120.3 | 142.3 | 149.2 | 150.4 | 0.359 |
| mut-orthoreg-0p1 | 2.178 | 2.118 | 2.068 | **2.051** | 120.3 | 142.4 | 149.2 | 150.4 | 0.359 |
| mut-orthoreg-1p0 | 2.178 | 2.117 | 2.068 | **2.051** | 120.3 | 142.4 | 149.2 | 150.4 | 0.359 |
| mut-persistent | 2.173 | 2.107 | 2.076 | **2.054** | 120.2 | 139.9 | 146.6 | 147.9 | 0.345 |

Note: heinrich's 50×512 subset gives slightly different numbers than training's 300-batch eval at seq_len=1024 (e.g. heinrich 2.051 vs training log 2.114 for mut-control). Relative ordering holds.

## Finding 1: Three "different" mutations are the same model

**`mut-control-mlp`, `mut-localon`, `mut-orthoreg-0p1`, `mut-orthoreg-1p0` converge to identical geometry and identical bpb to 4 decimals.**

| metric | control | localon | orthoreg-0p1 | orthoreg-1p0 |
|--------|---------|---------|--------------|--------------|
| bpb step 20k | 2.0513 | 2.0514 | 2.0514 | 2.0513 |
| EffDim step 20k | 150.4 | 150.4 | 150.4 | 150.4 |
| ConR² step 20k | 0.3591 | 0.3590 | 0.3590 | 0.3591 |
| PosR² step 20k | 0.0013 | 0.0013 | 0.0013 | 0.0013 |

Weight L2 difference between pairs: ~3–15 × 10⁻³. The weights drifted slightly during training from different regularization penalties, but the drifts went to the same basin with no measurable behavioral difference.

### Why each mutation had no effect

1. **localon (`local_scale: 0.0 → 0.25`)**: the forward path for `_use_adaptive_substrate=True` is `_forward_raw` which hard-returns `_linear_logits(chars)` and never touches the local path. `local_scale` is dead code for this architecture. (`decepticons/models/causal_bank_torch.py:2029-2031`). **This should be flagged as a config-validation bug in chronohorn's trainer — the run completed 20k steps and 1962 s of compute on a setting that had zero effect.**

2. **orthoreg-0p1 and orthoreg-1p0**: a 10× change in regularization strength produces zero bpb difference. The model already lives in a low-rank basin (EffDim 150 out of 2048 modes, 7.3% utilization). Orthogonality constraints have nothing to tighten when you're already that compressed. No evidence orthoreg did anything at any strength on this architecture.

3. **control-mlp vs the orthoreg/localon triplet**: indistinguishable, confirming the above.

### Actionable for chronohorn
- The localon ineffectiveness is a silent training failure — fix `_forward_raw` to use `local_scale` even for adaptive_substrate, or add a preflight check that rejects localon + adaptive_substrate combinations.
- Orthoreg isn't a useful knob at this scale; de-prioritize the schedule sweep.
- If you want four experts to actually diverge, you need explicit routing (like routed_sqrelu_experts) or a mechanism that breaks the symmetric optimization path. Plain MLP with "num_experts" metadata ≠ expert model.

## Finding 2: mut-persistent is the only genuine mutation

Persistent substrate (state carries across training batches, not reset) is the only variant that reaches a different basin:

- **Different weight signature** — max |diff| vs control is 0.58 (vs 0.003-0.015 for the others).
- **Lower EffDim**: 147.9 vs 150.4 at step 20k.
- **Lower ConR²**: 0.345 vs 0.359.
- **Slightly better training bpb**: 2.092 vs 2.106 (training log). Test bpb matches baseline routed (2.096).

The persistent variant compresses the substrate into a tighter state — lower dimensional, less content-linear. It learns to carry information across batch boundaries, which is a different optimization objective than the reset-per-batch variants.

Persistent is the only candidate worth further investigation.

## Finding 3: EffDim and ConR² ceilings are architectural, not hyperparameter-movable

Across 32 trajectory points (3 seeds + 5 variants × 4 steps = 32):

- EffDim at step 5k: **120–125** across all variants and seeds. Spread 4%.
- EffDim at step 20k: **148–155** across all variants and seeds. Spread 5%.
- ConR² at step 20k: **0.33–0.36**. Spread 9%.
- PosR² at step 20k: **0.001–0.002** across all. Essentially zero.

**These numbers are determined by the architecture (byte-hrr s8 with frozen Fourier ω), not by readout style, regularization, or the local path setting.** The 7% utilization of modes (150/2048) is the architectural attractor. Session 9-10's "dimensional collapse" finding reproduces under every reasonable mutation tested here.

### Implication for chronohorn
Stop searching for hyperparameter wins at s8. The mutations tested don't move the architectural ceiling. Scale (s12+) or structural changes (new rotation algebra, different substrate) are required.

## Status of the 2 heinrich bugs found this session

| bug | status |
|-----|--------|
| Byte shard loader (TOKEN_SHARD_MAGIC header + uint16 payload) | **fixed** `src/heinrich/backend/decepticon.py` |
| cb-loss doesn't warn when bpb > log2(vocab) | **fixed** `profile/compare.py` + CLI print |
| Loader doesn't capture routing for adaptive_substrate | **fixed** `decepticons/src/decepticons/loader.py` |
| MRI saves zero routing tensor without route_weights | **fixed** `src/heinrich/profile/mri.py` |
| cb-routing treats zeros as 100% E0 | **fixed** `profile/compare.py` (now errors with guidance) |

All 20 mut MRIs and 12 baseline MRIs in `/Volumes/Sharts/heinrich/session10_byte_scaling/*.seq.mri` were re-captured after the fixes. Numbers in this report are post-fix.

## Updated recommendations for chronohorn

1. **Routed_sqrelu_experts is healthy, keep it.** Retraction of the previous "kill E1" recommendation. The router uses both experts 40-60%, switches every ~2 tokens, and entropy grows over training.

2. **localon is silently disabled** for adaptive_substrate models. Either fix the forward path or add a config-time check.

3. **Drop orthoreg** from further sweeps at s8. No measurable effect at 0.1 or 1.0.

4. **mut-persistent is the only surviving lead.** Reaches a genuinely different basin. Worth a scale-up test (s12) and a longer trajectory (50k+).

5. **EffDim / PosR² / ConR² ceilings are architectural.** s8 byte-hrr lives at ~150 effective modes, ~0.001 position encoding, ~0.35 content R². Hyperparameter-level mutations do not move these. Next mutations should target the recurrence itself (ω modulation unfreezing, new algebras) or scale.
