# Session 11: 3-Seed Trajectory Forensics — byte-hrr-s8-trajectory

> **Retraction notice (post-session):** this report claims router collapse at step 20k (E0 = 100%, E1 = 0%). That claim is **false**. It came from a routing-tensor capture bug — heinrich's loader never captured route_weights for adaptive_substrate models, and the pre-allocated zero tensor was misread as a measurement. Actual routing is healthy (E0/E1 in 46%–59%, entropy 0.31–0.34). See `session11_variants_report.md` for the corrected analysis. Keep this document for its bpb / EffDim / ConR² numbers; ignore every line about expert routing.

**For chronohorn.** Twelve sequence MRIs across seeds 42/43/44 × steps 5k/10k/15k/20k, byte-level (vocab 256), `substrate_mode=frozen`, 17.8M params, 20k training steps.

Captures under `/Volumes/Sharts/heinrich/session10_byte_scaling/byte-hrr-s8-trajectory-seed{42,43,44}-step{5,10,15,20}k.seq.mri/`.

## Note on data

The first pass of this analysis used the wrong validation file. Heinrich's `load_val_sequences` was reading the chronohorn byte shard format incorrectly: the shards store bytes as uint16 after a 1024-byte header, but the loader was reading raw uint8 with no header skip. This turned valid byte data into nonsense for the model (zero-padded high bytes plus header leakage), which produced a ~10× loss inflation and distorted every downstream metric. Fix landed in `src/heinrich/backend/decepticon.py` — the loader now detects the `TOKEN_SHARD_MAGIC=20240520` header, skips it, and reads payload as uint16. Numbers below are from the re-captured MRIs with the correct data.

## Headline

**Training is 0.3% reproducible across 3 seeds** — final bpb 2.0745 / 2.0691 / 2.0676, essentially identical. Internal geometry (EffDim, ConR², substrate L2) is equally tight. Position encoding does not emerge at this scale. One expert is permanently dead.

## Seed-Level Convergence

### Training log
| seed | train_eval_loss | test_eval_loss | test_bpb (training) | train_time |
|------|-----------------|----------------|----|-----------|
| 42 | 1.4824 | 1.4526 | 2.0957 | 1009 s |
| 43 | 1.4841 | 1.4522 | 2.0951 | 1033 s |
| 44 | 1.4844 | 1.4537 | 2.0973 | 1005 s |

### Heinrich re-measured bpb (50 seqs × 512 bytes, same val shard)
| seed | 5k | 10k | 15k | 20k |
|------|-----|-----|-----|-----|
| 42 | 2.2066 | 2.1333 | 2.0840 | **2.0745** |
| 43 | 2.2050 | 2.1303 | 2.0800 | **2.0691** |
| 44 | 2.2074 | 2.1381 | 2.0818 | **2.0676** |

Heinrich's 20k measurement differs from training-logged bpb by about 1.4% (small subset vs 300-batch eval). Spread across 3 seeds at step 20k: 0.0069 bpb = 0.3%.

## Substrate Trajectory

| seed | step | EffDim | PosR² | ConR² |
|------|------|--------|-------|-------|
| 42 | 5k | 124.9 | 0.001 | 0.322 |
| 42 | 10k | 142.2 | 0.001 | 0.332 |
| 42 | 15k | 146.3 | 0.001 | 0.336 |
| 42 | 20k | **149.5** | 0.001 | **0.342** |
| 43 | 5k | 124.9 | 0.001 | 0.314 |
| 43 | 10k | 139.8 | 0.001 | 0.332 |
| 43 | 15k | 144.9 | 0.001 | 0.344 |
| 43 | 20k | **148.0** | 0.001 | **0.343** |
| 44 | 5k | 124.7 | 0.001 | 0.327 |
| 44 | 10k | 140.8 | 0.001 | 0.332 |
| 44 | 15k | 145.6 | 0.001 | 0.325 |
| 44 | 20k | **148.9** | 0.001 | **0.332** |

### Finding 1: EffDim growth is front-loaded
5k → 10k: +17 to +18 EffDim units. 15k → 20k: +3 each seed. **~75% of the dimensional unlock happens in the first 10k steps.** The remaining 10k buys ~4 more effective dimensions per seed.

### Finding 2: Position encoding never emerges at s8
PosR² stays at 0.001 ± 0.0005 across all 12 points. **The substrate state contains effectively zero linear information about position.** Confirms tex Finding 4 — position requires scale (s12+) or architectural intervention, not more time.

### Finding 3: ConR² plateaus by step 15k
All three seeds flatten ConR² at 0.32–0.34 by step 15k. Additional training past 15k adds EffDim but not content-linear predictivity. The substrate geometry keeps expanding, but the new dimensions don't carry more loss-predictive content.

### Finding 4: Seed convergence is within 1–3% on every geometry metric
| Metric | seed42 | seed43 | seed44 | spread |
|--------|--------|--------|--------|--------|
| bpb @ 20k | 2.0745 | 2.0691 | 2.0676 | **0.3%** |
| EffDim @ 20k | 149.5 | 148.0 | 148.9 | **1.0%** |
| ConR² @ 20k | 0.342 | 0.343 | 0.332 | **3.3%** |
| EffDim @ 5k | 124.9 | 124.9 | 124.7 | **0.2%** |

**The early-training EffDim spread is 0.2% — effectively deterministic.** Seeds diverge slightly over training but re-converge.

## W_omega Forensics (all 12 points)

```
Weight L2: 0.0   (all steps, all seeds)
Fourier correlation: 1.0
Fourier drift: 0.0015
```

`_adaptive_omega_proj.weight` is exactly zero throughout training for every seed. Bias is pure Fourier init (`linspace(0, 2π, 2048)`). **Frozen-HRR confirmed — all content learning happens in embedding + linear_in_proj + linear_readout, not in the substrate rotation modulator.** Matches session 9-10 tex Finding 2.

## Router Collapse (all 12 points)

```
E0: 100.0%   E1: 0.0%
Switch rate: 0.00%
Routing margin: 0.0000
Routing entropy: -0.0000
```

Identical in every one of the 12 points (data-independent — a property of the weights, not the val data). `routed_sqrelu_experts=2` with `balance_coeff=0.01` collapses by step 5k and never recovers. E1 is permanent dead weight: 511×2304 ≈ 1.18M params plus 256×511 ≈ 131K output-projection params = **~1.3M unused params per model** (7.5% of the 17.8M param count).

### Actionable for chronohorn
1. **Drop to 1 expert** → save 1.3M params with no bpb cost. The simplest fix.
2. **Raise balance_coeff** — 0.01 is not enough here. Try 0.1 or an annealing schedule.
3. **Router init audit** — the router weight might land in a single-expert basin at init. Check init distribution.
4. **Noise during routing** — adding gumbel or straight-through noise to the router could prevent the single-expert lock-in.

## Heinrich bug fixed (data loader)

The loader bug silently inflated byte-level loss measurements by ~10×. Impact was:
- `profile-cb-loss` overall_bpb wrong for every byte-level MRI captured so far.
- Trajectory `cb-trajectory` reported non-monotone ConR² (e.g. seed43 going 0.37 → 0.35 at step 20k). Artifact of reading header + zero-padded high bytes as data. **The real ConR² is monotone-plateau for all seeds.**
- `cb-substrate-local` per-position L2 unaffected (doesn't depend on loss).
- `cb-routing` unaffected (doesn't depend on val data — it's a weight property).
- `cb-omega-forensics` unaffected (reads checkpoint directly).

Belt-and-braces sanity check added in `profile-cb-loss`: warns when `overall_bpb > log2(vocab)`, which guarantees the reader sees when the model is worse than random guessing.

Existing byte-level MRIs in `/Volumes/Sharts/heinrich/session10_byte_scaling/*.seq.mri` have bad loss data and need re-capture if loss-based analysis is planned. Session 9-10 findings that used loss-based tools (e.g. bpb numbers in the tex) should be verified with the fixed loader.

## Recommendations for chronohorn

1. **Seed budget**: Final bpb variance is 0.3% at this config. Multi-seed is useful for error bars only. For per-config validation, one seed is enough.
2. **Kill E1**: routed_sqrelu_experts=2 is a parameter hole at this config. Either fix the routing or halve the expert count.
3. **Don't train byte-hrr-s8 past 15k**: diminishing returns on EffDim growth and flat ConR² beyond 15k. 5–10k saves 50% of compute with 97% of the final quality.
4. **Position is a scale problem, not a time problem**: confirmed at 3-seed scale. s12+ or architectural intervention required for position encoding.
5. **Frozen-HRR validates**: content R² climbs to 0.33–0.34 with zero modulation of ω-weight. Validates the frozen-Fourier thesis from session 9-10.
