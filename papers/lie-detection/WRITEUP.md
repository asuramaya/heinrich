# LLM truth directions are a family, not a feature

An audit of four mechanistic-interpretability claims about truth, using heinrich's 5-test falsification pipeline on Qwen2-7B.

## Summary of results

Across four papers (Marks & Tegmark 2023, Anthropic sleeper-probes 2024, Bürger et al. "truth is universal" 2024, Li et al. ITI 2023), the same linear probe methodology produces claims that reliably fail at least one of the five tests. The common failure mode is **cross-dataset transfer**: a direction that hits 100% in-domain collapses to 50–65% on other truth datasets. The common success: in-domain directions are real, functional, and at layer 20+ survive surface transformations like negation.

Our claim: mean-difference contrastive probes find a **family of related domain-specific truth directions**. Each direction works in-domain, partially shares structure with its neighbors, and fails to be the single universal axis the field writes about. Citations of the form "we found a direction for X" without running the five tests produce narratives that the data does not support.

## The 5-test pipeline

For any claim of the form "direction d at layer L classifies property P":

1. **In-domain accuracy.** Train/test split. Does it classify held-out examples?
2. **Bootstrap stability.** Does the direction stay anchor-stable across resamples (p5 cosine > 0.7)?
3. **Permutation null.** Does the signal exceed the null where labels are shuffled (d > perm_p95)?
4. **Cross-dataset transfer + vocab projection.** Does the direction generalize to other datasets of the same property? Does its lm_head projection name the concept?
5. **Causal ablation.** If you zero the layer or head or direction, does the model's property-handling change?

Tests 1-2 are standard. Test 3 was added after a synthetic stress test in FINDING_03 showed our earlier noise-floor test was broken — it had ~100% false-negative rate under realistic signal-to-noise ratios. Tests 4 and 5 are where most published claims fail.

## The four audits

### Marks & Tegmark 2023 (FINDING_01-06)

Claim: a linear direction learned from contrastive cities statements classifies truth across M&T's 4 datasets at Qwen-7B layer 20.

Result: in-domain accuracy 97-100% across layers 16-24. Bootstrap stable. Permutation null clears. **Cross-dataset fails:** cities direction transfers to companies at 55%, to common_claim at 58%. Within-model cosines across the 4 datasets: 0.21-0.42 — not a shared geometry.

Verdict: method works, interpretation stretched. The direction is "anti-false on cities," not "truth."

### Anthropic sleeper-probes 2024 (FINDING_07)

Claim: simple linear probes hit 99% AUROC on sleeper-agent detection.

Result: applied to vanilla (non-sleeper) Qwen-7B on M&T data, every layer fails bootstrap stability. The 99% is sleeper-specific — a property of the trigger-conditioning pipeline, not a generic LLM mechanism. The paper's method works for what it claims; its applicability to non-sleeper cases is unsupported.

Verdict: qualified.

### Bürger et al. "truth is universal" 2024 (FINDING_06, revisited in 11)

Claim: truth direction is universal across datasets and models.

Result (within-Qwen-7B test): train on pooled union of 4 datasets, classify each individually. Unified direction achieves 82.8% average at L16 vs 87.5% per-dataset. At L20+, unified drops to 78% (per-dataset stays at 85%). The pool direction has cosine 0.83 with cities, 0.55-0.60 with companies/common_claim — dominated by whichever domain has the most examples. Cross-model test requires Procrustes alignment (plumbing not built).

Verdict: partially supported. "Universal" is ~80% true in Qwen-7B — defensible on cities/neg_cities, fails on companies.

### Li et al. ITI 2023 (FINDING_11)

Claim: ~48 specific attention heads in a 7B model carry the bulk of truth-probe signal; steering along those heads' directions improves TruthfulQA 32.5 → 65.1.

Result: per-head attention capture built. Three individual heads on Qwen-7B cities (L16H23, L20H14, L24H20) hit **100% held-out accuracy** — signal concentration replicated.

But:
- **Cross-dataset:** those same heads collapse to 50-62% on common_claim.
- **Causal ablation:** zeroing L16H23 during forward gives identical layer-level accuracy to baseline. Zeroing all 5 top heads gives the same accuracy as zeroing 5 random heads. The "truth heads" are redundantly distributed statistical correlates, not causal substrate.
- **Cities ↔ neg_cities transfer:** at L16, cosine −0.017 (orthogonal, transfer 49%); at L20, transfer 95%. The L16 "100% accurate" head was surface-pattern; L20 tracks propositional content.

Verdict: signal concentration real. Universality, causality, and "truth" label all falsified. ITI's mechanism is per-domain linear-probe discovery of redundant correlates, not intervention on a truth feature.

## What this means

### The correction mid-session

An earlier draft of FINDING_11 concluded: "vocab projection eliminates all false positives; statistical tests alone are inadequate." The functional test (held-out classification) overturned this — the directions that failed vocab projection classified held-out examples at 100%. Vocab projection is a sanity check, not a gate. Residual-stream features are not required to align with lm_head rows.

This correction is the kind of thing the 5-test pipeline catches. The pipeline only works if you run the whole pipeline.

### Scale

One robust_feature verdict in the entire session: Qwen1.5-14B L36 (FINDING_08). All four tests pass; vocab projection at that scale does name `True`/`False`; direction marks "False" variants on the false side. Below 14B in the Qwen family, directions are functional without being lexically grounded. The field's habit of citing probe results at 7B and comparing to vocab-level interpretations is reading a signal into scale-gated behavior.

### Layers

L16 captures surface. L20+ captures proposition. A direction trained on cities at L16 fails on neg_cities (negation flips the pattern). The same training at L20 transfers at 95% (negation does not flip the signal). Layer-level analysis that doesn't separate these is mixing two different things.

### Per-head

Attention-head-level analysis does not yield cleaner interpretability than layer-level. ITI's "48 truth heads" reframed: these are the 48 heads whose output happens to align most strongly with the layer-level mean-diff direction. The layer has the signal; the heads carry it redundantly. Ablation confirms this: removing them does not remove the signal.

## What heinrich now enables

Any mech-interp claim of the form "direction d at layer L classifies X" can be audited end-to-end in ~30 seconds on an MLX-loaded model, at layer resolution or per-head resolution, with held-out classification, bootstrap, permutation, vocab projection, cross-dataset transfer, and causal ablation.

- `forward_pass(capture_heads={L}, ablate_heads={L: {h1}}, return_residual, residual_layer)` — single entry point for all probes.
- `_capture_per_head_attention` — inlines TransformerBlock, captures or ablates per-head residual contributions. Bit-identical to normal forward.
- Test plumbing in `tests/test_measurement_integrity.py` catches regressions.

## Scale test (14B)

Ran the full per-head pipeline on Qwen1.5-14B Chat at L24/32/36 (FINDING_08 had identified L36 as the first robust_feature layer). Result: the 7B findings replicate without change.

| metric | Qwen-7B L20 | Qwen-14B L36 |
|--------|-------------|--------------|
| layer in-domain test acc | 100% | 100% |
| top head in-domain acc | 100% | 98% |
| cities → neg_cities transfer | 95% | 96% |
| cities → common_claim transfer | 61% | 57% |
| ablate top 3 heads Δ acc | 0% | 0% |
| per-head vocab names truth | no | no |

Top heads at 14B (L24H35 d=5.56, L36H36 d=5.34, L32H25 d=4.23) — individually strong probe signal. Ablating any of them, or all three together, produces exactly 0% change in classification accuracy across all measured layers. Vocab projection at per-head resolution still yields garbage tokens (`['‖', '猗', 'ucz', 'van', ...]`). Cross-dataset transfer does not improve with scale — common_claim remains at ~57%.

FINDING_08's "robust at 14B L36" vindication was specifically about the **layer-level** direction's lm_head alignment at 14B (top tokens `True`, `False` appeared). That is consistent with the 14B layer-level direction being cleaner. But per-head direction geometry and causal redundancy at 14B match 7B exactly.

**Scale does not fix the ITI mechanism interpretation.** At every tested scale the pattern is: real in-domain signal, signal concentration in specific heads, cross-dataset collapse, no causal contribution from the top heads.

## Open items

- **Procrustes alignment across hidden dims.** Required for Bürger's cross-model universality test. 2-3 hours of plumbing.
- **ITI's own TruthfulQA result.** The 32.5 → 65.1 improvement is a real behavioral effect. Our audit predicts this is template-specific confidence modulation on TruthfulQA's prompts. Testing requires generating TruthfulQA completions with and without head intervention.
- **More papers.** The shape of the findings is repetitive across four papers. A fifth (CCS, RE, Azaria & Mitchell direct, Truth Neurons) might add a different failure mode or confirm the pattern.

## Reproducibility

Every finding has a script in `/tmp/` (to be promoted into `src/heinrich/` as MCP tools). Every claim has a JSON artifact. Every direction file can be regenerated from the Qwen-7B checkpoint + the cities/companies/common_claim/neg_cities CSVs in `data/`. No external compute required; 7B runs on a 64GB Mac in MLX-4bit.

## Unsupervised test: PCA on the pooled residual manifold

Labeled mean-diff is one way to find a direction. CCS (Burns et al., not yet formally audited) and related methods find directions unsupervised. If truth is a dominant structural axis in residual space, PCA on pooled residuals should surface it near the top.

Test: at Qwen-7B L20, concatenate 320 residuals (80 per class × 2 classes × 4 datasets) — 640 residuals × 3584 dims. Run SVD. Measure classification accuracy of each principal component on each dataset.

Result:

| PC | % variance | cities | neg_cities | companies | common_claim | avg |
|----|-----------|--------|------------|-----------|--------------|-----|
| 0 | 24.8% | 55.6% | 50.0% | 56.9% | 53.8% | 54.1% |
| 1 | 12.9% | 52.5% | 50.0% | 55.0% | 50.0% | 51.9% |
| 2 | 10.0% | 61.2% | 50.0% | 53.1% | 56.2% | 55.2% |
| 3 | 7.5% | 60.0% | 50.0% | 53.8% | 55.6% | 54.8% |
| **4** | **6.2%** | **87.5%** | **88.8%** | **56.2%** | **62.5%** | **73.8%** |
| 5 | 3.0% | 52.5% | 90.6% | 50.6% | 55.6% | 62.3% |
| 6 | 2.0% | 90.6% | 99.4% | 52.5% | 52.5% | 73.8% |

Cosine of PC4 with the labeled unified mean-diff: +0.835. PC4 is the same direction, discovered without labels, at 4th-rank variance.

**The top 4 PCs (52% of total variance) classify truth at chance.** Truth is a *trace axis* at Qwen-7B L20 — on the order of 6% of total residual variance, below language family (10.2% per FINDING_04 analogue on Qwen-0.5B) and below what are likely topic/register axes. No PC is universal: PC4 is cities/neg_cities; PC6 is cities-specific; PC5 is neg_cities-specific.

This matches the Session 4/5 finding pattern: safety is 0.5% of displacement variance, comply is 0.9%, language identity is 10%. Truth sits in the same size class — a minor signal riding on much larger structure.

## CCS replication

The PCA test showed truth is not a top-variance axis. CCS (Burns et al. 2022) is the specific unsupervised method that uses paired statement + negation as consistency structure rather than variance ranking. We implemented CCS on cities/neg_cities pairs at Qwen-7B L20:

- Loss: `(p(stmt) + p(neg) - 1)² + min(p(stmt), p(neg))²`
- 10 random initializations, L-BFGS-B, 200 iterations each.
- Best loss: 0.000 — CCS fits the consistency constraint perfectly.

Classification accuracy:

| method | cities | neg_cities | companies | common_claim | avg |
|--------|--------|------------|-----------|--------------|-----|
| **CCS (unsupervised)** | 99.4% | 99.4% | 66.9% | 65.6% | **82.8%** |
| unified mean-diff (labeled) | 93.8% | 98.8% | 77.5% | 66.9% | 84.2% |
| cities mean-diff (labeled) | 97.5% | 90.6% | 63.1% | 60.6% | 78.0% |
| PC4 (unsupervised PCA) | 87.5% | 88.8% | 56.2% | 62.5% | 73.8% |

CCS beats cities-only mean-diff on average and solves the paired structure exactly (99.4% × 99.4%) — that is exactly what its loss optimizes. CCS does **not** transfer better than mean-diff to companies/common_claim.

Cosine(CCS direction, cities mean-diff): +0.396 — moderately aligned, not identical. Cosine(CCS, pooled mean-diff): +0.473. CCS finds a direction that is *related to* but *not the same as* the supervised mean-diff — they catch different aspects of the same signal.

**The paradigm doesn't matter.** Supervised mean-diff, unsupervised PCA, unsupervised CCS — three different optimization objectives all hit the same structural ceiling at Qwen-7B L20. Within-domain directions are strong; cross-domain transfer plateaus at 65-70%. No training method recovers a universal truth axis because none exists in the residual geometry to recover.

## Cross-model Procrustes test (Bürger's universality)

Bürger et al.'s claim is cross-**model** universality: learn a truth direction on model A, align via Procrustes to model B's activation space, and it should work on model B. We tested this on Qwen-7B L20 → Mistral-7B L16, paired on the same 640 statements across 4 datasets.

**Procedure:**
1. Capture residuals at both models (sequential, avoiding OOM).
2. Orthogonal Procrustes: SVD of cross-covariance on train split (512 stmts), R ∈ R^{3584 × 4096} with R^T R = I.
3. Train direction on Qwen; project through R; classify Mistral test set.

**Results:**

| dataset | Qw on Qw | Qw → Ms (via R) | Ms native | cos(Qw@R, Ms) |
|---------|---------|-----------------|-----------|---------------|
| cities | 96.88% | 65.62% | 87.50% | +0.954 |
| neg_cities | 96.88% | 50.00% | 100.00% | +0.978 |
| companies | 56.25% | 50.00% | 65.62% | +0.942 |
| common_claim | 75.00% | 50.00% | 71.88% | +0.968 |

Direction cosines after alignment are 0.94-0.98 — the two models' cities-trained directions are **geometrically near-identical** once aligned. But per-dataset classification via the projected direction drops to chance for 3 of 4 datasets. The cross-model direction *geometry* survives alignment; the *decision threshold* does not.

The unified (pooled 4-dataset) direction transfers better:

| test ds | Qw_uni(Qw) | Qw_uni@R(Ms) | Ms_uni native |
|---------|-----------|--------------|---------------|
| cities | 87.50% | 93.75% | 87.50% |
| neg_cities | 96.88% | 100.00% | 100.00% |
| companies | 71.88% | 65.62% | 65.62% |
| common_claim | 75.00% | 59.38% | 75.00% |

`cos(Qw_unified @ R, Ms_unified) = +0.985`. Cross-model unified direction transfer is close to native performance, with one collapse (common_claim drops 75% → 59%).

**Bürger's claim is partially supported.** The truth direction has shared geometry across Qwen-7B and Mistral-7B — cosines 0.94-0.98 after Procrustes alignment. Operational classification via a cross-model direction is degraded because threshold calibration is model-specific; per-narrow-dataset directions suffer more than pooled. A "universal truth direction" in the geometric sense is real; a cross-model plug-and-play probe is not.

## Final cross-cutting claim

Mean-difference contrastive probes at Qwen-7B and Qwen-14B find a **family of related, domain-specific truth directions**.

- They classify their own training domain at 100%.
- They survive surface transformations (negation) at propositional layers (L20+ at 7B, L24+ at 14B).
- They share ~80% of their structure across domains at 7B — partial universality, not full.
- They are redundantly distributed across attention heads; ablation doesn't remove them at any tested scale (0.5B, 7B, 14B).
- Their per-head direction in residual-stream space does not align with `lm_head` truth/false rows at any tested scale; layer-level directions align at 14B but not below.
- Cross-dataset transfer does not improve with scale: cities → common_claim stays at 57-61% from 7B to 14B.
- Truth is **not** a dominant axis in residual space — PCA finds it at ~6% of variance, below topic/register structure. Unsupervised methods that select top-variance directions will miss it.
- Cross-model geometry is partially shared: Qwen-7B and Mistral-7B truth directions align at cosine 0.94-0.98 after orthogonal Procrustes, but cross-model classification via a projected direction degrades per-dataset (survives for pooled).

The field's move from "we found a direction for X in 7B" to "this direction is X" is where the audit bites. The first claim is supportable. The second requires running the pipeline.
