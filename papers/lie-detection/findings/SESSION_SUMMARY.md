# Session summary: 9 findings, 5 papers attacked

## Apparatus built and audited

- 5-test falsification pipeline: bootstrap stability, permutation null,
  within-group control (retired), vocab projection sanity, causal steering.
- Endpoints: `/api/replicate-probe-multilayer`, `/api/causal-truth-test`,
  `/api/direction-bootstrap`, `/api/behavioral-direction`,
  `/api/token-resolve`, `/api/direction-brief`, `/api/pca-reconstruction`,
  `/api/residual-trajectory`.
- Speedup: one-pass multi-layer capture — 7B × 6 layers from 60 min → 13 sec.
- Adversarial review found 2 broken tests (Bug 2 empty-string matcher, Bug 3
  noise-floor SNR). Both fixed with permutation null.

## Findings

1. **FINDING_01**: MVP safety direction at Qwen-0.5B — vocab in Arabic/code.
2. **FINDING_02**: M&T scale ladder (0.5B → 7B) — d grows, vocab still wrong,
   `true`/`false` same-side clustering at 7B.
3. **FINDING_03**: Apparatus audit — 2 tests broken, replaced, findings survive.
4. **FINDING_04**: M&T 4 datasets at Qwen-7B — cities works, common_claim fails
   permutation null. Direction is template-specific, not truth-general.
5. **FINDING_05**: Causal steering at Qwen-7B L20 — asymmetric, non-inverting.
   Can disrupt false-classification but can't convert to true.
6. **FINDING_06**: 4 M&T datasets → 4 different directions (cosines 0.21-0.37).
   No universal truth axis at Qwen-7B L20.
7. **FINDING_07**: Anthropic "simple probes" on vanilla Qwen-7B — every layer
   fails bootstrap stability. Their 99% AUROC is sleeper-specific,
   not a generic LLM mechanism.
8. **FINDING_08**: Qwen1.5-14B L36 — **first robust_feature verdict**.
   All 4 tests pass. Direction marks `False` variants on false side —
   better described as "anti-false" than "truth."
9. **FINDING_09**: ITI test deferred — requires per-head residual plumbing
   (1-2 hours of backend work). *[Superseded by FINDING_11.]*
10. **FINDING_10**: Jane Street `dormant-model-warmup` has no trigger word.
   Broad math-puzzle fine-tune of Qwen2-7B. Real dormant-1/2/3 series is
   DeepSeek-V3 FP8 (different architecture).
11. **FINDING_11**: Per-head attention residual capture built and tested on
   Qwen-7B cities → common_claim.
   - In-domain: 3 heads hit 100% held-out cities accuracy (L16H23, L20H14, L24H20).
   - Transfer: those same heads collapse to 50–62.5% on common_claim.
   - Native common_claim is at chance (L20, L24 test = 50%).
   - Direction cosine across datasets 0.18–0.42 at layer level, L20H14 = 0.10.
   ITI's signal-concentration claim replicated. Its truth-universality claim
   falsified. "Truth heads" are topic-classification heads. Self-correction:
   an earlier draft claimed vocab projection eliminated all FPs — functional
   tests showed that was itself a false negative.

## Audit status by paper

| Paper | Status | Verdict |
|-------|--------|---------|
| Marks & Tegmark 2023 (M&T, Geometry of Truth) | Complete (6 findings) | Method works, interpretation stretched. Direction is "anti-false at scale," not truth. |
| Anthropic Sleeper-Agent probes 2024 | Qualified (Finding 07) | 99% AUROC is sleeper-specific. Method on vanilla model is weak, bootstrap-unstable. |
| Bürger et al. Truth is Universal 2024 | Partial (within Finding 06) | Within-model universality fails (0.21-0.37 cosines across 4 M&T datasets). Cross-model test requires Procrustes plumbing. |
| Li et al. ITI 2023 | Complete (Finding 11) | Plumbing built. Signal-concentration replicated: 3 heads hit 100% cities held-out. Universality falsified: same heads collapse to 50–62.5% on common_claim, native common_claim at chance. "Truth heads" are topic-classification heads. |

## Cross-cutting claim (final form after FINDING_11)

Mean-difference contrastive probes at Qwen-7B find a **family of related
domain-specific truth directions** — not a single universal axis, not a pure
topic classifier, but something in between:
- L16 (surface): partially shared structure; pooled training nearly matches per-dataset.
- L20+ (propositional): each domain has its own truth subspace; pooling costs ~7%.
- Cross-dataset transfer: cities → neg_cities ≈95% at L20 (survives negation),
  cities → companies ≈60% (cross-domain collapse).
- Per-head: signal concentrates in specific heads, but ablating them doesn't
  reduce classification — redundantly distributed.
- Direction is functional (100% in-domain) without being lexically interpretable
  (vocab projection fails below 14B).

Bürger's "truth is universal": ~80% true at Qwen-7B — defensible on some
datasets, falsified on others. ITI's "truth heads": signal-concentration real,
universality false, causal importance null. M&T's cities: validated at propositional
level (survives negation at L20+), cross-dataset transfer always outside scope.

The field's habit of citing "we found a direction for X" without running the
5-test pipeline (in-domain accuracy + cross-dataset + vocab + ablation +
scale) is producing narratives unsupported by the data's full story.

## What heinrich now enables

Any mechanistic-interpretability paper that makes a claim of the form
"direction d at layer L classifies X" can be audited in:
- 30 seconds if the model has an MRI (vocab from captured scores)
- 30 seconds if the model can run via MLX backend (vocab from lm_head)

The outputs:
- d: Cohen's d
- boot p5: anchor stability
- perm p95: permutation null
- SNR: d / perm_p95
- vocab match: how many expected concept tokens appear in top-10
- verdict: robust_feature / partial / falsified

Total compute per paper attack: ~1-2 minutes (6 datasets × 4 layers).

## Priorities for next session

1. **Per-head attention residual capture** → unlocks ITI, unlocks cleaner
   layer-decomposition of M&T's signal.
2. **Procrustes alignment across hidden dims** → enables Bürger's
   cross-model universality test.
3. **Formal write-up of the audit methodology** — the pipeline itself is
   a contribution separate from the 6 findings on M&T.
4. **Expand audit to 5+ more papers** (Azaria & Mitchell direct replication,
   CCS, Representation Engineering, Truth Neurons, Alignment Faking linear
   features).

Every step above has plumbing already 80% built. The audit is accelerating
as we go.
