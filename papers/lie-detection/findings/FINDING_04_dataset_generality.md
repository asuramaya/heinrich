# Finding 04: M&T's truth direction is dataset-specific, not general

**Paper:** Marks & Tegmark 2023, *The Geometry of Truth*
**Test:** Apply the corrected 5-test pipeline at Qwen-7B L20 across M&T's
four factual-statement datasets.
**Date:** 2026-04-18
**Compute:** 25 seconds (one forward pass per prompt set, parallel across datasets).

## Question

If M&T's claim is "LLMs linearly represent the truth of factual statements,"
the direction should generalize across different kinds of factual statements.
Does it?

## Result

| Dataset | Content | Cohen's d | Boot p5 | perm p95 | SNR | Vocab tells us |
|---------|---------|-----------|---------|----------|-----|----------------|
| cities | "X is in Y" (countries) | **4.51** | **0.97** | 0.93 | **4.84** | `true, false, 错` on same side |
| neg_cities | "X is not in Y" (countries) | **5.56** | **0.98** | 0.91 | **6.10** | `true` on false-statement side |
| companies | "X has HQ in Y" / "X engages in Z" | 1.48 | **0.43** | 0.83 | 1.78 | code fragments both sides |
| common_claim | "Oxygen is purple", "Spiders walk on water" | 2.01 | 0.72 | **2.78** | **0.72** | `often` on true side — but below perm null |

## What these numbers say

1. **`cities` + `neg_cities`: strong, stable signal.** d ≈ 5, SNR ≈ 5-6, bootstrap rock-solid.
2. **`companies`: weak and anchor-unstable.** d=1.48, boot_p5=0.43 (fails < 0.7).
3. **`common_claim`: no signal above noise floor.** d=2.01 but the permutation null's 95th percentile is **2.78** — the direction on general-knowledge claims is indistinguishable from a random relabeling.

## Why neg_cities is the critical test

"The city of Paris is **not** in China" — labeled TRUE.
"The city of Paris is **not** in France" — labeled FALSE.

Both sentences start with the same city-country template. They ONLY
differ on which country appears in the NEGATED position. If M&T's direction
were really a "truth direction," it should treat "Paris-not-China" (true)
like other true statements, i.e., on the POSITIVE side.

It does. d=5.56, top_neg includes ` true`, ` True`. The direction responds
to negated claims the same way as affirmative ones. **This rules out the
trivial-surface-pattern explanation** (that it's tracking surface coherence
of city-country pairs).

What survives: the direction tracks "is this claim in the school of
evaluation-worthy factual assertions?" regardless of affirmation/negation
polarity. But ONLY within the cities-family format.

## Why common_claim is the killer

These are statements like "Spiders can use surface tension to walk on water"
(true) and "Oxygen is actually purple" (false). They're still factual
but don't follow the `[subject] is [property]` template that cities/neg_cities do.

On this more realistic test of "factual truth representation," **the
direction extracted from cities does not generalize.** Cohen's d=2.01 is
below the permutation-null p95 of 2.78 — meaning shuffled labels can
produce equal or greater d. The direction isn't capturing truth; it's
capturing "does this prompt follow the cities-template pattern."

## What this finding means

M&T's "Geometry of Truth" direction is:
- Robustly present in their training-style factual statements (cities, neg_cities).
- Polarity-invariant within that template (survives negation).
- Does NOT generalize to non-template factual claims.
- Does NOT have semantic truth-vocabulary in its logit-lens projection.

The right characterization: **Qwen-7B has a layer-20 direction that tracks
whether a claim follows a known "evaluate-this-format" template.** Not
truth. Not epistemic state. A template-recognition axis.

For M&T's core claim — "LLMs linearly represent truth" — this is a
significant qualification. The direction exists, it's robust, it steers
causally (likely, based on their intervention results — we haven't
reproduced that). But it represents a narrower phenomenon than the title.

## Predictive consequences

If the finding generalizes to other linear-probe truth-detection papers:
- Azaria & Mitchell's 71-83% accuracy probes: likely dataset-specific.
- Bürger et al.'s "universal 2D subspace": likely trains-on-one-format,
  generalizes-to-same-format.
- Sleeper-agent probes' 99% AUROC: likely captures the specific "deployment"
  context format used in training, not deception more generally.

The prediction: any paper that reports a probe trained on format X and
tested on held-out X will show degradation on format Y. The field's
habit of reporting cross-dataset accuracy within the same paper family
(TruthfulQA train, TruthfulQA test) masks this.

## What to do next

1. **Test ITI, sleeper-agent probes, Azaria&Mitchell on the same 4 datasets.**
   This would produce a replication table across papers, not just across datasets.
2. **Causal steering at Qwen-7B L20.** Does ±α·d actually flip the model's
   behavioral treatment of cities vs common_claim statements? If the
   direction only causally matters for the cities template, that's the
   cleanest confirmation.
3. **Test at larger scale (13B, 70B)** — M&T's "sufficient scale" regime.
   Does the common_claim failure close, or persist?

## Commands

```bash
# 4-dataset grid in 25 seconds
python3 -c "..." # see above snippet
```

Full structured output in `papers/lie-detection/data/dataset_grid_qwen7b_L20.json`.
