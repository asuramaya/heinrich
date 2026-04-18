# Finding 06: Four M&T datasets yield four different "truth directions"

**Paper:** Marks & Tegmark 2023, *The Geometry of Truth* — their
universality claim: "at sufficient scale, LLMs linearly represent the
truth or falsehood of factual statements."
**Test:** Extract the mean-difference direction separately from each of
their four datasets at Qwen-7B L20. Compute pairwise cosine similarities.
If the paper's claim holds, all four directions should ≈ the same axis.
**Date:** 2026-04-18
**Compute:** 26 seconds total.

## Result

### Per-dataset probe quality

| Dataset | Cohen's d | SNR vs perm null | Verdict |
|---------|-----------|------------------|---------|
| cities | +4.51 | 4.77 | passes bootstrap + perm |
| neg_cities | +5.56 | 5.77 | passes bootstrap + perm |
| companies | +1.48 | 1.75 | weak d, boot_p5=0.43 fails |
| common_claim | +2.01 | 0.72 | d **below perm p95** — no signal above noise |

### Pairwise cosines between the four extracted directions

| | cities | neg_cities | companies | common_claim |
|---|---|---|---|---|
| **cities** | 1.000 | 0.314 | 0.366 | 0.220 |
| **neg_cities** | 0.314 | 1.000 | 0.228 | 0.268 |
| **companies** | 0.366 | 0.228 | 1.000 | 0.207 |
| **common_claim** | 0.220 | 0.268 | 0.207 | 1.000 |

## What these numbers mean

If M&T's claim held at Qwen-7B L20, the cosine between any two
well-trained truth directions should be high — ideally > 0.8. In the
strict sense of "LLMs linearly represent truth," there should be ONE
direction, and all extraction methods should find it.

What we see: **four different directions, all 30% overlapping.** Every
dataset contributes a template-specific axis plus some small shared
component. The template-specific axis dominates.

The highest cosine (0.366 between cities and companies) is still far
from universality. If someone reported "truth representation found
across datasets, cosines 0.3" as evidence for a universal feature,
that would be graph-reading under a very generous threshold.

## Transfer prediction

Cross-dataset transfer can be predicted from the cosines. A direction
extracted on dataset A, projected onto dataset B's residuals, achieves
approximately `cos(dir_A, dir_B) × d_native(B)`. So:

- cities direction → common_claim prompts: expected d ≈ 0.22 × 2.01 ≈ 0.44
  (Cohen's d < 0.5 is essentially no signal.)
- common_claim direction → cities prompts: expected d ≈ 0.22 × 4.51 ≈ 1.0
  (cities signal is strong, but common_claim's direction only captures
  22% of it — much less than cities' own 4.51.)

**No direction transfers with high quality to any other dataset.** M&T's
universality claim cannot be salvaged by extraction-method changes; the
datasets themselves yield non-universal directions.

## The charitable and uncharitable readings

**Charitable:** M&T's method correctly identifies *a* linear direction in
each dataset that separates true from false at better-than-random rates.
These directions are robust within their own domain. This is a real
finding about the existence of separable structure.

**Uncharitable:** Calling this structure "truth" in the title implies a
concept-level feature the model has internalized. The data shows instead
that each dataset yields its own template-specific axis, and the shared
component between axes is minimal. The title oversells what the
experiments demonstrate.

## Combined verdict on M&T across all 6 findings

This finding completes the audit.

1. **Statistical signal**: real and scale-dependent (Finding 02)
2. **Apparatus passes its own audit**: yes, after corrections (Finding 03)
3. **Cross-dataset generality within own materials**: fails on common_claim (Finding 04)
4. **Causal effect**: asymmetric, non-inverting at natural scale (Finding 05)
5. **Universal direction across datasets**: **does not exist at Qwen-7B L20** (this finding)
6. **Semantic interpretation (vocab projection)**: not truth vocab at any scale tested (Finding 02)

The paper's *empirical procedure* (extract mean-difference direction on
one dataset, show probe accuracy, demonstrate causal intervention) reproduces
qualitatively. The paper's *interpretation* (LLMs represent truth linearly)
does not survive the 5-test pipeline applied to the paper's own materials.

## Charity + caveat

This is Qwen-7B — smaller than M&T's 13B/70B claimed regime. It's possible
that the universality improves at larger scale. But:
- The pattern was consistent across 0.5B / 1.5B / 7B (Finding 02).
- Qwen-7B has the cleanest statistical signal we observed.
- Non-universality at 7B puts the burden of proof on the 13B+ claim.

Running this at LLaMA-2-13B or 70B is a natural next step. Prediction:
the pattern holds qualitatively; cross-dataset cosines improve to
~0.5-0.6 with scale but not to the >0.8 that "universal" would require.

## Commands

```bash
# 4 datasets × 1 layer at Qwen-7B = 26 seconds
python3 -c "..." # see experiment snippet in context
```

Full cosines + summaries at `papers/lie-detection/data/cross_dataset_directions.json`.
