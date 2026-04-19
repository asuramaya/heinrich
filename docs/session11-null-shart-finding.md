# Null-shart hypothesis — refuted at Qwen-0.5B, all tested layers

## Claim under test

From `theory_of_sharts.tex`:

> "Null-shart ... The 47% of residual dimensions that are inert to ablation
> but still active during computation."

Operationalized: a "null shart" is a dim d with high activation magnitude
and near-zero KL when d is zeroed. The theory claims ~47% of residual
dims fit this.

## Protocol

Model: Qwen2.5-0.5B-Instruct, hidden=896, 24 layers.
Layers tested: L5, L10, L15, L20.
Prompts: 20 (7 benign, 7 harmful, 6 neutral narrative).
Per dim d ∈ [0, 896): zero at last-token residual, re-run tail
(L{n+1}..L23 + RMSNorm + lmhead), compare to baseline.
Measures:
- `mag(d)` = mean `|residual[d]|` across prompts
- `kl(d)` = mean `KL(baseline || ablated)` (nats)
- `top1_flip(d)` = fraction of prompts whose argmax changed
- `logit_l2(d)` = mean L2 distance in logit space
- Noise floor: `KL(baseline || second-baseline-run)` — measured at 0.0
  (MLX f16 tail is deterministic, so any nonzero KL is real signal).

## Result: refuted at every layer tested

Cross-layer summary:

| Layer | Spearman(mag, KL) | Log-Pearson | Null-shart strict (%) | Max KL | Max-flip dim |
|-------|--------------------|-------------|------------------------|--------|--------------|
| L5    | **0.829**          | 0.863       | 0.8%                   | 3.21   | 208 (75%) |
| L10   | **0.789**          | 0.803       | 2.3%                   | 2.62   | 208 (65%) |
| L15   | **0.767**          | 0.831       | 4.9%                   | 0.74   | 208 (40%) |
| L20   | **0.889**          | 0.902       | 1.2%                   | 1.35   | 490 (60%) |

Spearman p-values at every layer: ≤ 10⁻¹⁷⁴.

Mag–KL stay strongly rank-correlated (ρ > 0.76) at every depth. The
strict null-shart fraction (mag ≥ median AND kl ≤ p25) peaks at 4.9%
(L15), far below the theory's 47%.

### Quadrant depletion is systematic

At L10 — detailed example — under independence, a quadrant spanning
the top-X% of mag and bottom-Y% of KL should contain `(1-X) × Y × N`
dims. Observed vs expected:

| mag cut | kl cut | Observed | Expected under H0 | Enrichment |
|---------|--------|----------|-------------------|------------|
| ≥ p50 | ≤ p50 | 89 | 224 | 0.40× |
| ≥ p60 | ≤ p40 | 33 | 143 | 0.23× |
| ≥ p75 | ≤ p25 | 12 | 56 | 0.21× |
| ≥ p90 | ≤ p10 | 2 | 9 | 0.22× |

Every cut shows 0.2–0.4× enrichment. The null-shart quadrant is
actively DEPLETED relative to independence. The residual space avoids
the (high mag, low KL) corner by a 2-5× margin.

### Permutation test at L10

With mag fixed and KL shuffled across dims 1000 times (H0: mag and KL
are independent), the expected count in the strict null-shart quadrant
is 112.1 ± 6.4 dims. Observed: 21 dims.
**z = −14.3, p < 0.001**. We observe *fewer* null-shart dims than chance.

## What actually governs Qwen-0.5B's residual at L5-L20

A few high-magnitude dims carry the ablation effect. At L5 and L10 the
same dim (208) flips the top-1 token on 65–75% of prompts. At L15 dim
208 still dominates (40%). Only at L20 does a different dim (490) take
over (60% flip rate).

| Layer | top-1 dim by flip rate | flip rate | KL | mag |
|-------|-----------------------|-----------|----|----|
| L5    | 208                   | 0.75      | 3.21 | 4.54 |
| L10   | 208                   | 0.65      | 2.62 | 4.68 |
| L15   | 208                   | 0.40      | 0.74 | 9.30 |
| L20   | 490                   | 0.60      | 1.35 | 34.08 |

A single residual dimension is load-bearing for decision-making across
most of the network's depth. This is the opposite of the null-shart
prediction: information is concentrated, not diffuse.

## What dim 208 is NOT

I initially projected the unit vector e_208 through the final
RMSNorm + lmhead and got top tokens that were all Chinese:
`的带领, 的钱, 的话, 力还是, 的男人, 的女儿, 的第一, 的安全`.

I overreached. **A causal steering test falsifies the "anti-Chinese
axis" interpretation.** Clamping dim 208 to ±5 or +10 during generation:

| Prompt | Baseline | dim208=+5 | dim208=+10 |
|--------|----------|-----------|------------|
| "The capital of France is" | "Paris. It is the largest city..." | "Paris.,. is the capital of France.?." | "is called,,,,,,,,,," |
| "Once upon a time" | "there was a young man named John" | "there was a little girl named Sarah" | "in a faraway land,, () was the king" |
| "The quick brown fox" | "jumps over the lazy dog" | "jumps over the lazy dog, is the first..." | "jumps over the\n(),,,\n,,," |

**Clamping dim 208 produces degenerate English, not Chinese.** At +5
the text is lightly perturbed. At +10 it breaks into comma-soup. Chinese
tokens never appear.

The lmhead-unit-vector projection is a descriptive artifact of the
RMSNorm + unembedding geometry, not a causal description of the dim's
function. What the projection says a dim "pushes" when considered in
isolation is not what the dim actually does in the full circuit. The
RMSNorm scales the residual relative to its *other* dims; if dim 208
is abnormally large, RMSNorm scales everything down — including the
lmhead's access to all other dims.

**Dim 208 is an RMSNorm-coupled magnitude anchor, not a language axis.**
Its role is to keep the residual's norm in a stable operating regime.
Zeroing it breaks the norm assumption; clamping it to an extreme value
breaks the regime. Either perturbation degrades output.

This is a separate finding from the null-shart refutation, but it came
from the same data and illustrates the shallow-optimism failure mode
the project's session logs repeatedly warn about: lmhead-projection
descriptive statistics are NOT causal claims, and mistaking them for
causal claims is the easiest way to produce false interpretability.

## Caveats on the main finding

1. **Single model family.** Qwen-0.5B is small; dense 7B and MoE models
   could have different dynamics. The 47% figure in the theory is not
   tied to a specific measurement, so we don't know what model it was
   meant to describe.
2. **Last-position only.** Earlier positions' dims aren't tested.
3. **Output-measured.** "KL on next-token distribution" catches effects
   visible at the final logit. A dim could carry information read by
   attention queries at later layers without affecting the final logit.
   A stronger test would measure whether *any* later intermediate state
   differs, not just the final output. That would be a much more
   expensive sweep but would close the null-shart loophole.

## Implications

The null-shart category as written (≈47% active-but-inert dims) is not
supported at Qwen-0.5B. What exists instead:

- A **strong monotone relation** between magnitude and causal effect
  (Spearman > 0.76 at every depth).
- **Power-law concentration** of causal effect in a few dims — at L5,
  zeroing dim 208 alone explains more KL than the next 500 dims combined.
- **No "inert carrier" reservoir** at anywhere near 47% of the hidden
  dimension. At strictest cuts, null-shart candidates are ≤ 5% of dims
  and the quadrant is *depleted* not *enriched* vs independence.

A narrower claim the data COULD support:
- **~1–5% of dims are high-magnitude but low-KL** — those dims exist at
  every layer. They may carry signal read by downstream components that
  don't feed back to logits. This would be a quantitative null-shart
  claim (few dims, not half) and remains testable by layer-internal
  probes (not just output-measured KL).

## What's next

- **Intermediate-state probes:** repeat the ablation but measure
  per-layer residual differences, not just final KL. If "high-mag,
  low-output-KL" dims have nonzero effect on some intermediate layer's
  state, they're null sharts of the narrower kind.
- **Bigger model:** repeat at 3B or 7B. If the null-shart fraction
  grows with dimensionality, the theory's 47% might describe the
  larger-model regime. If it stays 1-5%, the category is just wrong.
- **Theory update:** the paper's 47% citation is undocumented. Either
  it traces to a measurement on a specific model (worth including that
  citation and caveating scope) or it's theoretical (in which case the
  theory paper needs revision to present this empirical counterexample).

## Files

- `/tmp/heinrich/null_shart_L05.json`
- `/tmp/heinrich/null_shart_L10.json`
- `/tmp/heinrich/null_shart_L15.json`
- `/tmp/heinrich/null_shart_L20.json`
- `/tmp/heinrich/null_shart_depth_summary.json`
- `/Users/asuramaya/Code/heinrich/src/heinrich/discover/null_shart.py` — module
- `/Users/asuramaya/Code/heinrich/docs/null-shart-protocol.md` — protocol
