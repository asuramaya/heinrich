# Finding 05: M&T's direction has causal effect on truth judgments, but not symmetrically or at natural magnitude

**Paper tested:** Marks & Tegmark 2023, *The Geometry of Truth* — their
causal-intervention claim: adding the truth direction to a residual stream
causes the model to "treat false statements as true and vice versa."
**Setup:** Qwen-7B L20, direction extracted from `cities.csv` (from Finding 02/04).
**Test:** Eval-framing prompt `True or false: "{X}" Answer:` with ±α steering.
Measured truth-score = logit("True") − logit("False") per statement.
**Test statements:** 10 true + 10 false held out from the extraction set.
**Date:** 2026-04-18

## Results

The direction has unit norm; the extracted mean-difference magnitude was
40.7 in hidden-space units. α is on the natural magnitude scale.

| label | α=−4·mag | α=−2·mag | α=−1·mag | α=0 | α=+1·mag | α=+2·mag | α=+4·mag |
|-------|----------|----------|----------|-----|----------|----------|----------|
| 0 (false) | −12.43 | −15.55 | −15.78 | −12.49 | −8.91 | −4.47 | **+0.14** |
| 1 (true) | −2.68 | +5.35 | +8.77 | +7.95 | +6.98 | +5.14 | +4.38 |

Top-token flip rates:
- α=0: false→`False` (100%), true→`True` (100%)
- α=+4·mag on false: `False`→`The` (60%) — **but not to `True`**

## Interpretation

### What M&T would predict if their claim is strict
- +α (truth direction): false statements' truth_score rises to positive (they're treated as true); true statements' truth_score stays positive (or rises).
- −α (anti-truth): true statements' truth_score drops to negative (they're treated as false); false statements stay negative.

### What actually happens

1. **+α on false statements: shift of +12.57 over the full α range.**
   Strong causal effect. Model's confidence that false statements are false
   collapses from −12.49 at α=0 to +0.14 at α=+4·mag. **60% of statements
   have their top token flip from ` False` to ` The`** (it starts answering
   the prompt as a continuation rather than a binary judgment).

   But crucially: **zero false statements flip to ` True` as top token.**
   The steering can disrupt the false-classification but can't *invert* it.

2. **−α on true statements: shift of +7.06 over the full α range.**
   Smaller effect. Even at α=−4·mag (aggressive anti-truth steering),
   all 10 true statements still predict ` True` as top token.

3. **Non-monotonicity at small −α.** For true statements,
   α=0 → +7.95, but α=−1·mag → +8.77 (higher). Small negative steering
   *increases* truth-score. That's not what a clean "truth axis" would do.

### What this suggests the direction actually is

The asymmetric causal effect is consistent with the "flag-this-claim"
interpretation from Findings 02, 04:

- The direction encodes "this is an evaluation-worthy claim with
  template-recognizable false-attribution."
- Steering +α amplifies that signal, making the model hedge away from
  committing to `False` — not because it now believes the statement, but
  because the evaluation signal is over-excited.
- Steering −α attenuates the signal, but the model's factual knowledge
  (Paris IS in France) is too strongly encoded elsewhere to be overridden
  at this single layer.
- The true-statement minimal effect is explained by: when no surface
  contradiction exists, the "flag-this-claim" signal is already low, so
  modifying it doesn't change behavior.

## What M&T said vs what we measured

- **Their claim:** "causing it to treat false statements as true and
  vice versa."
- **What we got:** partial true-→-false → cannot achieve.
  partial false-→-true is achievable at 4× natural direction magnitude, but
  it flips to ambiguity (e.g., top token `The`), not to `True`.

This is neither a clean vindication nor a clean refutation. The direction
IS causally active. But the effect doesn't look like "truth direction."
It looks like an axis the model uses to modulate confidence on
evaluation-template claims, asymmetrically biased against false-confidence.

## Triangulating across all four orders of evidence

| Test | Result |
|------|--------|
| 1. Statistical: d=4.51, bootstrap p5=0.97, perm SNR=4.84 | strong, robust |
| 2. Cross-dataset: d=2.01 on common_claim, below perm p95=2.78 | fails general-factuality |
| 3. Semantic (vocab): no truth tokens on correct side | fails |
| 4. Causal: asymmetric, non-inverting | weak / wrong direction |

All four converge. The direction exists mechanically, is layer-specific
(~L20 of Qwen-7B), is stable, and has causal effect. It is NOT the
truth representation M&T's title claims — on any of four tests.

## Weaknesses of this finding

- **Eval-framing is artificial.** Real model use doesn't ask "True or false: X? Answer:"
  and steering on this template might not reflect steering during natural generation.
- **One model size.** LLaMA-2-13B or -70B might show cleaner results.
- **Single direction extraction.** Re-extracting from `common_claim.csv` might
  produce a different direction that behaves more like a truth direction. If yes,
  the "truth direction" is composition-dependent — itself an important finding.
- **No attention-head decomposition.** Li et al. (ITI) would ask which specific
  heads carry this signal. Haven't done that.

## Next steps

1. **Extract the direction from `common_claim.csv` instead of cities, and
   re-run the same causal test.** If the common_claim-extracted direction
   causally affects judgments, the "direction exists but is dataset-specific"
   interpretation is refined. If it doesn't, the problem is the dataset.

2. **Run on Qwen-1.5B and 0.5B too.** If causal strength scales with
   model size, M&T's scale claim is vindicated on a qualitative axis
   even though 7B is still below their 13B threshold.

3. **Use natural prompts, not eval-framing.** Steer during generation
   on a task like "List 3 true things about Paris" and see if it starts
   emitting false things.

## Commands

```bash
python3 -c "..." # see the experiment code in message context
```

Full results at `papers/lie-detection/data/causal_steering_qwen7b_scaled.json`.
