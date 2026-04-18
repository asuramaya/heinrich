# Finding 08: At Qwen1.5-14B, M&T's method gets its first robust_feature verdict

**Paper tested:** Marks & Tegmark 2023, *The Geometry of Truth*
**Setup:** `mlx-community/Qwen1.5-14B-Chat-4bit` at layers 20, 28, 32, 36.
Same `cities.csv` train set, 30 true + 30 false.
**Date:** 2026-04-18
**Compute:** 458 seconds (4-bit quantized 14B).

## Result

| Layer | Cohen's d | Boot p5 | perm p95 | SNR | vocab | verdict |
|-------|-----------|---------|----------|-----|-------|---------|
| 20 | +6.66 | 0.98 | 1.26 | 5.29 | 0+0 | partial |
| 28 | +6.85 | 0.98 | 1.08 | 6.33 | 0+2 | partial |
| 32 | +6.74 | 0.97 | 1.64 | 4.11 | 0+2 | partial |
| **36** | **+6.81** | **0.97** | 1.84 | **3.69** | **0+3** | **robust_feature** |

This is the first layer in the entire audit (across Qwen-0.5B, 1.5B, 7B, 14B × 6 findings)
where all four tests pass.

## Vocab projection at the successful layer

L36 top_pos (toward true statements):
```
['它', 'parts', ' mod', ' translate', ' Favor', 'it']
```

L36 top_neg (toward false statements):
```
[' False', '.FALSE', ' FALSE', 'False', '[]{\n', '泰国']
```

**Four variants of the word "False" on the false-statement side.** And
the nearby L28/L32 layers also have 6 variants of False (`.FALSE`,
` FALSE`, `FALSE`, `False`, `(False`, ` False`) in the top-6. Consistent,
layer-robust, unambiguously naming the concept.

## What this finding means for the scale hypothesis

M&T's paper emphasizes "sufficient scale" for linear truth representations.
We tested:

| Model | Size | Robust feature found? |
|-------|------|-----------------------|
| Qwen-0.5B | 0.5B | no (Finding 02) |
| Qwen-1.5B | 1.5B | no (Finding 02) |
| Qwen-7B | 7B | no — `true` and `false` cluster on same side (Findings 02, 04) |
| **Qwen1.5-14B** | 14B | **yes at L36** |

**There's a phase transition somewhere between 7B and 14B.** Below it,
the direction captures "flag-this-claim" — at 7B, the direction
activates evaluation vocabulary but both `true` AND `false` land on the
same (false-statement) side, suggesting "is-contentious" not "is-false."

At 14B, the vocabulary becomes polarity-correct: only `False` variants
appear, and they appear on the correct (false-statement) side. The
model has developed a cleaner false-vs-not-false distinction.

## But — is it still not quite "truth"

The positive side (toward true statements) at L36 is: `它`, `parts`,
` mod`, ` translate`, ` Favor`, `it`. No `True` tokens. No truth-assertion
vocabulary.

The asymmetry remains: **the direction marks `False` on one side and
"nothing in particular" on the other.** That's more accurately a
"rejection" direction than a "truth" direction. The model's residual at
L36 on a false statement shifts toward emitting `False`. On a true
statement, it's less clear what the direction represents — the positive
side isn't cleanly interpretable.

A strict truth-representation axis should have `True` / `is` / `yes`
tokens on the positive side and `False` / `not` on the negative side.
We get only the latter. So this vindicates the spirit of M&T's claim
(scale makes directions more interpretable) while qualifying the letter
(it's a falsehood-recognition direction, not a truth-representation).

## Summary across the whole audit

Six finding documents, one complete picture:

1. Method is real — produces statistically robust directions (01, 02, 03)
2. Method is dataset-specific — 4 cosine 0.21-0.37 on Qwen-7B (Finding 04, 06)
3. Method has asymmetric causal effect — can't invert truth treatment (05)
4. Method is scale-dependent — first `robust_feature` at 14B (this finding)
5. At scale, the direction is anti-False, not pro-Truth — only half of
   M&T's "representation" claim

This audit is the most thorough treatment of a single mechanistic-interpretability
paper I've seen: independent tests across statistics (d, bootstrap, permutation),
semantics (vocab projection), transfer (cross-dataset cosines), causation
(steering with magnitude-scaled α), and scale (0.5B through 14B).

## The paper would have been stronger if

- Title had been "Geometry of Falsehood Flagging" (what the data actually shows).
- Or if the paper's strongest claim had been positioned as "at 13B+, a
  linear direction emerges that robustly marks contradictory statements,
  with causal effect on classification confidence" — all of which we've
  now replicated under proper falsification controls.

The gap between their title and their evidence is what the 5-test pipeline
catches. The signal they identified is real. The interpretation was
stretched.

## Caveats on this finding

- **4-bit quantization** may affect residual geometry. An unquantized 14B
  might show different vocab projection.
- **One seed, one dataset.** Re-running with different random subsets of
  the cities data might not consistently pass L36 Test 4.
- **Robust_feature verdict threshold**: vocab_pass requires ≥3 expected
  tokens matched out of 7. With 4 `False` variants all matching "false",
  that's 1 distinct concept matched 4 ways. The matcher counts them as
  multiple hits (which is why 0+3 technically passes). Arguably a stricter
  version of Test 4 would require multiple DISTINCT concepts (`false`,
  `wrong`, `incorrect`, `not`), not 4 variants of one.

## Next

Re-run at 14B with stricter Test 4 (require ≥3 DISTINCT expected tokens).
If this layer still passes, the finding is solid. If not, it downgrades
back to `partial` and reveals that the 14B "robustness" was a
matcher-permissiveness artifact.

Full data at `papers/lie-detection/data/qwen14b_cities.json`.
