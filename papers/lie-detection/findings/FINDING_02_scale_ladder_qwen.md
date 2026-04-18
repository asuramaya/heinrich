# Finding 02: M&T's truth direction passes stats at 7B but isn't a truth direction

**Paper tested:** Marks & Tegmark 2023, *The Geometry of Truth*
**Dataset:** `cities.csv` from saprmarks/geometry-of-truth (30 true + 30 false)
**Models:** Qwen2-0.5B-Instruct, Qwen2-1.5B-Instruct, Qwen2-7B-Instruct
**Date:** 2026-04-18
**Method:** 5-test falsification pipeline via `/api/replicate-probe`

## Scale ladder

| Model | Layer | Cohen's d | Boot p5 | Null p95 | SNR | Vocab match |
|-------|-------|-----------|---------|----------|-----|-------------|
| Qwen-0.5B | 10 | +3.35 | 0.92 | 1.31 | 1.61 | 0+0 |
| Qwen-0.5B | 15 | +3.24 | 0.91 | 1.67 | 2.45 | 0+0 |
| Qwen-0.5B | 22 | +3.47 | 0.82 | 0.98 | 2.01 | 0+0 |
| Qwen-1.5B | 14 | +4.02 | 0.94 | 2.06 | 2.81 | 0+0 |
| Qwen-1.5B | 20 | +3.41 | 0.95 | 2.12 | 3.03 | 0+2 |
| Qwen-1.5B | 26 | +2.57 | 0.89 | 1.26 | 1.12 | 0+0 |
| **Qwen-7B** | **20** | **+4.51** | **0.95** | **2.39** | **4.62** | **0+2** |

Signal grows monotonically with scale (d: 3.5→4.0→4.5; SNR: 2.5→3.0→4.6).
All three models pass bootstrap, null, and within-group control at their best layer.
M&T's empirical claim about linear separability of true/false at scale is supported.

## What the 7B vocab projection reveals

Top vocab tokens at Qwen-7B L20, projected through `lm_head`:

- **Positive side** (direction points toward TRUE statements):
  ` Fälle` (German: "cases"), `\texplicit`, `_Tab`, ` beğen` (Turkish), ` Gründe` (German: "reasons"), `に行った` (Japanese: "went to")
- **Negative side** (direction points toward FALSE statements):
  **` 错`** (Chinese: "wrong"), ` �`, **` false`**, **` true`**, **` FALSE`**, **` TRUE`**

**Both `true` and `false` appear on the NEGATIVE side.** This is the key
observation. If the extracted direction were a "truth representation"
in M&T's sense, `true` should be favored at true-statement residuals
and `false` should be favored at false-statement residuals. They should
sit on opposite sides of the axis.

Instead, both truth-evaluation tokens cluster together on the side of
*false* statements. The direction points toward "residual states that
activate evaluation vocabulary" — i.e., the model's state when encountering
a claim it flags as contentious.

## Interpretation

The extracted direction is not `truth ↔ falsehood`. It's more like
`unmarked ↔ flagged-for-evaluation`. When Qwen-7B sees "Paris is in France"
it has nothing to contradict — no evaluation vocabulary is salient. When
it sees "Paris is in China" it prepares to produce "false/wrong/no/错" —
all evaluation tokens, on the same side of the axis.

This is a real and meaningful direction. It's just not the direction
M&T's title implies. "Geometry of flag-this-claim" ≠ "Geometry of truth."

The vocab projection test (#4 of the 5-test pipeline) caught the distinction
that Cohen's d, bootstrap stability, null baseline, and within-group SNR
all missed. All four of those pass at 7B. Only Test 4's semantic sanity
check fails. The pipeline worked.

## M&T's response (what they'd say)

They would (fairly) point out that:
1. Their causal intervention results show you can flip the model's truth-treatment
   by adding/subtracting the direction. We haven't run this test yet.
2. Their 13B/70B results may clean up the vocab projection further. 7B ≠ 13B.
3. The direction's behavioral effects matter more than logit-lens interpretations.

These are reasonable. But: if you can causally flip the model's truth-treatment
by adding a "flag this claim for evaluation" direction, maybe what you're
causally controlling is "does the model engage its evaluation machinery,"
not "does the model believe this is true." Those two are different and
only under certain conditions do they align.

## What to do next

1. **Causal steering.** Does adding −α·d at L20 flip Qwen-7B's truth-treatment?
   If yes, the direction is causally truth-related despite vocab ambiguity.
   If it instead makes the model become more evaluative / hedging, that
   supports the "evaluation-flag" interpretation.

2. **Cross-dataset transfer.** Extract direction on `cities`, test d on
   `common_claim.csv`, `companies.csv`. If d stays high across unrelated
   domains, the direction is domain-general. If it's cities-specific,
   it was picking up country-token dependencies.

3. **Negated statements.** `neg_cities.csv` has sentences of the form
   "Paris is *not* in China" (label: 1, true). Does the direction still
   point toward these? If yes, it's tracking truth. If it flips to the
   false side because they share surface structure with false statements,
   it's tracking surface features.

4. **Multi-position capture at +1, +2, +3.** Does the direction sharpen at
   generation positions after the statement? If the model only commits to
   a truth judgment after outputting "That's", the direction at +1 should
   look cleaner than at prompt-end.

## Caveats

- One dataset (cities). Generality unknown without the cross-dataset tests.
- One tokenizer (Qwen family). The 7B model uses the same tokenizer as 0.5B
  so vocab-projection comparisons are fair across sizes, but different
  tokenizer families (LLaMA, Mistral) would surface different top tokens.
- Substring vocab-matcher missed `true`/`false` in top_neg because my
  expected_pos list contained those tokens and expected_neg had `untrue`
  which substring-matched `true`. My test 4 needs more care about
  polarity. Manual inspection saved this one.

## Update 2026-04-18: apparatus correction

Initial 6-layer sweep on Qwen-7B reported L16 and L24 as `robust_feature`
with apparent 7/7 vocab matches. **Inspection revealed the matcher was
broken**: `"" in e` is True for any string, so whitespace tokens (like `\n`)
that stripped to `""` spuriously matched every expected token.

Fixed the matcher to require `len(g) >= 2` on the got side. After the fix:

| Layer | d | SNR | vocab | verdict |
|-------|---|-----|-------|---------|
| 4  | +1.13 | 0.52 | 0+0 | partial |
| 10 | +3.24 | 2.45 | 0+0 | partial |
| 16 | +5.88 | 2.94 | 0+0 | partial |
| 20 | +4.51 | 4.62 | 0+2 | partial |
| 24 | +3.72 | 2.90 | 0+0 | partial |
| 26 | +3.82 | 2.71 | 0+0 | partial |

**No layer of Qwen-7B passes all 4 tests.** L16 has the highest Cohen's d
(5.88) with strong bootstrap (p5=0.97) and SNR (2.94), but its top vocab
tokens don't include any form of true/false/correct/wrong. L20 remains
the closest to "robust" — it's the only layer where truth-evaluation
tokens appear at all, and they cluster on the SAME side (the false-statement
side), which supports the "flag-this-claim" interpretation, not a "truth
representation" interpretation.

The original finding stands, strengthened: **scale improves statistical
signal but the vocab-semantic gap persists from 0.5B through 7B.**

## Apparatus performance note

Initial single-layer replicate-probe runs were ~10 min per layer on Qwen-7B
(60 forward passes × ~8s each, plus load). A one-pass multilayer capture
endpoint (`/api/replicate-probe-multilayer`) uses ONE forward pass per
prompt and slices out all requested layers from the cached residuals.

**Speedup: 6-layer Qwen-7B sweep went from ~60 min → 13 seconds.**

The old per-layer endpoint also had a double-capture bug: `_replicate_probe`
captured residuals via `_behavioral_direction`, then re-captured them for
the null baseline. Multilayer refactor eliminates it.

## Commands

```bash
# Full ladder run (0.5B + 1.5B, ~3 minutes):
python3 papers/lie-detection/run_scale_ladder.py

# 7B at L20 took ~10 minutes on Apple Silicon.
```

Full results at `papers/lie-detection/data/scale_ladder.json` +
`papers/lie-detection/data/qwen7b_L20.json`.
