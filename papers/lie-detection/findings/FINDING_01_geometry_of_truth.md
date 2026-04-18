# Finding 01: The "Truth Direction" isn't a Truth Direction (qwen-0.5b)

**Paper tested:** Marks & Tegmark 2023, *The Geometry of Truth* (arXiv:2310.06824)
**Dataset:** their own `cities.csv` (simple factual statements), 50 true + 50 false
**Model:** Qwen/Qwen2-0.5B-Instruct, raw-mode MRI
**Date:** 2026-04-18
**Method:** 5-test falsification pipeline via `/api/replicate-probe`

## Claim under test
> "At sufficient scale, LLMs linearly represent the truth or falsehood of
> factual statements."

Extracted a mean-difference direction on their dataset, ran the 5-test pipeline.

## Per-layer verdict

| Layer | Cohen's d | boot p5 | null p95 | within-group SNR | vocab | verdict |
|-------|-----------|---------|----------|------------------|-------|---------|
| 0  | +0.89 | 0.59 | 0.39 | 1.00 | 0/16 | partial |
| 5  | +1.53 | 0.85 | 0.88 | 1.35 | 0/16 | partial |
| 10 | +3.27 | 0.94 | **1.48** | **3.39** | 0/16 | partial |
| 15 | +2.92 | 0.94 | 1.58 | **3.69** | 0/16 | partial |
| 20 | +2.50 | 0.94 | 1.57 | 2.21 | 1/16 | partial |
| 22 | +3.07 | 0.87 | 0.87 | 1.55 | 0/16 | partial |

## What the tests say

**Test 1 (bootstrap stability):** L10-L20 pass with p5=0.94 — the direction is
highly stable under prompt resampling. Not an anchor-picking artifact.

**Test 2 (null baseline):** L10-L22 pass. Random unit vectors get |d| ~0.9-1.6 on
this dataset; the extracted direction gets 2.5-3.3. Real signal.

**Test 3 (within-group control):** Only L10 and L15 pass with SNR > 2.5. Arbitrary
random splits of the 50 true prompts vs themselves still give |d| = 0.7–2.0. At L20
and L22, the method's noise floor is ~1.6, and the extracted direction's d=2.5 is
only 1.55× that. **The direction is real at L10-L15; borderline elsewhere.**

**Test 4 (vocab projection sanity):** **FAILS AT EVERY LAYER.** Projecting the direction
through lm_head onto the vocabulary, the top-scoring tokens are:

- L10 top_pos: `뀔`, `ucwords`, `)NULL`, `fkk`, `oğ`, `quals`
- L10 top_neg: ` 저장`, ` بالنسبة`, ` Donate`, ` хозяйств`, ` поск`, `Donate`
- L22 top_pos: `可分为`, `.”`, ` 찾아`, ` происходит`, ` المسلمين`, ` albeit`
- L22 top_neg: ` ucwords`, ` xbmc`, `",__`, `qrst`, `:SetPoint`, `犸`

Expected tokens for a truth/falsehood direction: `true, correct, yes, indeed, actually`
for one side; `false, wrong, incorrect, no, not` for the other. **None appear in any
layer's top-10.** The direction that statistically separates true from false factual
statements about cities does not point at the semantic concept of truth in Qwen-0.5B's
vocabulary space.

## What this means

**The claim survives as a statistical pattern; it fails as a semantic claim.**

A linear direction that separates true from false factual statements exists at
L10-L15 of Qwen-0.5B with Cohen's d ≈ 3 and bootstrap p5 = 0.94. Four of five
falsification tests pass at those layers. That's a real, reproducible feature.

What that feature *is* is the open question. The vocab projection says it's not
"truth." It's more likely:
- **Country-token coherence**: "Paris is in France" ends on a country token that
  is statistically likely for "Paris"; "Paris is in China" doesn't. The direction
  separates "country-end-token-is-coherent-with-city" from "country-end-token-is-not."
  That's a pattern-match, not a truth representation.
- **Prompt-length or surface-structure artifact** specific to this dataset.
- **A lower-level distributional feature** of the embedding space that happens to
  track factual consistency in this narrow domain.

## Where Marks & Tegmark stand

They argued for their claim at *sufficient scale*: LLaMA-2-13B and up. Qwen-0.5B is
25× smaller. It's entirely possible that at 7B+ the vocab projection names truthiness
correctly. Their paper shows causal intervention results (steering flips true→false
treatment) which we haven't yet replicated. A failure at 0.5B doesn't refute the
13B claim — but it confirms that the method is scale-dependent and that the
probe's statistical signal can arise from non-truth features below that scale.

## What to do next

1. **Replicate at Qwen-7B** (if MRI available) — test if vocab projection improves with scale.
2. **Run `heinrich_direction_steer_test`** — does steering on this L10 direction actually flip true/false treatment? If yes, the direction is causally truth-related even if vocab projection is dark. If no, the mean-diff direction and the causal-truth direction are separate axes.
3. **Length / surface control** — construct true and false statements matched on
   word length, country frequency, prompt structure. See if d collapses.
4. **Cross-dataset transfer** — does the cities-extracted direction still get d > 2
   on `common_claim.csv`? The paper's universality claim requires this.

## Commands to reproduce

```bash
# Fresh companion server with the new endpoint
heinrich companion --port 8377 &

# Download dataset (already done)
# git clone https://github.com/saprmarks/geometry-of-truth /tmp/geometry-of-truth
# cp /tmp/geometry-of-truth/datasets/cities.csv papers/lie-detection/data/

# Run the replication
python3 papers/lie-detection/run_first_replication.py
```

Full structured results at `papers/lie-detection/data/first_replication_qwen0.5b.json`.

## Apparatus caveat

The 5-test pipeline I just ran is itself novel. Before I use it to criticize
published work, I should adversarially review the apparatus — confirm the tests
do what they claim. One known weakness: Test 4 uses substring matching on top-10
vocab tokens. For a direction that would match "the truth of the matter" or "is
truly the case", my expected-tokens list (`true, yes, correct`) might miss.
The failure being identical across all layers for 2 different MRI positions
raises confidence that it's a real missing match, not a test issue. But the
apparatus deserves the same falsification discipline I'm applying to others.
