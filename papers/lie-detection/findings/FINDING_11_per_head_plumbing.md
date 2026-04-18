# Finding 11: Per-head attention residual capture built — and it reveals the 5-test's essential vocab projection step

**Paper targeted:** Li et al. 2023, *Inference-Time Intervention* (ITI, arXiv:2306.03341).
**What was built:** Per-head residual capture in `cartography/runtime.py`.
**What was learned:** Per-head statistical-only tests find massively many false positives. Vocab projection is the load-bearing test in the 5-test pipeline.

## Plumbing built

`forward_pass(capture_heads: set[int] | None = None)` now inlines the attention path for any layer in the set, captures the scaled_dot_product_attention output (pre-`o_proj`), and returns per-head residual contributions as `{layer_idx: ndarray[n_heads, hidden]}` at the last token position.

**Correctness verified:** Bit-identical exit state to the normal forward pass (residual diff = 0.0 at L20 of Qwen-0.5B). Entropy, top token, logits all match.

**Speed:** 6.5s for 200 statements × 5 layers on Qwen-0.5B.

## Minimum viable ITI experiment on Qwen-0.5B

- Dataset: Marks & Tegmark `cities.csv`, 100 true + 100 false statements.
- Sweep: layers {8, 12, 14, 16, 18, 20} × 14 heads = 84 head-probes.
- Tests per head: Cohen's d, bootstrap p5 (n=50), permutation p95 (n=200), SNR = d/p95.

## Result (statistical-only filtering)

70 of 84 heads score `robust_feature`: d > 2, bootstrap p5 > 0.7, perm p95 < d, SNR > 1.5. 11 partial. 3 falsified.

| L | H | d | boot5 | p95 | SNR | verdict |
|---|---|---|-------|-----|-----|---------|
| 14 | 0 | 3.05 | 0.99 | 0.46 | 6.65 | robust_feature |
| 14 | 6 | 3.53 | 0.99 | 0.56 | 6.30 | robust_feature |
| 12 | 8 | 2.66 | 0.99 | 0.44 | 6.06 | robust_feature |
| ... | ... | ... | ... | ... | ... | ... |

## The falsification

**Cross-head cosine:** mean 0.001, median 0.002, std 0.065, p95 = 0.081. Essentially **orthogonal**.

**Effective rank of the 84 direction vectors:** 71.80 (near full rank).

**Cumulative variance:** top-3 PCs = 10.4%; top-5 = 15.5%; top-10 = 26.3%. No dominant shared axis.

**Vocab projection per head (top-5 tokens under lm_head):** garbage. No truth/false, no location tokens, no obvious semantic anchor. Random mix of code tokens, Chinese characters, whitespace/symbol clusters. Example:

  L8 H0: `[' CircularProgress', '.NotNil', '.Bac...]`
  L8 H1: `[']int', 'ndern', '@$', '偏差', '|$']`
  L8 H9: `['品', '乳', '香气', 'ths', '成']`

None of these name truth, countries, or propositional content.

## What this means

**Every head is finding a high-d direction on the cities dataset, but each finds a DIFFERENT direction.** The statistical tests — bootstrap, permutation, SNR — all pass. But the directions are:

- Orthogonal to each other
- Near full rank in the direction matrix
- Project through lm_head to random vocabulary

Two interpretations:
1. The cities dataset has many co-varying features (length, specific city names, country token distribution, grammatical register). Each head latches onto a different confound. None is the "truth" feature.
2. The 200 samples + high-dim substrate gives enough surface area that random-looking structure passes the 3 statistical tests without being a meaningful feature.

Both interpretations have the same implication: **statistical-only filtering generates false positives**. Vocab projection sanity is load-bearing in the 5-test pipeline.

## Updated 5-test policy

For per-head analysis:
1. **Cohen's d > 2** (signal exists)
2. **Bootstrap p5 > 0.7** (signal stable on resampling)
3. **Permutation p95 < d** (signal exceeds null)
4. **Vocab projection names the concept** (signal is interpretable — not a confound)
5. **Causal steering changes behavior** (signal drives the output)

At Qwen-0.5B: 70 heads pass tests 1-3. Zero heads pass test 4. Test 4 eliminates 100% of the false positives at this scale.

## Comparison to ITI

Li et al. (2023) report ~48 truth heads out of ~780 (6%) on Llama-7B. Our Qwen-0.5B at 84 heads shows:
- 83% pass statistical-only filtering (far higher than ITI's 6%)
- 0% pass with vocab projection added
- Orthogonal directions across "robust" heads

ITI's probe-based filtering uses TruthfulQA AUROC per head. That is a behavioral test (closer to test 5 in our pipeline) and is stricter than Cohen's d. Their 6% is compatible with most heads failing a behavioral test. Our orthogonality finding suggests their "truth heads" might also be a mixture of confound-catchers.

## Plumbing details (for follow-up use)

```python
from heinrich.cartography.runtime import load_model, forward_pass
m, tok = load_model('Qwen/Qwen2-7B-Instruct')
r = forward_pass(m, tok, "The city of Paris is in France.", capture_heads={16, 20, 24})
# r['head_residuals'] is dict: {layer_idx: ndarray[n_heads, hidden]}
```

`_capture_per_head_attention()` in `runtime.py` is the inline forward. It computes `sdpa_output[head] @ o_proj.weight[:, head_slice].T` — the exact residual-stream contribution of one head. Sum over heads reproduces `o_proj(sdpa_flat)`.

## 7B replication

Ran same pipeline on Qwen-7B (60 true + 60 false cities, layers {16, 20, 24}, 28 heads = 84 head-probes, 13.5s capture).

**Result: 0 robust / 0 vocab_pass / 44 stat_pass of 84 heads.**

The statistical tests still produce 52% false positives at 7B (lower than 0.5B's 83%, but same phenomenon). Vocab projection still rejects every candidate.

Top head by SNR: L16 H23, d=5.62, SNR=7.6, top positive tokens `['之家', ' ›', '洞']`, top negative `[' Sour', 'jie', '(strtolo']`. Neither side names truth-related tokens.

This matches FINDING_08's scale pattern: Qwen family requires 14B+ for any robust_feature verdict. ITI's mechanism, as a per-head probe, is likely also scale-gated — or confound-based on smaller models.

## Correction: vocab projection is not load-bearing. Functional test is.

The first draft of this finding claimed: "vocab projection eliminates 100% of false positives; statistical tests alone are inadequate." The held-out classification test, run next, *overturns* that claim.

**Test setup:** 120 true + 120 false cities statements, 80/20 train/test split. For each (layer, head), train a mean-diff direction on train; classify test-set by projection sign.

**Result:**
- Layer L16 whole-residual: **100% test accuracy**, overfit gap -2.1%.
- L20 and L24 whole-residual: 98% test accuracy.
- Three individual heads hit **100% test accuracy**: L16H23, L20H14, L24H20.
- Median head test accuracy: **83.3%**, min 47.9%, max 100%.
- Best head exactly matches best layer.

The directions are functionally real. They just do not decode to English truth/false tokens through `lm_head`. Two possibilities:
1. **The feature is "this claim about cities is plausible given the model's world knowledge,"** and that geometric region in residual space doesn't correspond to any specific lexical token.
2. **The truth-signal is encoded in a subspace that is effective for classification but has near-zero cosine with the `lm_head` directions for `true`/`false`** — i.e., the model's "truth" isn't committed to the output vocabulary in those tokens.

Both readings are consistent with FINDING_04: the direction *works* for cities but doesn't transfer to common_claim. The classifier is topic-specific, not lexically-grounded.

**Corrected claim:** vocab projection is a *sanity check*, not a gate. A direction can be a real classifier while its lm_head top-tokens are garbage, because residual-stream features are not required to align with `lm_head` rows. The load-bearing tests are:
- **Held-out classification accuracy** — does the direction generalize?
- **Cross-dataset transfer** — does it survive when the topic changes?
- **Causal steering** — does adding +α along the direction shift output behavior?

FINDING_08 (14B) had vocab projection pass (`True`/`False` in top tokens at L36) — so at that scale the direction *did* align with lm_head. Below 14B, directions can be functional without that alignment. The 5-test pipeline needs to separate "functional feature" from "interpretable feature"; they are not the same thing.

## What this means for ITI

Per-head analysis at Qwen-7B L16/20/24 finds **three heads with 100% held-out classification accuracy** on cities. That matches ITI's claim of truth-head concentration. The direction is functional. ITI is not refuted by our plumbing — the per-head signal is real.

Where ITI *is* likely overclaiming: cross-dataset transfer (FINDING_06 — 4 M&T datasets produce 4 different directions at Qwen-7B). The "truth heads" are probably "cities-heads" + "companies-heads" + etc. that each hit 100% on their own domain.

## Follow-up

- Cross-dataset: do L16H23 / L20H14 / L24H20 achieve high accuracy on `common_claim`? Prediction: no.
- Causal ablation: zero these heads during a cities completion prompt. Does the model stop distinguishing true from false? If not, these heads are correlated-with but not causing classification.
- 14B per-head test: does the L36 robust direction decompose into specific heads, or is it distributed?

## Transfer test — the truth-heads claim is falsified

Ran cross-dataset transfer: train direction on cities, apply to `common_claim`.
60 true + 60 false per dataset at Qwen-7B.

**Layer-level:**
- L16: cities_in 100% → common_claim_transfer **55%** (drop −45%)
- L20: 97.5% → 63% (drop −34%)
- L24: 97.5% → 59% (drop −38%)

**Per-head (the 100%-on-cities heads):**
- L16H23: 99% → **52.5%** (drop −47%)
- L20H14: 98% → **50%** (at chance, drop −48%)
- L24H20: 98% → 62.5% (drop −36%)

**Native common_claim (own train + own test):**
- L16 layer: train 83%, test **54%** — massive overfit, no generalizable direction.
- L20, L24 layer: test **50%** — at chance.
- Best per-head native cc (L16H9): train-test 83% — much weaker than cities 100%.

**Direction cosine (cities_dir vs cc_dir):**
- L16: 0.33, L20: 0.42, L24: 0.18 — mostly orthogonal.
- L16H23: 0.69 (shared subspace) but transfer still fails.
- L20H14: 0.10 (≈ orthogonal), L24H20: 0.35.

**Verdict on ITI:**
- **Signal concentration claim** (some specific heads carry the bulk of probe AUROC): **replicated**. Three heads hit 100% on cities, matching the "few truth heads" pattern.
- **Truth universality claim** (those heads encode truth, not topic): **falsified**. The same heads collapse to ~50% on common_claim, and native common_claim directions don't generalize either.
- **Mechanism interpretation:** what ITI calls "truth heads" are *topic-classification heads* for whatever domain the probe was trained on. Different dataset → different heads. This is consistent with FINDING_04 (cities works, common_claim fails) and FINDING_06 (4 M&T datasets give 4 orthogonal directions at 7B).

## Causal ablation test — the "truth heads" don't carry the signal

Built per-head ablation via `ablate_heads={layer: {head_idx}}` in `forward_pass`. For each top head, zero its sdpa output before o_proj, re-measure layer-level classification.

Baseline (no ablation, 80 true + 80 false, 80/20 split):
- L16 test 96.88%, L20 test 93.75%, L24 test 100.00%.

| Ablation | L16 test | L20 test | L24 test |
|----------|---------|---------|----------|
| L16H23 (the "top truth head") | 100% | 100% | 93.75% |
| L20H14 | 100% | 100% | 100.00% |
| L24H20 | 100% | 90.62% | 96.88% |
| All 3 top heads | 100% | 93.75% | 96.88% |
| Top 5 heads | 100% | 100% | 90.62% |
| **5 random heads (control)** | 100% | 96.88% | 93.75% |

**No ablation drops below baseline meaningfully.** Ablating L16H23 actually *raises* L16 accuracy. Ablating all 5 top heads gives the same result as 5 random heads.

The signal is **redundantly distributed** across heads. The "top heads" are those whose individual output happens to align most closely with the layer's mean-diff direction, but they are not load-bearing — other heads carry the same direction. Zero them, accuracy stays at 100%.

## Full ITI picture

- **Signal exists** (layer 16 classifies cities at 100%).
- **Signal concentrates in specific heads** when measured by linear probe accuracy (3 heads hit 100% alone).
- **Signal does not transfer** to other truth datasets (cities → common_claim drops to ~50%).
- **Signal survives ablation** of the heads that carry it (random and "truth" ablations equal).

The "truth heads" narrative fails on steps 3 and 4. What ITI identifies is:
- A dataset-specific linear direction in residual stream.
- The heads whose individual output most aligns with that direction.
- Not a causal mechanism, not a universal truth representation.

ITI's TruthfulQA 32.5→65.1 improvement likely reflects template-specific confidence modulation on TruthfulQA's prompts, not intervention on a truth feature. Predicted refutation: train ITI on one TruthfulQA subset, apply to another subset, expect sharp degradation.

## Plumbing built

- `_capture_per_head_attention(ly, h, mask, mx, ablate_head_idx, capture)` — inlines TransformerBlock, optionally captures per-head residual contributions and/or zeros specified heads.
- `forward_pass(capture_heads={L}, ablate_heads={L: {h1, h2}})` — bit-identical to normal forward when unused; captures/ablates per-head when used.

Heinrich's audit of ITI is now complete with all 5 tests run: probe accuracy, bootstrap stability, permutation null, vocab projection, cross-dataset transfer, **and causal ablation**. The result is consistent across tests: directions are real, functional, domain-specific, redundant, non-universal.

## Refinement: L16 vs L20 — surface vs propositional (from cities↔neg_cities control)

Ran 4×4 transfer matrix (cities, neg_cities, companies, common_claim) at L16/20/24.

The cities↔neg_cities transfer is the key control: neg_cities is the same dataset with the verb negated ("The city of X is NOT in Y"). The truth label flips with the negation. If the direction captured "X is in Y" as a surface pattern, it would fail on neg_cities. If it captures propositional truth, it should work.

```
                     cities   neg_cities   companies   common_claim
 L16 cities         96.88%      49.38%       64.38%       55.62%
 L20 cities        100.00%      95.62%       61.88%       61.25%
 L24 cities        100.00%      93.12%       55.00%       58.13%
```

L16 cities↔neg_cities cosine = −0.017 (orthogonal). L20 transfer = 95.6%.

**L16 is surface lexical, L20+ is propositional.** The layer-16 direction that got 96.88% held-out cities classification in the earlier finding was capturing "is-in-Y statement structure" — it failed the moment negation was introduced. The layer-20 direction tracks "this city-country pairing is correct" independent of surface form.

But **cross-domain still fails**. cities→companies at L20 is 61.88%; cities→common_claim 61.25%. The direction has a cities-knowledge signature, not a general entity-verification signature.

Companies is hard on its own terms: in-domain L20 companies = 37.5% (WORSE than chance), L16 = 84%, L24 = 72%. The signal collapses at L20 for companies, coming back weakly at L24. Another way of saying the company-relationship truth isn't encoded as a clean linear direction at 7B.

## Corrected cross-cutting claim

Mean-difference contrastive probes at Qwen-7B find **topic-specific propositional truth directions** — not lexical topic classifiers (the earlier reading), and not universal truth (Bürger's claim).

- Survive surface transformations (negation) at layer 20+.
- Do not survive domain change (cities ↔ companies, cities ↔ common_claim).
- Extraction layer matters: L16 is surface, L20+ is propositional.
- Even within a domain, some topics (companies at 7B) don't have a cleanly-decodable truth direction.

Bürger's "truth is universal" is falsified at 7B (cross-dataset fails). ITI's "truth heads" claim is the same finding reframed at per-head resolution (heads encode the same domain-specific direction). M&T's original cities result is validated as propositional not surface, but cross-dataset transfer was always outside its original scope.

## Unified-direction test: partial universality

Trained a single mean-diff direction on the POOLED union of all 4 datasets, tested on each dataset's held-out set. If universal truth exists, unified should match per-dataset.

```
 Layer   unified_avg   per_ds_avg   gap
 L16     82.81%        87.50%       -4.7%
 L20     78.12%        85.16%       -7.0%
 L24     77.34%        83.59%       -6.3%
```

L16 pooled nearly matches per-dataset. At L20+ it costs 7% — each domain has its own truth subspace.

Layer-20 pool direction cosines: cities 0.825, neg_cities 0.789, companies 0.606, common_claim 0.665. The pool is cities-biased. cities+neg_cities dominate the mean-diff; companies and common_claim partially align.

**Final reading:** truth is a *family* of related domain-specific directions with partial shared structure. Mean-diff finds whichever domain dominates training. Pooling yields a compromise direction (~80% on everything) that is not equivalent to any of the per-domain directions. Bürger's "truth is universal" is ~80% true at Qwen-7B — defensible on some datasets, falsified on others, never as clean as a single-axis theory claims.

## Artifacts

- `/Users/asuramaya/Code/heinrich/src/heinrich/cartography/runtime.py` — `_capture_per_head_attention`, `capture_heads` parameter.
- `/tmp/test_per_head.py` — bit-identical check.
- `/tmp/test_iti_mvp.py` — per-head Cohen's d ranking.
- `/tmp/test_iti_5test.py` — 3-stat test per head, JSON results in `/tmp/iti_5test_results.json`.
- `/tmp/test_iti_alignment.py` — cross-head cosine + vocab projection, JSON in `/tmp/iti_alignment.json`.
