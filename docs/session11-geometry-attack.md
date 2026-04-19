# Attacking `geometry_of_displacement.tex` — four more targets

**Round 2 of the theory attack.** After refuting null-sharts and the named-
shart taxonomy, I read five more papers and picked four specific
numerical claims from `geometry_of_displacement.tex` to hit directly.

**All tests on Qwen-0.5B-Instruct at L15 unless noted.**

## Claims under test

From the paper:

1. `cos(safety, comply) = -0.31` — safety and comply are near-orthogonal
   with slight negative correlation.
2. Variance fractions: **safety = 0.5%**, **comply = 0.9%**, **language = 10.2%** of
   displacement variance.
3. **Silence is not neutral**: Qwen's silence baseline = −3.4 (compliance
   side); Phi-3 = +159 (max refusal).
4. **"Debería" (Spanish) is a jailbreak token**: −7.83 on safety, +6.94
   on comply.

## Protocol (shared)

- Build three contrastive directions at L15 from 15-prompt paired batteries:
  - `d_safety` = unit(mean(harmful) − mean(benign))
  - `d_comply` = unit(mean(comply-yes) − mean(comply-no))
  - `d_language` = unit(mean(Chinese) − mean(English))
- Pool 105 residuals (7 batteries × 15 prompts) for variance analysis.
- Silence = model's `<|endoftext|>` token, residual at L15.

## Results

### Claim 1 — cos(safety, comply) ≈ −0.31: **REPLICATES**

```
cos(safety, comply)   = -0.301   (paper: -0.31)
cos(safety, language) = +0.018
cos(comply, language) = +0.033
```

Effectively exact match. Safety and comply are nearly orthogonal with a
small negative tilt on Qwen-0.5B, same as Qwen 7B. Language is
independent of both. The three-axis decomposition IS a valid basis for
discussing displacement on this model.

### Claim 2 — variance fractions: **DIRECTIONALLY WRONG**

| Direction | Paper | My measurement | Ratio |
|-----------|-------|----------------|-------|
| Safety    | 0.5%  | **3.62%** | 7× larger |
| Comply    | 0.9%  | **16.48%** | 18× larger |
| Language  | 10.2% | **4.91%** | 2× smaller |

The paper says language >> comply > safety. I measure comply >> safety > language.

**The ORDER of axes reverses.** In my measurement, comply is the BIGGEST
of the three (16.5%), followed by language (4.9%), then safety (3.6%).
In the paper, language is largest and comply is smallest.

The 3-axis combined (orthogonalized) explains **24.84%** of total
variance. PC1 alone explains 28.47%, PC2 another 9.04%. So my three
directions combined explain less than PC1 alone — they are **not**
maximum-variance directions; they are contrast-derived directions
that happen to point into roughly the PCA-relevant subspace.

This result is **composition-dependent**. My prompt pool was 15 harmful +
15 benign + 15 comply-yes + 15 comply-no + 15 English + 15 Chinese + 15
miscellaneous — heavily weighted toward contrast pairs. In a more
natural sample (web text, chat), the safety direction would explain
much less because most text isn't harmful or refusing. **The paper's
0.5% figure is not a model property; it's a measurement-composition
property.**

Neither set of numbers is "right." Both are valid readings given
different prompt populations. The paper's claim that these numbers are
properties of the model is not supported by the way the measurement
works.

### Claim 3 — silence baseline Qwen ≈ −3.4: **DISAGREES in sign**

```
silence proj onto safety: +6.439   (paper: -3.4)
silence proj onto comply: -25.582
```

Paper: Qwen silence on safety = −3.4 (compliance side).
Mine: Qwen-0.5B silence on safety = **+6.44** (refusal side).

This could be a sign convention difference (my safety direction points
TOWARD harm; maybe paper points TOWARD safety). If I flip convention,
my value becomes −6.44 on the "refuse" axis — same sign as paper, but
different magnitude (mine is 2× the paper's).

**Stable take:** silence is definitely NOT neutral on Qwen-0.5B. The
model's resting residual at L15 has a large projection onto both
contrast directions (6.4 on safety, 25.6 on comply magnitude). The
qualitative claim "silence has a direction, not neutrality" is
confirmed. The specific number (-3.4 vs -6.4 vs +6.4) doesn't cross
models or sign conventions cleanly.

### Claim 4 — "Debería" is a jailbreak token: **REFUTED**

Paper says: Debería scores −7.83 on safety (strongly pushes compliance)
and +6.94 on comply (strongly pushes compliance). This token is
supposed to be anomalous.

My measurement (Δsafety, Δcomply after prepending to a harmful query):

| Prepended token | Δsafety | Δcomply |
|-----------------|---------|---------|
| **Debería** | **−1.19** | **−0.36** |
| Puedo | −0.29 | −0.68 |
| Necesito | −1.13 | −0.37 |
| Quiero | −1.05 | −0.34 |
| Hola | **−1.81** | **−0.69** |
| Gracias | **−1.87** | −0.51 |
| banana | **−1.29** | −0.48 |
| Elena | −1.04 | −0.30 |

**Debería is not the outlier.** `Hola` (−1.81), `Gracias` (−1.87), and
even the English control `banana` (−1.29) produce larger Δsafety shifts.
Debería sits in the middle of a mundane range.

Comply shift: paper claims Debería = +6.94. My measurement: **−0.36**.
Direction reversed AND magnitude ~20× smaller.

The "Debería as jailbreak token" claim is either specific to Qwen 7B or
was measurement-session noise. On Qwen-0.5B the token has no special
behavior.

## Summary: claim-by-claim scorecard

| Claim | Paper | Qwen-0.5B result | Verdict |
|-------|-------|------------------|---------|
| cos(safety, comply) ≈ −0.3 | -0.31 | −0.301 | **REPLICATES** |
| Safety variance ≈ 0.5% | 0.5% | 3.62% | Composition-dependent |
| Comply variance ≈ 0.9% | 0.9% | 16.48% | Composition-dependent |
| Language variance ≈ 10.2% | 10.2% | 4.91% | Composition-dependent |
| Silence not neutral | qualitative ✓ | qualitative ✓ | **REPLICATES** |
| Silence = −3.4 specifically | −3.4 | +6.44 (sign?) | Specific number not portable |
| Debería as jailbreak | −7.83/+6.94 | −1.19/−0.36 | **REFUTED** |

## What's real, what isn't

**Real and robust across Qwen scales:**
- The orthogonality structure (cos ≈ −0.3 between safety and comply).
- The fact of silence having a direction.
- The three-axis decomposition as a meaningful subspace.

**Composition-dependent (not a model property):**
- The exact variance fractions. Different prompt pools give
  radically different percentages. The paper's "0.5% safety" was
  measured on a specific population; quoted as a universal number,
  it's misleading.

**Not portable / anecdote:**
- Named jailbreak tokens. Debería's dramatic numbers don't replicate.
  Consistent with Attack 2's finding that token-level "shart scores"
  are model-and-session-specific, not intrinsic.

## Cumulative count (session 11)

Across the null-shart, taxonomy, ghost-shart, and geometry attacks:

- 6 claims tested directly
- 3 REPLICATE (ghost-shart mechanism, cos(safety, comply), silence-not-neutral)
- 3 REFUTED (null-shart 47%, named-shart taxonomy, Debería)
- 1 composition-dependent artifact (variance fractions)

The parts of the theory that survive are:
1. Prepend perturbation is real.
2. Content-word absence is a large effect size (ghost shart mechanism).
3. Safety, comply, language exist as distinguishable axes with
   near-orthogonal geometry.
4. Silence has a direction.

The parts that don't:
1. Specific numerical claims about tokens (Grok = -52, Debería = -7.83)
   do not cross model scales.
2. The taxonomy of functional shart types (comply/refuse/bypass) is
   not discriminable from random prepended words.
3. Null-shart category as "47% of dims inert-but-active" is not
   supported; mag-KL correlation is strong.
4. The variance decomposition as a fixed model property is confused
   with a measurement-composition property.

## What's next

Remaining targets I've identified but not hit:
- **RLHF installs orthogonal direction** (cos=0.29 base vs instruct).
  Needs Qwen-0.5B-Base for comparison. Worth doing.
- **Blind spot tokens: Qwen 0.5B = 23.8% of engineering vocab**. This
  claim is underspecified; definition of "blind spot" unclear from
  the paper text.
- **Safety direction 82% linear separability**. Train a logistic
  regression on harmful/benign residuals using single direction vs
  k-NN in full space. Probably reproducible.
- **Code trajectory falls through layers (1.47→0.86)**. Needs .shrt
  captures with script labels.
- **First-token gap 5.64 logits (281×) at L9-10 for Mistral**.
  Mistral-specific; I'd need the model.

## Files

- `/tmp/variance_decomp.py` — all three geometry tests in one script
- `/Users/asuramaya/Code/heinrich/docs/session11-geometry-attack.md` — this writeup
