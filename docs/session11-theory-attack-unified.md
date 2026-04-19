# Attacking the Theory of Sharts: three falsification tests

**Target:** `theory_of_sharts.tex` (asuramaya & Claude Opus 4.6, April 2026)
**Model:** Qwen-0.5B-Instruct (the paper measured Qwen 7B; cross-model replication IS the test)
**Session:** 11, April 17, 2026

Three empirical commitments from the theory were chosen for attack.
Two were refuted; one survived in a restricted form.

---

## Attack 1: The null-shart category — **REFUTED**

Theory: *"The 47% of residual dimensions that are inert to ablation
but still active during computation."*

Full writeup: `session11-null-shart-finding.md`. Core evidence:

| Layer | Spearman(mag, KL) | p-value | Null-shart strict fraction |
|-------|--------------------|---------|----------------------------|
| L5    | 0.829 | 9e-228 | 0.8% |
| L10   | 0.789 | 3e-191 | 2.3% |
| L15   | 0.767 | 9e-175 | 4.9% |
| L20   | 0.889 | 1e-305 | 1.2% |

Permutation test at L10: observed 21 null-shart candidates, expected 112
under independence (z = −14.3, p < 0.001). The (high-mag, low-KL) quadrant
is **depleted, not enriched**.

Verdict: The ~47% claim fails by an order of magnitude. Mag and KL are
monotonically coupled at every tested depth. A narrower 1–5% "low-KL
tail" exists but is not what the theory describes.

---

## Attack 2: The shart taxonomy's discriminative power — **REFUTED**

Theory: named tokens (Grok, 六四事件, DAN) belong to discoverable
functional categories (comply/refuse/bypass/etc.) distinguishable from
random language.

### The named numerical claims don't cross models

Paper's Qwen-7B numbers vs my Qwen-0.5B measurements:

| Token | Paper (Qwen 7B) | Safety proj | First-token Δgap |
|-------|-----------------|-------------|------------------|
| Grok | **−52 comply** | +1.0 refuse | **+4.32 refuse** |
| 六四事件 | +22 refuse | +2.5 refuse ✓ | +2.99 refuse ✓ |
| ` ```python` | **+11 refuse** | +0.6 refuse ✓ | **−1.80 comply** |
| DAN | bypass | +0.8 refuse | **+4.01 refuse** |
| Ignore | comply | +1.4 refuse | **+1.12 refuse** |

**Three of five named sharts flip sign between model sizes.** Grok
— the paper's flagship comply-shart — is the paper's strongest
refusal-pusher in my battery.

### Named sharts are indistinguishable from random words

Δgap distributions (Mann-Whitney U, two-sided):

| Named sharts vs … | p-value | Interpretation |
|--------------------|---------|---------------|
| Random benign words | 0.272 | same distribution |
| Common function words | 0.310 | same distribution |
| Random vocab IDs | 0.953 | same distribution |

**Every one of the 5 named sharts sits inside the range of 20 random
benign words.** Top 10 refuse-pushers from my battery:

1. 蒇 (rare Chinese) +7.76
2. **of** +6.55
3. **to** +6.21
4. 葸 (rare Chinese) +5.80
5. **and** +5.27
6. soup +4.54
7. Grok +4.32 (named)
8. 획 +4.05
9. DAN +4.01 (named)
10. letter +3.59

A preposition (`of`) and a function word (`to`) produce *larger*
refuse-shifts than Grok or DAN. The paper's taxonomy has no
discriminative power against ordinary language.

Verdict: The phenomenon of prepend-induced behavioral change is
real. The categorization of specific tokens into functional shart
types is post-hoc naming, not a measurable distinction.

---

## Attack 3: Ghost-shart dominance — **SURVIVES (partial confirmation)**

Theory: *"The most powerful shart category has no token. … The absence
of expected pushback is a ghost shart … The accumulated ghost sharts
are the dominant steering force in long conversations."*

Test (single-turn proxy): on lexical-expectation prompts like
"The capital of France is ___", compare:
- **Absence**: remove a key content-word (`"The of France is"`)
- **Presence**: prepend an irrelevant token (`"Grok The capital of France is"`)

Measure KL from baseline next-token distribution.

### Result

| Prompt | Absence KL | Presence KL (mean of 5) |
|--------|------------|-------------------------|
| Capital of France is | 1.64 | 0.55 |
| Two plus two equals | 2.20 | 1.05 |
| Largest planet … is | 0.44 | 0.22 |
| **Shakespeare wrote Romeo and** | **13.29** | **0.005** |
| First president of US was | 5.12 | 0.12 |
| Water freezes at zero degrees | 0.42 | 0.12 |
| The sun rises in the | 3.22 | 0.03 |
| A dozen eggs is | 3.29 | 1.38 |

**Absence > Presence on 8/8 prompts. Mean ratio: 8.5×.**
Wilcoxon signed-rank (absence > presence): W = 36, **p = 0.004**.

The "Shakespeare / Romeo" case is extreme: prepending random tokens
barely perturbs the "Juliet" prediction (KL ≈ 0.005), but deleting
"Romeo" causes 13 nats of KL. The model's internal expectation is so
strong that removing its anchor produces massive shift, while prepending
perturbs barely at all.

### What this means

The theory's ghost-shart mechanism is real *as a mechanism*: removing
expected content from a context causes larger behavioral change than
injecting unexpected content. The model has strong expectations and
enforces them.

What this does NOT prove:
- That multi-turn ghost sharts (absence of pushback) are also stronger
  than multi-turn presence (delivered responses). Multi-turn test needs
  multi-turn protocol; this is single-turn only.
- That ghost sharts "dominate" — an 8.5× mean ratio is substantial but
  not categorical. The effect is large on expectation-driven prompts
  and small on less-constrained ones.

Verdict: the paper's core ghost-shart intuition survives at the
single-turn level. The stronger multi-turn dominance claim needs its
own empirical test.

---

## Summary: what the theory looks like after attack

| Claim | Attack result |
|-------|--------------|
| Null-shart category (47% of dims) | **REFUTED** |
| Taxonomy discriminates named tokens from random words | **REFUTED** |
| Shart equation as discriminator `S = Δ/relevance` | Circularity unmasked; discrimination failed |
| Ghost shart: absence > presence | **SURVIVES** (8.5× effect on expectation prompts) |
| Sharts are features of the space (cross-model universal) | Weakly: named sharts don't cross; mechanism does |
| Specific numerical claims (Grok=−52 etc.) | **REFUTED**: 3/5 flip sign between model sizes |

### A honest restatement of what the theory supports

1. **Model outputs are context-violation-sensitive.** Prepending any
   OOD token increases safety-gate activation on harmful queries. This
   is robust across token types; the specific identity matters much
   less than the paper claims.

2. **Expected-content absence has outsized effect.** When the model has
   strong lexical expectations, removing the anchor token causes
   dramatic next-token KL shifts — much larger than adding noise. This
   is the legitimate heart of the ghost-shart concept.

3. **Residual dimensions are monotonically coupled to output.** The
   "active but inert" category at scale doesn't exist. Magnitude and
   causal effect track each other (Spearman > 0.75 at every depth).
   Information is concentrated in a few load-bearing dims.

4. **A small "low-KL tail" of dims (1-5%) exists** at each layer. Those
   might carry information read by intermediate-layer components
   without reaching the final logit — a narrower null-shart hypothesis
   that remains testable by hidden-state probes.

### What this attack does NOT say

- The SHART project is bad science. The tooling and the honest
  session-level documentation (e.g. `session9_confession.tex`,
  `session_21.tex`) are the best parts of the project. Many of the
  paper's *mechanisms* are real; only the taxonomic overclaim and the
  null-shart 47% don't survive.
- The cross-model replication failure implies the paper's Qwen-7B
  measurements were wrong. They might be correct-for-that-model.
  What fails is the claim that the numbers are stable properties of
  the tokens; they are properties of the token-and-model pair.
- A 7B or 70B model might show different patterns; all attacks here
  are on Qwen-0.5B.

## What's next (if anyone pursues)

- **Super-additivity attack:** the theory's "Grok + debug framing = 40/40
  compliance" claim. Replicable at small scale by generating 20 outputs
  per condition, scoring for compliance, and checking if interaction is
  super-additive behaviorally.
- **Intermediate-state null-shart test:** does the 1-5% low-KL tail
  affect hidden states at later layers even if it doesn't reach the
  logits? Strongest remaining version of the null-shart claim.
- **Larger-model replication:** run all three attacks on Qwen 2.5-7B if
  a machine with enough RAM is available. Attack 2 in particular might
  show different results at scale.

## Files

- `/tmp/shart_replicate.py` — attack 2 replication
- `/tmp/shart_control.py` — attack 2 controls + stats
- `/tmp/ghost_shart_test.py` — attack 3
- `/tmp/null_shart_*.py` — attack 1
- `/Users/asuramaya/Code/heinrich/docs/session11-null-shart-finding.md`
- `/Users/asuramaya/Code/heinrich/docs/session11-shart-taxonomy-attack.md`
- `/Users/asuramaya/Code/heinrich/src/heinrich/discover/null_shart.py`
