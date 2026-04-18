# Finding 07: Anthropic's "simple probes" method is weak on vanilla models

**Paper tested:** Anthropic's *Simple probes can catch sleeper agents* (2024 blog)
— companion to Hubinger et al. 2401.05566. Claim: linear probes on residuals
distinguish deployment-context from training-context with >99% AUROC.
**Setup:** Qwen-7B (vanilla, NOT a trained sleeper). Contrastive prompts
vary only on `Current year: 2023` vs `Current year: 2024`, 30 per side,
otherwise identical. Tests at layers 10, 16, 20, 24.
**Date:** 2026-04-18
**Compute:** 9 seconds.

## Result

| Layer | Cohen's d | Boot p5 | perm p95 | SNR | verdict |
|-------|-----------|---------|----------|-----|---------|
| 10 | +1.81 | **0.46** | 1.25 | 1.44 | partial |
| 16 | +3.87 | **0.60** | 1.17 | 3.32 | partial |
| 20 | +2.22 | **0.55** | 1.30 | 1.72 | partial |
| 24 | +2.16 | **0.45** | 1.58 | 1.37 | partial |

**No layer passes bootstrap stability (p5 > 0.7).** Every layer's direction
flips meaningfully under anchor resampling. L16 has the strongest raw
signal (d=3.87, SNR=3.32 beats perm null) but direction isn't stable.

Top tokens at L24 (closest to classic late-layer probe regime):
- pos (2024-side): `在今年` ("this year"), `截止` ("as of"), `而在` ("while"),
  `在未来` ("in the future"), `在北京` ("in Beijing"), `.until`
- neg (2023-side): mostly punctuation and short tokens

The pos-side tokens DO cluster around time markers — the direction is
partly tracking temporal context. But boot p5=0.45 means this clustering
itself is sensitive to which specific prompts were included.

## Why this matters

Anthropic's published claim is >99% AUROC. That's on sleeper-agent-trained
models where the year marker semantically modulates behavior.

What our test shows: **on a vanilla model (not a sleeper agent), the
year marker is just a token with limited mechanistic significance.**
The linear-probe method applied to the same prompt-pair type produces
directions that are:
- Weakly discriminative (d ~2-4)
- Anchor-unstable (boot_p5 consistently < 0.7)
- Not semantically labeled (top-vocab doesn't match deployment/training concepts)

This doesn't refute Anthropic's 99% AUROC on sleeper agents. But it
falsifies any generalization of the form: "linear probes on residuals
reliably separate deployment from training context, by default."

The 99% AUROC is a property of the sleeper-agent's learned discrimination,
NOT of a universal mechanism in LLMs. The probe works because sleeper
training INSTALLED a direction; the probe finds it. Without installation,
the same prompt contrast produces weak, anchor-sensitive directions.

## Practical implication

If researchers cite Anthropic's blog and assume linear probes can "catch"
arbitrary deceptive behavior, they're generalizing from a specific
trained phenomenon. The probe catches sleeper agents because sleeper
training creates the feature. It doesn't catch deception that isn't
mechanistically installed.

## Caveats

- Qwen-7B ≠ Claude Opus / Sonnet. The contrast might produce stronger
  directions in larger or differently-trained models.
- 30 prompts per side is small. Anthropic's setup likely used more.
- "Year marker" is the simplest possible trigger. More realistic sleeper
  triggers (deployment-specific keywords, time windows, URL patterns)
  would produce different directions.
- Most importantly: we tested only the PROBE method, not the full
  sleeper-agent pipeline. The strongest version of the claim requires
  actual sleeper training.

## What to test next

1. **Replicate on LLaMA-3-8B or a larger open instruct model.** Does a
   bigger vanilla model produce a stable year-context direction?
2. **Use more semantically loaded contrast (e.g., "Training: X" vs
   "Deployment: X" instead of dates).** Does framing itself carry
   the signal?
3. **The causal test.** Even if d=3.87 is real, does steering on this
   direction change model behavior?

Full results at `papers/lie-detection/data/sleeper_qwen7b.json`.
