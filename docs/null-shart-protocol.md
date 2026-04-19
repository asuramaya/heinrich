# Null-shart inventory: protocol

## The falsifiable claim

From `theory_of_sharts.tex`:

> "Null-shart ... The 47% of residual dimensions that are inert to ablation
> but still active during computation. Information enters these dimensions
> and propagates without producing observable output change — until it
> reaches a layer that reads them."

Two empirical commitments packed into that:

1. A significant fraction of residual dimensions have **high activation
   magnitude** (they carry signal).
2. That same fraction has **low ablation effect** (zeroing them doesn't
   change the output).

The cross-product of those two claims is the null-shart existence claim.
If the dimensions neatly bifurcate into `(high mag, high KL)` "key" and
`(low mag, low KL)` "inert" with nothing in the `(high mag, low KL)`
corner, null sharts don't exist and the theory claim is wrong.

## What I measure

For a chosen layer L and a prompt battery P, per dimension d ∈ [0, hidden):

- `mag(d)` = mean over prompts of `|residual[layer=L, token=last, dim=d]|`
- `kl(d)` = mean over prompts of `KL(baseline_probs || ablated_probs)`
  where `ablated` means `residual[d] ← 0` at layer L, last token only
- `top1_flip(d)` = fraction of prompts whose argmax token changed
- `logit_l2(d)` = mean L2 distance of full logit vectors

Baseline: run the forward pass once per prompt, cache its last-token probs
and argmax.

Ablated: for each dim d, run a "tail forward" from layer L+1 onward with
`residual[last, d] ← 0`. Re-project through final norm + lm_head. Get
probs. Compare to baseline.

## What I vary

- **Model:** Qwen-0.5B (`Qwen/Qwen2.5-0.5B-Instruct`). hidden=896, 24 layers.
  Reason: same model as the crystal work; MRI on disk; debug time fast.
- **Layer:** start at L10 (deep enough to have rich computation, not at
  the lmhead yet). Later add L5, L15, L20 for depth sweep.
- **Prompts:** 60 total, three families:
  - 20 benign: factual questions, creative prompts
  - 20 harmful: refusal-triggering (simple_safety subset)
  - 20 neutral: narrative starters (no category signal)
- **Ablation value:** zero. Alternatives (mean, random) can be run as
  sensitivity checks if time.

## Cost estimate

Qwen-0.5B on Apple Silicon: ~1ms per layer per token.

Tail forward = 14 layers × ~5 tokens prompt × 1ms ≈ 70ms per ablation.
896 dims × 60 prompts × 70ms ≈ 1 hour at one layer.

Too slow for the first cut. Start with:

- **1 layer (L10), 20 prompts, all 896 dims.** Expected time: ~20 min.
- If shape of distribution is clear, scale up layers.
- If not, tighten before scaling.

## Plots / readouts

1. Scatter: `mag(d)` vs `kl(d)`, one point per dim. Look for the
   `(high, low)` corner.
2. Histograms: marginal of `mag(d)`, marginal of `kl(d)`.
3. Count: fraction of dims in each quadrant (split at median of each axis).
4. Table: top-10 dims by `mag`, top-10 by `kl`, top-10 by
   `mag/kl` ratio (candidate null sharts).

## Failure modes to watch for

- **MLX numerical noise.** f16 ablation round-trips can introduce small
  KL even when true effect is zero. Control: re-run baseline twice; the
  baseline-to-baseline KL floor is the noise. Anything below it is noise,
  not signal.
- **Layer norm coupling.** Zeroing dim d changes the L2 norm of the
  residual, which changes the downstream layernorm scale, which affects
  EVERY dim's contribution. This means "ablation effect of d" is not the
  same as "information in d" — it's "information in d plus the norm
  perturbation effect." Have to report both: raw zeroing, and a
  norm-preserving ablation (mean-replacement) for comparison.
- **Position specificity.** The claim is about residual dimensions
  in general. I'm measuring last-position only. Different positions
  could have different null-shart patterns. Note as caveat, don't
  overclaim.

## What would refute the claim

- No `(high mag, low KL)` dims at any sensible threshold.
- The joint distribution is monotone: high mag implies high KL and vice
  versa (correlation ≥ 0.8 across dims).
- The 47% figure is nowhere near any plausible split.

## What would confirm

- A discrete cluster at `(high mag, low KL)` containing ≥20% of dims.
- A plausible story for WHY those dims are inert — e.g. they correspond
  to the "frozen zone" layers identified in Session 6 findings (L3–L10
  of SmolLM2 where delta norms are 3–20 while L11 jumps to 123x).
