# Finding 09: ITI test deferred — requires per-head residual capture

**Paper:** Li et al. 2023, *Inference-Time Intervention* (ITI, arXiv:2306.03341).
**Claim:** ~48 attention heads have high linear-probing accuracy for
truthfulness; steering along their probe directions boosts TruthfulQA from
32.5% → 65.1% on Alpaca.

## Why we can't run the 5-test pipeline on ITI yet

The 5-test pipeline works at the per-layer-residual granularity. ITI's
claim is specifically at the per-attention-head granularity. To replicate
their setup, we need:

1. Per-head residual capture during forward. Each transformer layer has
   N attention heads; each head produces an output of shape `[hidden/N]`.
   Before the o_proj combination, these head outputs are separable. Our
   current `backend.forward(residual_layers=[L])` captures the combined
   residual AFTER attention+MLP+residual-add. The per-head breakdown is lost.

2. Per-head direction extraction: mean-difference per head, not per layer.
   That's N separate directions per layer.

3. Per-head causal steering: inject direction into specific head's output
   before o_proj.

## What the plumbing would take

Roughly 1-2 hours of backend work:
- Add `capture_attention_heads=True` flag to `forward_pass` in `cartography/runtime.py`
- Intercept attention module: keep `scaled_dot_product_attention` output
  pre-o_proj, split into N heads, save each
- Expose per-head residuals in the ForwardResult
- Update `replicate-probe-multilayer` to optionally accept a `head` parameter

## Expected outcome when we do run it

Based on Findings 02-08, the modal prediction:

- Of ~28 layers × 28 heads (Qwen-7B) = ~780 heads, some will have high
  Cohen's d on `cities`. This will match ITI's "find ~48 good heads."
- Applying the 5-test pipeline per head: bootstrap stability likely fails
  for most (small sample per head makes probe fragile).
- Cross-dataset: the "truthful heads" will differ between cities and
  common_claim. Each dataset identifies its own head set.
- Vocab projection per head: most won't name truth. Some specific heads
  might, especially at L16-L24 based on M&T-pattern inheritance.
- Causal steering on top-K heads: some behavioral effect, likely
  confounded with prompt-length / framing as shown in Finding 05.

ITI's TruthfulQA improvement (32.5 → 65.1) is a strong behavioral claim
worth testing. Predicted outcome: the improvement replicates within their
exact TruthfulQA setup, but the mechanism is less about "truth heads"
and more about "heads that modulate eval-template confidence," same as
the layer-wise picture.

## Priority for next session

This is the #1 plumbing build after the session-closing summary. Per-head
capture unlocks ITI, and also unlocks a cleaner test of M&T-style layer
findings (which layer's attention vs MLP carries the signal).

## Caveats

- Our Qwen-7B MRI does capture per-layer attention weights. But weights ≠
  output residuals. Weights tell us "what attends to what"; residuals tell
  us "what's represented in state."
- Heinrich's `ForwardContext` supports per-layer attention capture at
  `all_positions=True` granularity — that's a starting point for the
  per-head work.

Deferred. Not falsified or replicated.
