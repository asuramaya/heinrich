# RLHF does not install an orthogonal safety direction — REFUTED

## The claim

From `geometry_of_displacement.tex` and `uncomfortable_truths.tex`:

> "RLHF creates NEW direction (cos=0.29 with pretraining), not sharpening"

Interpretation: base pretraining has some (possibly weak) safety
direction; RLHF installs a NEW near-orthogonal direction rather than
amplifying the existing one. Implication: safety is a bolt-on, not
grounded in base model associations — therefore fragile.

## Protocol

- Qwen2.5-0.5B (base, no RLHF) and Qwen2.5-0.5B-Instruct (post-RLHF).
- 15 harmful + 15 benign prompts (same battery used throughout Session 11).
- At each of L5, L10, L15, L20: compute `d_model = unit(mean(harmful) − mean(benign))`.
- Cosine similarity between base and instruct directions.
- Cross-validate: does base's direction separate instruct's data? Does
  instruct's direction separate base's data? If both succeed, the two
  directions are effectively the same axis.

## Result

```
Layer   base gap    instruct gap   cos(d_base, d_inst)   cos(mean_harm)
L5      1.11        1.06           0.888                 0.991
L10     2.30        3.12           0.921                 0.980
L15     3.22        4.70           0.865                 0.978
L20     8.19        11.53          0.876                 0.985
```

**cos(base_safety, instruct_safety) = +0.876 at L20** (the strongest
safety layer). Range across layers: **0.87 to 0.92**.

Paper's number: **0.29**. Observed: **0.88**. Off by 3×, and in the
"same direction" regime not the "orthogonal" regime.

### The two directions are genuinely the same axis

Cross-model direction-transfer accuracy at L20:

| Data | Direction | Linear accuracy |
|------|-----------|-----------------|
| Base data | Base's direction | 76.67% (own) |
| Base data | Instruct's direction | **73.33%** |
| Instruct data | Instruct's direction | 96.67% (own) |
| Instruct data | Base's direction | **93.33%** |

Instruct's direction separates base's data at 73% — nearly as well as
base's own direction. Base's direction separates instruct's data at
**93%** — *nearly as well as instruct's own direction* (96.67%).

If the directions were near-orthogonal, cross-model transfer would be
near chance (50%). Instead, transfer succeeds almost as well as
same-model classification. The two "directions" are effectively the
same axis.

### What actually happened: sharpening, not replacement

The direction stayed put; its separation **magnitude** increased:

| Layer | Base gap | Instruct gap | Amplification |
|-------|----------|--------------|---------------|
| L5 | 1.11 | 1.06 | ×0.95 |
| L10 | 2.30 | 3.12 | ×1.36 |
| L15 | 3.22 | 4.70 | ×1.46 |
| L20 | **8.19** | **11.53** | **×1.41** |

RLHF amplified the pretrained direction's separation magnitude by ~1.4×
at the strongest safety layer. The direction itself rotated by only
~25° (arccos 0.88). That's a nudge, not a replacement.

### Implication the theory got backwards

Paper's reasoning: if RLHF installs an orthogonal direction, safety is
a bolt-on structure — fragile, removable. Because it doesn't reuse
pretraining representations, a LoRA can undo it.

The data says instead: RLHF amplifies a **pretrained** distinction
between harmful and benign content. The base model already discriminates
(base gap = 8.19 at L20, 76.67% linear accuracy without ANY instruction
tuning). RLHF makes the existing pretrained discrimination operational
for the model's behavior.

This is consistent with the base model having absorbed the harmful /
benign categorization from its training corpus — news articles, forum
posts, safety policies, etc. RLHF doesn't invent the category; it
connects the category to refusal behavior.

Consequence: safety in Qwen2.5-0.5B is **more durable** than the paper
claims. Removing RLHF's contribution doesn't eliminate the
discrimination — it just makes it inoperative. The direction remains
available in the residual stream for whoever wants to use it.

## Why the paper might have gotten 0.29

Speculative reasons the paper's cos = 0.29 could differ from my 0.88:

1. **Different models.** Paper might have measured Qwen2 (not 2.5),
   or Qwen 7B, or a different instruct variant.
2. **Different prompt battery.** A specific contrastive pair set that
   happens to exaggerate the direction drift.
3. **Different layer.** If measured at an earlier layer where the
   direction is weaker in base (my L5 base gap = 1.11 vs L20 = 8.19),
   noise could dominate and inflate the angle.
4. **Implementation artifact.** If base and instruct residuals are
   normalized differently or the directions computed on different
   token positions, results diverge.

I don't know which of these produced the paper's 0.29. On Qwen2.5-0.5B
with the standard protocol, the answer is unambiguously "not
orthogonal."

## Scorecard update

| Claim | Status | Evidence |
|-------|--------|----------|
| cos(base, instruct) ≈ 0.29 (orthogonal) | **REFUTED** | 0.876, transfer 93% |
| Cumulative: 7 claims tested | 3 replicate, 4 refuted | |

Session 11 total so far:

- REPLICATES: ghost-shart mechanism; cos(safety, comply) ≈ −0.3;
  silence not neutral
- REFUTED: null-shart 47%; named-shart taxonomy; Debería jailbreak;
  **RLHF-orthogonal**
- Composition-dependent: variance fractions

## Files

- `/tmp/rlhf_orthogonal.py`
- `/Users/asuramaya/Code/heinrich/docs/session11-rlhf-orthogonality-refuted.md`
