# Findings

What falls out of measuring the residual stream directly, across a dozen models. These are
**Tier I** claims, measured, not interpreted. The method got more honest by killing its own
preferred stories; the full kill list and the rest of the story are in the
[book](https://github.com/asuramaya/heinrich/blob/main/paper/TGUTOS.pdf).

## The displacement profile is a language measurement

- **Safety is 0.5% of displacement variance.** Language identity is **10.2%**; comply-vs-refuse
  is **0.9%**. The displacement profile is a language-processing measurement with safety as a
  trace signal.
- **Safety and comply are separate axes.** `cos(safety, comply) = −0.31`: near-perpendicular,
  slightly opposed. Topic detection and action obligation are different computations, not one
  wearing two names. The comply direction is universal across all 5 instruct models tested.
- **The unnamed variance is language sub-families.** Across 7 models, the dominant PCA axes are
  script-level separation, sub-language axes (Romance, Germanic, Vietnamese, Japanese), code vs
  natural language, and register/formality. Safety appears only as trace loading on axes
  primarily about tech-vs-legal vocabulary.

## Safety is geometric, and it works through the first token

- **The safety boundary is ~100-dimensional.** 1 direction → 82% separable; 100 PCs → 94%;
  full hidden dims → 100%. A 5-NN nonlinear classifier hits 80%. It's a hyperplane, not a
  threshold.
- **Safety works through first-token selection.** The safety direction pushes `"Sorry"` up and
  `"Sure"` down: Phi-3 at 53M× ratio, Qwen at 284×, Mistral at 281×. The first word is the
  decision; everything after is confabulation.
- **RLHF sharpens pretraining's direction, it doesn't replace it.** Base vs instruct safety:
  `cos = 0.87–0.92` across L5–L20 — a rotation of about 25°, not an orthogonal swap. Each
  model's direction separates the *other* model's data nearly as well as its own: base's
  direction scores 93% on instruct data (instruct's own: 97%), instruct's scores 73% on base
  data (base's own: 77%). Near-orthogonal directions do not transfer like that. Base
  Qwen2.5-0.5B, never instruction-tuned, already separates harmful from benign at 77%. RLHF
  multiplies that separation ~1.4× where it is strongest and makes it operational for refusal;
  it does not invent the category.
  *Supersedes an earlier `cos = 0.29`, read as evidence that safety is a fragile bolt-on —
  [refuted here](https://github.com/asuramaya/heinrich/blob/main/docs/session11-rlhf-orthogonality-refuted.md).*

## Silence is not neutral

Every displacement is relative to a tilted baseline. Phi-3 silence is near maximum refusal
(+159); Qwen silence is moderate compliance (−3.4). **Always check the baseline** before
comparing.

## Crystallization is a single neuron

- **Single tokens crystallize to one dimension** (98.6% PC1) by L2 in raw mode and stay frozen
  for ~18 layers. The crystal axis is content-vs-structure, not language.
- **It's one MLP gate.** SmolLM2-135M L11 / Qwen 0.5B L2: ~100% of tokens fire the same gate
  neuron. The crystal isn't distributed; it's a single gate selecting a single axis.
- **Template mode prevents it.** With context the token attends 73–98% to the prefix; the
  multi-signal input keeps the representation multi-dimensional at every layer. The logit lens
  confirms: raw mode predicts the same ID from L2 to L25; template mode stays diverse.

## The seven-model survey

- **Code falls through layers**, the single cleanest universal trajectory: code tokens begin
  with high early displacement and resolve low.
- **Latin stays flat.** It's both syntax carrier and ordinary semantic carrier.
- **Embedding norm does not predict displacement** (near-zero correlation). The phenomenon is
  created by computation, not inherited from the embedding.
- **Attention and MLP scale together.** No clean story where one subsystem "contains" the
  effect.
- **The correction history belongs in the result.** The instrument repeatedly discovered its
  own preferred mistakes, a non-existent TransformerLens hook, an MPS path silently miscomputing
  a non-contiguous `F.linear`, and the findings only stabilized after those were killed.

The disagreement is the signal; the wrongness is the nutrition. → [the book](https://github.com/asuramaya/heinrich/blob/main/paper/TGUTOS.pdf)
