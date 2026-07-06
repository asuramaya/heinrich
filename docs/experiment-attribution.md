# Experiment: Attention or MLP? — attributing the commit

*Follow-up to [`experiment-the-between.md`](experiment-the-between.md), which ruled out
attention-to-content and the position-0 sink as the carrier of the ~3/4 readout commit.*

## Question

The commit is a *readout rotation*: the state comes to project onto the answer's output
direction, late. The residual stream is additive (`h <- h + attn_out + mlp_out`), so we
can attribute the answer's readout, per layer, to attention vs MLP by **direct logit
attribution** — project each sublayer's last-token output onto the answer's unembedding
direction (folded through the final RMSNorm gain). This reads the sublayer *output*, not
which token it attended to, so it sidesteps every attention-weight confound from the
between experiment. L\* comes from the blessed self-checking lens
(`heinrich.profile.readout.lens_logits`); a null answer (`bicycle`) is the control.

## Result (smollm2-135m, 8 capital prompts, `scripts/experiment_attribution.py`)

- **Both sublayers write the answer at the commit, answer-specifically.** At L\* (mean
  rel 0.84): MLP→answer **+35.7** vs MLP→null **+6.9**; attention→answer **+28.4** vs
  attention→null **−7.6**. The commit is not carried by one sublayer alone.
- **MLP is the larger, later contributor; attention primes at the onset.** Attention's
  answer-push peaks at rel **0.77** (the commit onset) and fades; the MLP contribution
  grows through 0.80→0.96. A retrieval-then-write two-step, consistent with factual
  recall as a late MLP key→value lookup (Geva et al.; ROME).
- **Nothing until the band.** The answer readout is flat/negative through the first ~0.8
  of depth, then both sublayers write it in the L24–26 band — the layer-space image of
  homing's flat-then-cliff.

## Confound caught (the null control did its job)

At the **final layer (L28, rel 0.93)** the raw attribution is scale-dominated: MLP→null
jumps to **+119**, *larger* than MLP→answer (+77). The final MLP pushes many tokens, not
the answer specifically — so final-layer DLA magnitudes are not answer-specific. The null
control flagged it; the clean, answer-specific signal is the **commit band L24–26**,
where answer-vs-null separates (MLP pushes the answer up and the null down). Trust the
band, not the last layer.

## What it means

The between experiment ruled out attention-to-content and the sink. This one locates the
carrier: the commit is a **sublayer write in the ~3/4 band, MLP-led with an attention
prime at the onset.** The "relation" that carries the decision is therefore (at least
partly) in the MLP's *learned weights* — a key→value lookup — which is a different kind
of between than the attention graph, and one the current object-isolating frame also
can't decompose. Refined next step: does the answer-writing MLP direction correspond to a
small set of neurons (a localized lookup) or a distributed write? Heinrich already stores
`gate`/`up` activations, so the neuron-level attribution is a read away.

## Status

Run on smollm2-135m (capitals). Data: `docs/data/experiment-attribution-smollm2-135m.json`.
Not yet cross-model; the final-layer scale confound is documented, not fixed.
