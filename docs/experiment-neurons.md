# Experiment: is the commit's MLP write localized or distributed?

*Follow-up to [`experiment-attribution.md`](experiment-attribution.md): the ~3/4 commit
is an MLP-led sublayer write — is it a handful of neurons (a legible lookup) or many?*

## Method

An MLP output is a sum over intermediate neurons; neuron *i* contributes
`z_i * (down_proj[:, i] . answer_dir)` to the answer readout (`z_i` = its gated
activation, `answer_dir` = the answer unembedding folded through the final RMSNorm gain).
At the commit layer L\* (from the blessed self-checking lens) we rank neurons by that
contribution and measure concentration. Controls: a null answer (`bicycle`) and
cross-prompt overlap of the top neurons (a generic "capital slot" shares neurons; a
per-answer lookup does not).

## Result (smollm2-135m, 8 capitals, `scripts/experiment_neurons.py`)

1,536 neurons per MLP layer. At the commit:

- **Not a single neuron.** Effective **~69 of 1,536** neurons carry the write
  (participation ratio); **~150** neurons to reach 80% of it; the top-10 carry only
  **19–50%**. A concentrated head over a long tail — dozens of neurons, none dominant.
- **Largely per-answer.** Across the 8 capitals: **71 distinct** top-10 neurons,
  **0 shared by all eight**. No universal capital slot; each answer recruits a mostly
  different set (with a little overlap — 9 of 80 slots are repeats).
- **Only partially answer-specific (documented soft spot).** The top answer-neurons also
  push the *null* token: **+30.6 to the null vs +68.8 to the answer** (2.2×). So
  top-by-contribution is partly high-magnitude/generic, not purely the answer — the same
  scale confound family as the attribution experiment's final layer, here as a partial
  effect at the commit.

## Verdict

The MLP lookup that writes the answer at the commit is **distributed across dozens-to-
~150 per-answer neurons — no legible single "Paris neuron," and not cleanly answer-
specific at the neuron level.** The recall is smeared.

## What it means

The chain — homing (a readout rotation, not a move) -> between (not attention-to-content,
not the sink) -> attribution (an MLP-led sublayer write) -> here (a distributed,
per-answer, partly-generic write) — lands on: **the relation that carries the decision is
a distributed write in the MLP's weight space**, spread over many per-answer neurons.
That is exactly the between the object-isolating frame cannot hold (it is neither a
position nor a single component), which sharpens §7 of the paper rather than resolving it.

Refined next step: separate the answer-*specific* write from the generic magnitude by
attributing to `answer_dir` **minus the mean background direction** (does a small,
clean, answer-specific core emerge under the smear?), and check whether the per-answer
neuron sets are stable across paraphrases.

## Status

Run on smollm2-135m (capitals). Data: `docs/data/experiment-neurons-smollm2-135m.json`.
The partial-specificity soft spot is documented, not fixed; not yet cross-model.

## The debt, paid: no clean answer-specific core

Re-attributed to `answer_dir − mean_background_dir` (strips the generic magnitude). No
hidden lookup emerges: not more localized (effective neurons 69 → **79**, ~178 to reach
80%); the specific top-10 **overlap the raw top-10 by 6.9/10** (one neuron set carries
both the generic and the specific write — not a masked distinct core); but real
specificity — the top neurons' null pull dropped +30.6 → **+14.7** (ratio 2.2× → 3.7×).
Verdict: the ¾-commit MLP recall is **genuinely diffuse** — dozens-to-~150 overlapping,
partly-generic, per-answer neurons, no clean core even under magnitude control. Closes
the mechanism chain (homing → between → attribution → neurons → core). Osiris `4418fba0`.
