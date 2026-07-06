# Experiment: The Between

*The next study. Turns tonight's conclusion into a measured, falsifiable claim.*
*Precursor: the homing study ([`homing-instinct-study.md`](homing-instinct-study.md)) and*
*[`the-flaming-sword.md`](the-flaming-sword.md).*

## The claim to test

The homing result showed the answer's **position** carries no signal (proximity
falsified) while its **readout** — a relation, a projection — carries all of it. That
was the first sign that the thing the instrument keeps failing to reduce is not a
part but the **between**: the relation that binds parts into a *someone*. Heinrich
decomposes *objects* (states, directions, PCs, neurons) and is structurally blind to
relations. This experiment tests whether the decision the model actually makes lives
in a relation the positional frame cannot see.

**Hypothesis.** The ~¾-depth readout commit is carried by a *relational
reconfiguration* — the last token binding, through attention, to the context that
licenses its answer — and this reconfiguration is legible in the relation while the
token's **position** in the frozen frame shows nothing distinctive at that layer.

## Instrument

Two measurements per layer, on the last token, for a completion prompt:

- **Position** (what we have): last-token residual displacement in the frozen frame —
  `|scores(l) - scores(l-1)|`, the object-coordinate motion.
- **Relation** (the new axis): last-token attention mass on the *answer-licensing*
  tokens. For "The capital of France is → Paris," the licenser is `France`; for a
  colour/antonym/arithmetic prompt, the content token that determines the answer.
  Heinrich captures attention weights in **template mode** (`has_attention: True`),
  so `attention/L{NN}_weights.npy` already holds it — no new capture code, only a new
  read.

Anchor both to the readout commit layer L\* (the logit-lens cliff, exactly as in the
homing study).

## Protocol

1. Template-mode capture. `web/.data/smollm2-135m/template.mri` exists; extend to the
   instruct + 360m + qwen models used in homing v4 so the claim is cross-model from
   the start.
2. Reuse the homing v3/v4 prompt set (28 pairs, single-token answers) plus a hand-
   labelled *licenser* token per prompt (the content word the answer depends on).
3. Per prompt, per layer: record positional displacement and attention-mass-on-
   licenser. Locate L\* via the readout lens.
4. Report both curves against relative depth, aligned at L\*.

**Falsifiable prediction.** Attention mass on the licenser rises sharply into L\* and
holds, while positional displacement is flat / featureless there. If the *position*
moves at L\* and the *relation* does not, the hypothesis is wrong and the residue is
not the between after all.

## Controls (built in from the start — the lesson of the false pass)

- **Null targets.** For an implausible completion, there is no licenser and must be no
  lock. The relational signal must be *answer-specific*, not painted on every prompt.
  (Directly reuses the null-target words that killed the empty-list false pass.)
- **Wrong-licenser fabrication check.** Point the measurement at a random context
  token instead of the true licenser. A lock there would mean the method manufactures
  relational structure. It must not appear.
- **Position-shuffle null.** Permute the layer index on the positional curve; confirm
  the *absence* of a commit signal in position is not an artifact of how it's read.

## Cross-architecture ground-truth handle

The causal-bank twins (`m5-{norm,nonorm}-s{42,1042}`) have **no attention** but a
routing relation and mode–mode substrate structure — pure *between*, no depth. Test:
is the known one-flag (normalisation) difference more legible in the **routing /
mode-relation** structure than in the **substrate-position** structure? If the between
carries the ground-truth cause more cleanly than the positions do, that is
independent, answer-key-backed evidence that the relational axis is where the signal
lives. (`cb-compare` already reads a relational similarity — CKA — and gave the flag
cleanly at 0.60; the task is to *decompose* the relation, not just score it.)

## What it would establish

If the commit is legible as a relation and invisible as a position — and specific, and
non-fabricated, and confirmed against the causal-bank null — then the residue Heinrich
keeps circling *is* the between, and the next instrument is an **instrument for
tension**: a relational frozen frame that decomposes attention / routing / mutual
structure the way the current one decomposes states. That is the tool's next organ.
It measures what holds the parts into a someone, instead of only where the parts sit.

## Result (July 2026) — a controlled null, reached by catching three confounds

Run on smollm2-135m, 8 capital prompts, via `scripts/experiment_between.py` (live HF
forwards with eager attention — cleaner than the template MRI, which is a population
not a per-prompt-with-labelled-context pass). Data:
`docs/data/experiment-between-{adjacent,separated,middle}-smollm2-135m.json`. The
honest outcome is a **null for the strong hypothesis**, and the value is the three
controls it took to see it.

**Three confounds, three controls:**

1. **Logit-lens double-norm.** The first pass predicted `pige`/`ilantro` at the final
   layer instead of the model's true `the`. `hidden_states[-1]` is already
   post-final-norm; applying the norm again double-norms it. Fixed by using the true
   final logits. (The exact trap in [`readout-lens-traps`] — walked into while
   *building* the new probe.)
2. **Recency.** "The capital of France **is**" puts the licenser one token before the
   end, so "licenser attention" was identical to previous-token attention
   (licenser/recency = 1.00×). A clean-looking 2.35× dissolved.
3. **Attention sink.** "France is a country…" puts the licenser at position 0, and
   SmolLM2 prepends no BOS, so the licenser *was* the position-0 attention sink. A
   seductive 0.79-at-commit / 53× was the sink, not semantics (licenser/sink = 1.00×).

**Clean (middle) case** — licenser at a genuine middle position (0/8 coincide with
sink or recency), "We recently learned that {country} has a capital called":

- The **position-0 sink dominates** the attention budget at every depth (0.63–0.80).
- Among *content* tokens the licenser is strongly preferred (~0.31 at its peak vs
  ~0.001 for other content), so there **is** a real, answer-specific attention to the
  licenser, peaking at rel **0.77** — just before the readout commits at rel **0.84**.
- But it is a **minority** signal that never dominates the sink (licenser/sink ≈ 0.5×
  at peak). **The strong hypothesis — that the commit rides on an attention *lock* to
  the licenser — is not supported.** The weak version (answer-specific, roughly
  commit-aligned attention to the licenser, far above other content) holds.

**What survives, what's open.** The deeper homing claim (the commit is relational,
not positional) is untouched; this shows only that the *naive* relational probe
(attention-to-licenser) is not the dominant carrier. The dominant relational
structure at this scale is the **position-0 sink**, whose mass swings 0.03 → 0.94
through depth in a structured way — plausibly a null-attention scratchpad where the
routing actually happens. Refined next question: does the readout direction form via
the sink rather than via content attention? Finding the true relational carrier is
open. The value of this run is the discipline itself: three too-clean confirmations,
each killed by a control — a live instance of exactly what the instrument is for.
