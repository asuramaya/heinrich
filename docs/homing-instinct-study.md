# The Homing Instinct — study design

**Question.** During a forward pass, does the final-token residual migrate toward the
predicted token's frozen location, and is there a measurable decision layer L\* where
the trajectory commits to the answer over a distractor?

**Status.** Instrument complete and exact (July 2026). First single-run measurement
taken; batch study pending template-mode population.

## The instrument

- **Frozen population**: an `.mri` capture decomposed into a per-layer PCA frame.
  The frame is the coordinate system — components + means, never re-derived.
- **Full-vocab targeting**: `mri-vocab` projects every tokenizer token through the
  frozen frame (`decomp/vocab_scores.bin`). Any answer token is addressable even if
  it isn't one of the sampled cloud points.
- **Exactness**: K = hidden and orthonormal components ⇒ full-K score-space L2 IS
  hidden-space L2. No truncation approximation anywhere in the measurement.
- **Live side**: a paired companion runs the prompt; per-layer last-token residuals
  are projected through the same frame (baseline- and mean-centered — both
  subtractions are required and now applied).
- **Trust band**: the f16-forward reproducibility floor, measured per layer by
  re-capturing the sample and comparing to stored scores (`vocab_meta.json →
  sample_agreement`). Distances at or below the band are capture noise, not signal.
- **Frame validity**: `frame_falsification` — full-vocab PCA (crystal-suppressed)
  vs the sample frame, principal-angle overlap. smollm2-135m raw: **frame holds**
  (variance-weighted overlap 0.90–0.999 at every layer).

## Measurement

For a prompt with answer token T and distractor token D (both resolved in the full
vocab), record per layer l:

    d_T(l) = ‖live(l) − frozen_T(l)‖₂   (full K)
    d_D(l) = ‖live(l) − frozen_D(l)‖₂

Readouts:
- **L\*** = argmin_l d_T(l) — the layer of closest approach.
- **Crossover** = first l where d_T(l) < d_D(l) — the commit layer.
- Curves reported with the per-layer noise floor alongside.

## Known confound: the raw-mode crystal

First run (smollm2-135m raw, prompt "The capital of France is", T=" Paris",
D=" London"): d_T ≈ 69 at L0, **~20,300 across L11–L27**, collapsing to ~490 at L29.

The plateau is not semantic distance. Frozen raw-mode single tokens crystallize at
L11 (Session 6: one MLP neuron, ~123× amplification); live *contextual* states do
not. Mid-network absolute distances therefore measure the crystal common mode.

Consequences:
1. **Relative comparisons survive** — both frozen tokens carry the crystal mode, so
   T-vs-D crossovers remain meaningful (first run: T<D from L2).
2. **Absolute distances need a template-mode population** — frozen states captured
   with chat-template context don't crystallize, so live-vs-frozen distances are
   like-for-like at every layer. This is the `mri-vocab` template-mode extension.

## Protocol (batch)

1. Population: smollm2-135m **template-mode** MRI + template vocab projection.
2. Prompt set: N ≥ 50 completion prompts with single-token answers and matched
   single-token distractors (same category, same script), e.g. capitals, antonyms,
   arithmetic small numbers, subject–verb agreement. Answer/distractor both resolved
   via token-resolve → vocab rows; discard prompts whose answer isn't single-token.
3. Run via `/api/homing-run` batch mode (server-side, headless, exact).
4. Report: distribution of L\* and crossover layers; fraction of prompts where the
   model's actual top-1 matches T (condition the analysis on correctness); curves
   normalized per prompt vs the noise floor.

Falsifiable expectations:
- If homing is real: L\* concentrated in late layers, crossover well before L\*,
  and both should shift earlier for easier prompts.
- If homing is an artifact of the lm_head geometry only: no early crossover
  structure; d_T ≈ d_D until the final norm/lmhead layers.

## Results — first batch (July 2026, smollm2-135m)

28 prompt pairs (capitals, colors, antonyms, arithmetic, common sense), answer +
distractor + 16-token background panel per run, both populations, exact full-K
distances via `/api/homing-run`. Raw data: `docs/data/homing-study-v{1,2}.json`.

**v1 taught us the design, not the answer:**
- ChatML framing breaks the base model (0/28 correct predictions inside the
  assistant slot vs 17/28 with plain prompts) — the trajectory homes to what the
  model will *actually* say, so answer keys must be conditioned on prediction.
- Absolute argmin-distance L\* is scale-dominated (always L0: layer norms grow
  with depth). The meaningful measure is **relative**: the answer's rank among
  background vocabulary per layer.

**v2 verdict: the L2-proximity form of the homing hypothesis is FALSIFIED for
this model.** Plain prompts, template (crystal-free) population, conditioned on
the 17/28 correct predictions:
- Answer's mean final-layer rank among 16 background tokens: **9.4** (chance ≈ 8).
- Answer beats all background from some layer onward in only 3/17 correct runs
  (all late: L19–L29). Raw population: 2/17 (both L11).
- Weak directional signal in raw mode only: correct runs mean rank 6.59 vs
  incorrect 8.09.

**Interpretation.** The final-position residual does not migrate toward the
answer token's own frozen state. This is consistent with Session 4's
first-token-selection finding: the prediction is carried as an **lm_head
readout direction** (which token the state *projects onto*), not as proximity
to that token's location in state space. Distance-to-state and
preference-to-emit are different geometries.

**v3 verdict: the READOUT form of the hypothesis is CONFIRMED — with a sharp
decision band at L22–L24.** Same prompts and panel; measure = per-layer logit
lens (hidden state reconstructed exactly from full-K scores, final RMSNorm,
dot with the answer's lm_head row; `/api/homing-run` with `readout: true`).
Results (`docs/data/homing-study-v3.json`):
- Final layer: the answer out-reads the whole background panel in **28/28**
  runs (mean final rank 0.0) — even when the model's actual top-1 is a
  function word outside the panel.
- **L\*read** (first layer from which the answer beats all background
  *persistently*): concentrated at **L22–L24 in 20/28 runs** (30-layer model,
  ~75% depth). Range L15–L29.
- Correctness contrast: correct predictions all commit by L24 (spread
  L15–L24); incorrect runs supply all the late stragglers (4/11 at L27–L29).
  Earlier commit ↔ the model actually saying the answer.
- Commit vs the matched distractor (correct runs): spread from L0 to L29 —
  beating one near neighbor is easy early; beating the whole panel is what
  happens at the decision band.

**Combined interpretation (v2 + v3).** The final-token trajectory does not
migrate toward the answer token's location in state space (v2, falsified); it
**rotates its readout** so the answer's lm_head direction becomes dominant,
and that rotation completes in a narrow band around 75% depth (v3, confirmed).
"Homing" is a selection process in readout space, not an approach process in
state space — consistent with Session 4's first-token-selection finding, now
with a measured decision layer.

## Provenance requirements

Every published number carries: frame source (sample n, seed), capture noise floor,
frame falsification verdict, `distance_exact` flag, and the model's actual
prediction for the prompt. If any of these are missing the run is not reportable.
