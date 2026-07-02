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

**Next falsifiable variant (v3):** replace L2-to-frozen-exit with the live
state's per-layer projection onto the answer's **lmh virtual-layer row**
(already in `vocab_scores.bin`, row index nL−1) vs the background panel — i.e.,
measure when the readout direction, not the state distance, commits. If that
also fails to show a commit layer before the final layers, the "decision layer"
concept itself is wrong for this model class.

## Provenance requirements

Every published number carries: frame source (sample n, seed), capture noise floor,
frame falsification verdict, `distance_exact` flag, and the model's actual
prediction for the prompt. If any of these are missing the run is not reportable.
