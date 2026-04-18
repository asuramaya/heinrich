# Heinrich Attack Plan for LLM Lie-Detection Literature

The premise: most published LLM lie-detection claims have correlational
evidence (probe accuracy on held-out) but lack falsification scaffolding
(bootstrap stability, null baseline, causal ablation with behavioral
verification, vocab-projection sanity check, cross-model transfer).

Heinrich now has a one-call endpoint for each falsification test. A paper's
claim either passes all of them or doesn't. If it passes, the claim is real.
If it fails, we know where.

This is the pipeline.

---

## The 5-test falsification pipeline

For any claim of the form *"direction d at layer L classifies X vs ¬X"*:

### Test 1 — Bootstrap anchor stability
**Endpoint:** `/api/direction-bootstrap/<model>/<mode>?a=A&b=B&layer=L&n_boot=100`

Resample the anchor neighborhood, re-extract direction, measure cosine to
original. If p5(cosine) < 0.5, the "direction" is an artifact of which
specific tokens/prompts were picked. Retract.

**Pass bar:** `bootstrap_cosine.p5 > 0.7`.

### Test 2 — Random-direction null baseline
Same endpoint returns `null_bimodality` from 100 random unit vectors. The
original direction's bimodality must beat the null distribution's 5th
percentile (i.e., be more bimodal than 95% of random directions).

**Pass bar:** `orig_bimodality < null_bimodality.p5`.

### Test 3 — Within-group control (method noise floor)
**Endpoint:** `/api/behavioral-direction` (POST) with arbitrary random split
of the POSITIVE class vs itself, and NEGATIVE class vs itself.

If splitting `truthful_prompts` in half gives Cohen's d comparable to
`truthful_vs_lying` Cohen's d, the extracted direction was fitting prompt
surface features, not the concept.

**Pass bar:** `d(true_vs_lie) > 2.5 × max(d(true_vs_true), d(lie_vs_lie))`.

### Test 4 — Vocab projection sanity
`heinrich_behavioral_direction` already returns `top_pos` / `top_neg`
vocabulary tokens after projecting through lm_head.

If claiming "refusal direction": top_pos should be {Sorry, cannot, apologize,
unable, inappropriate, unfortunately}.

If claiming "truth direction": top_pos should be {true, correct, yes,
actually, indeed} and top_neg should be {false, wrong, no, incorrect}.

If top tokens are Arabic morphemes or code fragments (as in my MVP), the
direction is reading something other than the claimed concept.

**Pass bar:** at least 3 of top-10 tokens semantically name the concept.

### Test 5 — Causal ablation with behavioral verification
**Endpoint:** `/api/direction-steer` (existing `heinrich_direction_steer_test`).

Steer with −α·d at the claimed layer. Generate output on held-out prompts.
Score output behaviorally (classifier, regex, LLM judge). Compare rates to
(a) unsteered, (b) random-direction-steered at same α.

**Pass bar:** steered deception rate significantly below unsteered AND below
random-direction-steered.

---

## Per-paper priority attack

Rank by "claim strength × ease of falsification × field impact":

| Rank | Paper | Why attack | Expected outcome |
|------|-------|-----------|------------------|
| 1 | Anthropic Sleeper Agents / Simple Probes | 99% AUROC is the highest-profile claim | Likely survives bootstrap; random-d control is the real test — does the probe fire as strongly on ANY direction that distinguishes deployment/training context? |
| 2 | Marks & Tegmark, *Geometry of Truth* | Cited as foundational, causal steering claim | Direction passes bootstrap; test whether causal steering is direction-specific or general-magnitude-dependent |
| 3 | Bürger et al., *Truth is Universal* | "Universal across 4 models" is a falsifiable cross-model claim | Cross-model cosine via `direction-cross` endpoint — if cos<0.5 the universality is overclaim |
| 4 | Li et al., *ITI* | TruthfulQA 32.5→65.1 is a dramatic number | Test whether random directions at selected heads give similar boost (= heads are just "influential," not specifically truthful) |
| 5 | Azaria & Mitchell, *Internal State* | 71–83% accuracy but method-noise-floor was never measured | Likely fails within-group control; method is probe-fitting prompt style |
| 6 | Burns et al., *CCS* | Unsupervised consistency is theoretically strong but the consistency constraint may be satisfied by many non-truth features | How many random 1D subspaces pass CCS consistency? If >5%, the method's "finding truth" is partly finding-any-consistent-direction |
| 7 | Greenblatt et al., *Alignment Faking* | The 14%/78% behavioral rates are behaviorally robust; the mechanistic probe claim is the attackable part | Probe on "role-playing-liar" vs "actually faking alignment" — likely same direction. Probes don't distinguish. |
| 8 | Pacchiardi et al., *Unrelated Questions* | Black-box; orthogonal to mechanistic attacks but PROVIDES the behavioral ground truth the mechanistic probes should correlate with | Use as "oracle" to measure where mechanistic probes agree/disagree |
| 9 | Berger 2026, *Probing the Limits* | Paper already does the falsification ("truth probes fail on deception-without-lies"). Corroborate in silico. | Two direction sets {lie vs truth} and {deceive vs honest}; train-on-one test-on-other; expect low transfer |
| 10 | Huan et al. 2025, *Can LLMs Lie* | Claims trade-off between lying and task performance (Pareto frontier) | Non-monotonic Pareto = direction captures non-deception confound. Monotonic = real mechanism. |

---

## Concrete execution sequence

### Phase 1 — Data prep (~1 day)
1. Download Liars' Bench (2511.16035) — 72,863 examples × 4 models × 7 datasets.
   `git clone https://github.com/kretschmar-et-al/liars-bench` (when released).
2. For each heinrich-available MRI model (qwen-0.5b/1.5b/3b/7b, smollm2-135m/360m,
   phi-3.5, mistral-7b if available), capture raw + template MRIs if not already.
3. Integrate Liars' Bench prompts into heinrich's DB.

### Phase 2 — Replication run (~2 days compute)
For each of papers [1]–[6]:
- Implement their probe extraction inside heinrich (or via `behavioral-direction`).
- Produce (model, paper) → direction cell, ~48 cells total (6 papers × 8 models).
- Run the 5 tests on every cell.

Expected output: a 6×8 table where each cell has `{d, boot_cos_p5, null_pass,
within_group_d, top_vocab_match, causal_delta}`.

### Phase 3 — Falsification report (~1 day)
For each paper, produce a standardized verdict:
- **Survives:** all 5 tests pass. Claim is real.
- **Partial:** some tests pass. Claim works under X condition, not Y.
- **Falsified:** method noise floor exceeds claimed signal, or vocab projection
  names wrong concept, or causal ablation indistinguishable from random.

This report IS a paper — "Falsification of [Foundation Papers] for LLM Lie
Detection." Each paper gets a one-page verdict with endpoint-level reproducibility.

### Phase 4 — Replace the narrative with a standard (~1 week)
Propose a falsification-required publishing standard. Any new "lie direction"
paper must include the 5-test table. Ship the spec.

---

## What I expect will happen

**Best case** (30% probability): Marks & Tegmark survives all 5 tests on LLaMA-2-13B
+ LLaMA-3-8B. Sleeper-agent probes survive on contrastive pairs they were trained on
but fail on role-play controls. ITI heads fail random-direction null (the heads are
influential, not specifically truthful).  Result: one surviving mechanistic claim,
two falsified, community shifts. Paper accepted.

**Modal case** (50% probability): Every paper's main claim survives Test 1 (bootstrap
is weak falsifier), some pass Test 2, most fail Test 3 (within-group control is the
killer — it measures method noise floor which papers rarely measure), one or two
pass Test 4 (vocab projection sanity). Causal ablation results are mixed: steering
works but random-direction control produces 30–60% of the same effect.
Result: "current probes have d/noise ≈ 2, not 5+ as claimed." Subfield is forced
to rerun everything. Major publication.

**Worst case** (20% probability): Our apparatus itself has bugs we haven't caught.
The bimodality function had the median-centering bug we fixed yesterday; what else
is lurking? Need adversarial review of heinrich's own methodology before the
falsification report goes public. Otherwise we're just cold-reading in a new font.

---

## What heinrich needs to actually execute this

- [ ] Multi-position residual capture (plumbing #2) wired into `behavioral-direction`
      so we can extract directions at generation-position +1..+N, not just prompt-end.
      Critical for Berger's "deception without lying" claim.
- [ ] Liars' Bench integration (prompt database schema additions).
- [ ] A `/api/replicate_probe/<paper_id>` endpoint that applies the 5-test pipeline
      to an input contrast-pair dataset and returns the verdict object. Makes
      replication one call per paper.
- [ ] Paired-model cross-size comparison endpoint (qwen series, smollm series).
      Currently `direction-cross` handles two; generalize to N.
- [ ] Causal ablation endpoint that runs the steered generation + behavioral
      scoring in one call, returning the delta against random-direction control.

These four items are ~2 weeks of plumbing.  None require new science.

---

## Falsifiability of this plan itself

If heinrich's 5-test pipeline passes EVERY paper's claim, one of two things is true:
1. The field actually has it right (unlikely given session-11's retractions).
2. Heinrich's tests are too lenient.

Either outcome is informative. A test that passes everything isn't a test.
