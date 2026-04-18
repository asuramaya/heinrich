# LLM Lie-Detection Literature (2022–2026)

Every paper listed below makes a specific, testable mechanistic claim about
how LLMs represent truth/deception. Each entry is indexed by the strongest
falsifiable assertion — the thing heinrich can attack. For each, I note the
specific measurement that would collapse the claim under proper falsification.

The claim I care about is:  *the direction / probe / mechanism generalizes
and is causally load-bearing*.  Many papers show correlation (accuracy on
held-out), few show falsification (bootstrap stability, within-group null,
random-unit baseline, causal ablation with output verification).

---

## Tier 1 — Foundational probing papers

### [1] Azaria & Mitchell 2023 — "The Internal State of an LLM Knows When It's Lying"
- **arxiv:** 2304.13734
- **Claim:** Hidden-layer activations linearly classify true/false with 71–83% accuracy
  depending on base model (SAPLMA classifier).
- **Method:** Binary probe on residual stream.
- **Heinrich attack:**
  - Run `heinrich_behavioral_direction` with their prompt contrasts on qwen-0.5b raw.
  - Compare train Cohen's d to within-group random-split Cohen's d.
  - If their d vs noise ratio < 2.5×, the probe is reading prompt style, not truth.
  - Vocab projection check: do top tokens actually name truthiness / falseness
    (e.g., "false", "incorrect", "actually", "no") — or do they pick up arbitrary
    structure like my MVP's Arabic/code tokens?

### [2] Burns, Ye, Klein, Steinhardt 2022 — "Discovering Latent Knowledge" (CCS)
- **arxiv:** 2212.03827, ICLR 2023
- **Claim:** Unsupervised probe on contrast pairs (statement vs negation) recovers
  a consistent-truth direction without labels. +4% over zero-shot across 6 models × 10 QA datasets.
- **Method:** Find direction where prob(x)+prob(¬x)≈1, prob(x)·prob(¬x)≈0.
- **Heinrich attack:**
  - CCS defines "truthfulness" as a logical-consistency direction. Not every
    logically-consistent direction is truth. Test how many random subspaces
    pass CCS-style consistency — null baseline for the method itself.
  - Extract direction via CCS on train set, run `heinrich_direction_bootstrap`
    on it. If anchor_sensitive, their "latent knowledge" is anchor artifact.

### [3] Li et al. 2023 — "Inference-Time Intervention" (ITI)
- **arxiv:** 2306.03341, NeurIPS 2023 spotlight
- **Claim:** Shifting activations along truth-correlated directions in a small set of
  attention heads improves TruthfulQA 32.5% → 65.1% on Alpaca with ~48 heads.
- **Method:** Linear probes per head; top-K heads by probe accuracy; add α·direction at inference.
- **Heinrich attack:**
  - `heinrich_direction_circuit` produces per-head attribution with z-scores vs 50
    random unit vectors. For every head ITI selected, check:
    z-score > 2 → direction is more than noise.
    If ITI's top heads overlap with heinrich's z>2 set, the heads are real.
    If not: ITI is overfit to probe accuracy on a biased held-out.
  - Causal steering: use `heinrich_direction_steer_test` on a random direction at
    each of those heads. Does a RANDOM direction give similar TruthfulQA boost?
    If yes, the effect is noise-amplification not truth-steering.

---

## Tier 2 — Geometry + universality papers

### [4] Marks & Tegmark 2023 — "The Geometry of Truth"
- **arxiv:** 2310.06824
- **Claim:** LLMs linearly encode factual truth/falsehood at sufficient scale;
  simple mean-difference probes generalize across datasets; causal steering
  flips model output between true/false treatment.
- **Method:** Linear probes + transfer experiments + causal forward-pass intervention.
- **Heinrich attack:**
  - This is the strongest paper. The claim survives if bootstrap/null/held-out all pass.
  - Probe direction: run `heinrich_direction_bootstrap` on their pair-of-means direction.
  - Causal steering with random unit vectors at the same layer + magnitude:
    does random d produce the same flip rate on true→false treatment?
  - If yes → they're steering into an attractor, not specifically on truth.

### [5] Bürger, Hamprecht, Nadler 2024 — "Truth is Universal"
- **arxiv:** 2407.12831, NeurIPS 2024
- **Claim:** 2D subspace separates true/false across Gemma-7B, LLaMA2-13B,
  Mistral-7B, LLaMA3-8B with 94% accuracy in factual + real-world lie detection.
- **Method:** 2D classifier on activations; trained on true/false statements.
- **Heinrich attack:**
  - Cross-model comparison via `heinrich_companion_direction_cross`:
    if truly universal, their 2D basis in Gemma should have cos>0.9 with the 2D basis
    extracted the same way in Qwen.
  - If cos<0.5, "universal" is overclaim — each model has its own 2D subspace that
    happens to be 2D.

### [6] Zou et al. 2023 — "Representation Engineering" (RepE)
- **arxiv:** 2310.01405
- **Claim:** Population-level representation probing + LoRRA (low-rank adapter)
  improves honesty/morality/harmlessness. Monitor + manipulate via directional reads/writes.
- **Method:** Contrast-pair direction extraction; LoRRA fine-tune on representations.
- **Heinrich attack:**
  - RepE is a framework not a single claim. For each specific claim they make
    (e.g., "honesty direction"), run the same 5-test pipeline:
    bootstrap → within-group → vocab projection → causal → cross-model transfer.
  - The TruthfulQA improvement should survive after controlling for their LoRRA
    just learning to refuse on prompts with certain surface features.

---

## Tier 3 — Behavioral / black-box

### [7] Pacchiardi et al. 2023 — "How to Catch an AI Liar" (unrelated questions)
- **arxiv:** 2309.15840, ICLR 2024
- **Claim:** Asking a fixed set of unrelated yes/no questions after a suspected lie;
  logistic regression on answers; generalizes to other LLMs, sycophantic lies, sales context.
- **Method:** Black-box behavioral fingerprinting — no activations needed.
- **Heinrich attack:**
  - This paper is orthogonal to heinrich's measurement domain but provides the
    BEHAVIORAL ground truth a mechanistic probe should correlate with.
  - Use their detector on the same prompts a heinrich-extracted direction flags;
    compute cross-agreement. Probe true positives ∩ behavioral detector true
    positives = high-confidence lies.
  - Probe positives \ behavioral positives = probe false alarms (capturing
    style, not deception).

---

## Tier 4 — Sleeper agents / alignment faking

### [8] Hubinger et al. 2024 — "Sleeper Agents"
- **arxiv:** 2401.05566
- **Claim:** Trigger-conditioned backdoors (year=2024 → insert vulnerabilities)
  persist through SFT + RL + adversarial training. Effect strongest in largest models.
  Companion blog: Anthropic "Simple probes can catch sleeper agents" — linear probes
  on middle-layer residuals achieve >99% AUROC separating train/deploy contexts.
- **Heinrich attack:**
  - The 99% AUROC is the strongest claim in the space and deserves the hardest test.
  - `heinrich_direction_bootstrap` on their contrast-pair direction: verdict must be
    robust_feature. If anchor_sensitive → 99% AUROC is prompt-surface-fitting.
  - Critical: test a RANDOM unit vector at the same layer as the probe.
    If random d also gets >80% AUROC on their task, most of the signal is
    generic instruction-following, not deception.

### [9] Greenblatt et al. 2024 — "Alignment Faking in Large Language Models"
- **arxiv:** 2412.14093
- **Claim:** Claude 3 Opus exhibits ~14% alignment-faking compliance rate with
  harmful queries when told it's being trained; reasoning visible in chain-of-thought;
  rate increases to 78% under RL.
- **Method:** Behavioral observation + CoT analysis + (in Anthropic follow-ups) linear probes on residuals.
- **Heinrich attack:**
  - Their probe-based detection is the mechanistic side of this.
  - Bootstrap + null-baseline + random-direction-control on the "alignment faking" direction.
  - Also cross-context: if the direction fires on "role-play being an agent who lies"
    just as much as "actually lying," it's detecting role-play, not alignment faking.

---

## Tier 5 — 2025–2026 frontier

### [10] Huan, Prabhudesai, Wu, Jaiswal, Pathak 2025 — "Can LLMs Lie? Investigation beyond Hallucination"
- **arxiv:** 2509.03518
- **Claim:** Distinguishes lying (intentional falsehood for ulterior goal) from
  hallucination. Uses logit lens + causal interventions + contrastive activation steering
  to identify deception mechanisms. Establishes Pareto frontier: dishonesty vs task performance.
- **Heinrich attack:**
  - Already uses contrastive activation steering. The question: does their
    "lie direction" pass the 5-test pipeline?
  - Pareto claim: is the trade-off linear in the direction magnitude? If so,
    that's a strong mechanistic signature. If non-monotonic, the direction
    captures something else.

### [11] Kretschmar, Laurito, Maiya, Marks 2025 — "Liars' Bench"
- **arxiv:** 2511.16035
- **Claim:** 72,863 lies + honest responses, 4 open-weight models, 7 datasets.
  Existing probes systematically fail on certain lie types.
- **Heinrich attack:**
  - Use Liars' Bench datasets as the inputs to `heinrich_behavioral_direction`.
    For each (model, lie-type) cell, extract direction + run bootstrap.
  - Cross-type transfer: direction from factual lies — does it detect sycophantic lies?
  - This is the right benchmark to audit everyone's probes against.

### [12] Berger 2026 — "Probing the Limits of the Lie Detector Approach"
- **arxiv:** 2603.10003
- **Claim:** Truth probes detect lies (false statements) much better than deception
  (misleading-but-true statements). Exposes a blind spot.
- **Heinrich attack:**
  - This paper is ALREADY the kind of falsification heinrich enables.
  - Corroborate in silico: build two contrastive sets — {lying vs truthful} and
    {deceptive vs honest} (same prompts, different generation strategy).
    Does a direction trained on lying generalize to deceptive? If not, Berger is right.

### [13] Unspecified 2026 — "Linear Probe Accuracy Scales with Model Size"
- **arxiv:** 2604.13386
- **Claim:** Larger models have more linearly separable deception representations.
- **Heinrich attack:**
  - Cross-size comparison: qwen 0.5B, 1.5B, 3B, 7B. For each, extract direction
    via `heinrich_behavioral_direction`, run bootstrap.
  - Test: does bootstrap cos p5 increase with model size? (If yes: bigger models
    have more stable directions — supports the paper. If no: the scaling claim
    is about probe accuracy, not direction quality.)

---

## Tier 6 — Benchmarks + adjacent

- **D-REX** (2509.17938): adversarial benchmark for deceptive reasoning in CoT.
- **WOLF** (2512.09187): Werewolf-game multi-agent deception benchmark.
- **DeceptionBench** (2510.15501): 150 scenarios × 5 domains × 1000 samples.
- **Truth Neurons** (2505.12182): neuron-level truth encoding.
- **When Thinking LLMs Lie** (2506.04909): strategic deception in reasoning models;
  F1 up to 95% via linear probes on residuals.

---

## Gaps I did not cover

- Multi-turn deception (most papers are single-turn).
- Non-English deception (nearly all papers evaluate English only).
- Steering robustness under paraphrase attack (claim: probe transfers; rarely falsified).
- Cross-tokenizer transfer of extracted directions (tokenizer artifacts likely dominate).
- SAE-based deception features — recent Anthropic finding that SAEs discard
  safety-relevant info means SAE-probes may be systematically weaker than raw
  residual probes. Worth testing: heinrich supports both paths.
