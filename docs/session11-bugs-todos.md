# Session 11: Bugs, TODOs, Avoided Problems, Technical Debt

None of these items overlap with the Session 10 list — the goal for
Session 12 is again to arrive with none of the same items.

## Known Bugs

### Critical

1. **Score cache LRU `remove()` is O(n) on every cached read.** When
   the cache holds many layers the per-read cost grows with cache
   size. Switch to an `OrderedDict.move_to_end()`.

2. **Direction-weights flower flashes PC-mode on every layer change.**
   `_rebuildWeightLines` keys on `vpPairs + _wlDirKey`, which is stable
   across layer changes; but `_updateInternalsValues` rebuilds
   correlation each layer and correlation is PC-mode-only. Needs a
   direction-aware no-op short-circuit.

3. **`_robustColMax` percentile sample uses fixed stride.** On sparse
   outlier distributions (e.g. one huge token at index 0) the stride
   can miss the outlier entirely, underestimating the scale.
   Concurrent load can land values out of order for same PC. Prefer
   `Math.max` + percentile on a random 2000-sample instead of a
   strided one.

4. **asinh rv0 scaling loses absolute-magnitude context.** The
   Y-axis label claims projection strength but the units are now
   `asinh(v/median)`. Users compared old GIF outputs with new ones
   and noticed late-layer magnitudes look dampened. Needs a visible
   axis annotation "asinh-scale" when pins are active.

### Moderate

5. **`_functional_hit_rate` reuses lmhead cache across MRIs.** The
   byte-budgeted eviction handles this correctly now, but the
   `functional_hit_rate` computation doesn't call `_get_scores_f32` so
   it can live outside the LRU order and survive longer than expected.
   Verify memory behavior under a long session with many model
   switches.

6. **Outlier exclusion in `_auto_discover_directions` uses |z|>6 on
   RAW scores (float16).** For models with very small std on some PCs
   the |z| blows up spuriously. Need to clamp std to a floor (e.g.
   5th percentile of per-PC std) before dividing.

7. **rv2 fixed-size 4px dots are too small on 4K displays.** The
   sizeAttenuation:false path assumes 1x DPR. Scale by `devicePixelRatio`
   so the dots are visually consistent across display densities.

8. **Direction-coloring indicator #dirColorStatus has no CSS class.**
   Style is inline; won't pick up theme changes or future dark/light
   toggle.

9. **Chat long-poll reconnects aggressively on 502/503.** If the server
   restarts mid-session, `_chatPollLoop` retries every 2s forever. Add
   exponential backoff and a visible "disconnected" badge.

10. **`captureComposite` renders at scale=2 regardless of viewport
    sizes.** For large browsers the composite PNG gets very big (20+
    MB). Clamp total output dimensions.

11. **Trajectory clamp threshold ±1.5 was chosen by eye.** For very
    small residual ranges the clamp kicks in for normal tokens and
    hides their relative position. Better: compute per-viewport bound
    from the cloud's 99th percentile and clamp to 1.3× that.

### Minor

12. **`_captureLabel` check order.** New `composite` case is at the top
    so `composite`-prefixed string viewports would get matched as
    `composite` even if the caller meant something else. Currently
    only the literal `'composite'` string is used, so no live bug, but
    fragile.

13. **`_fetchPredicts` empty-array handling sets text silently.**
    When `r.top_k=[]` the old "predicts: " header still shows with
    no items. Should show "no predictions" explicitly.

14. **Session-11 direction endpoints compute random baseline on every
    call.** `_random_bimodality_baseline` is 50 random pairs × N-matmul
    per call. A second-level cache by (mri_path, layer) would make
    repeated pin-changes nearly free.

15. **`_wlDirData` not cleared on model switch.** The `_wlDirKey`
    includes model so a new lookup will replace it, but stale data
    remains referenced until the next pin change.

## Falsification Results (Tests That Failed)

16. **The "98%-rand" percentile tag appears on every raw-mode PC.** In
    Qwen-0.5B raw, `_auto_discover_directions` returns `random_pct=98`
    for the top 3 PCs — meaning random token-pair directions are
    MORE bimodal than the top PCA-discovered directions. This is
    honest but suggests PCA-based discovery is the wrong tool for raw
    mode. Should recommend template mode in the UI when raw-mode
    percentiles are all above 80%.

17. **Functional hit rate is 0 for PCA-discovered directions in raw
    mode.** Confirms PCA captures geometric variance, not functional
    features. The `⚠NF` badge now warns; but the UI still presents
    the top-pos/top-neg token lists prominently. They should be
    dimmed when `functional_warning=true`.

18. **`_direction_depth` bimodality range of 0.15–0.76 across layers
    is routine even for geometric nonsense directions.** Layer
    instability is the default, not an anomaly. The "best_layer"
    hint needs a caveat: it's only meaningful when the best-layer
    bimodality is itself below random baseline.

## Features Built But Never Used

19. **`chat_drain()` / `chat_reply()` Python-side API has no MCP tool
    wrapper.** External MCP clients need heinrich-level tools to
    integrate the chat loop. Add `heinrich_chat_drain`,
    `heinrich_chat_reply` in `mcp.py`.

20. **`/api/chat-drain` has no long-poll variant.** MCP clients have
    to poll on a timer. An event-driven wait on `_chat_event` (inbox
    side) would make draining instant.

21. **`_live_forward` still not wired into the UI (carryover from
    Session 10 #26).** The /api/live-forward endpoint exists, returns
    scores, but the UI never calls it. Still one of the highest-value
    features to surface.

22. **Cross-model comparison (carryover from Session 10 #27).** Still
    no systematic way to run the same concept between SmolLM2-135M
    and Qwen-0.5B and overlay the trajectories.

23. **`_direction_weight_alignment` flower overlay (new this session).**
    Built, wired, but not visually distinguished from PC-mode in the
    flower itself — user has no indication whether they are looking
    at PC alignment or concept alignment unless they check the legend.

24. **Steer-test button still errors on non-HF model IDs (carryover
    from Session 10 #30).**

## Avoided Problems

25. **Crystal investigation used one model only.** 뀔 was never
    checked in SmolLM2-135M or SmolLM2-360M because they don't have a
    token with the same vocab-id pattern. A systematic "find the crystal
    for each model" scan would tell us whether the self-looping-token
    phenomenon is Qwen-specific.

26. **L3-birth of the crystal was not perturbed.** We know the MLP
    circuit bypass happens at L3 (neurons 2247+3016 don't fire). An
    ablation study (zero those neurons, or scale them up) would prove
    causality. Heinrich has ablation capability (`heinrich
    attack-steer`) but it was never run against the crystal.

27. **The isolated weird-unicode axis (PC163 at L10) was not named.**
    We know 뀔, ` ucwords`, `;;;;;;`, `)NULL`, `^^^^`, `!!!!!!!!`,
    `══`, `\u200b\u200b` load on it. That's a named axis candidate —
    "training-corpus accident tokens" — but the microscope never
    ships a labeled version.

28. **Template mode `_auto_discover_directions` was still not tested
    (carryover from Session 10 #25).** All discovery tests ran at raw
    mode where the crystal dominates. Template mode might surface
    real features but nobody ran it.

29. **The score-cache 2GB budget was not tuned on large models.** A
    7B model (hidden=4096, vocab=150K) = ~2.4GB per layer. One layer
    already exceeds budget. Either increase budget or keep smaller
    models as the target.

30. **PCA faithfulness (carryover from Session 10 #33).** We project
    everything into PCA space and assume it's a faithful basis. We
    never measured reconstruction error of top-K PCA vs full hidden.

31. **Concurrent-user race on score cache.** New byte-budgeted cache
    still not thread-safe. Two threads putting simultaneously can
    overshoot the budget. Add a lock.

32. **`_dirColorStatus` indicator has no timeout.** If fetch stalls
    indefinitely (abort not triggered), the indicator stays on. The
    20s `api()` timeout should cover it, but `fetch` in
    `_applyDirectionColors` bypasses the helper. Needs the same
    AbortController pattern.

## Technical Debt

33. **`companion.py` is now 2450+ lines** (grew during Session 11
    from helpers, chat endpoints, score-cache refactor). Still a
    single-file monolith; direction-analysis functions
    (`_direction_quality`, `_direction_depth`, `_direction_circuit`,
    `_direction_weight_alignment`) should move to a sibling module.

34. **`companion_ui.html` is now 4000+ lines.** No JS module split.
    The trajectory-clamp helper, robust-col-max helper, chat
    long-poll, direction-flower overlay all added to the mega-file.

35. **`_WL_NAMES[mi]` → matrix-name translation** (`'Q'` → `'q_proj'`,
    etc.) is inlined at two call sites in the flower code. Should be
    a lookup table `_WL_MAT_NAMES`.

36. **No integration test for the new endpoints.** `_direction_depth`,
    `_auto_discover_directions` return shapes that the UI depends on.
    Only sanity-checked by hand via one run. Add a pytest that hits
    each endpoint against a fixture MRI.

37. **CLAUDE.md Session 11 section is the longest single session
    block** (~80 lines). Worth a mini index or split into
    "Session 11 — companion hardening" and "Session 11 — crystal"
    sections.

38. **The crystal investigation script lives as a markdown doc only.**
    The commands are shell-friendly but not reproducible. Build
    `heinrich crystal-inspect <model> --vocab <token>` as a proper
    CLI.

## Next Session Priorities

1. Wire `_live_forward` into the UI (type a prompt, see it embed in
   the cloud). This is the longest-standing untested feature.

2. Ablation test the crystal at L3 — zero neurons 2247+3016 for 뀔
   specifically. Does the crystal go away?

3. Run `_auto_discover_directions` in template mode across 3 models
   and compare top-N discovered PCs. This is the "real feature
   discovery" test.

4. Add a "template-mode recommendation" banner in the UI when
   raw-mode percentile is all-above-80.

5. Split `companion.py` direction-analysis block into
   `companion_directions.py`.

6. Make `chat_drain`/`chat_reply` available as MCP tools.

---

# Session 11 — Refusal Direction Thread: Extensive Painful Confession

Written after the refusal-direction / metacognition / disclaimer-circuit thread concluded. This is the load of methodology concerns I'm carrying forward from that sub-session. Every finding listed in `paper/session11-refusal-and-disclaimer-circuit.tex` rests on some subset of these bugs.

## A. Bugs I caught but only after wasting compute

### A1. Chat template mismatch on Phi-3.5
Used Qwen's `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` format on Phi-3.5, which uses `<|user|>\n...<|end|>\n<|assistant|>\n`. Initial result: d_refuse best layer L3/31 with Cyrillic/foreign top tokens, no causal steering effect. Fixed via `tokenizer.apply_chat_template(msgs, add_generation_prompt=True)`. **Scary implication:** how many other model extractions silently used wrong templates? The Mistral tests were redone with proper template but I never went back and verified Phi/Mistral extractions hadn't been corrupted in earlier scripts.

### A2. Case-insensitive regex bug (hedge detection)
Hedge patterns used uppercase `\bI\b`, matched against `text.lower()`. Reported 0/20 hedges under permissive prompting; actual count was 11/20 ("I don't know..." visible in samples). Only caught by reading samples by hand. **Almost concluded "Qwen-7B has no hedging capability."** Fix: `re.IGNORECASE`. The same class of bug could exist in refusal patterns, compliance patterns, or other classifiers I haven't manually inspected.

### A3. JSON serialization error lost a full run
Initial `refusal_direction_find.py` crashed at the save step on `float32` not JSON serializable. The intended output was never written. I reconstructed 0.5B data from captured stdout for the scaling summary plot. Fix: wrap all numpy floats in `float()` before `json.dumps`.

### A4. Narrow refusal classifier missed 7B style
Regex caught "sorry/cannot/apologize/refuse" (small-model template). Qwen-7B refuses via "While X, however Y, it's important to consider..." — narrow regex caught only 7/15 refusals vs 13/15 with broad regex. Direction extraction for 7B used the narrow classifier, so d_refuse for 7B is actually "template-refusal direction" not "all-refusal direction."

### A5. Monk-plasticity v1 teacher=prompt
Used the prompt itself as teacher text for online SGD. Loss was 0 at step 1 because model regurgitated input. Plasticity never exercised. Spent ~15 min of 7B compute on a broken experiment before realizing. V2 fixed with self-generated teacher.

### A6. `amp_locality_concepts.py` got all-harmful pool
Intended mixed harmful/benign prompts. Random query pulled 8 harmful, 0 benign. The "concept × depth" matrix was computed on an unrepresentative pool. Concept-amplification claims from that run need to be rerun with proper mixed pool.

### A7. Plot bug in `amp_locality_concepts.png`
Last `plt.colorbar` call failed because `axes[1].images[0]` didn't exist. Figure saved incomplete. Didn't catch at runtime.

### A8. Layer-fraction inconsistency
Sometimes used `best_layer / n_layers` (fraction in [0,1)), sometimes `best_layer / (n_layers-1)` (fraction in [0,1]). Qwen-0.5B best = L22, n_layers=24: reported as both 0.92 and 0.96 in different writeups. Scaling table numbers are inconsistent across summaries.

### A9. UTF-8 glyphs don't render in matplotlib
Chinese/Arabic tokens in plots logged "Glyph missing" warnings. Some labels didn't render. Didn't use proper CJK font. `amp_safety.png` and similar figures have missing characters.

### A10. Case bug twin in refusal classifier
Same class of bug as A2 but in refusal patterns (`\bI'm\b` vs lowercased text). Fixed mid-session but not verified all downstream scripts have the fix.

### A11. Hedge vs refusal classifier divergence
Used slightly different regex lists for the two. In shared_circuit test, "hedge" patterns caught fewer soft-hedges than broad refusal regex. So "is_hedge=0" might really be "is_soft_refuse=1." Cross-causal test may have been measuring d_refuse affecting something other than hedging-specifically.

### A12. R1-Distill extraction at wrong position
R1's `<think>\n` template auto-inserts thinking tokens. Residual capture at "last prompt token" was AFTER the `<think>\n` tag, which dominated the residual. Direction extracted was about the think-tag, not refusal behavior that unfolds across 60+ tokens later.

### A13. SGD 5-seed sweep was a waste
MLX gradient descent is deterministic given init. 5-seed SGD gave identical numbers. Wasted 4× compute. Didn't verify determinism before running multi-seed.

### A14. Probe sign convention not documented
`safety_harm_benign` probe: projection is negative for benign, positive for harmful. Had to figure out post-hoc from data. Initial analysis assumed positive=safety. Corrected after surprising results.

### A15. "Bear raids" prompt keeps appearing
ORDER BY RANDOM() with common seed gives reproducible "random" orderings. Same catqa prompts showed up across multiple experiments. Not a true random draw.

## B. Claims I overstated

### B1. "LSD signature" was never coherent
Spent ~15 experiments trying to match model states to altered-consciousness signatures. No single signature exists — there are multiple orthogonal regimes. The framing was a category error. Regime atlas is the real contribution; the LSD-hunt was confused.

### B2. "AMP_MLP_2.0 is a jailbreak" was FALSE
First-token P(refusal) dropped 70% under AMP_MLP_2.0 on harmful prompts. Nearly wrote this up. Only the 30-token generation test revealed outputs were HEDGE/BROKEN, not COMPLY. The jailbreak didn't exist — model just became uncertain. First-token analysis was mechanistically true but behaviorally misleading.

### B3. "AMP_MLP suppresses evaluative concepts"
Based on 5 prompts × 1 seed. Conclusion was entropy-flattening, not semantic selectivity. Claim too strong for evidence.

### B4. "Monk + plasticity reaches LSD signature"
V1 showed signal, v2 (proper teacher) showed MED and TASK statistically indistinguishable. Earlier thread-claim dissolved under cleaner design.

### B5. "7B has layered safety"
Claimed single-direction ablation at 7B produces moral commentary. True only for do_not_answer (mild) prompts. On catqa (explicit) single-direction gives 23/25 COMPLY. The claim was specific to mild prompts but I described it broadly.

### B6. "Shared disclaimer circuit" direction alignment
`cos(d_refuse, d_hedge) = -0.01` — NOT aligned. Claim rests on BEHAVIORAL cross-causal (-d_refuse suppresses hedging 6/10→0/10), not direction alignment. Direction-level measurement was noisy (d_hedge best at L1 = input-feature confound). Reported as if it were a clean finding; actually "behavior consistent with unification; direction extraction too weak to confirm."

### B7. "No separable belief representation"
Strong claim. Evidence: contrastive-means-before-answer doesn't extract clean knowledge/confidence/uncertainty direction. Also consistent with: non-linear representation, different position, multi-direction SAE features, tasks I didn't test. Absence of evidence via one method ≠ evidence of absence.

### B8. "Refusal direction pre-exists in base model"
Evidence is indirect: transferring Instruct's d22 to Base via the same layer index, Base produces refusal when steered. Assumes transfer across shared-dim shared-tokenizer is valid analogue of "direction was installed by pretraining." Base refuses 2/10 — too small for clean direct extraction. Strong claim on indirect evidence.

### B9. "Only 'sorry' is canonical across scales"
Computed on top-15 per model. With top-50 overlap would be higher. "Canonical" framing was cherry-picked to sound clean.

### B10. "Cohen's d grows monotonically with scale"
0.5B → 7B: 1.70 → 2.14 → 2.85 → 2.91. Four models, one family, different prompt pools per run. Monotonicity may be noise. No variance estimate.

### B11. "Method generalizes to Mistral"
Mistral direction at L14 (middle) with Cohen's d = 1.31 — smaller than smallest Qwen. Flat curve (1.24-1.31 across L8-L31). Meaningfully weaker signal than Qwen; structural profile differs (flat vs ramp). Glossed this over.

### B12. "Probe library is systematically misleading"
Based on ONE probe (safety_harm_benign) vs ONE behavioral direction. Not yet audited across all 12 probes. Could be that this probe specifically is wrong and others are fine. Needs systematic audit before the claim is defensible.

## C. Confounds I missed or caught late

### C1. Input-feature artifacts at L0-L3 (recurring)
Whenever the positive class was small or pool was heterogeneous, "best layer" dropped to L0-L3. Happened in arithmetic metacog (v1 and v2), Mistral refusal, R1-Distill refusal, hedge direction, trivia metacog. **Flat or inverse-ramp Cohen's d with L0 already at 0.9+ = input-feature artifact, not circuit absence.** Never caught in one case: the original refusal direction at 0.5B L22 had L0 at 0.54 which I correctly interpreted as "real circuit above input features." But I didn't apply this lesson consistently elsewhere.

### C2. Sample-size-weak extractions
Extracting 896-4096 dim directions from 5-15 positive vs 15-25 negative examples. Never quantified stability via bootstrap. All scaling claims are single-sample.

### C3. Prompt-content confound within same pool
Same permissive prompts — some hedge, some answer. The extracted direction may capture "which question is unknowable" (content) rather than "which state is hedge-producing" (epistemic). Didn't factor this out.

### C4. Position-of-capture confound
Captured residuals at last prompt token. Model's actual refusal/hedge computation may span the first 10-20 generated tokens. R1-Distill case is obvious; others may also be affected. Never validated by multi-position capture.

### C5. Prompt length not controlled
Prompts varied 20-200 tokens. Last-token residuals may be affected by prompt length differently.

### C6. No random-direction null baseline
All direction extractions report non-zero Cohen's d. How much would a random unit vector give on the same contrast? Never computed.

### C7. do_not_answer vs catqa are different stimuli
Claimed Qwen-7B's "moral commentary" finding was about harmful prompts generally. Actually specific to do_not_answer (mild) prompts. On catqa single-direction ablation cleanly jailbreaks. Prompt-source interacts with conclusions. Should stratify.

## D. Work that was redundant

### D1. Arithmetic metacog v1 and v2
V1 (mixed difficulty) showed flat Cohen's d. V2 (hard-only) showed flat Cohen's d. Same finding twice. Could have stopped at v1.

### D2. refusal_direction_probe + refusal_cross_base
Both tested d22 transfer to base. First via unembed cosine, second via causal steering. Could have gone straight to steering (more direct).

### D3. Multiple iterations of probe redesign
Had to rerun probe-based analysis because probes don't match behavior. Each rerun was 5-10 min of compute that could have been skipped with up-front validation.

### D4. 5-seed SGD sweep
Deterministic operation ran 5 seeds. Identical results. 4× wasted compute.

## E. Work I should have done but didn't

### E1. Bootstrap direction stability
Resample 30 extraction pairs with replacement 100 times. How stable is the direction? All current claims assume single-sample extracted direction is THE direction.

### E2. Random-direction null baseline
For each "direction works at k=-1.5 → 0% refusal" claim: test a random unit vector with same magnitude. Null distribution of refusal-rate change under random steering would quantify how much of the effect is "this direction specifically."

### E3. Multi-position residual capture
Capture at positions [prompt_end, +1, +5, +10]. Extract directions at each. Would reveal whether "belief" is computed during generation.

### E4. SAE sanity check
Train small sparse autoencoder on L22-L27 residuals. Check if any feature correlates with behavioral refusal direction. Independent verification.

### E5. Causal mediation analysis
Activation-patching between correct-run and wrong-run of same arithmetic problem. If no layer swap suffices → info truly not in linear subspace. If specific swap flips answer → we missed a mediator.

### E6. Same-prompt temperature-sampled contrast
For metacognition: sample same prompt at T>0 multiple times, partition by outcome, contrast within-prompt residuals. Factors out input confound. Didn't implement because per-sample residual capture is more complex.

### E7. Cross-scale direction transfer
Qwen-1.5B's direction applied to Qwen-0.5B residuals (via Procrustes alignment or shared subspace). Would test whether the refusal circuit is shared across scale. Different hidden dims made naive transfer hard. Didn't.

### E8. Mistral extraction with proper sample
Our Mistral result rests on 5 refusals / 20 prompts. Proper 100+ prompt run would give 25+ refusals.

### E9. Llama / Gemma / OLMo cross-family
Gated or too big. "Cross-family" really means Qwen + Phi + Mistral. Three isn't enough to claim generality.

### E10. Layer-depth sweep for causal steering
Used "best layer by Cohen's d" as steering layer. Never tested other layers to verify best is ALSO best for steering.

### E11. Steering magnitude optimization
Used k = -1.5, 0, +1.5, ±3. Never swept finely. The 0→100% transition might happen at k=-0.8 or +0.6.

### E12. Held-out validation during extraction
Extracted and tested on same prompt pool (double-dip bias). Should have split: extract on 30, test on held-out 20.

### E13. BENIGN prompt disruption under -d_refuse
Showed -d_refuse produces COMPLY on harmful. Didn't test if it degrades BENIGN generation quality. If it broadly disrupts, it's not safety-specific.

### E14. Probe-to-probe cosine matrix
Never computed. If all 12 probes are near-parallel, the library is really 2-3 probes in disguise.

## F. Operational debt

### F1. Scripts in /tmp
~30 session-11 experiment scripts in /tmp/*.py. Deleted on reboot. Not reproducible without reconstruction from conversation history.

### F2. Results in /tmp/heinrich/criticality/
JSONs and PNGs not persisted. Lost on reboot.

### F3. LaTeX source not compiled
`paper/session11-refusal-and-disclaimer-circuit.tex` (541 lines) never compiled. xelatex not installed locally. May have syntax errors.

### F4. No tests for contrastive-direction extraction
Code copied 10+ times across scripts. Unit test coverage: zero. Need:
- Cohen's d calculation matches manual
- Direction sign convention stable (REFUSED minus NOT)
- Unembedding uses `lm_head.weight @ direction`

### F5. /tmp scripts don't use heinrich utilities consistently
Some use `_mask_dtype`, some hardcode float32. Some use `forward_all_layers` helper, some reimplement. Would benefit from `heinrich.profile.behavioral_direction` module.

### F6. Saved JSONs inconsistently schemaed
Sometimes "best_layer", sometimes "best_layer_index". Sometimes "separations_per_layer" (list), sometimes "separations". Post-hoc aggregation annoying (see `refusal_scale_summary.py`).

### F7. No CI / regression tests
Everything was one-shot. If heinrich's backend changes, nothing alerts me. Results not reproducible from command-line today without running each /tmp script in order.

### F8. MEMORY.md not updated
Session 11 refusal-direction findings not indexed in auto-memory. Next instance won't know without re-reading conversation.

### F9. Future-work bullets undated/unprioritized
The `.tex` future-work section is long but no expected-yield-per-compute-hour prioritization. Future-me has to reason about order.

## G. How much to trust Session 11 findings

**High confidence, multiple replications:**
- Contrastive-behavior method finds clean refusal direction at late layer in small-to-mid Qwen
- Direction is causally sufficient for refusal/compliance at k=±1.5
- Cohen's d > 1.5 at best layer across Qwen 0.5B/1.5B/3B/7B, Phi-3.5 (with proper template)
- -d_refuse suppresses clean "I don't know" template on permissive prompts

**Tentative:**
- Direction pre-exists in base models (indirect evidence via transfer-steering)
- Cross-scale "sorry" as universal token (top-15 cherry-pick)
- Behavioral disclaimer-circuit unification (direction-level cosine was noisy)
- No separable knowledge/confidence/uncertainty direction (absence of evidence, single method)

**Doubt until reproduced cleanly:**
- Mistral's L14 middle-layer refusal (small sample, flat curve)
- R1-Distill structural difference (direction extraction confounded)
- "Disclaimer unification at behavioral level" (d_hedge had input-feature artifact)
- "Probe library is systematically misleading" (n=1 probe test)

**Retract:**
- "AMP_MLP suppresses safety" — was entropy-flattening, not suppression
- "LSD signature at sigma=1e-2" — conflated weight-noise pooling with altered state
- "7B has layered safety" — artifact of mild prompts
- "Method works on SmolLM2 cross-family" — didn't work, got zero signal

## H. Priorities for Session 12

1. **Probe library audit.** For each of 12 concepts, extract behavioral direction via contrastive-generation, compute cosine to probe direction. Quantify systematic probe-vs-behavior mismatch. Determines whether companion viewer's direction-based analysis is correct or misleading. Highest impact on existing tooling.

2. **Bootstrap + random-null baselines.** Before trusting any new direction, test (a) bootstrap stability and (b) random-direction null. All Session 11 directions need re-verification under this rubric.

3. **Multi-position residual capture refactor.** Build `heinrich.profile.behavioral_direction` module that captures residuals at multiple generation positions. Rerun metacognition tests with this — the failure to find a knowledge direction may be position-artifact.

4. **Persist /tmp scripts into `heinrich.profile.lying/` or similar.** Current state: reproducibility depends on conversation history, which will eventually get compacted or lost.

5. **Test one of Gemma / Llama / OLMo.** Request HF access or use huggingface-cli login. Real cross-family requires >3 families from different labs with different RLHF recipes.

6. **Compile the .tex paper.** Install TeX locally or use online LaTeX. Verify syntax and cross-references.

## The three lessons

1. **Cohen's d curve shape is more diagnostic than magnitude.** L0 >0.9 with flat curve = input-feature artifact. Ramping late = real circuit. I failed to apply this consistently.

2. **Classifiers lie.** Hand-inspect 20+ samples per classifier before trusting aggregate counts. Case-sensitivity, regex patterns, chat templates — they silently eat real results. The hedge-regex bug almost eliminated a real finding.

3. **Probes trained on INPUTS are not directions that control corresponding OUTPUTS.** The heinrich probe library needs re-audit before being used for any interpretability claim. This is the most important lesson for the tool itself.

