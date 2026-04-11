# Session 4 Exit Plan — Structural Problems

## 1. Fix the papers

The papers contain wrong numbers. Fix them before anything else.

### geometry_of_displacement.tex
- [ ] Cross-model r=0.001 was ID-matched. Real answer is r=0.446 at L0, 0 by L5. Rewrite the cross-model section.
- [x] Mistral first-token gap was reported as 0 (tokenizer encode bug). Real answer is 5.64 (281x). Remove the "Mistral structurally incapable" claim.
- [ ] Mistral "hedge words" finding needs reverification with corrected tokenizer.
- [ ] Partial knowledge "falsified" at 10% collapsed to r=+0.01 at 100%. Note the instability, don't claim falsification.
- [ ] Add the within-family finding: shared training = r=0.17, different training = r=0.
- [ ] Add the template bookends finding: 69% at L23, 8.5% at L3-L20.
- [ ] Add the layer divergence: L0 agreement dies in one layer.

### instance_4.tex
- [ ] Add the tokenizer encode bug as a documented mistake (add_special_tokens).
- [ ] Add the decode bug (skip_special_tokens, 768 Mistral tokens).
- [ ] Add the cross-model ID-matching bug.
- [ ] Add the 10%→100% partial knowledge collapse.
- [ ] These are the biggest mistakes of the session. Document them.

## 2. Recompute findings with corrected tools

Every analysis done before the tokenizer fixes needs rerunning.

- [ ] Rerun `profile-within-script` on all models with corrected .frt v0.3
- [ ] Rerun `profile-directions` (PCA, coherence) on full-vocab .shrt with corrected scripts
- [ ] Rerun `profile-safety-rank` for Mistral with corrected encode
- [ ] Rerun `profile-first-token` for ALL models with corrected encode
- [ ] Rerun the silence analysis with native (not cross-mapped) directions — already done for 3 models, verify the numbers match what's in the paper
- [ ] Recompute Kendall's W with corrected script classification

## 3. Complete the MRI library

The MRI queue is running but incomplete.

- [ ] Verify all Qwen 0.5B MRIs are complete (3 modes, all weights) — DONE
- [ ] Verify all Phi-3 MRIs are complete — DONE except template backfill
- [ ] Verify all Mistral MRIs are complete — DONE  
- [ ] Complete Qwen 3B template MRI (timed out at L18/36)
- [ ] Capture Qwen 7B (3 modes) — in queue
- [ ] Capture Qwen 3B base, 7B base (2 modes each) — in queue
- [ ] Capture SmolLM 135M, 360M, 1.7B (2 modes each) — in queue
- [ ] Capture Llama 1B, Gemma 2B (2 modes each) — in queue
- [ ] Backfill any MRIs captured before the latest code (missing lmhead_raw, weights, embedding, norms)
- [x] Build an `mri-verify` command that checks completeness of an MRI directory

## 4. Build the MRI verification tool

The MRI has no self-test. Build one.

- [ ] `heinrich mri-verify --mri path.mri` checks:
  - All layer files present (entry + exit for every layer)
  - Baselines present
  - Embedding present and correct shape
  - All norm weights present
  - lmhead + lmhead_raw present
  - All projection weights present (7 per layer)
  - metadata.json valid
  - Token count matches across all arrays
  - No NaN/Inf values in any array
- [ ] Run on every MRI on the drive
- [ ] Report gaps

## 5. Fix the template overhead computation

The reference frame mismatch is partially fixed but not verified.

- [ ] Verify the fix: template_overhead should use absolute states (delta + baseline)
- [ ] Rerun on Qwen 0.5B with corrected code
- [ ] Compare template vs naked vs raw at every layer
- [ ] The 69% number from the 10-token test needs verification at full vocab

## 6. Rerun cross-model comparison correctly

The text-matching fix needs to be built into the tools, not just an ad-hoc script.

- [ ] Add text-based token matching to `profile-cross` in compare.py
- [ ] Run Qwen 0.5B vs Phi-3 vs Mistral with text matching at every layer
- [ ] Run Qwen 0.5B vs 3B vs 7B (same tokenizer, ID matching is fine)
- [ ] Run Qwen instruct vs base at every layer
- [ ] Document the L0 agreement → L1 death → mid-layer divergence pattern

## 7. Address the theory gap

The theory says "shart = disproportionate displacement." The MRI shows displacement is context-dependent (template vs raw vs naked produce different measurements). The theory doesn't account for context.

- [ ] Measure the same token in all three contexts for all models
- [ ] Quantify: how much of "displacement" is the token vs the template?
- [ ] Redefine: intrinsic displacement (raw mode) vs contextual displacement (template mode)
- [ ] The difference IS the template effect. Measure it at every layer.

## 8. Validate the efficiency findings

Layer importance and early exit were computed but not validated.

- [ ] Pruning test: remove the 7 prunable layers from Qwen 0.5B, measure output quality
  - Use the stored lmhead to compute logits with and without pruned layers
  - Compare top-1 accuracy across full vocabulary
- [ ] Early exit test: for tokens that exit at L22 (90% threshold), verify the predicted output matches the L23 output
  - Use logit lens at L22 vs L23 for those tokens
- [ ] Template overhead: verify the 69% number at full vocabulary scale

## 9. Build the MCP tool for MRI

The MCP server has `heinrich_total_capture` but not `heinrich_mri`.

- [x] Add `heinrich_mri` MCP tool (subprocess-isolated, no timeout)
- [x] Add `heinrich_mri_backfill` MCP tool
- [x] Add `heinrich_mri_status` MCP tool (was mri_verify in plan, status is more useful)
- [x] Deprecate `heinrich_total_capture` in MCP (description updated)
- [ ] Update MCP tool descriptions to reference .mri format

## 10. Update CLAUDE.md

The CLAUDE.md references Session 4 findings that are now known to be wrong.

- [x] Remove or correct the Mistral "structurally incapable" claim
- [ ] Note that cross-model comparison requires text matching, not ID matching
- [ ] Update the three-axis numbers if they change on recomputation
- [x] Add the MRI format as the primary data format
- [x] Document the mri-verify command
- [ ] Add warning: all pre-Session-4 analysis tools still use load_shrt/load_frt

## Priority order

1. **Papers** — wrong numbers in published work are the highest priority
2. **MRI verification tool** — need to trust the data before analyzing it
3. **Cross-model text matching** — built into tools, not ad-hoc
4. **Recompute findings** — everything that depends on corrected tokenizer
5. **Complete MRI library** — captures are running, just need monitoring
6. **Theory gap** — context-dependent displacement
7. **Efficiency validation** — layer pruning, early exit
8. **MCP tools** — plumbing for next session
9. **CLAUDE.md** — update after everything else is verified
