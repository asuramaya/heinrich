# Session-12 Bugs, Problems, and Meta-TODOs

Captured over the course of session 12 across Heinrich, Decepticons, and the
Sharts MRI library. Each entry notes: **what's wrong**, **impact**, **status**,
and **next step**.

---

## Bugs found and fixed

### B1. Raw-mode CB MRI shape mismatch (`profile/mri.py:441,497`) — **FIXED**
- **What**: `_capture_mri_causal_bank` pre-allocated `substrate_states` at
  `cfg.n_modes` but adaptive-substrate models return
  `_last_features = concat(substrate_modes, x_embed)` whose trailing axis is
  `n_modes + embed_dim`. Captures errored with
  `ValueError: could not broadcast input array from shape (3456,) into shape (3072,)`.
- **Impact**: Every raw-mode capture on a byte-level adaptive-substrate model
  with mismatched n_modes/embed_dim failed outright. Most session-11 raw CB
  MRIs for s12 models never existed.
- **Status**: Fixed by probing shape from the first forward and allocating
  actual shape. Metadata `hidden_size` now reflects captured width. Commit
  `057707e`.

### B2. Non-adaptive forward-path parallel implementation (`decepticons/loader.py`) — **FIXED**
- **What**: The non-adaptive branch of `CausalBankInference.forward_captured`
  reimplemented the model's forward inline: `_linear_states`, manual band
  readout or `_linear_logits`, `_local_logits`, hand-rolled logit composition.
  Drifted from the training forward whenever plugins were added.
- **Impact**: Same class of bug as session-11's hash_memory silent-bypass.
  Future plugins in `_linear_logits` / `_local_logits` would silently drop
  out of MRI captures without errors.
- **Status**: Session-11 P2 deferred. Now collapsed. Model-side stashes:
  `_last_states_nonadaptive`, `_last_x_embed_nonadaptive`,
  `_last_band_logits_list`, `_last_local_logits`. Loader reads stashes.
  Regression fixture `test_forward_captured_matches_training_forward_nonadaptive`
  asserts max|Δ|=0 on a non-adaptive config with local path enabled. Verified
  bit-identical on gated-delta-cr-s8, lasso-pos-b2-s8, byte-bands-s8.
  Decepticons commit `a3fb77e`.

### B3. Val-data mismatch on 4 byte-level MRIs — **FIXED**
- **What**: 4 session-11 MRIs (`byte-bands-s8-50k`, `byte-curriculum-s8-50k`,
  `byte-patch4-s8-50k`, `byte-seqlen4096-s8-50k`) had stale
  `provenance.val_data` pointing at `fineweb_val_000000.bin` (sp-tokenized,
  vocab 1024/8192) while the models are byte-level (vocab 256). Pre-session-11
  heinrich silently mangled the data via the byte-shard format bug; after the
  fix, byte models correctly reject with `IndexError: index out of range in self`.
- **Impact**: 4 recapture failures in the session-12 bulk pass. Any byte
  model trained with a stale val_data path will now error cleanly.
- **Status**: `heinrich mri-recapture` auto-corrects byte-level models to
  prefer a `_bytes.bin` sibling if the recorded val_data is non-byte-level.
  Fixed during session 12, commit `057707e`.

### B4. Binary `two_band_partition` heuristic too strict — **FIXED**
- **What**: Heuristic was `top8 < 0.01 AND max_tail > 0.05`. W2-B at step 50k
  had `top8 = 0.030` (modest leak into PC0-7) but `PC8-19 = 0.415` (strongly
  concentrated position). Ratio 14, clearly partitioned. Binary flag flipped to NO.
- **Impact**: Under-reporting of partitions with any leak, including
  well-established partitioned models. Misled session-12 W2-B analysis
  temporarily.
- **Status**: Replaced with continuous `partition_score =
  max_tail_pos_r2 / max(top_pos_r2, 0.005)` and categorical `partition_verdict
  ∈ {absent, leaky, partial, partitioned}`. Thresholds: tail < 0.02 → absent;
  score < 2 → leaky; score ∈ [2,5) → partial; score ≥ 5 → partitioned.
  Back-compat `two_band_partition` bool now triggers on partial OR partitioned.
  4 new unit tests covering every verdict. Commit `3a404f0`.

### B5. Dataclass config is frozen → can't monkey-patch `local_scale=0` at inference — **WORKED AROUND**
- **What**: `CausalBankConfig` is a frozen dataclass. Setting
  `model.config.local_scale = 0.0` at inference raises
  `FrozenInstanceError: cannot assign to field 'local_scale'`.
- **Impact**: Forensic test "ablate local path at inference" must monkey-patch
  the `_local_logits` method instead. Works but more invasive.
- **Status**: Work-around is documented. If chronohorn wants an
  `--inference-local-scale 0` override path it can be added to `CausalBankModel`
  as a second, mutable override attribute. Not blocking.

---

## Corrupted or broken checkpoints in the library

### CK1. `byte-hrr-frozen-s12-50k.checkpoint.pt` — NaN weights (training diverged)
- **What**: 21 of 30 tensors NaN-saturated. `linear_embedding.weight`,
  `linear_readout.*`, router bias, all expert weights — all NaN at 100% density.
- **Impact**: Every forward pass produces NaN substrate and NaN logits. The
  session-11 MRI of this model showed all-NaN substrate (`884736/884736 NaN`).
  Heinrich captured the NaN faithfully; the model is cooked.
- **Status**: Session-12 saw the file size shrink from session-11's 332MB to
  session-12's 158MB, suggesting a clean retrain. Needs re-verification via
  `mri-health` against the new file.
- **Next**: Chronohorn should add a NaN-detection tripwire at checkpoint save.
  A single `torch.any(torch.isnan(param))` check at save-time would have
  prevented this.

### CK2. 2 W2-A intermediate checkpoints corrupted (zip-read fail)
- **What**: `s12w2-cr-s12-scandead-20k_step5000.checkpoint.pt` and
  `_step15000.checkpoint.pt` fail to load with
  `PytorchStreamReader: failed finding central directory. Checkpoint file is corrupted.`
- **Impact**: Can't MRI step5k / step15k of W2-A; trajectory has gaps.
- **Status**: Unresolved. Both files exist on the shared volume but are
  unreadable.
- **Next**: Chronohorn should re-export or retry. Consider checksum verification
  in the chronohorn drain/pull pipeline.

---

## Library hygiene issues

### H1. Frontier-class contamination (sp1024 models in byte frontier)
- **What**: `cb-delta-s12-h8-s32-*` models report `1.807/1.808 bpb` in the
  chronohorn DB but are actually sp1024-tokenized. The DB value is derived by
  chronohorn's ingestion using a hardcoded `2.436 bytes/sp1024-token`
  conversion because `dataset.test_bytes_per_token` is `null` in the JSON.
- **Impact**: Direct comparison to byte-level frontier values (1.785, 2.022)
  is apples-to-oranges. These models are not byte-level.
- **Status**: Chronohorn has started segregating via `_illegal_sp1024/` subdir
  in `session12/`. Good first step but not enforced at query time.
- **Next**: Chronohorn DB query should WHERE-clause on `vocab_size = 256` OR
  on an explicit `family='byte'` tag. Enforce `test_bytes_per_token` non-null
  at ingestion for any sp-tokenized entry.

### H2. MRI library still has pre-session-11 stale MRIs outside session11/
- **What**: `heinrich mri-check-fixups` reports stale MRIs (fix_level < 2) in
  `session10_byte_scaling/`, `session_archive/`, `old/`, etc. Session-12
  `mri-recapture` only processed `session11/` directory in the bulk pass.
- **Impact**: Cross-session analyses using those older MRIs could mix
  fix_level=0 data (wrong loss values) with fix_level=2.
- **Status**: Tool exists; operational pass on other directories pending.
- **Next**: Run `heinrich mri-recapture --dir /Volumes/Sharts/heinrich/session_archive --execute`
  and same for `session10_byte_scaling`. Verify with `mri-check-fixups`
  showing 0 stale.

### H3. 3 MRIs still un-recapturable by naming convention
- **What**: `byte-hrr-persistent-warm.seq.mri`, `byte-seqlen4096-cold.seq.mri`,
  `byte-seqlen4096-warm2048.seq.mri`. Idiosyncratic names don't match
  `<base>.checkpoint.pt` or `<base>-{M}k_step{N000}.checkpoint.pt` patterns.
- **Impact**: These 3 stay at fix_level=None indefinitely unless recaptured
  manually.
- **Status**: Listed as skipped by `mri-recapture`.
- **Next**: Either (a) manually capture with explicit `--model` path or
  (b) add more naming patterns to `_resolve_recapture_source` if a
  generalizable rule emerges.

---

## Meta-TODOs for Heinrich

### T1. Factor the Q3 context-knee test into a CLI tool
- **What**: The "effective context length" test (per-position loss curve
  with random-prefix sequences) is currently an inline Python script run
  twice in session 12. It's the cleanest diagnostic for context-ceiling
  questions.
- **Why**: Every future architecture experiment needs this measurement.
  Sliding-window attention, Mamba pilots, hash-memory-fixed — all questions
  about whether mechanism X unlocks long context route through this test.
- **Proposed**: `heinrich profile-cb-effective-context --model <ckpt>
  [--val <bytes.bin>] [--seqlen 512] [--n-trials 30]`. Outputs per-ctx-bucket
  bpb table + identifies the knee (first bucket transition below threshold).
- **Priority**: HIGH. The most repeatedly-useful forensic we've built.

### T2. Factor the Q4 substrate-ablation tests into CLI tools
- **What**: Three tests used in session 12 but written inline:
  1. Substrate truncation (zero modes beyond rank-K, measure bpb)
  2. Substrate zero-out (zero all substrate modes, x_embed tail only)
  3. Local-path ablation (monkey-patch `_local_logits`)
- **Why**: Answer "is this model over-parameterized / feedforward /
  substrate-primary vs local-primary" without writing new scripts each time.
- **Proposed**: `heinrich profile-cb-ablations --model <ckpt> --ablate
  <substrate|local|truncate:K>`. Outputs bpb delta per ablation.
- **Priority**: MEDIUM. Can be done after the Mamba pilot.

### T3. Session-11 book retraction note on EffDim undercount
- **What**: Session 11 book quotes EffDim=8-15 on substrates with two-band
  partitions. Those numbers are under-counted by 2-5× on any substrate with
  `ConR² > 0.1 AND PosR² > 0.01` simultaneously (the two-band signature).
- **Why**: Readers may interpret quoted EffDim as working dimensionality
  and misjudge architectural capacity.
- **Proposed**: Add a retraction paragraph to
  `docs/session11_book.tex` in the retractions chapter referencing
  `profile-cb-pc-bands` as the replacement tool for two-band substrates.
- **Priority**: LOW but should happen before session-11 book is cited.

### T4. MRI metadata should carry `architectural_family`
- **What**: The session-12 dichotomy (substrate-primary vs local-primary)
  isn't inferable from current metadata without re-reading the checkpoint
  config. Every analysis tool ends up reconstructing it ad-hoc.
- **Proposed**: Add `metadata.model.family` field derived at capture time
  from `(local_scale > 0, substrate_mode, half_life_max, adaptive_substrate)`.
  Values: `substrate-primary`, `local-primary`, `hybrid`.
- **Priority**: MEDIUM. Quality-of-life; not blocking.

### T5. `provenance.model_path` absolute paths break on volume remount
- **What**: `mri-recapture` relies on the recorded `model_path`. If the
  Sharts volume is remounted at a different path (e.g., running on another
  host), the recorded absolute path is invalid.
- **Proposed**: Store both the absolute path AND a path-relative-to-Sharts.
  `mri-recapture` tries both, preferring the one that exists.
- **Priority**: LOW. Would only bite in multi-host setups.

### T6. `heinrich profile-cb-additivity` sometimes silently rounds ω/geometry
- **What**: `_cb_additivity_metrics` uses a 5000-sample SVD even for
  large MRIs with 25k+ tokens. Tail-PC position R² measurements may be
  noisy. OK for quick check, not for publication-grade.
- **Proposed**: Add `--svd-samples N` flag (default 5000; pass 25000 to
  SVD the whole substrate).
- **Priority**: LOW. Useful tightening; not blocking.

### T7. `profile-cb-pc-bands` noise bars not surfaced
- **What**: The partition score is deterministic (fixed SVD sample, fixed
  train/test split seed), but per-PC R² has sampling noise that's not
  reported.
- **Proposed**: Add `--n-bootstrap K` flag; bootstrap the train/test split
  and report per-band pos_r2 ± SEM.
- **Priority**: LOW. Would add confidence to marginal verdicts (e.g., the
  W2-B step-20k "partial" call).

### T8. Silent partial-forward-capture failures
- **What**: When `forward_captured` returns some `None` values for keys the
  analysis expects (e.g., `band_logits` on a single-band model), downstream
  tools sometimes silently use the None rather than triggering a clean
  `no band info in this MRI` error.
- **Proposed**: Audit `profile/compare.py` for `result.get(...)` calls and
  ensure they either assert-non-None or document a skip path.
- **Priority**: LOW but growing; accumulates silently.

### T9. Tool fingerprint doesn't include chronohorn SHA
- **What**: `provenance.tool_fingerprint` records heinrich + decepticons
  SHAs. Chronohorn's commit (which determines training pipeline, dataset
  preprocessing, etc.) is not recorded on the MRI.
- **Proposed**: The MRI's source checkpoint was produced by a specific
  chronohorn commit; that SHA lives in the result JSON, but not the MRI.
  Propagate at capture time if discoverable from the result_json path.
- **Priority**: MEDIUM. Makes cross-session analyses fully self-describing.

### T10. No automated test for the Q3 knee result
- **What**: The "16-byte substrate-primary ceiling" finding is the load-bearing
  session-12 result. Nothing in the test suite would catch a regression if
  future decepticon changes accidentally extend or restrict this.
- **Proposed**: Add an integration test `test_cb_effective_context.py` that
  loads a tiny substrate-primary model and asserts the knee is at the
  expected position. Shorthand: the substrate-primary family is defined
  partly by this knee; the test anchors it.
- **Priority**: LOW. Nice-to-have regression guard for a finding we want to
  track.

---

## Chronohorn-side asks

### C1. NaN-detection tripwire at checkpoint save
- See CK1. Prevents future NaN-corrupt checkpoints.

### C2. Checksum verification on checkpoint pull/transfer
- See CK2. Prevents corrupted checkpoints in the Sharts library.

### C3. Enforce `test_bytes_per_token` non-null in result JSON ingestion
- See H1. Prevents frontier-class contamination.

### C4. Add chronohorn SHA to result JSON `provenance`
- See T9. Propagate commit-level reproducibility upstream.

### C5. Record `architectural_family` tag at train time
- See T4. Set to `substrate-primary` if `local_scale == 0`, else
  `local-primary`. Tag in both result JSON and checkpoint metadata.

### C6. Retire the "speed arm" nomenclature until architecture-family-aware
- Session 11 and early session 12 used "speed arm" as if it were a single
  architectural family. It's actually two families (substrate-primary,
  local-primary) with fundamentally different prediction machinery and
  different bpb ceilings. Any future narrative should distinguish them.

---

## Known structural limits (not bugs, just ceilings)

### L1. Substrate-primary family: 16-byte effective context ceiling
- **What**: substrate-primary models (learnable-s8, learnable-s12,
  scan-ablation-s8, mut-hashmem-fixed, mut-control-mlp) all plateau at a
  16-byte effective context regardless of `half_life_max` (tested at 2 and 512)
  or `linear_modes` (tested at 2048). 256× longer half-life moves the knee
  only 2× (8 → 16 bytes).
- **Why it matters**: The EMA-based gradient routing can't propagate
  long-range dependency into the predictor. Training, scale, and half-life
  tuning all fail to break through.
- **Implication**: The substrate-primary family is capped near 1.78 bpb
  (8-gram byte English entropy). To get past this ceiling, need a
  fundamentally different integration mechanism (selective gating with
  per-position Δ routing, sliding-window attention, etc.).

### L2. Local-primary family: `local_window` context ceiling
- **What**: local-primary models (byte-lasso-pos-b2-s12, cr-s8-scandead,
  cr-hrrangle-s12) have effective context equal to `local_window` + a small
  EMA augment. Current `local_window=8` → 16-byte knee. Saturation ~2.02 bpb.
- **Implication**: Bumping `local_window` to 32 could push knee to 32-64 but
  with quadratic FLOPs on the local conv — still O(n·k) overall, just
  higher k. Saturation ceiling estimated at ~1.5 bpb if k=64.

### L3. Hash-memory: no context extension
- **What**: `mut-hashmem-fixed-20k` (post-fix, actually-trained hash memory)
  has the SAME 16-byte knee and 2.09 bpb saturation as `mut-control-mlp-20k`
  (no hash). Hash memory is decorative for context.

### L4. Gated-delta-substrate with `local_scale > 0`: hides substrate work
- **What**: All gated-delta models in the library have `local_scale ∈ {0.25}`.
  Local conv does 99% of the prediction (ablating local drives bpb 2.0 → 4.9).
  The substrate's contribution to context is invisible in this configuration.
- **Implication**: Before declaring gated-delta a failed long-context
  mechanism, it needs to be tested in substrate-primary (`local_scale=0`)
  configuration. That experiment has never been run.

---

## The one remaining experiment

**`substrate_mode=gated_delta, local_scale=0.0, half_life_max=512, learnable-ω,
routed-experts, s8, 50k steps`** — the untested substrate-primary gated-delta
configuration. Outcome:

- If knee ≤ 16 bytes → gated-delta is ceiling-bound like frozen-HRR; the
  causal-bank architecture family is context-ceilinged at 1.78 bpb regardless
  of mechanism. Pursuit shifts to hybrid architectures (windowed attention).
- If knee > 32 bytes → gated-delta selective gating escapes the EMA ceiling.
  Mamba-class is the lever. Scale it up.

One flag change from the existing scan-ablation recipe. Estimated cost:
30 min of A4000 compute.
