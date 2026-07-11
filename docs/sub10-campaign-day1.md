# The sub-1.0 campaign, day 1 — heinrich's ledger (2026-07-10/11)

Raw material for the paper. Every number below has a committed golden in
`docs/data/` and a ruling in the Osiris graph; nothing here is from memory.

## The ladder (closed, five rungs, all double-witnessed at the 4th decimal)

enwik8 store → enwik8 held-out queries, fo-noadapt-enwik90-50k base 2.0367:

| store | delta (their/our pinned) | λ bid | mix |
|-------|--------------------------|-------|-----|
| 8M    | −0.0144                  | .05   | 2.0223 |
| 16M   | −0.0187                  | .05   | 2.0180 |
| 32M   | −0.0367                  | .10   | 2.0000 |
| 64M   | −0.0488                  | .10   | 1.9880 |
| 94M   | −0.0686                  | .20   | 1.9682 |

λ doubles every ~two store doublings (demand curve, grid-geometry-proof:
two independent calibrations disagree by ≤1 rung on the knee, never on the
law). Goldens: `knn-ladder-*.txt`.

## The oracle purse and the gate verdicts

Per-position binary oracle over {base, pure kNN} bounds every λ(x) gate.
Purse scales: 0.27 bpb (8M) → 0.39 (94M) → 1.12–1.19 (novel folds).
- Threshold gates: dead everywhere (~1–6%); on the pure-smoothing control
  they work (29%) with INVERTED signs — diffuse no-match votes are the value.
- Learned binary switch: dead (overfits).
- Learned soft λ(x) (15 quadratic features, cal-trained on mix NLL):
  alive, small — 45.6% vs scalar 43.3% (world), 53.6% vs 52.5% (control),
  and it turns itself OFF at home (mean λ 0.007) — the honest-gate check.
- BLENDING BEATS SWITCHING AT EVERY RUNG. E3's bar is 45.6%.
Goldens: `knn-ladder-{8m,94m}-gate.txt`, `knn-square-*.txt`.

## The relevance square (64M, the campaign's pivotal control)

| store → queries   | delta  | reading |
|-------------------|--------|---------|
| enwik8 → enwik8   | −0.055 | genuine in-domain retrieval (small) |
| world → world     | −0.48  | entangled |
| enwik8 → world    | −0.62  | pure smoothing — BEATS the relevant store |
| world → enwik8    | −0.00  | no demand at home; λ(x) self-switches off |

One sentence: λ-mixing collects in-domain precision and calibration
repair, never cross-domain knowledge; relevance SUBTRACTS at novelty
(confident cross-language matches displace helpful diffuse votes). The
λ(novelty) demand curve is a calibration-repair demand curve. Purse
anatomy identical across stores (letters ~70%, whitespace ~2× overweight).
Fold discriminator: kNN-alone vs base — {world, wikiml} calibration folds;
math the first knowledge fold (alone 2.85 << base 3.31, mix −0.77).

## The arrow through depth (transformers, 4 models)

Current-token identity dissolves monotonically emb→final (0.42→0.07);
next-token recovery rises ~4× to a peak AT the commit band (~¾–⅚ depth,
matching homing L*), then rotates into the output basis. Retrospective
columns EMPTY at every depth: the residual stream carries no verbatim
past — the KV cache is the externalized archive. One axis unifies the
triptych: where the verbatim past lives (bank in-state → organ evicts →
transformer KV-external → kNN datastore unbounded).
Golden: `arrow-depth-smollm2-135m.txt`; ruling a84a9a18.

## Eviction (evict-first, to a reserve)

5-point trajectory (fo-adapt-50k, 10k…50k, capture_bpb inline):
retrospective info AT the trained floor by step 10k (first 20% of
training), holds a ~⅓-frozen RESERVE while loss falls 40k more steps.
Crowding-out dead; "evict to a reserve, then learn."
Golden: `eviction-trajectory-5point.txt`; ruling 5d044cce. E4 (three-arm
evict-init) is pre-registered off this; heinrich supplies the subspace
(`profile-cb-retro-subspace`).

## Spectral ON/OFF (two-witness, no dissenting row)

Organ-ON bodies whitened flat (spread ~0.03); organ-OFF trained body
AMPLIFIES the kernel imprint (spread 1.44, 69% variance in slow band).
THE ORGAN WHITENS; BARE TRAINING ANTI-WHITENS. Frozen kernel steepens
mildly under any honest slice (their old flattening baseline was a
corpus-head artifact — third corpus-position confound killed).
Goldens: `spectral-bands-{trained,frozen}.txt`; rulings 1b9ea361/f35a6c04.

## Instrument debt paid / folded this cycle

profile-cb-spectral-bands · capture_bpb in pc-information ·
--gate (oracle + threshold + learned λ(x) + purse-by-class) ·
--query-corpus/--lam-grid/globbed shard-dialect corpora (hash-verified
against chronohorn's reader — the session-11 bug caught pre-run) ·
profile-cb-retro-subspace (E4's arms B/C are one projection each).

## Open at handoff (successor's queue, Osiris thread carries detail)

1. Blessed window on SW's "CHAIN EXITED" mail: enwik8→math and
   enwik8→wikiml per-fold controls (full gate/by-class reports).
2. E4: frozen init capture from SW → retro-subspace export → arm C
   calibration by measured R² → capture pass on 1k/2k/5k checkpoints.
3. Surface table freeze pends code fold + enwik9-150M (their side).
4. enwik9-era co-witness rungs (my timely ceiling ~94M, resets at full
   clocks).
