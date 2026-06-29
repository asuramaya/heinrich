# Observatory viewport audit

A punch list of every viewport and interactive feature in the companion SPA,
its state, and what finishing it needs. Derived by cross-referencing the
endpoints the SPA calls against what the Worker serves (Phase-1, June 2026).

**The shape of the gap:** the *recorded-data* layer is complete; everything
unfinished is the *direction-compute* layer (all `direction-*` endpoints are
501 — Phase-2 browser compute), plus the rv2 stub, rv1's producer fix, and the
Tier-C live layer.

## Viewports

| Viewport | State | Notes / what it needs |
| --- | --- | --- |
| Cloud A/Δ/B (vp0–2) | ✅ complete | `cloud-bundle`; out-of-range PC guard fixed |
| Trajectory A/Δ/B (vp3–5) | ✅ complete | PC-major slabs; stale-buffer RangeError fixed |
| Internals/Flower A/Δ/B (lv0–2) | ✅ complete | `weight-align`; content-aware framing fixed |
| Prism dual-browser (br0) | ✅ complete | renders PC-pair scatter from loaded data |
| PC Spectrum (br0bot) | ✅ complete | `serve-meta` `full_k` + `pc_vars`; carries the merged Superposition overlay |
| **rv0 Direction Depth** | ⛔ empty | needs `direction-depth` (501) + 2 pinned tokens. **Browser-computable** from `token-pca` (per-token per-layer PC vectors): project the A↔B separation through layers client-side. |
| **rv1 Neurons** | ◐ data-gated | builds + colors by pinned-token `neuron-field` (works). Needs `intermediate_size`; with the worker TOKN-header patch reverted, it must come **producer-side** (`decompose`/`publish` → `decomp/meta.json`). |
| **rv2 Interference** | ⛔ stub | intentionally emptied (Superposition merged into PC Spectrum). Free creative slot; design TBD. |

## Interactive features (token panel + overlays)

| Feature | State | Endpoint (status) |
| --- | --- | --- |
| Token A/B pin + PC bars + neuron canvas | ✅ complete | `token-bundle`, `neuron-field`, `gate-heatmap` (200) |
| Token hover detail | ✅ complete | `token-hover` (200) |
| **Cloud direction-coloring** | ⛔ broken | `direction-project` (501) — color the cloud by an A↔B direction. *High visual value.* |
| **Direction Analysis** (Analyze / sepRun) | ⛔ broken | `direction-quality`, `direction-nonlinear` (501) |
| **Discover Features** (Discover) | ⛔ broken | `direction-discover` (501) |
| Direction circuit / brief / weights | ⛔ broken | `direction-circuit`, `direction-brief`, `direction-weights` (501) |
| Token predictions (token cards) | ⛔ broken | `token-predicts` (501) — `lmhead @ exit` |
| Run / Live (forward, chat, steer) | ⛔ disconnected | `live-forward`, `live-chat`, `direction-steer` (501) — **Tier C** (white-box inference; browser-tiny or BYO origin) |
| Token search resolve | ⚠ gap | `token-resolve` — SPA calls it; Worker does not wire it (token-by-text lookup). Verify/wire from `tokens.json`. |
| Snapshot (PNG) / GIF capture | ✅ local | `capture-result` / `capture-upload` are POST no-ops in a static deploy; capture works client-side |
| Analysis chat | ◐ benign | `chat` (MCP-backed; no-op without an agent) |

## Phase-2 work, grouped

The `direction-*` family is one logical unit — **directions over loaded
arrays**. Porting it to browser WebGPU/JS unlocks, in one pass:

1. `direction-project` → cloud direction-coloring (highest visual payoff)
2. `direction-depth` → **rv0** Direction Depth
3. `direction-quality` / `-nonlinear` → Direction Analysis panel
4. `direction-discover` → Discover Features
5. `direction-circuit` / `-brief` / `-weights` → circuit/brief panels

All operate on `token-pca` / `pc-full` / `weight_alignment` data the Worker
already serves. The math: a direction is the (mean) A↔B difference vector in
PC space; project scores onto it, measure separation, rank PCs. This is the
SPA-modifying Tier-B port called out in `observatory.md`.

**Tier C** (live forward/chat/steer + token-predicts) stays separate: needs
white-box inference — transformers.js for ≤3B in-browser, or a BYO origin.

## Immediate plan

1. **Producer `intermediate_size`** — `decompose`/`publish` write it into
   `decomp/meta.json`; document in `ARTIFACT_FORMAT.md`. Unblocks rv1 without
   the worker patch.
2. **rv0 Direction Depth** — first browser-compute: A↔B separation through
   layers from `token-pca`. Proves the Tier-B pattern on one viewport.
3. **rv2** — design once 1–2 land.
