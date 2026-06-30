# The Heinrich Observatory — producer / consumer / edge

This document describes how Heinrich is *presented and deployed*, and where the
line falls between the heavy research tool and the public-facing reader.

## The one idea

Heinrich is two programs wearing one coat:

- **The producer** — `mri`, `mri-decompose`, `eval`, `audit`, direction
  discovery, the causal-bank forensics, the splat/vision pipeline. It runs
  models, needs weight access and a GPU, and writes artifacts. This is the
  MCP tool surface — the "monster." It runs *where the models are* (a laptop,
  a fleet box, a CUDA machine).

- **The consumer** — everything that *reads* a captured artifact and presents
  or analyzes it **without a model**: the companion viewer, plus most of the
  `profile-*` family (chain, cross, survey, pca-anatomy, layer-deltas,
  logit-lens, gates, cb-loss, cb-routing, …). These are pure readers of
  `.mri`/`.npz` files. They run *anywhere* — including a browser at the edge.

Heinrich's design already drew this line: **"the tool captures state, not
interpretation; interpretation is left to the reader."** Capture vs. analyze.
The DB / `.mri` artifact is the single source of truth. The Observatory just
makes the *consumer* half publishable.

```
PRODUCER (local / fleet, GPUs)            CONSUMER (edge, browser)
  heinrich mri / decompose / eval           the observatory:
  audit / discover / splat                    cloud · trajectory · flower
  MCP tool suite                              · spectrum · neurons
        │                                     + profile-* as panels
        │                                            ▲
        └────────  .mri artifact (the contract)  ────┘
                   published to R2, read at the edge
```

You are **not** porting Heinrich into R2. You are giving Heinrich a *publish
target*, and the `.mri` artifact is the API between the two halves.

## What fits the edge shape, and what does not

| Capability | Where it runs | Why |
| --- | --- | --- |
| Cloud / trajectory / flower / spectrum / neurons | **edge + browser** | reads decomp blobs; pure byte-range reads |
| `profile-*` read-side analysis (deltas, logit-lens, pca-anatomy, cb-*) | **edge + browser** (port) | model-free analysis of captured artifacts |
| Direction project / quality / discover, predictions | **browser (WebGPU)** | numpy over already-loaded arrays; movable client-side |
| Capture (`mri`, `shart`, `sht`) | **producer only** | forward pass through live weights |
| `eval` / `audit` (scorers, behavioral) | **producer only** | runs the model + judge models |
| Live forward / steer (white-box) | **producer or browser-tiny** | needs residual access + injection; ≤3B via transformers.js, else BYO origin |
| Splat / vision GPU pipeline | **producer only** | 3DGS + diffusion, heavy GPU |

The rule: **if it needs the weights, it stays with the producer.** If it reads
a recording, it can live at the edge. Which side a viewer reaches is no longer
hard-coded — it's negotiated per session through the capability manifest below,
so the same SPA is read-only on the edge and full-power against a local node.

## Three access modalities, one substrate

The same R2 artifact serves three audiences:

1. **Agents** → MCP (as today). Full producer + consumer, driven by Claude.
   Can also read/write the published R2 artifacts.
2. **Humans** → the Observatory: a public gallery of published model MRIs.
   Art piece on the surface (fly through a real model's residual stream),
   full read-side analysis underneath. Free, edge-served, infinite scale,
   zero paid compute (see [`ARTIFACT_FORMAT.md`](../web/ARTIFACT_FORMAT.md)).
3. **Researchers** → `pip install heinrich`, capture your own model, publish
   your own MRIs against the open artifact format.

This makes Heinrich a **platform**: a producer tool + an open artifact format
+ a reference reader. The viewer being an art piece is the on-ramp to the
format — not a tech demo that grew up, but the *reference implementation* of
an open format.

## Deployment topology

```
 your machine / fleet            Cloudflare                 any browser
 ┌───────────────────┐           ┌──────────────┐           ┌──────────┐
 │ heinrich mri      │  publish  │ R2 bucket    │  range    │ SPA      │
 │ heinrich decompose├──────────▶│ (artifacts)  │◀─────────▶│ + WebGPU │
 │ heinrich publish  │           │ Worker (api) │  reads    │ compute  │
 └───────────────────┘           └──────────────┘           └──────────┘
   GPU, weights, DB               zero egress, immutable      no install
```

- **Producer**: captures + decomposes + publishes. Needs GPU/weights.
- **R2**: immutable, content-addressable, zero-egress artifact store.
- **Worker**: translates `/api/*` → R2 GET + byte-range; serves the SPA.
  No model, no paid compute — see the Worker in `web/worker/index.js`.
- **Browser**: renders (Three.js) and computes the interactive analysis
  (WebGPU/JS). Optionally runs a tiny model client-side for live mode.

## The publish loop

```
heinrich mri          --model X --mode raw --n-index 2000 --output runs/X.mri
heinrich mri-decompose --mri runs/X.mri --n-components 0   # full PC range (= hidden_size); never truncate — the full residual geometry IS the MRI
heinrich publish      --mri runs/X.mri --bucket heinrich-mri    # → R2 (S3 API)
```

`heinrich publish` (`src/heinrich/observatory/publish.py`) is the seam: it
computes the worker-native sidecars (`tokens.json`, `norms.json`,
`baselines.json`), selects only the consumer-served files (decomp blobs +
sidecars — never the multi-GB raw weights/activations), uploads them, and
upserts `models.json`. It talks to R2 over the **S3 API** (boto3), so the
producer stays pure Python — no Node/wrangler dependency. Creds come from
`R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY`.

```
heinrich publish --mri runs/X.mri --dry-run            # report the plan
heinrich publish --mri runs/X.mri --local-dir export/  # export layout, no network
heinrich publish --mri runs/X.mri --bucket B --with-hover   # also mlp/ (token-hover)
```

`pip install heinrich[publish]` pulls boto3. The viewer reads exactly what
`publish` uploads — that set is the [artifact contract](../web/ARTIFACT_FORMAT.md).

## The capability manifest (the inverted contract)

The boundary between edge and producer is **declared, not detected**. Both
backends answer `GET /api/capabilities`, and the *one* SPA composes its UI from
that — so the edge Worker never has to chase feature parity, and the divergence
between "cloud viewer" and "local viewer" can't grow (there's nothing to keep in
sync; there's one client reading one contract).

```
worker (edge)  → { backend:"worker", live:false, weights:false, mcp:false, models:"r2"   }
heinrich local → { backend:"local",  live:true,  weights:true,  mcp:true,  models:"both"  }
```

The same `companion_ui.html`, pointed at the Worker, is read-only; pointed at a
local `heinrich companion`, it lights up — verified, zero code difference. Adding
a feature sorts into exactly one tier, once: **model-free over the artifact**
(precompute or browser-compute, works everywhere) or **needs the live model**
(a capability bit, lit only where a backend advertises it).

### The on-ramp ladder — install only the rung you reach for

| Rung | Backend | Unlocks | Cost |
| --- | --- | --- | --- |
| **cloud** | Worker + browser | cloud/trajectory/flower/spectrum/neurons, browser-computed direction analysis, precomputed predictions | zero install |
| **slim local** | `heinrich companion` (numpy, full `.mri`) | circuit, nonlinear-kNN, weight-direction — heavy *artifact* analysis | light, **no GPU** |
| **full local** | `heinrich` + torch, model loaded | live forward, steer, model chat | the full instrument |

A "🔓 unlock" nudge (and the gated notes) open a panel offering the cheapest rung
plus a **deep-link handoff**: the cloud SPA's state already lives in
`location.hash` (`m/d/l/a/b/p`), so the link is just `localhost:8377` + that hash
— the spawned local instance restores the exact view (model · layer · pins). With
`heinrich companion --gallery <public-base>` (`models:"both"`), the gallery model
you were on is proxied locally, so it travels with you and the SPA stays
single-origin (no mixed-content / Private-Network-Access).

## Roadmap

- **Phase 1 — done.** Recorded-data layer served from R2: cloud, trajectory,
  flower, spectrum, neurons, token interaction. Real model verified
  (SmolLM2-135M, 30 layers).
- **Phase 2 — done.** The interactive `direction-*` and `token-predicts` work at
  the edge — browser-compute (sliding-window direction analysis, cross-compare)
  plus producer-side precompute (`falsification.json`, `token_predicts.bin`).
  No WebGPU required yet; plain JS over already-loaded arrays.
- **Phase 3 — `heinrich publish` (done).** Capture → publish → render is one
  clean loop over the S3 API. Format in [`ARTIFACT_FORMAT.md`](../web/ARTIFACT_FORMAT.md).
- **Phase 3.5 — capability inversion (done).** `/api/capabilities` is the boundary;
  the SPA is capability-driven; the on-ramp ladder + unlock/deep-link handoff +
  `models:"both"` gallery proxy let a cloud visitor descend into a local instance.
- **Phase 4 — the Observatory proper.** Public gallery landing page over
  `models.json`; `profile-*` analyses as panels; transformers.js live mode for
  ≤3B models (a browser-resident `live:true` node — another point on the lattice).
  Open refinement: per-model `has_weights` from serve-meta, so the slim-local tier
  hides cleanly on proxied (lean) gallery models instead of showing-then-failing.

## What to resist

The MCP monster being complex is *correct* — agents absorb complexity. The web
face is popular **because it is curated and beautiful, not complete.** Cramming
the full tool surface into the browser kills the property that makes it
popular. Keep the monster as the producer; grow the consumer deliberately, one
gorgeous read-side view at a time.
