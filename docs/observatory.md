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
a recording, it can live at the edge.

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
heinrich mri-decompose --mri runs/X.mri --n-components 48
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

## Roadmap

- **Phase 1 — done.** Recorded-data layer served from R2: cloud, trajectory,
  flower, spectrum, neurons, token interaction. Real model verified
  (SmolLM2-135M, 30 layers).
- **Phase 2 — browser compute.** Port the interactive `direction-*` and
  `token-predicts` endpoints to WebGPU/JS in the SPA. This is where the SPA
  is modified (Tier-B compute moves client-side).
- **Phase 3 — `heinrich publish` (done).** Capture → publish → render is one
  clean loop: `publish` emits sidecars, selects the consumer subset, uploads to
  R2 over the S3 API, and upserts the manifest. Format versioned in
  [`ARTIFACT_FORMAT.md`](../web/ARTIFACT_FORMAT.md).
- **Phase 4 — the Observatory proper.** Public gallery landing page over
  `models.json`; `profile-*` analyses as panels behind the viewer; optional
  BYO-origin / transformers.js live mode for ≤3B models.

## What to resist

The MCP monster being complex is *correct* — agents absorb complexity. The web
face is popular **because it is curated and beautiful, not complete.** Cramming
the full tool surface into the browser kills the property that makes it
popular. Keep the monster as the producer; grow the consumer deliberately, one
gorgeous read-side view at a time.
