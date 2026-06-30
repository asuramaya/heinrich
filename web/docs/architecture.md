# Architecture &amp; the Observatory

Heinrich is two programs wearing one coat.

- **The producer** — `mri`, `decompose`, `eval`, `audit`, the causal-bank forensics, the
  MCP suite. Runs models, needs weight access and a GPU, writes artifacts. The "monster."
- **The consumer** — everything that *reads* a captured artifact without a model: the
  companion viewer plus most of the `profile-*` family. Runs anywhere, including a browser.

```
PRODUCER (local / fleet, GPUs)            CONSUMER (edge, browser)
  heinrich mri / decompose / eval           the Observatory:
  audit / discover / MCP suite                cloud · trajectory · flower
        │                                     · spectrum · neurons
        └──────  .mri artifact  ──────────────────────┘
                 published to R2, read at the edge
```

You are **not** porting Heinrich into R2. You give it a publish target, and the `.mri` is the
API between the halves. → [The `.mri` artifact](/artifact)

## The Observatory

The public consumer is a static [Three.js SPA](https://hcirnieh.com/observatory) + an R2
bucket of immutable decomposition blobs + a thin Worker that does only object reads and HTTP
byte-range requests. No model, no server compute, no install. The local `companion` is **the
same SPA**, run directly against a live producer.

## The capability manifest — the inverted contract

The boundary between edge and producer is **declared, not detected**. Every backend answers
`GET /api/capabilities`, and the one SPA composes its UI from that — so the edge Worker never
chases feature parity, and the divergence between "cloud viewer" and "local viewer" can't grow:
there's one client reading one contract.

```
worker (edge)  → { backend:"worker", live:false, weights:false, mcp:false, models:"r2"   }
heinrich local → { backend:"local",  live:true,  weights:true,  mcp:true,  models:"both"  }
```

The same `companion_ui.html`, pointed at the Worker, is read-only; pointed at a local
`heinrich companion`, it lights up — zero code difference. Adding a feature sorts into exactly
one tier, once: **model-free over the artifact** (works everywhere) or **needs the live model**
(a capability bit, lit only where a backend advertises it).

## The on-ramp ladder — install only the rung you reach for

| Rung | Backend | Unlocks | Cost |
| --- | --- | --- | --- |
| **cloud** | Worker + browser | cloud / trajectory / flower / spectrum / neurons, browser-computed direction analysis, precomputed predictions | zero install |
| **slim local** | `heinrich companion` (numpy, full `.mri`) | circuit, nonlinear-kNN, weight-direction — heavy *artifact* analysis | light, no GPU |
| **full local** | `heinrich` + torch, model loaded | live forward, steer, model chat | the full instrument |

A "🔓 unlock" nudge in the cloud viewer hands its exact view down to a spawned local
instance via a deep-link (state rides in `location.hash`), and
`heinrich companion --gallery https://hcirnieh.com` (`models:"both"`) proxies the gallery
model so it travels with you and the SPA stays single-origin.

## What was rejected, and why

No paid GPU of any kind, and no black-box inference providers. Black-box returns text/logprobs
only — it cannot expose residuals or accept steering vectors, which *is* the instrument.
Everything runs on the visitor's GPU; deployment is one static artifact + an R2 bucket, zero
marginal cost, zero ops. → full write-up in
[`docs/observatory.md`](https://github.com/asuramaya/heinrich/blob/main/docs/observatory.md).
