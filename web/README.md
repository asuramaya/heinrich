# Heinrich Observatory

A serverless, edge-served reader for Heinrich MRI artifacts — fly through a
language model's residual stream in the browser. The art-piece front door to
the [Heinrich](../README.md) model-forensics toolkit.

No model, no server compute, no install for the viewer: a static SPA + an R2
bucket of immutable decomposition blobs + a thin Worker that does only object
reads and HTTP byte-range requests. All heavy analysis runs in the browser.

- **Architecture & positioning:** [`docs/observatory.md`](../docs/observatory.md)
- **The open artifact format (the contract):** [`ARTIFACT_FORMAT.md`](ARTIFACT_FORMAT.md)

## The shape

```
PRODUCER (GPU/weights)          R2 (artifacts)          BROWSER (no install)
heinrich mri / decompose  ──▶   immutable blobs   ◀──▶  Three.js SPA + WebGPU
heinrich publish                Worker = /api → R2        compute (Phase 2)
```

Heinrich's producer half (capture, eval, audit — the MCP suite) runs where the
models are. The consumer half (this viewer, plus the read-side `profile-*`
analyses) reads recordings and runs at the edge. You don't port the producer to
R2 — the `.mri` artifact *is* the API between them.

## Layout

```
web/
  wrangler.jsonc      Workers Assets (SPA) + R2 binding (MRI → heinrich-mri)
  worker/index.js     /api/* → R2 reads; everything else → public/index.html
  public/index.html   the companion SPA (generated from src/heinrich/companion_ui.html)
  upload.sh           push .data/** into R2 (local sim or remote)
  .data/              MRI artifacts (gitignored) — synthetic or captured
ARTIFACT_FORMAT.md    the producer↔consumer contract
```

## Run it locally (no Cloudflare account)

```bash
# A. synthetic data — pure stdlib, no model, no installs
python3 ../scripts/cf_synth_mri.py --out .data --model synth-mini --mode raw

#    …or a real capture (needs the heinrich producer + a model)
heinrich mri --model HuggingFaceTB/SmolLM2-135M --mode raw --n-index 2000 \
  --output .data/smollm2-135m/raw.mri
heinrich mri-decompose --mri .data/smollm2-135m/raw.mri --n-components 48
python3 ../scripts/cf_mri_prep.py --out .data        # sidecars + manifest

# B. load into the local R2 simulation + run the edge stack locally
bash upload.sh local
wrangler dev          # http://localhost:8787 — pick a model, fly
```

`wrangler dev` runs the whole thing offline: miniflare simulates R2 (sqlite +
blobs under `.wrangler/state`), the same code path that runs at the edge.

## Deploy to the edge

```bash
wrangler login
wrangler r2 bucket create heinrich-mri
bash upload.sh remote                 # or `heinrich publish` once it lands
wrangler deploy                       # add a custom domain route in the dashboard
```

## Status

- **Phase 1 (done):** recorded-data layer — cloud, trajectory, flower,
  spectrum, neurons, token interaction. Verified on SmolLM2-135M (30 layers,
  2000 tokens), served entirely from R2 range reads.
- **Phase 2:** browser-compute the interactive endpoints (`direction-*`,
  `token-predicts`) via WebGPU; this is where the SPA gains client-side compute.
- **Phase 3:** `heinrich publish` — fold sidecar generation + selective upload
  into one producer verb (currently `scripts/cf_mri_prep.py` + `upload.sh`).
- **Phase 4:** the gallery — a landing page over `models.json`, `profile-*`
  analyses as panels, optional BYO-origin / transformers.js live mode (≤3B).

## Costs

R2 has zero egress, artifacts are immutable (cache-forever), and byte-range
reads bound transfer to the slice viewed — so a published gallery is ~free to
host at any scale. The producer's GPU cost is paid once, at capture.

The SPA is a generated copy of `src/heinrich/companion_ui.html`
(`cp` it into `public/index.html`); edit the source, not the copy.
