# Heinrich companion — serverless (Cloudflare) prototype

Serves the MRI visualizer as a static SPA + an R2 bucket of immutable decomp
blobs. No paid compute, no model: the Worker only does R2 GETs, byte-range
reads, and CLDB repacking. All heavy compute (geometry, eventual live forward)
runs in the visitor's browser (WebGPU).

## Layout

```
web/
  wrangler.jsonc      Workers Assets (SPA) + R2 binding (MRI → heinrich-mri)
  worker/index.js     /api/* → R2; everything else → public/index.html
  public/index.html   the companion SPA (copy of src/heinrich/companion_ui.html)
  upload.sh           push .data/** into R2 (local sim or remote)
  .data/              synthetic MRIs (gitignored) — built by the generator
```

## Run it (local, no Cloudflare account needed)

```bash
# 1. synthesize MRIs (pure stdlib, no installs)
python3 ../scripts/cf_synth_mri.py --out .data --model synth-mini --mode raw

# 2. load them into the local R2 simulation
./upload.sh local

# 3. dev server (first run downloads workerd)
wrangler dev
# open http://localhost:8787 → pick synth-mini → cloud should render
```

## Deploy to hcirnieh.com

```bash
wrangler login                              # interactive (run via `! wrangler login`)
wrangler r2 bucket create heinrich-mri
./upload.sh remote
wrangler deploy
# then add a custom domain route for hcirnieh.com in the dashboard or wrangler
```

## Status

MVP boot path wired: `/api/models`, `decomp-meta`, `serve-meta`,
`decomp?layer=N`, `cloud-bundle`. Interactive endpoints (direction-*,
neuron-field, attn, weight-align, live-*) return 501 until ported — the
direction/geometry ones move to browser WebGPU; live-* is in-browser
transformers.js (tiny models only). See the project memory for the full plan.
