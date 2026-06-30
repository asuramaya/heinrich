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
heinrich mri-decompose --mri .data/smollm2-135m/raw.mri --n-components 0  # full PC range (= hidden_size)
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
wrangler deploy                       # add a custom domain route in the dashboard

# Publish the lean consumer subset to R2. Two paths:
#  (a) S3 API — pure Python, portable, no node:
pip install 'heinrich[publish]'
export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=...
heinrich publish --mri .data/smollm2-135m/raw.mri --bucket heinrich-mri
#  (b) no S3 keys — export the lean layout, push it with your wrangler session:
heinrich publish --mri .data/smollm2-135m/raw.mri --local-dir .r2-export
#   then: for each file under .r2-export, wrangler r2 object put heinrich-mri/<key> --file <f> --remote
```

`heinrich publish` uploads only the lean consumer subset (decomp blobs +
sidecars + `falsification.json` / `token_predicts.bin` — never the multi-GB raw
weights) and upserts `models.json`. `--local-dir` / `--dry-run` need no
credentials; path (b) reuses the same OAuth login as `wrangler deploy`.

## Run the full instrument locally (the maximal node)

The *same* SPA, served by a local `heinrich companion`, is the full-power node —
live forward / steer / chat plus the heavy artifact analysis the edge can't do:

```bash
heinrich companion --mri-root .data                       # full power over local captures
heinrich companion --mri-root .data --gallery https://hcirnieh.com   # + proxy the public gallery
```

It advertises `GET /api/capabilities {live:true, weights:true, mcp:true,
models:both}`; the edge Worker advertises the minimal node — and the SPA composes
its UI from whichever it's talking to (see
[the capability manifest](../docs/observatory.md#the-capability-manifest-the-inverted-contract)).
The cloud viewer's "🔓 unlock" nudge hands its exact view down to a local instance
via a deep-link (state rides in `location.hash`).

## Status

- **Phase 1–3 (done):** recorded-data layer (cloud/trajectory/flower/spectrum/
  neurons/token interaction), the interactive layer at the edge (browser-computed
  `direction-*` + cross-compare, producer-precomputed `falsification.json` /
  `token_predicts.bin`), and `heinrich publish`. Verified on SmolLM2-135M.
- **Phase 3.5 (done):** capability inversion — `/api/capabilities` is the boundary;
  the on-ramp ladder (cloud → slim-local numpy → full-local torch) + unlock/deep-
  link handoff + `models:"both"` gallery proxy.
- **Phase 4:** the gallery landing page over `models.json`; `profile-*` panels;
  a browser-resident `live:true` node via transformers.js (≤3B).

## Costs

R2 has zero egress, artifacts are immutable (cache-forever), and byte-range
reads bound transfer to the slice viewed — so a published gallery is ~free to
host at any scale. The producer's GPU cost is paid once, at capture.

## Surfaces

`hcirnieh.com` serves three things off one Worker:

- `/` — the immersive landing (`public/index.html`, authored).
- `/observatory` — the viz SPA, a generated copy of `src/heinrich/companion_ui.html`
  (`cp` it into `public/observatory/index.html`; edit the source, not the copy).
- `/docs/` — the VitePress doc site. Source in `docs/`; build before deploy:

```bash
cd docs && npm install && npm run build   # → ../public/docs (gitignored)
cd .. && wrangler deploy                   # ships landing + viz + docs
```
