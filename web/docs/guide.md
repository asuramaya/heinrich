# Install &amp; the pipeline

Heinrich is a Python package. The producer half needs weight access and (ideally) a GPU;
the consumer half runs anywhere.

```bash
pip install heinrich
pip install 'heinrich[publish]'   # adds boto3 for `heinrich publish` over the S3 API
```

Python 3.10+, `numpy` + `safetensors` as the core deps. MLX backend on Apple Silicon,
HuggingFace `transformers` as the fallback, a decepticon backend for causal-bank checkpoints.

The whole loop is four verbs: **capture → decompose → view → publish.**

```
heinrich mri          --model X --mode raw --n-index 2000 --output runs/X.mri
heinrich mri-decompose --mri runs/X.mri --n-components 0      # full PC range (= hidden_size)
heinrich companion     --mri-root runs                        # fly through it locally
heinrich publish       --mri runs/X.mri --bucket heinrich-mri # → R2, appears in the Observatory
```

## 1 · Capture: `heinrich mri`

Records the complete residual state of a model: the displacement at the entry and exit
position of **every layer**, plus weights, norms, attention, and MLP gate/up activations.

```bash
heinrich mri --model HuggingFaceTB/SmolLM2-135M --mode raw --n-index 2000 \
  --output runs/smollm2-135m/raw.mri
```

- **`--mode`**, the baseline the displacement is measured against:
  - `raw`: single tokens spliced directly, silence baseline. Shows crystallization.
  - `naked`: BOS baseline, no chat template.
  - `template`: full chat-template context; captures attention + entry/exit at different positions.
- **`--n-index`**, how many tokens to capture. Full vocabulary is supported (150K+ tokens, no sampling bias); a few thousand is plenty for a public gallery.
- **Backends auto-detect.** A `.checkpoint.pt` is loaded as a causal bank via `decepticons.loader`; everything else is MLX or HuggingFace.

The output is a `.mri/` directory, the [primary data format](/artifact).

::: tip Different models have different silence
Baseline entropy varies (Qwen 0.5B ≈ 0.92, 3B ≈ 3.91). Don't compare absolute deltas across
models, use ranks or within-model relative. The tool warns on high baseline entropy and
unconverged statistics.
:::

## 2 · Decompose: `heinrich mri-decompose`

PCA the residual stream per layer, keep the top-K principal components plus each token's
projection, and write the transposed binary indexes the viewer reads with O(1) seeks.

```bash
heinrich mri-decompose --mri runs/smollm2-135m/raw.mri --n-components 0
```

`--n-components 0` means the **full** PC range (= `hidden_size`). The full residual geometry
*is* the MRI. A capped PC range defeats the point, and `publish` warns if you ship a
truncated one. Decompose also precomputes the falsification tables and the captured-vocab
logit lens so the edge needs no model. → [The `.mri` artifact](/artifact)

## 2.5 · Extend to the full vocabulary: `heinrich mri-vocab`

An existing decomposition only carries the tokens it captured, usually a few thousand. Run
`mri-vocab` against it to project *every* tokenizer token through that same frozen PCA frame:

```bash
heinrich mri-vocab --model HuggingFaceTB/SmolLM2-135M --mri runs/smollm2-135m/raw.mri
heinrich mri-vocab --model HuggingFaceTB/SmolLM2-135M --mri runs/smollm2-135m/raw.mri \
  --gate-summary --top-n 50   # + the full-vocab neuron field (top-50 important neurons/layer)
```

This is what makes the cloud, the trajectories, and pin-anything-in-the-vocabulary work at
full scale instead of a sample. It writes `vocab_*` files alongside the decomposition; a
`.mri` that skips this step still publishes and still renders, just at sample scope. →
[The `.mri` artifact](/artifact)

## 3 · View: `heinrich companion`

The local viewer is the **maximal node**: the same SPA the Observatory serves, but with the
model loaded for live forward / steering / chat and the heavy artifact analysis the edge
can't do.

```bash
heinrich companion --mri-root runs                          # full power over local captures
heinrich companion --mri-root runs --gallery https://hcirnieh.com   # + proxy the public gallery
```

It opens at `http://localhost:8377` and declares its capabilities at
`/api/capabilities`; the SPA composes its UI from that. → [Architecture](/architecture)

## 4 · Publish: `heinrich publish`

Ships only the lean consumer subset (decomp blobs + sidecars + `falsification.json` /
`token_predicts.bin`, never the multi-GB raw weights) to R2, and upserts `models.json`.

```bash
# (a) S3 API, pure Python, portable
export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=...
heinrich publish --mri runs/smollm2-135m/raw.mri --bucket heinrich-mri

# (b) no network, export the lean layout, push it however you like
heinrich publish --mri runs/smollm2-135m/raw.mri --local-dir export/

# (c) audit what's actually at the edge against what would be published, no upload
heinrich publish --mri runs/smollm2-135m/raw.mri --bucket heinrich-mri --check
```

The model then appears in the [Observatory](https://hcirnieh.com/observatory), served from
the edge at near-zero cost. The set that `publish` uploads *is* the
[artifact contract](/artifact). `--check` is worth running after any producer change that
touches which files get published. A bucket left half-upgraded is worse than one left alone:
every request still succeeds, it just serves the old shape.

## Beyond the loop

- `heinrich run` / `heinrich eval`, the scorer pipeline over HF benchmarks. → [CLI](/cli)
- `heinrich audit <model_id>`, a full behavioral security audit.
- `heinrich serve`, the MCP stdio server (the agent surface). → [MCP](/mcp)
