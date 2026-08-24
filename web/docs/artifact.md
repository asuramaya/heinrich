# The `.mri` artifact

The `.mri` is the contract between the producer (GPU + weights) and the consumer (the edge +
the browser). The producer writes a multi-GB directory; `heinrich publish` selects the few-MB
lean subset the viewer actually reads. The full spec lives in
[`web/ARTIFACT_FORMAT.md`](https://github.com/asuramaya/heinrich/blob/main/web/ARTIFACT_FORMAT.md).

Two data scopes ship inside one `.mri`. **Sample** scope is the original: a few thousand
tokens, chosen at capture time. **Vocab** scope, added later via `heinrich mri-vocab`,
projects every tokenizer token through the sample's own frozen PCA frame, so the viewer can
address the full vocabulary instead of a sample. Vocab scope is what the Observatory serves
by default today; sample scope is the fallback for a `.mri` that never ran `mri-vocab`.

## Object layout (what publish ships)

```
<model>/<mode>.mri/
  metadata.json
  decomp/
    meta.json · tokens.json
    all_scores.bin    [HEI2]   variance + capped scores, all layers
    pc_scores.bin     [PCSC]   PC-major index   (cloud queries, O(1) per-PC seek)
    token_scores.bin  [TOKS]   token-major index (per-token queries, O(1) seek)
    token_neurons.bin [TOKN]   per-token neuron field (gate×up)
    gate_heatmap.npy           [N × layers] f16
    weight_alignment.json      per-layer weight↔PC alignment (the flowers)
    neuron_importance.json     top neurons per layer
    falsification.json         per-layer 50-pair random bimodality + top-50 bimodal PCs
    token_predicts.bin [TPRD]  captured-vocab logit lens [N × L × K]
  norms.json · baselines.json  worker-native sidecars (from *.npz)

  # vocab scope (optional, projects the sample's frame onto every tokenizer token)
  decomp/vocab_scores.bin        [VSCR]  every token, row-major, all layers
  decomp/vocab_pc16.bin          [VP16]  every token, layer-major (the cloud's own view)
  decomp/vocab_ids.npy · vocab_tokens.json · vocab_scripts.json   row → id / text / script
  decomp/vocab_meta.json         provenance + agreement check against the sample rows
  decomp/vocab_gate_heatmap.npy  full-vocab depth curve
  decomp/vocab_token_neurons.bin [VTKN]  top-50 neuron field, every row
  decomp/vocab_neuron_importance.json    per-layer top-50 neuron indices
```

The producer also writes raw per-layer activations, `weights/`, and `attention/`, the
**consumer never reads those.** The raw `.mri` can be multi-GB; the published subset is a few
MB, or a few hundred MB with vocab scope at full vocabulary size.

## Binary indexes

Magic-headed blobs give O(1) seeks for each access pattern. Sample scope:

- **`pc_scores.bin`** (`PCSC`): `[K × layers × tokens]` f16. One seek per PC, the cloud viewports.
- **`token_scores.bin`** (`TOKS`): `[tokens × layers × K]` f16. One seek per token, pin / spectrum.
- **`token_neurons.bin`** (`TOKN`): `[tokens × layers × intermediate]` f16. The neuron field.

Vocab scope adds the same shapes at full vocabulary size, plus a transpose: `vocab_pc16.bin`
(`VP16`) is layer-major so "all tokens, one PC, one layer" is one contiguous read instead of a
strided scan through the row-major `vocab_scores.bin`. Full binary layouts (headers, offsets,
every wire format) are in [`ARTIFACT_FORMAT.md`](https://github.com/asuramaya/heinrich/blob/main/web/ARTIFACT_FORMAT.md) §2 and §7.

All little-endian; the Worker parses half-floats via `DataView.getFloat16`.

## The Worker contract (the consumer HTTP API)

`web/worker/index.js` exposes the read API. Each endpoint maps to an object read or a
byte-range read, no compute.

| Endpoint | Reads | Response |
| --- | --- | --- |
| `GET /api/capabilities` | (static) | the capability manifest, see [Architecture](/architecture) |
| `GET /api/models` | `models.json` | JSON array |
| `GET /api/serve-meta/<m>/<mode>` | header + variance block | `{full_k, n_layers, n_tokens, pc_vars}` |
| `GET /api/vocab-pc-columns/...?layer=L&pcs=A,B,C` | `vocab_pc16.bin`, one layer | `VPCL` binary, the LOD-friendly single-layer route |
| `GET /api/vocab-token-neurons/...?rows=A,B` | `vocab_token_neurons.bin` | `VTNB` binary, the sample fallback for a pin outside it |
| `GET /api/gate-heatmap`, `/api/neuron-field`, `/api/weight-align`, `/api/token-resolve` | the blobs above | binary / JSON |

**Browser-computed** in the SPA (no server compute): `direction-project/quality/brief/discover/depth`,
cross-model compare. **Producer-only** (gated off at the edge via `/api/capabilities`):
circuit / nonlinear / weight-direction / steer, live forward / chat.

## Why it scales

R2 has **zero egress**, artifacts are immutable (`cache-forever`), and byte-range reads bound
transfer to the slice viewed. A published gallery is close to free to host at any scale; the
GPU cost is paid once, at capture. The Cloudflare edge cache also sits in front of every
artifact endpoint now: the first request for a given `(layer, PC)` column pays the R2 read,
every later request for that same column, from anyone, is served from the edge. Browser-side,
a sliding window streams only a direction's *support* PCs, so the same viewer that runs a
135M model stays interactive on a full-hidden one.
