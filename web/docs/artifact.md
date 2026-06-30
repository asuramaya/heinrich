# The `.mri` artifact

The `.mri` is the contract between the producer (GPU + weights) and the consumer (the edge +
the browser). The producer writes a multi-GB directory; `heinrich publish` selects the few-MB
lean subset the viewer actually reads. The full spec lives in
[`web/ARTIFACT_FORMAT.md`](https://github.com/asuramaya/heinrich/blob/main/web/ARTIFACT_FORMAT.md).

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
```

The producer also writes raw per-layer activations, `weights/`, and `attention/` — **the
consumer never reads those.** The raw `.mri` can be multi-GB; the published subset is a few
MB + the optional neuron index.

## Binary indexes

Three magic-headed blobs give O(1) seeks for the three access patterns:

- **`pc_scores.bin`** (`PCSC`) — `[K × layers × tokens]` f16. One seek per PC → the cloud viewports.
- **`token_scores.bin`** (`TOKS`) — `[tokens × layers × K]` f16. One seek per token → pin / spectrum.
- **`token_neurons.bin`** (`TOKN`) — `[tokens × layers × intermediate]` f16. The neuron field.

All little-endian; the Worker parses half-floats via `DataView.getFloat16`. The transposed
layout is the difference between 48 sequential reads and millions of page faults.

## The Worker contract (the consumer HTTP API)

`web/worker/index.js` exposes the read API. Each maps to an object read or a byte-range read —
no compute.

| Endpoint | Reads | Response |
| --- | --- | --- |
| `GET /api/capabilities` | (static) | the capability manifest — see [Architecture](/architecture) |
| `GET /api/models` | `models.json` | JSON array |
| `GET /api/serve-meta/<m>/<mode>` | header + variance block | `{full_k, n_layers, n_tokens, pc_vars}` |
| `GET /api/pc-full/...?pc=N` | `pc_scores.bin` slab | f16 per layer |
| `GET /api/token-pca/...?token=N` | `token_scores.bin` row | f32 |
| `GET /api/falsification/<m>/<mode>` | `falsification.json` | `{random_baseline, top_pcs}` |
| `GET /api/token-predicts/...?token=&layer=&k=` | `token_predicts.bin` + tokens.json | `{top_k:[{text,prob,logit,mri_idx}]}` |
| `GET /api/gate-heatmap`, `/api/neuron-field`, `/api/weight-align`, `/api/token-resolve` | the blobs above | binary / JSON |

**Browser-computed** in the SPA (no server compute): `direction-project/quality/brief/discover/depth`,
cross-model compare. **Producer-only** (gated off at the edge via `/api/capabilities`):
circuit / nonlinear / weight-direction / steer, live forward / chat.

## Why it scales

R2 has **zero egress**, artifacts are immutable (`cache-forever`), and byte-range reads bound
transfer to the slice viewed. A published gallery is ~free to host at any scale; the GPU cost
is paid once, at capture. Browser-side, a sliding window streams only a direction's *support*
PCs — so the same viewer that runs a 135M model stays interactive on a full-hidden one.
