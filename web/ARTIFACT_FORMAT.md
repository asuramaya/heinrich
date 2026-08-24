# The Heinrich MRI artifact format

The contract between the **producer** (`heinrich mri` / `mri-decompose`) and the
**consumer** (the Observatory: Worker + browser SPA). A publish target that
conforms to this spec can be rendered by the reference reader with no model and
no server compute: just object reads and HTTP byte-range requests.

Design principle: **every consumer query is either a plain GET of a precomputed
JSON, or a fixed-stride byte-range read of an immutable binary index.** No
numpy parsing in the Worker; no recomputation server-side.

All integers are little-endian. `f16` = IEEE half (`<e`), `f32` = `<f`.

Two data scopes coexist in one `.mri`. §1-6 describe the **sample** scope: a
few thousand tokens, chosen at capture time, the format this spec originally
covered. §7 describes the **vocab** scope: every tokenizer token, projected
through the sample's own frozen PCA frame after the fact via `heinrich
mri-vocab`, optional and additive, and what the Observatory actually serves by
default today. A `.mri` published without §7's files still works; the viewer
falls back to sample scope and says so.

---

## 1. Object layout (R2 keys)

```
models.json                                  # gallery manifest (array)
<model>/<mode>.mri/
  metadata.json                              # capture provenance
  decomp/
    meta.json                                # decomposition metadata
    tokens.json                              # worker-native token sidecar
    all_scores.bin       [HEI2]              # variance + capped scores, all layers
    pc_scores.bin        [PCSC]              # PC-major index (cloud queries)
    token_scores.bin     [TOKS]              # token-major index (per-token queries)
    token_neurons.bin    [TOKN]              # per-token neuron field (gate×up)
    gate_heatmap.npy                         # [N × n_real_layers] f16
    weight_alignment.json                    # per-layer weight↔PC alignment (flowers)
    neuron_importance.json                   # top neurons per layer
    falsification.json                       # per-layer: 50-pair random bimodality
                                             #   baseline + top-50 bimodal PCs
                                             #   (direction discover + percentile)
    token_predicts.bin   [TPRD]              # captured-vocab logit lens:
                                             #   [N × L × K] (u32 mri_idx, f16 prob,
                                             #   f16 logit), token-major O(1) seek
    L{NN}_scores.npy / _variance.npy / _components.npy
    emb_scores.npy / lmh_scores.npy (+ _variance)   # virtual boundary layers
    vocab_scores.bin      [VSCR]              # (optional, §7) full vocab, row-major
    vocab_pc16.bin         [VP16]              # (optional, §7) full vocab, layer-major
    vocab_ids.npy                              # (optional, §7) row → token id
    vocab_tokens.json                          # (optional, §7) row → text
    vocab_scripts.json                         # (optional, §7) row → script
    vocab_meta.json                            # (optional, §7) provenance + agreement check
    vocab_gate_heatmap.npy                     # (optional, §7) [n_rows × n_real_layers] f16
    vocab_token_neurons.bin [VTKN]              # (optional, §7) top-N neuron field, every row
    vocab_neuron_importance.json               # (optional, §7) per-layer top-N indices
  norms.json / baselines.json                # worker-native sidecars (from *.npz)
  mlp/L{NN}_gate.npy, _up.npy                # (optional) for token-hover neurons
```

The producer also writes `tokens.npz`, `norms.npz`, `baselines.npz`, raw
per-layer activations, `weights/`, and `attention/`. **The consumer never reads
those.** `heinrich publish` uploads only the files above. The raw `.mri` can
be multi-GB; the published subset is a few MB plus the optional neuron index.

### `models.json`
```json
[{ "model": "smollm2-135m", "mode": "raw", "n_layers": 30,
   "n_tokens": 2000, "version": "0.7", "architecture": "transformer" }]
```

### `<model>/<mode>.mri/metadata.json`
```json
{ "architecture": "transformer", "version": "0.7",
  "capture": { "mode": "raw", "n_tokens": 2000, "intermediate_size": 1536 },
  "model": { "name": "...", "n_layers": 30, "hidden_size": 576 } }
```

### `decomp/meta.json`
`n_layers` (total, incl. virtual), `n_real_layers`, `n_sample`, `n_components`,
`intermediate_size` (MLP width; the Neurons viewport reads it), `layers`
(ordered list, drives variance ordering), `virtual_layers`, `sample_indices`,
`method`. `mri-decompose` writes `intermediate_size`; `heinrich publish`
back-fills it from the `token_neurons.bin` TOKN header for older decompositions.

Layer ordering is canonical and shared by every binary index:
`[L00 … L{n-1}, emb, lmh]` → `total_layers = n_real_layers + 2`.

---

## 2. Binary indexes

Each begins with a 4-byte magic so the consumer can validate.

### `all_scores.bin`: magic `HEI2`
```
header  <4sIIII = magic, total_layers, n_tokens, score_k, var_k     (20 bytes)
then    f32[total_layers * var_k]            # per-layer variance (row-major)
then    f16[total_layers * n_tokens * score_k]   # capped scores
```
`score_k = min(n_components, 50)`. Variance carries full `var_k`. The variance
block alone is enough to synthesize `serve-meta` (§4).

### `pc_scores.bin`: magic `PCSC` (PC-major; O(1) per-PC seek)
```
header  <4sIII = magic, total_layers, n_tokens, full_k              (16 bytes)
then    f16[full_k][total_layers][n_tokens]   # one contiguous slab per PC
```
One PC slab = `total_layers * n_tokens * 2` bytes at offset `16 + pc*stride`.
Drives the cloud (§4 `cloud-bundle`) and `pc-full` / `pc-column`.

### `token_scores.bin`: magic `TOKS` (token-major; O(1) per-token seek)
```
header  <4sIII = magic, n_tokens, total_layers, full_k              (16 bytes)
then    f16[n_tokens][total_layers][full_k]   # one row per token
```
One token row = `total_layers * full_k * 2` bytes at offset `16 + tok*stride`.

### `token_neurons.bin`: magic `TOKN` (per-token neuron field)
```
header  <4sIII = magic, n_tokens, n_real_layers, intermediate       (16 bytes)
then    f16[n_tokens][n_real_layers][intermediate]   # precomputed gate×up
```
Virtual layers excluded (no MLP). Drives `neuron-field`.

---

## 3. JSON sidecars (worker-native, emitted at publish)

The consumer must not parse `.npz`. The producer flattens them to JSON:

- **`decomp/tokens.json`**: `{ token_ids:[int], scripts:[str], token_texts:[str] }`
  (length `n_tokens`; from `tokens.npz`).
- **`norms.json`**: `{ <key>: { mean, std, min, max, shape } }` (from `norms.npz`).
- **`baselines.json`**: `{ <key>: { norm, mean, std, shape } }` (from `baselines.npz`).
- **`decomp/weight_alignment.json`**: `[{ layer, matrices:[{ name, alignment:[full_k] }] }]`
  (already precomputed by `mri-decompose`; flowers read it directly).
- **`decomp/neuron_importance.json`**: top neurons per layer.

---

## 4. Consumer HTTP API (the Worker contract)

The reference Worker (`web/worker/index.js`) exposes these. Each maps to object
reads above. **Binary responses** are raw little-endian; **JSON** as noted.

| Endpoint | Reads | Response |
| --- | --- | --- |
| `GET /api/capabilities` | (static literal) | JSON `{backend, live, weights, mcp, write, models, …}`, the inverted contract |
| `GET /api/models` | `models.json` | JSON array |
| `GET /api/decomp-meta/<m>/<mode>` | `decomp/meta.json` | JSON |
| `GET /api/serve-meta/<m>/<mode>` | `all_scores.bin` hdr+var block, `pc_scores.bin` hdr | JSON `{full_k, n_layers, n_real_layers, n_tokens, pc_vars}` |
| `GET /api/decomp/<m>/<mode>?layer=N` | `L{NN}_scores.npy` + `tokens.json` | JSON `{scores[][], scripts, texts, token_ids}` |
| `GET /api/cloud-bundle/...?full=&medium=&step=` | `pc_scores.bin` slabs | `CLDB` binary (§5) |
| `GET /api/pc-full/...?pc=N` | `pc_scores.bin` slab | `<III n_layers,n_tok,pc>` + f16 |
| `GET /api/pc-column/...?pc=N&layer=L` | `pc_scores.bin` | raw f16[n_tok] |
| `GET /api/token-pca/...?token=N` | `token_scores.bin` row | `<II n_layers,full_k>` + f32 |
| `GET /api/token-layer/...?token=N&layer=L` | `token_scores.bin` | `<II K,layer>` + f16 |
| `GET /api/token-bundle/...?full=&hover=&layer=` | `token_scores.bin` (+ `mlp/`) | `TKBD` binary (§5) |
| `GET /api/neuron-field/...?token=N` | `token_neurons.bin` row | raw f16[n_real_layers·intermediate] |
| `GET /api/gate-heatmap/<m>/<mode>` | `gate_heatmap.npy` | raw `.npy` |
| `GET /api/token-bio/...?token=N` | `gate_heatmap.npy` row | JSON `{token_idx, max_per_layer}` |
| `GET /api/weight-align-all/<m>/<mode>` | `weight_alignment.json` | JSON array |
| `GET /api/weight-align/...?layer=N` | `weight_alignment.json` | JSON (one layer) |
| `GET /api/falsification/<m>/<mode>` | `falsification.json` | JSON `{random_baseline[][], top_pcs[][]}` |
| `GET /api/token-predicts/...?token=N&layer=L&k=K` | `token_predicts.bin` slice + `tokens.json` | JSON `{top_k:[{text, prob, logit, mri_idx}]}` |
| `GET /api/token-resolve/<m>/<mode>?text=...` | `tokens.json` | JSON `{idx, text}` (cross-model compare) |
| `GET /api/norms` · `/api/baselines` | sidecar | JSON |
| `GET /api/vocab-meta/<m>/<mode>` | `vocab_meta.json` | JSON (§7) |
| `GET /api/vocab-tokens/<m>/<mode>` | `vocab_tokens.json` | JSON array of text |
| `GET /api/vocab-scripts/<m>/<mode>` | `vocab_scripts.json` | JSON array of script |
| `GET /api/vocab-ids/<m>/<mode>` | `vocab_ids.npy` | raw `.npy` |
| `GET /api/vocab-gate-heatmap/<m>/<mode>` | `vocab_gate_heatmap.npy` | raw `.npy` |
| `GET /api/vocab-neuron-importance/<m>/<mode>` | `vocab_neuron_importance.json` | JSON array (§7) |
| `GET /api/vocab-pc-bundle/...?full=&medium=&step=` | `vocab_pc16.bin` slabs | `VPCB` binary (§7) |
| `GET /api/vocab-pc-columns/...?layer=L&pcs=A,B,C` | `vocab_pc16.bin` slabs, one layer, ≤256 PCs | `VPCL` binary (§7) |
| `GET /api/vocab-pc-column/...?layer=L&pc=P` | `vocab_pc16.bin` slab | `<III layer,pc,n_rows>` + raw f16[n_rows] |
| `GET /api/vocab-token/...?row=R` | `vocab_scores.bin` row | `<II n_layers,K>` + f32 |
| `GET /api/vocab-token-bundle/...?rows=A,B` | `vocab_scores.bin` rows | `VTKB` binary (§7) |
| `GET /api/vocab-token-neurons/...?rows=A,B` | `vocab_token_neurons.bin` rows | `VTNB` binary (§7) |

**Browser-computed** in the SPA over already-loaded arrays (no server compute):
`direction-project/quality/brief/discover/depth`, cross-model compare. **Producer-
only** (advertised via `/api/capabilities`, gated off at the edge): `direction-
circuit/nonlinear/weights/steer`, live forward/chat. **Live channels** (`poll`,
`chat-poll`, `live-status`) tag `static:true` so the SPA's MCP loops never start.
The SPA composes all of this from the capability manifest. See
[`docs/observatory.md`](../docs/observatory.md#the-capability-manifest-the-inverted-contract).

`.npy` parsing in the Worker: read the v1.0 header (`\x93NUMPY`, 2-byte version,
`uint16` header length at offset 8, dict header), payload at `10 + headerLen`;
`<f2` → `DataView.getFloat16`, `<f4` → `getFloat32`.

---

## 5. Composite response formats

### `cloud-bundle`: magic `CLDB`
```
header  <4sIIIIIII = magic, version, n_full, n_med, n_layers, n_tok, n_sample, step
        u32[n_full]                    # full PC ids
        u32[n_med]                     # medium PC ids
        f16[n_full * n_layers * n_tok] # full slabs (all tokens)
        f16[n_med  * n_layers * n_sample] # medium slabs (every step-th token)
```
Out-of-range PCs (≥ `full_k`) are dropped, not errored.

### `token-bundle`: magic `TKBD`
```
header  <4sIII = magic, version, layer, n_tokens
per token: <IIIIII = token_idx, flags, n_layers, full_k, hover_k, hover_inter
payloads (entry order):
  if flags&1: f32[n_layers * full_k]   # full PCA row
  if flags&2: f16[hover_k]             # hover layer row
  if flags&4: f16[hover_inter]         # hover neuron field (gate×up at layer)
```

---

## 6. Versioning

- `version` on `metadata.json` / `models.json` tracks producer schema.
- Binary magics (`HEI2` v2, `PCSC`, `TOKS`, `TOKN`, `CLDB` v1, `TKBD` v1,
  `VSCR`, `VP16`, `VTKN`, `VPCB` v1, `VPCL`, `VTKB` v1, `VTNB` v1) carry their
  own version field where present; bump on layout change.
- Artifacts are immutable and content-addressable → `Cache-Control: immutable`.

---

## 7. Full-vocabulary extension (vocab scope)

Produced after the fact by `heinrich mri-vocab --model <hf_id> --mri <path>`,
against an *existing* decomposition. It does not re-derive the PCA frame;
it projects every tokenizer token through the sample's frozen components and
means. Optional: a `.mri` published without these files still works, sample
scope only.

```
decomp/vocab_scores.bin        [VSCR]  every token, row-major, all layers
decomp/vocab_pc16.bin          [VP16]  the transpose: every token, one layer/PC at a time
decomp/vocab_ids.npy                   row → tokenizer id (int32)
decomp/vocab_tokens.json               row → decoded text (JSON array, not object)
decomp/vocab_scripts.json              row → script (JSON array, same order as vocab_tokens.json)
decomp/vocab_meta.json                 provenance, agreement check against the sample rows
decomp/vocab_gate_heatmap.npy          [n_rows × n_real_layers] f16, max|gate·up| per token/layer
decomp/vocab_token_neurons.bin [VTKN]  top-N signed gate·up per token/layer, every row
decomp/vocab_neuron_importance.json    per-layer top-N neuron indices + contributions
```

`vocab_row` means the same row in every one of these files. They share one
row ordering, fixed by `_vocab_token_list` at capture time (dedup'd by
decoded text, first token id wins).

### `vocab_scores.bin`: magic `VSCR` (row-major; O(1) per-token seek)
```
header  <4sIII = magic, n_rows, n_layers, K                        (16 bytes)
then    f16[n_rows][n_layers][K]           # one row per token
```
One token row = `n_layers * K * 2` bytes at offset `16 + row*stride`. Same
shape as `token_scores.bin`, full vocabulary instead of the sample. The stored
components are orthonormal and `K` is commonly the full `hidden_size`, so
distances computed here equal hidden-space distances exactly (up to f16
rounding). `vocab_meta.json`'s `distance_exact` says whether this particular
capture reaches that bar.

### `vocab_pc16.bin`: magic `VP16` (layer-major; O(1) per-(layer,PC) seek)
```
header  <4sIII = magic, n_layers, n_pcs, n_rows                    (16 bytes)
then    f16[n_layers][n_pcs][n_rows]       # one column per (layer, PC)
```
The transpose of `vocab_scores.bin`, truncated to the top `n_pcs` **display**
components. Measurement stays exact in `vocab_scores.bin`; this file exists
because "all tokens, one PC, one layer" (what the cloud viewports need) is a
full strided scan of the row-major file otherwise. One column =
`n_rows * 2` bytes at offset `16 + (layer*n_pcs + pc)*n_rows*2`.
`n_pcs` defaults to the sample decomposition's own PC ceiling, not a fixed
number, so every PC pair the UI can select is covered.

### `vocab_token_neurons.bin`: magic `VTKN`
```
header  <4sIII = magic, n_rows, n_real_layers, top_n               (16 bytes)
then    f16[n_rows][n_real_layers][top_n]  # signed gate·up, top_n columns only
```
Full-fidelity per-token neuron fields at vocab scale were costed at
8.8-68GB/model and rejected. This is the affordable version: every row, but
only each layer's `top_n` most-important neurons (by full-vocab importance,
not the sample's 2,000-token estimate). `vocab_neuron_importance.json` maps
slot index back to the real neuron index per layer. A slot outside the
top-`top_n` is genuinely uncaptured, not zero; treat it as missing, not off.

### `vocab_meta.json`
```json
{ "n_rows": 48660, "n_layers": 32, "n_real_layers": 30, "n_components": 576,
  "dtype": "float16",
  "layout": "VSCR: 16-byte header <4sIII (magic, n_rows, n_layers, K), row-major [n_rows, n_layers, K]",
  "distance_exact": true,
  "frame": { "source": "sample decomposition (frozen)", "n_sample": 2000, "mode": "raw" },
  "sample_agreement": { "n_rows_checked": 2000, "worst_median_rel": 0.03, "layers": ["…"] },
  "elapsed_s": 105.1 }
```
`sample_agreement` is the built-in falsification check: every sample-scope
row has a vocab-scope counterpart (same token), and the two are compared.
`worst_median_rel` above ~0.10 means the forward path drifted from the
original capture and the vocab projection shouldn't be trusted.

### `VPCB`: vocab-pc-bundle (mirrors `CLDB`, §5, byte-identical past the magic)
```
header  <4sIIIIIII = magic, version, n_full, n_med, n_layers, n_rows, n_sample, step
        u32[n_full]                     # full PC ids
        u32[n_med]                      # medium PC ids
        f16[n_full * n_layers * n_rows] # full slabs (every row)
        f16[n_med  * n_layers * n_sample] # medium slabs (every step-th row)
```
One ingest path in the client reads both `CLDB` and `VPCB`; only the magic
differs. `?full=` with no PC ids is a 32-byte header-only probe, used to
check the endpoint exists before committing to a real fetch.

### `VPCL`: vocab-pc-columns (single layer, many PCs, the LOD-friendly one)
```
header  <4sIII = magic, layer, n_pcs, n_rows                       (16 bytes)
        u32[n_pcs]                      # PC ids, this layer only
        f16[n_pcs * n_rows]             # one slab per requested PC
```
Capped at 256 PCs per request server-side. Exists because `VPCB` costs
`n_layers×` the bytes for an analysis that only needs one layer. Direction
coloring, the depth profile, Discover Features, and the reveal-zoom LOD all
read through this instead. Concurrent requests for overlapping PC lists on
the same layer share the underlying fetch client-side (`_vpcColInflight`);
different features computing "top PCs at this layer" by different rankings
routinely overlap without either one knowing about the other.

### `VTKB`: vocab-token-bundle (mirrors `TKBD`, §5, the pinned-pair round trip)
```
header  <4sIIII = magic, version, n_rows, n_layers, K
        u32[n_rows]                     # row ids, request order
        f32[n_rows * n_layers * K]      # full row, upcast from the f16 store
```

### `VTNB`: vocab-token-neurons-bundle
```
header  <4sIIII = magic, version, n_rows, n_layers, top_n
        u32[n_rows]                     # row ids, request order
        f16[n_rows * n_layers * top_n]  # stays f16 on the wire, no upcast
```
Half the bytes of `VTKB` for the same row count. There's no reason to widen
a top-N reduction that's already this small.

A conforming producer in any language can publish to this format; a conforming
reader can render any such artifact. That decoupling is the point.
