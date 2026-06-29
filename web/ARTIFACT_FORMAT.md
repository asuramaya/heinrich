# The Heinrich MRI artifact format

The contract between the **producer** (`heinrich mri` / `mri-decompose`) and the
**consumer** (the Observatory: Worker + browser SPA). A publish target that
conforms to this spec can be rendered by the reference reader with no model and
no server compute — just object reads and HTTP byte-range requests.

Design principle: **every consumer query is either a plain GET of a precomputed
JSON, or a fixed-stride byte-range read of an immutable binary index.** No
numpy parsing in the Worker; no recomputation server-side.

All integers are little-endian. `f16` = IEEE half (`<e`), `f32` = `<f`.

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
    L{NN}_scores.npy / _variance.npy / _components.npy
    emb_scores.npy / lmh_scores.npy (+ _variance)   # virtual boundary layers
  norms.json / baselines.json                # worker-native sidecars (from *.npz)
  mlp/L{NN}_gate.npy, _up.npy                # (optional) for token-hover neurons
```

The producer also writes `tokens.npz`, `norms.npz`, `baselines.npz`, raw
per-layer activations, `weights/`, and `attention/`. **The consumer never reads
those** — `heinrich publish` uploads only the files above. The raw `.mri` can
be multi-GB; the published subset is a few MB + the optional neuron index.

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
`intermediate_size` (MLP width — the Neurons viewport reads it), `layers`
(ordered list, drives variance ordering), `virtual_layers`, `sample_indices`,
`method`. `mri-decompose` writes `intermediate_size`; `heinrich publish`
back-fills it from the `token_neurons.bin` TOKN header for older decompositions.

Layer ordering is canonical and shared by every binary index:
`[L00 … L{n-1}, emb, lmh]` → `total_layers = n_real_layers + 2`.

---

## 2. Binary indexes

Each begins with a 4-byte magic so the consumer can validate.

### `all_scores.bin` — magic `HEI2`
```
header  <4sIIII = magic, total_layers, n_tokens, score_k, var_k     (20 bytes)
then    f32[total_layers * var_k]            # per-layer variance (row-major)
then    f16[total_layers * n_tokens * score_k]   # capped scores
```
`score_k = min(n_components, 50)`. Variance carries full `var_k`. The variance
block alone is enough to synthesize `serve-meta` (§4).

### `pc_scores.bin` — magic `PCSC`  (PC-major; O(1) per-PC seek)
```
header  <4sIII = magic, total_layers, n_tokens, full_k              (16 bytes)
then    f16[full_k][total_layers][n_tokens]   # one contiguous slab per PC
```
One PC slab = `total_layers * n_tokens * 2` bytes at offset `16 + pc*stride`.
Drives the cloud (§4 `cloud-bundle`) and `pc-full` / `pc-column`.

### `token_scores.bin` — magic `TOKS`  (token-major; O(1) per-token seek)
```
header  <4sIII = magic, n_tokens, total_layers, full_k              (16 bytes)
then    f16[n_tokens][total_layers][full_k]   # one row per token
```
One token row = `total_layers * full_k * 2` bytes at offset `16 + tok*stride`.

### `token_neurons.bin` — magic `TOKN`  (per-token neuron field)
```
header  <4sIII = magic, n_tokens, n_real_layers, intermediate       (16 bytes)
then    f16[n_tokens][n_real_layers][intermediate]   # precomputed gate×up
```
Virtual layers excluded (no MLP). Drives `neuron-field`.

---

## 3. JSON sidecars (worker-native, emitted at publish)

The consumer must not parse `.npz`. The producer flattens them to JSON:

- **`decomp/tokens.json`** — `{ token_ids:[int], scripts:[str], token_texts:[str] }`
  (length `n_tokens`; from `tokens.npz`).
- **`norms.json`** — `{ <key>: { mean, std, min, max, shape } }` (from `norms.npz`).
- **`baselines.json`** — `{ <key>: { norm, mean, std, shape } }` (from `baselines.npz`).
- **`decomp/weight_alignment.json`** — `[{ layer, matrices:[{ name, alignment:[full_k] }] }]`
  (already precomputed by `mri-decompose`; flowers read it directly).
- **`decomp/neuron_importance.json`** — top neurons per layer.

---

## 4. Consumer HTTP API (the Worker contract)

The reference Worker (`web/worker/index.js`) exposes these. Each maps to object
reads above. **Binary responses** are raw little-endian; **JSON** as noted.

| Endpoint | Reads | Response |
| --- | --- | --- |
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
| `GET /api/norms` · `/api/baselines` | sidecar | JSON |

**Compute endpoints** (`direction-*`, `token-predicts`) return `501` — they are
Phase-2 browser-compute, not server-served. **Live endpoints** (`poll`,
`chat-poll`, `live-status`) return benign no-ops in a static deployment.

`.npy` parsing in the Worker: read the v1.0 header (`\x93NUMPY`, 2-byte version,
`uint16` header length at offset 8, dict header), payload at `10 + headerLen`;
`<f2` → `DataView.getFloat16`, `<f4` → `getFloat32`.

---

## 5. Composite response formats

### `cloud-bundle` — magic `CLDB`
```
header  <4sIIIIIII = magic, version, n_full, n_med, n_layers, n_tok, n_sample, step
        u32[n_full]                    # full PC ids
        u32[n_med]                     # medium PC ids
        f16[n_full * n_layers * n_tok] # full slabs (all tokens)
        f16[n_med  * n_layers * n_sample] # medium slabs (every step-th token)
```
Out-of-range PCs (≥ `full_k`) are dropped, not errored.

### `token-bundle` — magic `TKBD`
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
- Binary magics (`HEI2` v2, `PCSC`, `TOKS`, `TOKN`, `CLDB` v1, `TKBD` v1) carry
  their own version field; bump on layout change.
- Artifacts are immutable and content-addressable → `Cache-Control: immutable`.

A conforming producer in any language can publish to this format; a conforming
reader can render any such artifact. That decoupling is the point.
