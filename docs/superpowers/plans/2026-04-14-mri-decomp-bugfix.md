# MRI + Decompose Bugfix Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 13 bugs in MRI capture, mri-decompose, and companion that cause OOM on models larger than SmolLM2-135M and break the viewer after the BIN_K cap commit.

**Architecture:** Three independent fix groups ordered by risk. Group A (companion/decompose point fixes) is safe and testable without a model. Group B (decompose memory) is medium-risk with clear before/after. Group C (MRI capture loop) is high-risk, touching the core capture path with no automated test coverage. Each commit should be independently correct.

**Tech Stack:** Python 3.10+, numpy, MLX (lazy-eval GPU), sklearn (randomized SVD)

**Key context:**
- MLX is a **lazy evaluation** framework. Operations build a computation graph. The graph is only evaluated when results are needed (e.g., converted to numpy, or explicitly forced). If you run 24 layers without forcing evaluation, the graph holds ALL 24 layers of intermediates in memory.
- The MRI capture has two file layouts: **flat** (`L{NN}_exit.npy` in MRI root) and **nested** (`layers/L{NN}/exit.npy`). Nested is used when `use_mmap=True` (captures >1GB). Both layouts must be handled everywhere.
- `ops.float32` is `mx.float32` on MLX and `torch.float32` on HF. Both backends define this attribute on their ops namespace.
- The capture loop returns 8 values from `layer_decomposed`: `h_exit, attn_weights, attn_scores, h_pre_mlp, attn_output, gate_val, up_val, mlp_out`. Each has different shapes and extraction logic.

---

## Bug–Task Map

| Bug | Description | Task |
|-----|------------|------|
| #10 | Score binary validation broken by BIN_K cap | 1 |
| #7 | Unbounded neuron cache in companion | 2 |
| #9 | Duplicate #tokpanel CSS | 3 |
| #13 | Dead code in delta PCA | 3 |
| #6 | Full SVD in companion PCA | 3 |
| #8 | Hardcoded /Volumes/sharts | 3 |
| #12 | all_scores accumulates full K (6.8 GB) | 4 |
| #11 | Neuron profiles load full gate*up as float32 (2.9 GB) | 5 |
| #1 | Mmap stat paths crash on nested layout | 6 |
| #2 | Resume check only finds flat layout | 6 |
| #3 | mlp_out never saved | 6 |
| #5 | Gate/up file handle churn (140K open/close) | 6 |
| OOM | MLX graph accumulates across all layers | 7 |
| #4 | Embedding gradient: 3 forward passes per batch | 8 |
| #9 | Batch size 32 hardcoded | 8 |

---

## Group A: Companion + Decompose Point Fixes (low risk)

### Task 1: Fix score binary validation (Bug #10)

The commit `a6148b6` capped the binary blob to BIN_K=min(K,50,max_k) PCs but didn't update the companion validation. `meta["n_components"]` stores full K (e.g. 576). Binary header stores BIN_K (e.g. 50). The companion compares them at line 781 and rejects every binary where hidden_size > 50 — which is every model.

**Files:**
- Modify: `src/heinrich/companion.py:781-782`

- [ ] **Step 1: Fix the validation**

In `src/heinrich/companion.py`, find lines 781-782:

```python
                        if meta.get("n_sample") != bnN or meta.get("n_components") != bnK:
                            self._send_json({"error": f"Binary/meta mismatch: bin has {bnN}tok/{bnK}pc, meta has {meta.get('n_sample')}/{meta.get('n_components')}. Re-run decomposition."})
```

Replace with:

```python
                        if meta.get("n_sample") != bnN:
                            self._send_json({"error": f"Binary/meta mismatch: bin has {bnN} tokens, meta has {meta.get('n_sample')}. Re-run decomposition."})
```

**Why the n_components check is wrong:** The binary blob intentionally caps to BIN_K (`min(K, 50, max_k)`) per commit a6148b6. The meta stores the full K (hidden_size). These SHOULD differ. The file size check on line 778 already validates structural integrity (header + variance block + score block = expected bytes). The n_sample check ensures the binary was generated from the same decomposition run.

- [ ] **Step 2: Commit**

```bash
git add src/heinrich/companion.py
git commit -m "fix: score binary validation — n_components check broken by BIN_K cap"
```

---

### Task 2: Bound neuron cache (Bug #7)

`_neuron_result_cache` at line 310 stores ~90KB per entry (30 layers x 1536 intermediate x 2 bytes float16) with no eviction. Hover across 2000 tokens = 180MB. No bound.

**Files:**
- Modify: `src/heinrich/companion.py:310,365`

- [ ] **Step 1: Add cache bound and eviction**

At line 310, replace:

```python
_neuron_result_cache: dict[str, bytes] = {}  # "mri_path:token_idx" → result bytes
```

with:

```python
_NEURON_CACHE_MAX = 2000  # ~180MB at 90KB/entry
_neuron_result_cache: dict[str, bytes] = {}  # "mri_path:token_idx" → result bytes
```

At line 365 (inside `_neuron_field`), replace:

```python
    _neuron_result_cache[cache_key] = data  # ~90KB per token, cached in memory
```

with:

```python
    if len(_neuron_result_cache) >= _NEURON_CACHE_MAX:
        # Evict oldest 25%
        keys = list(_neuron_result_cache.keys())
        for k in keys[:len(keys) // 4]:
            del _neuron_result_cache[k]
    _neuron_result_cache[cache_key] = data
```

- [ ] **Step 2: Commit**

```bash
git add src/heinrich/companion.py
git commit -m "fix: bound neuron cache to 2000 entries with 25% eviction"
```

---

### Task 3: Minor companion + decompose fixes (Bugs #6, #8, #9 CSS, #13)

Four independent one-line fixes. Each is trivially correct.

**Files:**
- Modify: `src/heinrich/companion.py:185,605,722`
- Modify: `src/heinrich/companion_ui.html:18`
- Modify: `src/heinrich/profile/compare.py:5473-5475`

- [ ] **Step 1: Randomized SVD in companion PCA (Bug #6)**

In `src/heinrich/companion.py`, replace line 185:

```python
        U, S, _ = np.linalg.svd(centered, full_matrices=False)
```

with:

```python
        from sklearn.utils.extmath import randomized_svd
        U, S, _ = randomized_svd(centered, n_components=min(10, centered.shape[1]),
                                  random_state=42)
```

**Why:** `np.linalg.svd` is O(n*d^2). For 150K tokens x 896 hidden, that's enormous. `randomized_svd` is O(n*k*d) — 18x faster. `mri_decompose` already uses it. The companion PCA only needs 3 components (for 3D projection) so `n_components=10` is generous.

- [ ] **Step 2: Extract hardcoded /Volumes/sharts (Bug #8)**

In `src/heinrich/companion.py`, add to the `CompanionHandler` class (after line 723):

```python
    mri_root = "/Volumes/sharts"

    def _mri_path(self, model: str, mode: str) -> str:
        return f"{self.mri_root}/{model}/{mode}.mri"
```

Then replace every occurrence of `f"/Volumes/sharts/{model}/{mode}.mri"` in `do_GET` with `self._mri_path(model, mode)`. There are 18 occurrences — lines 745, 803, 812, 822, 840, 853, 866, 878, 891, 902, 913, 933, 978, 984. Also replace the 3 direct Path constructions: lines 761, 830, 939, 954, 969.

In `run_companion` (line 1083), add `mri_root` parameter:

```python
def run_companion(port: int = 8377, mri_root: str = "/Volumes/sharts"):
    CompanionHandler.mri_root = mri_root
```

- [ ] **Step 3: Remove duplicate #tokpanel CSS (Bug #9)**

In `src/heinrich/companion_ui.html`, delete line 18:

```css
#tokpanel{position:fixed;right:8px;top:8px;width:280px;max-height:60vh;overflow-y:auto;background:rgba(10,10,10,0.95);border:1px solid #222;border-radius:4px;padding:8px;z-index:20;font-size:9px;backdrop-filter:blur(4px);cursor:default}
```

Line 37's `#tokpanel` definition is the active one. Line 18's `position:fixed` properties are stale from before the panel was moved into the grid layout.

- [ ] **Step 4: Remove dead code in delta PCA (Bug #13)**

In `src/heinrich/profile/compare.py`, replace lines 5473-5475:

```python
    for li in range(n_layers):
        entry_path = _find_exit(li)  # reuse resolver (entry has same path pattern)
        exit_path = _find_exit(li)
```

with:

```python
    for li in range(n_layers):
        exit_path = _find_exit(li)
```

Line 5474 (`entry_path = _find_exit(li)`) is dead code — the result is immediately overwritten at line 5479. The comment "entry has same path pattern" is wrong (entry files are `entry.npy`, not `exit.npy`).

- [ ] **Step 5: Commit**

```bash
git add src/heinrich/companion.py src/heinrich/companion_ui.html src/heinrich/profile/compare.py
git commit -m "fix: randomized SVD, configurable mri_root, stale CSS, dead delta PCA code"
```

---

## Group B: Decompose Memory Fixes (medium risk)

### Task 4: Cap all_scores accumulation to BIN_K (Bug #12)

`all_scores` at line 5299 accumulates full K columns (e.g. 896 for Qwen 0.5B) per layer. For 26 layers x 150K tokens x 896 x 2 bytes = 6.8 GB. But the binary blob at line 5554 only uses the first BIN_K=50 columns. The full-K scores are already saved per-layer to disk at line 5338. Fix: only keep BIN_K columns in memory.

**Files:**
- Modify: `src/heinrich/profile/compare.py:5294-5299, 5313-5314, 5342-5343, 5380-5381, 5396-5397, 5461-5468, 5553-5555`

- [ ] **Step 1: Compute BIN_K before the main loop**

After line 5294 (`print(f"  {K} components...")`), add:

```python
    # Compute BIN_K early — full scores saved per-layer to disk,
    # only BIN_K columns accumulated in memory for the binary blob
    _est_total_layers = n_layers + 2  # real layers + emb + lmh virtual layers
    _max_blob_bytes = 100 * 1024 * 1024
    _max_k = max(3, _max_blob_bytes // (_est_total_layers * n_sample * 2))
    BIN_K = min(K, 50, _max_k)
```

- [ ] **Step 2: Cap accumulation at all 4 append sites**

**Site 1 — missing layer fallback (line 5313-5314):**

Replace:
```python
            all_variances.append(np.zeros(K, dtype=np.float32))
            all_scores.append(np.zeros((n_sample, K), dtype=np.float16))
```
with:
```python
            all_variances.append(np.zeros(BIN_K, dtype=np.float32))
            all_scores.append(np.zeros((n_sample, BIN_K), dtype=np.float16))
```

**Site 2 — main layer loop (lines 5342-5343):**

Replace:
```python
        all_variances.append(variance)
        all_scores.append(scores)
```
with:
```python
        all_variances.append(variance[:BIN_K])
        all_scores.append(scores[:, :BIN_K].copy())  # .copy() detaches from full array
```

**Site 3 — virtual layer missing fallback (lines 5380-5381):**

Replace:
```python
            all_variances.append(np.zeros(K, dtype=np.float32))
            all_scores.append(np.zeros((n_sample, K), dtype=np.float16))
```
with:
```python
            all_variances.append(np.zeros(BIN_K, dtype=np.float32))
            all_scores.append(np.zeros((n_sample, BIN_K), dtype=np.float16))
```

**Site 4 — virtual layer after SVD (lines 5396-5397):**

Replace:
```python
        all_variances.append(variance)
        all_scores.append(scores)
```
with:
```python
        all_variances.append(variance[:BIN_K])
        all_scores.append(scores[:, :BIN_K].copy())
```

- [ ] **Step 3: Remove late BIN_K computation, simplify blob write**

Replace lines 5461-5468:
```python
    # BIN_K: cap for binary blob and delta PCA (viewer initial load)
    # Cap blob to ~100MB (JS doubles to float32 on parse)
    total_layers = len(all_scores)
    max_blob_bytes = 100 * 1024 * 1024
    max_k = max(3, max_blob_bytes // (total_layers * n_sample * 2))
    BIN_K = min(K, 50, max_k)
    print(f"  Blob: {total_layers} layers × {n_sample} tokens × {BIN_K} PCs = "
          f"{total_layers * n_sample * BIN_K * 2 / (1024*1024):.0f}MB", file=sys.stderr)
```

with:
```python
    total_layers = len(all_scores)
    print(f"  Blob: {total_layers} layers x {n_sample} tokens x {BIN_K} PCs = "
          f"{total_layers * n_sample * BIN_K * 2 / (1024*1024):.0f}MB", file=sys.stderr)
```

Replace lines 5553-5555 (blob write):
```python
    var_block = np.concatenate([v[:BIN_K] for v in all_variances]).tobytes()
    score_block = np.concatenate([s[:, :BIN_K].view(np.uint16).ravel()
                                  for s in all_scores]).tobytes()
```

with (arrays already capped, no slicing needed):
```python
    var_block = np.concatenate(all_variances).tobytes()
    score_block = np.concatenate([s.view(np.uint16).ravel()
                                  for s in all_scores]).tobytes()
```

- [ ] **Step 4: Commit**

```bash
git add src/heinrich/profile/compare.py
git commit -m "fix: cap all_scores to BIN_K columns — saves ~7GB for Qwen 0.5B decompose"
```

---

### Task 5: Chunk gate/neuron computation in decompose (Bug #11)

Two sites in `mri_decompose` load full gate/up arrays as float32, creating 2.9 GB temporaries for Qwen 0.5B.

**Site A — gate heatmap (lines 5448-5457):** `np.load(mmap_mode='r')` then `g * u` creates a full float32 intermediate via numpy's upcast.

**Site B — neuron importance (lines 5516-5518):** `.astype(np.float32)` explicitly copies the entire mmap into RAM.

**Files:**
- Modify: `src/heinrich/profile/compare.py:5448-5457, 5515-5518`

- [ ] **Step 1: Chunk gate heatmap**

Replace lines 5451-5457:
```python
            if gp.exists() and up.exists():
                g = np.load(str(gp), mmap_mode='r')
                u = np.load(str(up), mmap_mode='r')
                gate_heat[:, li] = np.abs(g * u).max(axis=1).astype(np.float16)
            elif gp.exists():
                g = np.load(str(gp), mmap_mode='r')
                gate_heat[:, li] = np.abs(g).max(axis=1).astype(np.float16)
```

with:
```python
            if gp.exists() and up.exists():
                g = np.load(str(gp), mmap_mode='r')
                u = np.load(str(up), mmap_mode='r')
                _chunk = 4096
                _col = np.zeros(n_sample, dtype=np.float32)
                for _s in range(0, n_sample, _chunk):
                    _e = min(_s + _chunk, n_sample)
                    _col[_s:_e] = np.abs(g[_s:_e].astype(np.float32) * u[_s:_e].astype(np.float32)).max(axis=1)
                gate_heat[:, li] = _col.astype(np.float16)
            elif gp.exists():
                g = np.load(str(gp), mmap_mode='r')
                _chunk = 4096
                _col = np.zeros(n_sample, dtype=np.float32)
                for _s in range(0, n_sample, _chunk):
                    _e = min(_s + _chunk, n_sample)
                    _col[_s:_e] = np.abs(g[_s:_e].astype(np.float32)).max(axis=1)
                gate_heat[:, li] = _col.astype(np.float16)
```

**Why 4096:** At 4096 rows x 4864 intermediate x 4 bytes float32 = 76 MB per chunk for gate AND up. Two chunks (g + u) = 152 MB peak. Compared to 2.9 GB for the full array.

- [ ] **Step 2: Chunk neuron importance**

Replace lines 5515-5518:
```python
            if gp.exists() and up_p.exists():
                g = np.load(str(gp), mmap_mode='r').astype(np.float32)
                u = np.load(str(up_p), mmap_mode='r').astype(np.float32)
                contrib = np.abs(g * u).mean(axis=0)  # [intermediate]
```

with:
```python
            if gp.exists() and up_p.exists():
                g = np.load(str(gp), mmap_mode='r')
                u = np.load(str(up_p), mmap_mode='r')
                _inter = g.shape[1]
                _contrib = np.zeros(_inter, dtype=np.float64)
                _chunk = 4096
                for _s in range(0, n_sample, _chunk):
                    _e = min(_s + _chunk, n_sample)
                    _gc = g[_s:_e].astype(np.float32)
                    _uc = u[_s:_e].astype(np.float32)
                    _contrib += np.abs(_gc * _uc).sum(axis=0)
                contrib = _contrib / n_sample
```

Note: uses float64 accumulator to avoid precision loss from summing many float32 values.

- [ ] **Step 3: Commit**

```bash
git add src/heinrich/profile/compare.py
git commit -m "fix: chunk gate/neuron computation — avoids 2.9GB float32 intermediates"
```

---

## Group C: MRI Capture Fixes (high risk)

### Task 6: Fix save section — mmap paths, resume, mlp_out, file handles (Bugs #1, #2, #3, #5)

These four bugs all live in the save section (lines 912-1334). They share the same root cause: the code was written for flat layout, then mmap was added with nested layout, but the save/resume/stat code wasn't updated. Fixing them separately would mean editing the same 300 lines four times. One coherent edit.

**Files:**
- Modify: `src/heinrich/profile/mri.py:36` (add helper), `912-930` (resume), `1062-1087` (gate/up alloc), `1182-1191` (gate/up write), `1278-1334` (save/stat)
- Test: `tests/test_profile_mri.py`

- [ ] **Step 1: Write test for `_find_layer_file`**

Add to `tests/test_profile_mri.py`:

```python
class TestFindLayerFile:

    def test_flat_layout(self, tmp_path):
        from heinrich.profile.mri import _find_layer_file
        d = tmp_path / "test.mri"
        d.mkdir()
        np.save(d / "L00_exit.npy", np.zeros((5, 4), dtype=np.float16))
        assert _find_layer_file(d, 0, "exit") == d / "L00_exit.npy"

    def test_nested_layout(self, tmp_path):
        from heinrich.profile.mri import _find_layer_file
        d = tmp_path / "test.mri"
        d.mkdir()
        ldir = d / "layers" / "L00"
        ldir.mkdir(parents=True)
        np.save(ldir / "exit.npy", np.zeros((5, 4), dtype=np.float16))
        assert _find_layer_file(d, 0, "exit") == ldir / "exit.npy"

    def test_nested_preferred_over_flat(self, tmp_path):
        from heinrich.profile.mri import _find_layer_file
        d = tmp_path / "test.mri"
        d.mkdir()
        np.save(d / "L00_exit.npy", np.zeros((5, 4), dtype=np.float16))
        ldir = d / "layers" / "L00"
        ldir.mkdir(parents=True)
        np.save(ldir / "exit.npy", np.ones((5, 4), dtype=np.float16))
        # Nested takes priority
        result = _find_layer_file(d, 0, "exit")
        assert result == ldir / "exit.npy"

    def test_missing_returns_none(self, tmp_path):
        from heinrich.profile.mri import _find_layer_file
        d = tmp_path / "test.mri"
        d.mkdir()
        assert _find_layer_file(d, 0, "exit") is None
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_profile_mri.py::TestFindLayerFile -v`
Expected: ImportError — `_find_layer_file` doesn't exist yet.

- [ ] **Step 3: Add `_find_layer_file` helper**

In `src/heinrich/profile/mri.py`, after line 38 (after `_is_mlx_backend`), add:

```python
def _find_layer_file(mri_dir: Path, layer: int, name: str) -> Path | None:
    """Find a per-layer .npy file in nested or flat layout.

    Nested: mri_dir/layers/L{NN}/{name}.npy  (used by mmap captures >1GB)
    Flat:   mri_dir/L{NN}_{name}.npy         (used by in-memory captures)
    """
    nested = mri_dir / "layers" / f"L{layer:02d}" / f"{name}.npy"
    if nested.exists():
        return nested
    flat = mri_dir / f"L{layer:02d}_{name}.npy"
    if flat.exists():
        return flat
    return None
```

- [ ] **Step 4: Run test — expect PASS**

Run: `pytest tests/test_profile_mri.py::TestFindLayerFile -v`
Expected: 4 passed.

- [ ] **Step 5: Fix resume check (lines 916-921)**

Replace lines 916-921:
```python
            existing_exit = sum(1 for i in range(n_layers)
                                if (out_dir_check / f"L{i:02d}_exit.npy").exists())
            if existing_exit == n_layers:
                test_path = out_dir_check / "L00_exit.npy"
                test_arr = np.load(test_path, mmap_mode='r')
```

with:
```python
            existing_exit = sum(1 for i in range(n_layers)
                                if _find_layer_file(out_dir_check, i, "exit") is not None)
            if existing_exit == n_layers:
                test_path = _find_layer_file(out_dir_check, 0, "exit")
                test_arr = np.load(test_path, mmap_mode='r')
```

- [ ] **Step 6: Fix gate/up allocation — keep file handles open (lines 1062-1087)**

Replace lines 1064-1087 (the gate_files/up_files allocation):
```python
    gate_files = {}
    up_files = {}
    if output:
        mlp_out_dir = Path(output) / "mlp"
        mlp_out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_layers):
            for name, store in [("gate", gate_files), ("up", up_files)]:
                fpath = mlp_out_dir / f"L{i:02d}_{name}.npy"
                # Write numpy header, then keep file open for batch writes
                header = np.lib.format.header_data_from_array_1_0(
                    np.zeros((n_tokens, intermediate_size), dtype=np.float16))
                with open(fpath, 'wb') as f:
                    np.lib.format.write_array_header_1_0(f, header)
                store[i] = {"path": fpath, "header_size": 128}  # .npy v1 header is 128 bytes typically
        # Measure actual header size from first file
        _test_path = mlp_out_dir / "L00_gate.npy"
        with open(_test_path, 'rb') as f:
            np.lib.format.read_magic(f)
            np.lib.format.read_array_header_1_0(f)
            _header_bytes = f.tell()
        for i in range(n_layers):
            gate_files[i]["header_size"] = _header_bytes
            up_files[i]["header_size"] = _header_bytes
```

with:
```python
    gate_handles = {}
    up_handles = {}
    _gate_header_size = 0
    if output:
        mlp_out_dir = Path(output) / "mlp"
        mlp_out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_layers):
            for name, store in [("gate", gate_handles), ("up", up_handles)]:
                fpath = mlp_out_dir / f"L{i:02d}_{name}.npy"
                header = np.lib.format.header_data_from_array_1_0(
                    np.zeros((n_tokens, intermediate_size), dtype=np.float16))
                with open(fpath, 'wb') as f:
                    np.lib.format.write_array_header_1_0(f, header)
                fh = open(fpath, 'r+b')
                fh.seek(0)
                np.lib.format.read_magic(fh)
                np.lib.format.read_array_header_1_0(fh)
                _gate_header_size = fh.tell()
                store[i] = {"fh": fh, "header_size": _gate_header_size}
```

- [ ] **Step 7: Fix gate/up write in capture loop (lines 1182-1191)**

Replace lines 1182-1191:
```python
            # Write gate/up batch directly to file (no memory accumulation)
            if output and i in gate_files:
                row_bytes = intermediate_size * 2  # float16
                offset = gate_files[i]["header_size"] + batch_start * row_bytes
                with open(gate_files[i]["path"], 'r+b') as f:
                    f.seek(offset)
                    f.write(batch_gates[i].tobytes())
                with open(up_files[i]["path"], 'r+b') as f:
                    f.seek(offset)
                    f.write(batch_ups[i].tobytes())
```

with:
```python
            # Write gate/up batch via persistent file handles
            if output and i in gate_handles:
                row_bytes = intermediate_size * 2  # float16
                foffset = gate_handles[i]["header_size"] + batch_start * row_bytes
                gate_handles[i]["fh"].seek(foffset)
                gate_handles[i]["fh"].write(batch_gates[i].tobytes())
                up_handles[i]["fh"].seek(foffset)
                up_handles[i]["fh"].write(uv_np.tobytes())
```

**Important:** After the capture loop ends (after line 1219), add handle cleanup:
```python
    # Close gate/up file handles
    for i in gate_handles:
        gate_handles[i]["fh"].close()
        up_handles[i]["fh"].close()
```

**Note for Task 7:** When Task 7 rewrites the capture loop, the gate/up variable names change from `batch_gates[i]`/`batch_ups[i]` to `gv_np`/`uv_np`. The reference to `gate_handles`/`up_handles` stays the same. The write code will be integrated into the new per-layer block.

- [ ] **Step 8: Fix save section stat paths + add mlp_out save (lines 1278-1334)**

Replace lines 1278-1334 (the entire "Layer files" section) with:

```python
        # Layer files: mmap captures already on disk, in-memory need writing
        total_size = 0
        if use_mmap:
            for i in range(n_layers):
                exit_arrays[i].flush()
                entry_arrays[i].flush()
                pre_mlp_arrays[i].flush()
                mlp_out_arrays[i].flush()
                for name in ("exit", "entry", "pre_mlp", "mlp_out"):
                    f = _find_layer_file(out_dir, i, name)
                    if f:
                        total_size += f.stat().st_size
        else:
            for i in range(n_layers):
                for name, arr_dict in [("exit", exit_arrays), ("entry", entry_arrays),
                                        ("pre_mlp", pre_mlp_arrays), ("mlp_out", mlp_out_arrays)]:
                    p = out_dir / f"L{i:02d}_{name}.npy"
                    np.save(p, arr_dict[i])
                    total_size += p.stat().st_size

        # Attention arrays
        attn_dir = out_dir / "attention"
        mlp_dir = out_dir / "mlp"
        if use_mmap:
            for i in range(n_layers):
                attn_out_arrays[i].flush()
                attn_weight_arrays[i].flush()
                attn_logit_arrays[i].flush()
                ao = _find_layer_file(out_dir, i, "attn_out")
                if ao:
                    total_size += ao.stat().st_size
        else:
            attn_dir.mkdir(exist_ok=True)
            for i in range(n_layers):
                np.save(out_dir / f"L{i:02d}_attn_out.npy", attn_out_arrays[i])
                np.save(attn_dir / f"L{i:02d}_weights.npy", attn_weight_arrays[i])
                np.save(attn_dir / f"L{i:02d}_logits.npy", attn_logit_arrays[i])
                total_size += (out_dir / f"L{i:02d}_attn_out.npy").stat().st_size

        # Attention weights/logits and gate/up stats (paths identical in both layouts)
        for i in range(n_layers):
            total_size += (attn_dir / f"L{i:02d}_weights.npy").stat().st_size
            total_size += (attn_dir / f"L{i:02d}_logits.npy").stat().st_size
            total_size += (mlp_dir / f"L{i:02d}_gate.npy").stat().st_size
            total_size += (mlp_dir / f"L{i:02d}_up.npy").stat().st_size
```

**What changed:**
1. mmap: flushes mlp_out_arrays (was missing)
2. mmap: uses `_find_layer_file` for exit, entry, pre_mlp, mlp_out, attn_out (was using flat paths)
3. non-mmap: saves mlp_out (was missing entirely)
4. Removed double-counting of gate/up (old code had lines 1320-1322 AND 1327-1328 both counting them)

- [ ] **Step 9: Run all existing tests**

Run: `pytest tests/test_profile_mri.py -v`
Expected: All pass (including the new TestFindLayerFile).

- [ ] **Step 10: Commit**

```bash
git add src/heinrich/profile/mri.py tests/test_profile_mri.py
git commit -m "fix: save section — mmap paths, resume check, mlp_out save, file handle churn"
```

---

### Task 7: Break MLX graph in capture loop (the OOM fix)

This is the highest-risk change. The current capture loop (lines 1126-1200) accumulates 9 lists of lazy MLX tensors across all layers, then calls `stack_to_numpy` once at the end of each batch. For 24 layers, the entire computation graph stays in GPU memory. Fix: convert each intermediate to numpy INSIDE the layer loop, immediately after each layer. This forces evaluation per-layer, freeing the graph.

**The exact semantics of each return value from `layer_decomposed`:**

| Value | Shape | Position to extract | Notes |
|-------|-------|-------------------|-------|
| `h` (h_exit) | `[B, T, hidden]` | `-1` (exit) and `token_pos` (entry) | Same tensor, two positions |
| `attn_w` | `[B, heads, T, T]` | row `token_pos` → `[B, heads, T_seq]` | Softmax weights |
| `attn_scores` | `[B, heads, T, T]` or `None` | row `token_pos` → `[B, heads, T_seq]` | Raw logits, needs clip to fp16 range. `None` on HF backend. |
| `h_pre_mlp` | `[B, T, hidden]` | `-1` | MLP input after norm |
| `attn_output` | `[B, T, hidden]` | `-1` | Attention contribution (before residual add) |
| `gate_val` | `[B, T, inter]` or `None` | `-1` | After SiLU/GeLU activation |
| `up_val` | `[B, T, inter]` or `None` | `-1` | Up projection output |
| `mlp_out` | `[B, T, hidden]` | `-1` | down_proj(gate * up) |

For raw/naked mode: T=1, so `[:, -1, :]` and `[:, 0, :]` are the same position. The `len(shape) == 3` checks handle this.

**Files:**
- Modify: `src/heinrich/profile/mri.py:1108-1200`

- [ ] **Step 1: Add conversion helpers before the batch loop**

After line 1110 (`t_start = time.time()`), before line 1113 (`for batch_start`), add:

```python
    # --- Per-layer conversion helpers ---
    # MLX is lazy: operations build a graph, evaluation deferred until needed.
    # Converting to numpy via np.array() forces evaluation of that subgraph.
    # By converting per-layer instead of accumulating, the graph never grows
    # beyond one layer's worth of intermediates.
    def _at_pos(t, pos):
        """Extract position from [B, T, D]. No-op for [B, D]."""
        return t[:, pos, :] if len(t.shape) == 3 else t

    if is_mlx:
        import mlx.core as mx
        def _to_np(t):
            """MLX tensor -> numpy float32. Forces graph evaluation."""
            return np.array(t.astype(mx.float32))
    else:
        def _to_np(t):
            """Torch tensor -> numpy float32."""
            return t.float().cpu().numpy()
```

- [ ] **Step 2: Replace the layer loop and store section**

Replace lines 1126-1191 (from `h = ops.embed(inp)` through the gate/up file write) with:

```python
        h = ops.embed(inp)
        for i, ly in enumerate(model_inner.layers):
            h, attn_w, attn_scores, h_pre_mlp, attn_output, gate_val, up_val, mlp_out = \
                ops.layer_decomposed(ly, h, mask)

            # --- Convert to numpy per-layer, freeing the MLX graph ---
            # h is [B, T, hidden]. Exit = last position, entry = token_pos.
            # _to_np forces MLX graph evaluation; h becomes a concrete array
            # for the next layer.
            exit_arrays[i][batch_start:batch_end] = \
                (_to_np(_at_pos(h, -1)) - baseline_exit[i]).astype(np.float16)
            entry_arrays[i][batch_start:batch_end] = \
                (_to_np(_at_pos(h, token_pos)) - baseline_entry[i]).astype(np.float16)

            # h_pre_mlp, attn_output, mlp_out: [B, T, hidden] -> last pos
            pre_mlp_arrays[i][batch_start:batch_end] = \
                _to_np(_at_pos(h_pre_mlp, -1)).astype(np.float16)
            attn_out_arrays[i][batch_start:batch_end] = \
                _to_np(_at_pos(attn_output, -1)).astype(np.float16)
            mlp_out_arrays[i][batch_start:batch_end] = \
                _to_np(_at_pos(mlp_out, -1)).astype(np.float16)

            # Attention weights: [B, heads, T, T] -> [B, heads, T_seq] at token_pos
            attn_weight_arrays[i][batch_start:batch_end] = \
                _to_np(attn_w[:, :, token_pos, :]).astype(np.float16)

            # Attention scores: needs clip to float16 range. None on HF backend.
            if attn_scores is not None:
                attn_logit_arrays[i][batch_start:batch_end] = \
                    np.clip(_to_np(attn_scores[:, :, token_pos, :]),
                            -65504, 65504).astype(np.float16)
            else:
                attn_logit_arrays[i][batch_start:batch_end] = \
                    np.zeros((B, n_heads, T_seq), dtype=np.float16)

            # Gate/up: [B, T, inter] -> last pos. May be None.
            if gate_val is not None:
                gv_np = _to_np(_at_pos(gate_val, -1)).astype(np.float16)
                uv_np = _to_np(_at_pos(up_val, -1)).astype(np.float16)
            else:
                gv_np = np.zeros((B, intermediate_size), dtype=np.float16)
                uv_np = np.zeros((B, intermediate_size), dtype=np.float16)

            # Write gate/up via persistent file handles
            if output and i in gate_handles:
                row_bytes = intermediate_size * 2
                foffset = gate_handles[i]["header_size"] + batch_start * row_bytes
                gate_handles[i]["fh"].seek(foffset)
                gate_handles[i]["fh"].write(gv_np.tobytes())
                up_handles[i]["fh"].seek(foffset)
                up_handles[i]["fh"].write(uv_np.tobytes())
```

**What was removed:**
- The 9 list declarations (`layer_entries`, `layer_exits`, `layer_pre_mlps`, `layer_attn_outs`, `layer_mlp_outs`, `batch_attn_w`, `batch_attn_s`, `batch_gates`, `batch_ups`)
- The `ops.stack_to_numpy()` calls (lines 1169-1173)
- The post-loop store block (lines 1174-1191)
- The bfloat16 special-casing for gate/up (lines 1157-1158) — `_to_np` handles this via `.astype(mx.float32)` (MLX) or `.float()` (torch)

**What stays the same:**
- The embedding gradient block (lines 1193-1200) — untouched in this task, moved to Task 8
- The progress/checkpoint code (lines 1202-1218)

- [ ] **Step 3: Verify old tests still pass**

Run: `pytest tests/test_session5_code.py::TestDecomposedForward -v`
Expected: All 4 tests pass (these test the decomposed forward functions, not the capture loop).

Run: `pytest tests/test_profile_mri.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/heinrich/profile/mri.py
git commit -m "fix: break MLX graph per-layer — convert intermediates to numpy inside loop"
```

---

### Task 8: Adaptive batch size + embedding gradient batching (Bugs #4, #9)

Batch size is hardcoded at 32. For Qwen 0.5B template mode (24 layers x 896 hidden), the layer loop (even with per-layer eval) processes 32 tokens through 24 layers. With the Q@K recomputation at each layer, this is ~6GB of peak GPU memory per batch.

The embedding gradient is worse: it does a full forward pass (probe for top-1) + a full forward+backward (gradient computation) through ALL layers, per batch. That's ~3x the memory of the capture forward, and there's no per-layer eval possible inside it.

**Files:**
- Modify: `src/heinrich/profile/mri.py:1109, 1193-1200`

- [ ] **Step 1: Adaptive batch size**

Replace line 1109:
```python
    batch_size = 32  # all modes batchable — template has fixed length
```

with:
```python
    # Adaptive batch size: larger models need smaller batches.
    # mem_factor correlates with GPU memory per batch per layer.
    # 135M (576h, 30L) = 17280 -> 32; 360M (~960h, ~32L) = 30720 -> 8; 0.5B (896h, 24L) = 21504 -> 16
    _mem_factor = n_layers * hidden
    if _mem_factor > 30000:
        batch_size = 4
    elif _mem_factor > 20000:
        batch_size = 8
    elif _mem_factor > 10000:
        batch_size = 16
    else:
        batch_size = 32
    print(f"  Batch size: {batch_size} (layers*hidden={_mem_factor})")
```

- [ ] **Step 2: Separate embedding gradient into smaller batches**

Replace lines 1193-1200:
```python
        # Backward pass: embedding gradient
        emb_for_grad = ops.embed(inp)
        eg = ops.embedding_grad(emb_for_grad, mask)
        # eg is [B, seq_len, hidden] — take the token position
        if len(eg.shape) == 3:
            emb_grad_arrays[0][batch_start:batch_end] = eg[:, token_pos, :].astype(np.float16)
        else:
            emb_grad_arrays[0][batch_start:batch_end] = eg.astype(np.float16)
```

with:
```python
        # Embedding gradient: separate smaller batches.
        # The gradient computation does probe forward + forward + backward through
        # ALL layers. No per-layer eval possible. Use batch_size//4 to cap memory.
        _eg_batch = max(1, min(B, batch_size // 4))
        for _eg_s in range(0, B, _eg_batch):
            _eg_e = min(_eg_s + _eg_batch, B)
            _eg_inp = inp[_eg_s:_eg_e] if not isinstance(inp, np.ndarray) else ops.array(inp[_eg_s:_eg_e])
            _eg_mask = mask  # mask shape is (T, T), shared across batch
            _eg_emb = ops.embed(_eg_inp)
            _eg = ops.embedding_grad(_eg_emb, _eg_mask)
            if len(_eg.shape) == 3:
                emb_grad_arrays[0][batch_start + _eg_s:batch_start + _eg_e] = \
                    _eg[:, token_pos, :].astype(np.float16)
            else:
                emb_grad_arrays[0][batch_start + _eg_s:batch_start + _eg_e] = \
                    _eg.astype(np.float16)
```

**Why this works:** The `inp` variable is the full batch input tensor (MLX or torch). Slicing `inp[_eg_s:_eg_e]` extracts sub-batches. The mask is (T, T) — it doesn't depend on batch size, so it's reused.

- [ ] **Step 3: Commit**

```bash
git add src/heinrich/profile/mri.py
git commit -m "fix: adaptive batch size + sub-batched embedding gradient"
```

---

### Task 9: Verification

No automated test runs `capture_mri`. The only way to verify Tasks 6-8 is to run an actual capture and check the output.

- [ ] **Step 1: Run test suite**

```bash
pytest tests/test_profile_mri.py tests/test_session5_code.py -v
```
Expected: All pass.

- [ ] **Step 2: Run MRI capture on SmolLM2-135M (the model that works)**

```bash
cd /Users/asuramaya/Code/heinrich
python -m heinrich.cli mri --model smollm2-135m --mode raw --output /tmp/test_135m_raw.mri --n-index 100
```

Expected:
- Completes without error
- Output directory has `L{NN}_exit.npy` or `layers/L{NN}/exit.npy` files
- `metadata.json` exists with correct n_tokens and n_layers
- `mlp/L{NN}_gate.npy` files exist
- For 100 tokens, `use_mmap` should be False (estimated_bytes < 1GB), so flat layout

Verify: `python -c "from heinrich.profile.mri import load_mri; m = load_mri('/tmp/test_135m_raw.mri'); print(m['metadata']['capture']['n_tokens'], 'tokens,', m['metadata']['model']['n_layers'], 'layers'); print('exit_L0:', m['exit_L0'].shape); print('mlp_out_L0:', m.get('mlp_out_L0', 'MISSING'))"`

Expected: `100 tokens, 30 layers`, `exit_L0: (100, 576)`, `mlp_out_L0: (100, 576)` (not MISSING).

- [ ] **Step 3: Verify companion loads scores**

```bash
python -m heinrich.cli companion &
sleep 2
curl -s http://localhost:8377/api/scores/smollm2-135m/template | head -c 20 | xxd | head -1
kill %1
```

Expected: Binary data starting with `HEIN` magic (hex `4845 494e`), not a JSON error about n_components mismatch.

- [ ] **Step 4: Clean up**

```bash
rm -rf /tmp/test_135m_raw.mri
```
