# Concept Microscope v2: Full-Dimensional Analysis & Causal Testing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all 50-PC approximations with full 896-dimensional analysis, add causal intervention testing, and enable nonlinear feature detection — so the tool measures concepts properly instead of approximating them.

**Architecture:** Server-side `/api/direction-*` endpoints compute full-K projections and return results to the browser. The browser displays metrics and colors tokens using server-computed projections. Steering integrates the existing backend protocol (`steer_dirs`) with a new `/api/steer-test` endpoint. Nonlinearity is measured via k-NN vs linear probe accuracy comparison on server-computed projections.

**Tech Stack:** Python 3.10+, numpy (mmap), existing MLX/HF backends for steering, companion HTTP server, companion_ui.html (Three.js)

---

## File Map

| File | Role | Changes |
|------|------|---------|
| `src/heinrich/companion.py` | HTTP server + analysis endpoints | Add `/api/direction-project`, `/api/direction-steer`, `/api/direction-nonlinear` endpoints + computation functions |
| `src/heinrich/companion_ui.html` | 3D viewer + interaction | Replace 50-PC direction coloring with server-projected full-K coloring, add steer-test UI, add nonlinearity display |
| `src/heinrich/mcp.py` | MCP tool definitions | Add `heinrich_companion_direction` tool for full analysis |
| `tests/test_direction.py` | Tests for direction analysis | New file: test direction quality, projection, nonlinearity |

---

### Task 1: Full-K Direction Projection Endpoint

Replace the 50-PC browser-side direction coloring with server-computed full-896D projections for all tokens.

**Files:**
- Modify: `src/heinrich/companion.py` (add endpoint after `_direction_quality`)
- Test: `tests/test_direction.py` (new file)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_direction.py
"""Tests for full-K direction analysis."""
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch
import tempfile


def _make_fake_scores(n_tokens, n_pcs, tmpdir):
    """Create a minimal decomp directory with fake scores."""
    decomp = Path(tmpdir) / "decomp"
    decomp.mkdir()
    scores = np.random.randn(n_tokens, n_pcs).astype(np.float16)
    np.save(str(decomp / "L00_scores.npy"), scores)
    tokens_path = Path(tmpdir) / "tokens.npz"
    np.savez(tokens_path,
             token_texts=np.array([f"tok{i}" for i in range(n_tokens)]),
             scripts=np.array(["latin"] * n_tokens),
             token_ids=np.arange(n_tokens))
    meta = {"n_sample": n_tokens, "n_real_layers": 1,
            "layers": [{"layer": 0, "pc1_pct": 50, "intrinsic_dim": 10, "neighbor_stability": 0.5}]}
    (decomp / "meta.json").write_text(json.dumps(meta))
    return scores


def test_direction_project_returns_all_tokens():
    with tempfile.TemporaryDirectory() as tmpdir:
        scores = _make_fake_scores(100, 50, tmpdir)
        from heinrich.companion import _direction_project
        result = _direction_project(tmpdir, a=0, b=1, layer=0)
        assert "projections" in result
        assert len(result["projections"]) == 100
        # Verify projections are computed in full K (50 dims, not truncated)
        diff = scores[0].astype(np.float32) - scores[1].astype(np.float32)
        direction = diff / (np.linalg.norm(diff) + 1e-8)
        expected = scores.astype(np.float32) @ direction
        np.testing.assert_allclose(result["projections"], expected, atol=0.1)


def test_direction_project_normalization():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_fake_scores(100, 50, tmpdir)
        from heinrich.companion import _direction_project
        result = _direction_project(tmpdir, a=0, b=1, layer=0)
        proj = result["projections"]
        # Projections should be centered around the midpoint of A and B
        pa, pb = proj[0], proj[1]
        mid = (pa + pb) / 2
        span = abs(pa - pb) / 2
        assert span > 0  # tokens should be different
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_direction.py -v`
Expected: FAIL with "cannot import name '_direction_project'"

- [ ] **Step 3: Write the endpoint function**

Add to `src/heinrich/companion.py` after `_direction_quality()`:

```python
def _direction_project(mri_path: str, a: int, b: int, layer: int) -> dict:
    """Project all tokens onto the full-K direction between two tokens.

    Returns normalized projections [-1, +1] centered on the A-B midpoint,
    suitable for direct use as color values in the browser.
    """
    decomp = Path(mri_path) / "decomp"
    score_path = decomp / f"L{layer:02d}_scores.npy"
    if not score_path.exists():
        return {"error": f"No scores at L{layer}"}

    scores = np.load(str(score_path), mmap_mode="r")
    N, K = scores.shape

    if a >= N or b >= N:
        return {"error": f"Token index out of range (max {N-1})"}

    diff = scores[a].astype(np.float32) - scores[b].astype(np.float32)
    mag = float(np.linalg.norm(diff))
    direction = diff / (mag + 1e-8)

    proj = (scores.astype(np.float32) @ direction).tolist()

    return {"projections": proj, "magnitude": mag, "K": K,
            "proj_a": proj[a], "proj_b": proj[b]}
```

Add GET endpoint in `do_GET()` before `/api/commands`:

```python
        elif path.startswith('/api/direction-project/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                a = int(qs.get('a', ['0'])[0])
                b = int(qs.get('b', ['0'])[0])
                layer = int(qs.get('layer', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                result = _direction_project(mri_path, a, b, layer)
                if "projections" in result:
                    # Send as binary float32 for efficiency (600KB vs 3MB JSON)
                    import struct
                    proj = np.array(result["projections"], dtype=np.float32)
                    header = json.dumps({k: v for k, v in result.items()
                                        if k != "projections"}).encode()
                    body = struct.pack('<I', len(header)) + header + proj.tobytes()
                    self._send_bytes(body)
                else:
                    self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/direction-project/<model>/<mode>?a=N&b=N&layer=L"})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_direction.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/heinrich/companion.py tests/test_direction.py
git commit -m "feat: full-K direction projection endpoint"
```

---

### Task 2: Replace Browser 50-PC Coloring with Full-K Server Projections

**Files:**
- Modify: `src/heinrich/companion_ui.html` (`_applyDirectionColors` function)

- [ ] **Step 1: Write the server fetch + binary parse in JS**

Replace `_applyDirectionColors()` in companion_ui.html. The new version fetches `/api/direction-project/` and uses the server-computed full-K projections instead of computing locally from the 50-PC blob.

```javascript
let _dirColorActive=false;
let _dirColorReqId=0;
async function _applyDirectionColors(){
  if(pinnedToken<0||pinnedTokenB<0||!model||!mode||!nN){
    if(_dirColorActive){
      _dirColorActive=false;
      // Reset to script colors
      for(const v of views){
        _ensureColors(v);
        if(v.trajMarker){const tc=v.trajMarker.geometry.attributes.color?.array;
          if(tc){for(let i=0;i<nN;i++){const hex=CL[scripts[i]]||0xAAAAAA;tc[i*3]=((hex>>16)&0xFF)/255;tc[i*3+1]=((hex>>8)&0xFF)/255;tc[i*3+2]=(hex&0xFF)/255;}v.trajMarker.geometry.attributes.color.needsUpdate=true;}}
        if(v.trajGroup&&v._trajTokenMap){const lc=v.trajGroup.geometry.attributes.color?.array;
          if(lc){for(const{ti,start,count}of v._trajTokenMap){const hex=CL[scripts[ti]]||0xAAAAAA;const r=((hex>>16)&0xFF)/255,g=((hex>>8)&0xFF)/255,b=(hex&0xFF)/255;
            for(let j=start;j<start+count;j++){lc[j*3]=r;lc[j*3+1]=g;lc[j*3+2]=b;}}v.trajGroup.geometry.attributes.color.needsUpdate=true;}}
      }
      _markDirty();
    }
    return;
  }
  const reqId=++_dirColorReqId;
  try{
    const resp=await fetch(`/api/direction-project/${model}/${mode}?a=${pinnedToken}&b=${pinnedTokenB}&layer=${cL}`);
    if(_dirColorReqId!==reqId)return;// superseded
    const buf=await resp.arrayBuffer();
    // Parse: 4-byte header length + JSON header + float32 projections
    const dv=new DataView(buf);
    const hdrLen=dv.getUint32(0,true);
    const hdr=JSON.parse(new TextDecoder().decode(new Uint8Array(buf,4,hdrLen)));
    const proj=new Float32Array(buf,4+hdrLen);
    if(proj.length!==nN)return;// mismatch
    // Normalize: center on midpoint, scale by half-gap
    const pa=hdr.proj_a,pb=hdr.proj_b;
    const mid=(pa+pb)/2,span=Math.abs(pa-pb)/2||1;
    // Compute RGB per token
    const _dirR=new Float32Array(nN),_dirG=new Float32Array(nN),_dirB=new Float32Array(nN);
    for(let i=0;i<nN;i++){
      const t=Math.max(-1,Math.min(1,(proj[i]-mid)/span));
      const at=Math.abs(t);
      if(t>0){_dirR[i]=0.25+0.75*at;_dirG[i]=0.25*(1-at);_dirB[i]=0.15*(1-at);}
      else{_dirR[i]=0.15*(1-at);_dirG[i]=0.25*(1-at);_dirB[i]=0.25+0.55*at;}
    }
    for(const v of views){
      const col=v.geo.attributes.color?.array;
      if(col){for(let i=0;i<nN;i++){col[i*3]=_dirR[i];col[i*3+1]=_dirG[i];col[i*3+2]=_dirB[i];}v.geo.attributes.color.needsUpdate=true;}
      if(v.trajMarker){const tc=v.trajMarker.geometry.attributes.color?.array;
        if(tc){for(let i=0;i<nN;i++){tc[i*3]=_dirR[i];tc[i*3+1]=_dirG[i];tc[i*3+2]=_dirB[i];}v.trajMarker.geometry.attributes.color.needsUpdate=true;}}
      if(v.trajGroup&&v._trajTokenMap){const lc=v.trajGroup.geometry.attributes.color?.array;
        if(lc){for(const{ti,start,count}of v._trajTokenMap){
          for(let j=start;j<start+count;j++){lc[j*3]=_dirR[ti];lc[j*3+1]=_dirG[ti];lc[j*3+2]=_dirB[ti];}
        }v.trajGroup.geometry.attributes.color.needsUpdate=true;}}
    }
    _dirColorActive=true;_markDirty();
  }catch(e){/* silently fail — coloring is best-effort */}
}
```

- [ ] **Step 2: Test manually**

Restart companion, refresh browser, pin two tokens. Cloud should recolor after ~4 second server round-trip. Verify by pinning CJK vs Latin tokens — should show clean split identical to before but now using all 896 dimensions.

- [ ] **Step 3: Commit**

```bash
git add src/heinrich/companion_ui.html
git commit -m "feat: full-K direction coloring via server projection"
```

---

### Task 3: Nonlinearity Detection Endpoint

Measures whether a concept is linear (separable by a hyperplane) or nonlinear (needs curved boundary). Computes k-NN classification accuracy vs linear probe accuracy on the direction. Large gap = nonlinear feature.

**Files:**
- Modify: `src/heinrich/companion.py` (add `_direction_nonlinear` + endpoint)
- Modify: `tests/test_direction.py` (add test)

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_direction.py

def test_direction_nonlinear_linear_data():
    """A perfectly linear split should have knn ≈ linear accuracy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create data with a clean linear split on dim 0
        n = 200
        scores = np.zeros((n, 10), dtype=np.float16)
        scores[:, 0] = np.linspace(-5, 5, n)  # linear gradient
        scores[:, 1:] = np.random.randn(n, 9).astype(np.float16) * 0.1  # noise
        decomp = Path(tmpdir) / "decomp"
        decomp.mkdir()
        np.save(str(decomp / "L00_scores.npy"), scores)
        tokens_path = Path(tmpdir) / "tokens.npz"
        np.savez(tokens_path,
                 token_texts=np.array([f"tok{i}" for i in range(n)]),
                 scripts=np.array(["latin"] * n),
                 token_ids=np.arange(n))
        meta = {"n_sample": n, "n_real_layers": 1,
                "layers": [{"layer": 0, "pc1_pct": 50, "intrinsic_dim": 10, "neighbor_stability": 0.5}]}
        (decomp / "meta.json").write_text(json.dumps(meta))

        from heinrich.companion import _direction_nonlinear
        result = _direction_nonlinear(tmpdir, a=0, b=n - 1, layer=0, n_sample=100)
        assert "linear_acc" in result
        assert "knn_acc" in result
        # For perfectly linear data, both should be high and gap should be small
        assert result["linear_acc"] > 0.8
        assert abs(result["knn_acc"] - result["linear_acc"]) < 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_direction.py::test_direction_nonlinear_linear_data -v`
Expected: FAIL with "cannot import name '_direction_nonlinear'"

- [ ] **Step 3: Implement the function**

Add to `src/heinrich/companion.py` after `_direction_project()`:

```python
def _direction_nonlinear(mri_path: str, a: int, b: int, layer: int,
                         n_sample: int = 2000) -> dict:
    """Test whether a direction is linear or nonlinear.

    Splits tokens into A-side and B-side by projection sign.
    Compares linear probe accuracy vs k-NN accuracy.
    Large gap (knn >> linear) means the concept boundary is curved.
    """
    decomp = Path(mri_path) / "decomp"
    score_path = decomp / f"L{layer:02d}_scores.npy"
    if not score_path.exists():
        return {"error": f"No scores at L{layer}"}

    scores = np.load(str(score_path), mmap_mode="r")
    N, K = scores.shape

    if a >= N or b >= N:
        return {"error": f"Token index out of range (max {N-1})"}

    # Compute direction and project
    diff = scores[a].astype(np.float32) - scores[b].astype(np.float32)
    mag = float(np.linalg.norm(diff))
    direction = diff / (mag + 1e-8)
    proj = scores.astype(np.float32) @ direction

    # Labels: 1 = A-side (positive projection), 0 = B-side
    pa, pb = float(proj[a]), float(proj[b])
    mid = (pa + pb) / 2
    labels = (proj > mid).astype(np.int32)

    # Sample for speed
    rng = np.random.RandomState(42)
    n_sample = min(n_sample, N)
    idx = rng.choice(N, n_sample, replace=False)
    X = scores[idx].astype(np.float32)
    y = labels[idx]

    # Skip if too imbalanced
    pos = y.sum()
    if pos < 10 or (n_sample - pos) < 10:
        return {"linear_acc": 0.5, "knn_acc": 0.5, "gap": 0.0,
                "verdict": "too_few_samples", "n_sample": n_sample}

    # Train/test split
    n_train = int(n_sample * 0.7)
    perm = rng.permutation(n_sample)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Linear probe: logistic regression on full K dimensions
    # Use simple approach: project onto direction, threshold at midpoint
    proj_test = X_test @ direction
    linear_pred = (proj_test > mid).astype(np.int32)
    linear_acc = float((linear_pred == y_test).mean())

    # k-NN: 5-nearest-neighbors in full K space
    from numpy.linalg import norm
    k = 5
    # Compute pairwise distances (test vs train)
    # For n_sample=2000: 600×1400 × K distances — manageable
    knn_correct = 0
    for i in range(len(X_test)):
        dists = norm(X_train - X_test[i], axis=1)
        nn_idx = np.argpartition(dists, k)[:k]
        nn_labels = y_train[nn_idx]
        pred = 1 if nn_labels.sum() > k / 2 else 0
        if pred == y_test[i]:
            knn_correct += 1
    knn_acc = knn_correct / len(X_test)

    gap = knn_acc - linear_acc
    if gap > 0.1:
        verdict = "nonlinear"
    elif gap > 0.03:
        verdict = "slightly_nonlinear"
    else:
        verdict = "linear"

    return {"linear_acc": linear_acc, "knn_acc": knn_acc,
            "gap": gap, "verdict": verdict, "n_sample": n_sample, "K": K}
```

Add GET endpoint in `do_GET()`:

```python
        elif path.startswith('/api/direction-nonlinear/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                a = int(qs.get('a', ['0'])[0])
                b = int(qs.get('b', ['0'])[0])
                layer = int(qs.get('layer', ['0'])[0])
                n_sample = int(qs.get('n_sample', ['2000'])[0])
                mri_path = self._mri_path(model, mode)
                result = _direction_nonlinear(mri_path, a, b, layer, n_sample)
                self._send_json(result)
            else:
                self._send_json({"error": "Usage: /api/direction-nonlinear/<model>/<mode>?a=N&b=N&layer=L"})
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_direction.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/heinrich/companion.py tests/test_direction.py
git commit -m "feat: nonlinearity detection endpoint (k-NN vs linear probe)"
```

---

### Task 4: Wire Nonlinearity into Separation Panel

**Files:**
- Modify: `src/heinrich/companion_ui.html` (`_updateSeparation` function)

- [ ] **Step 1: Add nonlinearity fetch after direction-quality**

In `_updateSeparation`, after the direction-quality results are displayed, fire a second request for nonlinearity:

```javascript
    // After the existing direction-quality display code, add:
    // Nonlinearity test (async, updates panel when done)
    const nlDiv=document.createElement('div');
    nlDiv.style.cssText='padding:2px 4px;margin:2px 0;font-size:8px;color:#555';
    nlDiv.textContent='testing linearity...';
    list.appendChild(nlDiv);
    api(`direction-nonlinear/${model}/${mode}?a=${a}&b=${b}&layer=${cL}&n_sample=2000`).then(nl=>{
      if(_sepReqId!==reqId)return;
      if(nl.error){nlDiv.textContent='linearity: '+nl.error;return;}
      const vColor=nl.verdict==='linear'?'#4f4':nl.verdict==='nonlinear'?'#f44':'#cc4';
      nlDiv.textContent='';
      const vSpan=document.createElement('span');
      vSpan.style.cssText=`color:${vColor};font-weight:bold`;
      vSpan.textContent=nl.verdict.toUpperCase();
      nlDiv.appendChild(vSpan);
      nlDiv.appendChild(document.createTextNode(
        ` linear=${(nl.linear_acc*100).toFixed(0)}% knn=${(nl.knn_acc*100).toFixed(0)}% gap=${(nl.gap*100).toFixed(0)}%`));
    }).catch(()=>{nlDiv.textContent='linearity: failed';});
```

- [ ] **Step 2: Test manually**

Restart companion, refresh browser, pin "the"/的 — should show LINEAR (both accuracies high, small gap). Pin King/Queen — may show slightly_nonlinear or linear.

- [ ] **Step 3: Commit**

```bash
git add src/heinrich/companion_ui.html
git commit -m "feat: nonlinearity score in separation panel"
```

---

### Task 5: Causal Steer-Test Endpoint

Test whether a direction is CAUSAL — does steering along it change model output? Uses the existing backend steering protocol.

**Files:**
- Modify: `src/heinrich/companion.py` (add `_direction_steer_test` + endpoint)
- Modify: `tests/test_direction.py` (add test)

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_direction.py

def test_steer_test_returns_structure():
    """Steer test should return clean/steered outputs and a change metric."""
    # This test mocks the backend since we can't load a real model in unit tests
    from unittest.mock import MagicMock, patch
    with tempfile.TemporaryDirectory() as tmpdir:
        scores = _make_fake_scores(100, 50, tmpdir)

        mock_backend = MagicMock()
        mock_backend.generate.side_effect = [
            "The king was powerful",   # clean generation
            "The queen was beautiful",  # steered generation
        ]

        with patch('heinrich.companion._get_steer_backend', return_value=mock_backend):
            from heinrich.companion import _direction_steer_test
            result = _direction_steer_test(
                tmpdir, a=0, b=1, layer=0,
                prompt="The ruler was", alpha=2.0, max_tokens=20,
                model_id="test-model")
            assert "clean" in result
            assert "steered" in result
            assert "changed" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_direction.py::test_steer_test_returns_structure -v`
Expected: FAIL with "cannot import name '_direction_steer_test'"

- [ ] **Step 3: Implement the steer-test function**

Add to `src/heinrich/companion.py`:

```python
_steer_backend_cache = {}

def _get_steer_backend(model_id: str):
    """Get or create a model backend for steering tests."""
    if model_id not in _steer_backend_cache:
        from .cartography.runtime import get_backend
        _steer_backend_cache[model_id] = get_backend(model_id)
    return _steer_backend_cache[model_id]


def _direction_steer_test(mri_path: str, a: int, b: int, layer: int,
                          prompt: str, alpha: float = 2.0,
                          max_tokens: int = 30,
                          model_id: str = "") -> dict:
    """Test if steering along the A→B direction changes model output.

    Generates text with and without steering, compares outputs.
    If output changes meaningfully, the direction is causal.
    """
    decomp = Path(mri_path) / "decomp"
    score_path = decomp / f"L{layer:02d}_scores.npy"
    if not score_path.exists():
        return {"error": f"No scores at L{layer}"}

    # Load direction from full-K components
    comp_path = decomp / f"L{layer:02d}_components.npy"
    if not comp_path.exists():
        return {"error": f"No components at L{layer} — needed to map PC direction to hidden space"}

    scores = np.load(str(score_path), mmap_mode="r")
    components = np.load(str(comp_path))  # [K, hidden_dim]
    N, K = scores.shape

    # Direction in PC space
    diff = scores[a].astype(np.float32) - scores[b].astype(np.float32)
    mag = float(np.linalg.norm(diff))
    dir_pc = diff / (mag + 1e-8)

    # Map to hidden space: direction_hidden = components.T @ dir_pc
    direction = (components.T @ dir_pc[:components.shape[0]]).astype(np.float32)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # Get backend
    if not model_id:
        meta = json.loads((Path(mri_path) / "metadata.json").read_text())
        model_id = meta["model"]["name"]

    try:
        backend = _get_steer_backend(model_id)
    except Exception as e:
        return {"error": f"Cannot load model {model_id}: {e}"}

    # Generate clean
    clean = backend.generate(prompt, max_tokens=max_tokens)

    # Generate steered (A→B direction, positive alpha)
    steer_dirs = {layer: (direction, alpha)}
    steered_pos = backend.generate(prompt, steer_dirs=steer_dirs, max_tokens=max_tokens)

    # Generate steered (B→A direction, negative alpha)
    steer_dirs_neg = {layer: (direction, -alpha)}
    steered_neg = backend.generate(prompt, steer_dirs=steer_dirs_neg, max_tokens=max_tokens)

    # Simple change metric: character-level edit distance ratio
    def _change_ratio(a_text, b_text):
        if not a_text and not b_text:
            return 0.0
        common = sum(c1 == c2 for c1, c2 in zip(a_text, b_text))
        return 1.0 - common / max(len(a_text), len(b_text), 1)

    change_pos = _change_ratio(clean, steered_pos)
    change_neg = _change_ratio(clean, steered_neg)

    return {
        "prompt": prompt, "layer": layer, "alpha": alpha,
        "clean": clean,
        "steered_pos": steered_pos,
        "steered_neg": steered_neg,
        "change_pos": change_pos,
        "change_neg": change_neg,
        "changed": change_pos > 0.2 or change_neg > 0.2,
        "causal": change_pos > 0.2 and change_neg > 0.2,
    }
```

Add POST endpoint in `do_POST()`:

```python
        elif path == '/api/direction-steer':
            model_name = args.get("model", "")
            mode_name = args.get("mode", "")
            a = int(args.get("a", 0))
            b = int(args.get("b", 0))
            layer = int(args.get("layer", 0))
            prompt = args.get("prompt", "Once upon a time")
            alpha = float(args.get("alpha", 2.0))
            max_tokens = int(args.get("max_tokens", 30))
            if not model_name:
                self._send_json({"error": "model required"})
                return
            mri_path = self._mri_path(model_name, mode_name or "raw")
            model_meta = json.loads((Path(mri_path) / "metadata.json").read_text())
            model_id = model_meta["model"]["name"]
            result = _direction_steer_test(
                mri_path, a, b, layer, prompt, alpha, max_tokens, model_id)
            self._send_json(result)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_direction.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/heinrich/companion.py tests/test_direction.py
git commit -m "feat: causal steer-test endpoint"
```

---

### Task 6: Steer-Test UI in Separation Panel

**Files:**
- Modify: `src/heinrich/companion_ui.html` (add steer button + results display)

- [ ] **Step 1: Add steer test button after nonlinearity display**

```javascript
    // After the nonlinearity section in _updateSeparation, add:
    const steerBtn=document.createElement('button');
    steerBtn.style.cssText='margin:4px 0;padding:2px 6px;font-size:7px;background:#1a1a2a;border:1px solid #333;color:#88f;border-radius:2px;cursor:pointer';
    steerBtn.textContent='Test causality (needs model)';
    const steerResult=document.createElement('div');
    steerResult.style.cssText='font-size:7px;color:#666;margin:2px 0;white-space:pre-wrap';
    steerBtn.addEventListener('click',async()=>{
      steerBtn.disabled=true;steerBtn.textContent='steering...';
      try{
        const sr=await fetch('/api/direction-steer',{method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({model,mode,a,b,layer:cL,prompt:'Once upon a time',alpha:2.0,max_tokens:30})
        }).then(r=>r.json());
        if(sr.error){steerResult.textContent='error: '+sr.error;return;}
        const causalColor=sr.causal?'#4f4':sr.changed?'#cc4':'#c44';
        const causalLabel=sr.causal?'CAUSAL':sr.changed?'PARTIAL':'NOT CAUSAL';
        steerResult.textContent='';
        const cs=document.createElement('span');
        cs.style.cssText=`color:${causalColor};font-weight:bold`;
        cs.textContent=causalLabel;
        steerResult.appendChild(cs);
        steerResult.appendChild(document.createTextNode(
          `\nclean:  ${sr.clean.substring(0,80)}\n+steer: ${sr.steered_pos.substring(0,80)}\n-steer: ${sr.steered_neg.substring(0,80)}`));
      }catch(e){steerResult.textContent='error: '+e.message;}
      finally{steerBtn.disabled=false;steerBtn.textContent='Test causality (needs model)';}
    });
    list.appendChild(steerBtn);
    list.appendChild(steerResult);
```

- [ ] **Step 2: Test manually**

Pin two tokens. Click "Test causality" button. Wait ~10 seconds (model load + 3 generations). Should show CAUSAL/NOT CAUSAL with clean vs steered text.

- [ ] **Step 3: Commit**

```bash
git add src/heinrich/companion_ui.html
git commit -m "feat: causal steer-test button in separation panel"
```

---

### Task 7: Full-K Direction Depth Profile and Superposition

Replace the 50-PC depth profile (rv0) and superposition (rv2) with full-K server data.

**Files:**
- Modify: `src/heinrich/companion.py` (add `_direction_depth` endpoint)
- Modify: `src/heinrich/companion_ui.html` (modify `_buildRV0Variance`, `_buildRV2Attention`)

- [ ] **Step 1: Add server endpoint for depth profile**

```python
def _direction_depth(mri_path: str, a: int, b: int) -> dict:
    """Compute full-K direction strength at every layer.

    Returns per-layer: magnitude, concentration (pcs_50), token A and B
    projections, and population percentiles (10th, 50th, 90th).
    """
    decomp = Path(mri_path) / "decomp"
    meta = json.loads((decomp / "meta.json").read_text())
    n_layers = len(meta["layers"])

    layers = []
    for li in range(n_layers):
        score_path = decomp / f"L{li:02d}_scores.npy"
        if not score_path.exists():
            continue
        scores = np.load(str(score_path), mmap_mode="r")
        N, K = scores.shape
        if a >= N or b >= N:
            continue

        diff = scores[a].astype(np.float32) - scores[b].astype(np.float32)
        mag = float(np.linalg.norm(diff))
        direction = diff / (mag + 1e-8)

        # Project A, B, and sample of population
        pa = float(scores[a].astype(np.float32) @ direction)
        pb = float(scores[b].astype(np.float32) @ direction)

        # Sample percentiles
        step = max(1, N // 500)
        sample_proj = scores[::step].astype(np.float32) @ direction
        p10, p50, p90 = float(np.percentile(sample_proj, 10)), \
                         float(np.percentile(sample_proj, 50)), \
                         float(np.percentile(sample_proj, 90))

        # Concentration
        diff2 = diff ** 2
        total_d2 = float(diff2.sum())
        order = np.argsort(diff2)[::-1]
        cumul = np.cumsum(diff2[order]) / (total_d2 + 1e-8)
        pcs_50 = int(np.searchsorted(cumul, 0.5)) + 1

        # Top PC at this layer
        top_pc = int(order[0])
        top_share = float(diff2[order[0]] / (total_d2 + 1e-8))

        layers.append({
            "layer": li, "magnitude": mag,
            "proj_a": pa, "proj_b": pb,
            "p10": p10, "p50": p50, "p90": p90,
            "pcs_50": pcs_50, "top_pc": top_pc, "top_share": top_share,
        })

    return {"layers": layers, "n_layers": n_layers}
```

Add endpoint:
```python
        elif path.startswith('/api/direction-depth/'):
            parts = path.split('/')
            if len(parts) >= 5:
                model, mode = parts[3], parts[4]
                a = int(qs.get('a', ['0'])[0])
                b = int(qs.get('b', ['0'])[0])
                mri_path = self._mri_path(model, mode)
                result = _direction_depth(mri_path, a, b)
                self._send_json(result)
            else:
                self._send_json({"error": "Usage"})
```

- [ ] **Step 2: Modify rv0 to use server data**

Replace `_buildRV0Variance` to fetch `/api/direction-depth/` and plot the full-K magnitude, projections, and percentiles instead of computing from the 50-PC blob. The existing drawing code (lines, cursor, percentile bands) stays the same — just the data source changes.

- [ ] **Step 3: Modify rv2 to use server data**

The `_direction_depth` response includes `pcs_50` and `top_pc`/`top_share` per layer. Use these to draw the superposition tubes with accurate full-K concentration instead of 50-PC approximation.

- [ ] **Step 4: Test and commit**

Run: `pytest tests/test_direction.py -v && pytest tests/test_mcp.py -v`

```bash
git add src/heinrich/companion.py src/heinrich/companion_ui.html tests/test_direction.py
git commit -m "feat: full-K direction depth profile and superposition viewports"
```

---

### Task 8: MCP Tool for Full Direction Analysis

Expose the full direction analysis pipeline as an MCP tool so Claude can query it programmatically.

**Files:**
- Modify: `src/heinrich/mcp.py` (add tool definition + dispatch)

- [ ] **Step 1: Add tool definition to TOOLS dict**

```python
    "heinrich_companion_direction": {
        "description": "Full-dimensional direction analysis between two tokens. Returns magnitude, concentration, bimodality, nonlinearity, explained variance, and top tokens per side — all computed in the model's full hidden dimension. The companion viewer must be running.",
        "parameters": {
            "token_a": {"type": "integer", "description": "Token index A", "required": True},
            "token_b": {"type": "integer", "description": "Token index B", "required": True},
            "layer": {"type": "integer", "description": "Layer to analyze (default: auto-select peak)"},
            "test_nonlinear": {"type": "boolean", "description": "Also run k-NN vs linear probe test (slower)"},
        },
    },
```

- [ ] **Step 2: Add dispatch + implementation**

```python
        if name == "heinrich_companion_direction":
            return self._do_companion_direction(arguments)
```

```python
    def _do_companion_direction(self, args: dict[str, Any]) -> dict[str, Any]:
        """Full direction analysis via companion server."""
        import urllib.request
        port = args.get("port", 8377)
        a = args["token_a"]
        b = args["token_b"]
        layer = args.get("layer", 0)

        # Get direction quality
        url = f"http://localhost:{port}/api/direction-quality/{{}}/{{}}" \
              f"?a={a}&b={b}&layer={layer}"
        # We need model/mode — fetch from companion
        try:
            models_resp = urllib.request.urlopen(
                f"http://localhost:{port}/api/models", timeout=5)
            # Use first transformer model
            # Actually, let user specify or use whatever's loaded
        except Exception as e:
            return {"error": f"Companion not reachable: {e}"}

        # Simpler: just pass through to companion
        payload = json.dumps({"token_a": a, "token_b": b,
                              "layer": layer}).encode()
        req = urllib.request.Request(
            f"http://localhost:{port}/api/direction-quality/_current/_current"
            f"?a={a}&b={b}&layer={layer}",
            method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
        except Exception as e:
            return {"error": f"Direction analysis failed: {e}"}

        if args.get("test_nonlinear"):
            nl_req = urllib.request.Request(
                f"http://localhost:{port}/api/direction-nonlinear/_current/_current"
                f"?a={a}&b={b}&layer={layer}",
                method="GET")
            try:
                with urllib.request.urlopen(nl_req, timeout=60) as resp:
                    nl = json.loads(resp.read())
                result["nonlinearity"] = nl
            except Exception:
                result["nonlinearity"] = {"error": "failed"}

        result["hint"] = "Use heinrich_companion_show to navigate the viewer to this direction."
        return result
```

**Note:** The `_current` model/mode placeholder should be replaced with a mechanism to query the currently-loaded model from the companion. A simple approach: add `/api/current` endpoint that returns `{model, mode}`. Alternative: require model/mode as tool parameters.

- [ ] **Step 3: Test and commit**

Run: `pytest tests/test_mcp.py -v`

```bash
git add src/heinrich/mcp.py
git commit -m "feat: heinrich_companion_direction MCP tool"
```

---

## Execution Order

Tasks 1-4 are independent of Tasks 5-6. Task 7 depends on Task 1. Task 8 depends on Tasks 1 and 3.

**Recommended order:** 1 → 2 → 3 → 4 → 7 → 5 → 6 → 8

This delivers full-K coloring (Task 2) and nonlinearity (Task 4) early — the highest-impact features. Causal testing (Tasks 5-6) requires model loading and is slower to iterate on. The full-K depth profile (Task 7) completes the measurement stack. The MCP tool (Task 8) wires everything together.

## What This Doesn't Cover (Future Plans)

- **Multi-token concepts**: Requires contextual MRI capture from varied prompts (different capture mode)
- **Circuit tracing**: QK/OV composition graph from stored weight matrices (separate plan)
- **Scale testing**: MRI capture of 7B+ models (storage + capture time, no new tools)
- **Training dynamics**: Checkpoint-series MRI comparison (separate plan)
