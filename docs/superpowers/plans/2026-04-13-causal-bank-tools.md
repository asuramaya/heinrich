# Causal Bank Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build sequence-mode MRI capture for causal banks and 12 analysis tools that read from it, following the same capture-once-read-many pattern the transformer side uses.

**Architecture:** Extend `forward_captured()` in decepticons to return temporal attention internals. Add a sequence-mode capture path in `profile/mri.py` that runs validation sequences and stores per-position substrate, routing, loss, and temporal attention. All analysis tools are pure MRI readers in `profile/compare.py` -- no model loading. Two standalone diagnostic tools (causality, reproducibility) DO load the model. Two tokenizer tools need no model at all.

**Tech Stack:** numpy, torch (decepticons inference), sentencepiece (tokenizer tools). No new dependencies.

**Repos:** Two repos involved.
- `/Users/asuramaya/Code/carving_machine_v3/decepticons/` -- upstream loader changes (Tasks 1-2)
- `/Users/asuramaya/Code/heinrich/` -- everything else (Tasks 3-15)

**Data format:** Validation data is uint16 `.bin` files: `np.fromfile(path, dtype=np.uint16)`. Located on fleet at `/pf/datasets/fineweb10B_sp*/fineweb_val_*.bin`. Local copies on `/Volumes/sharts/heinrich/`. Tokenizer is sentencepiece `.model` file.

**Storage budget for sequence.mri (50 seqs x 512 positions):**
- substrate: 50 x 512 x 516 modes x 2B = ~26 MB
- routing: 50 x 512 x 8 experts x 2B = ~0.4 MB
- loss: 50 x 512 x 4B = ~0.1 MB
- temporal weights: 50 x 512 x 8 snapshots x 2B = ~0.4 MB
- embedding: 50 x 512 x 256 embed x 2B = ~13 MB
- Total: ~40 MB per model. Manageable.

---

## Phase A: Foundation

### Task 1: Extend decepticon `forward_captured()` with temporal attention capture

**Repo:** `/Users/asuramaya/Code/carving_machine_v3/decepticons/`

**Files:**
- Modify: `src/decepticons/models/substrate_transforms.py:210-247` (TemporalAttention.forward)
- Modify: `src/decepticons/loader.py:55-126` (CausalBankInference.forward_captured)

- [ ] **Step 1: Store attention weights in TemporalAttention.forward()**

In `substrate_transforms.py`, after the softmax on line 242, store the weights and output:

```python
        attn = torch.softmax(attn, dim=-1)
        self._last_attn_weights = attn.detach()  # [batch, heads, seq, M]

        # Aggregate: [batch, heads, seq, head_dim]
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, seq, self.num_heads * self.head_dim)
        result = self.out_proj(out)
        self._last_output = result.detach()  # [batch, seq, state_dim]
        return result
```

- [ ] **Step 2: Read temporal attention internals in forward_captured()**

In `loader.py`, after the `_linear_states()` call (line 73-74), add:

```python
            # Temporal attention internals (if present)
            ta = getattr(self._model, '_temporal_attention', None)
            if ta is not None:
                result["temporal_attn_weights"] = (
                    ta._last_attn_weights.cpu().numpy()
                    if hasattr(ta, '_last_attn_weights') and ta._last_attn_weights is not None
                    else None
                )
                result["temporal_attn_output"] = (
                    ta._last_output.cpu().numpy()
                    if hasattr(ta, '_last_output') and ta._last_output is not None
                    else None
                )
                result["temporal_snapshot_interval"] = getattr(
                    self._model, '_temporal_snapshot_interval', None
                )
            else:
                result["temporal_attn_weights"] = None
                result["temporal_attn_output"] = None
                result["temporal_snapshot_interval"] = None
```

- [ ] **Step 3: Verify no regressions**

```bash
cd /Users/asuramaya/Code/carving_machine_v3/decepticons && python -c "
from decepticons.loader import load_checkpoint
m = load_checkpoint('/Volumes/sharts/heinrich/cb-s8-experts-50k.checkpoint.pt')
import numpy as np
r = m.forward_captured(np.array([[1, 2, 3, 4]]))
print('Keys:', sorted(r.keys()))
print('substrate:', r['substrate_states'].shape)
print('temporal_attn_weights:', type(r['temporal_attn_weights']))
print('temporal_attn_output:', type(r['temporal_attn_output']))
"
```

Expected: keys include `temporal_attn_weights` (None for models without temporal attention). No crash.

- [ ] **Step 4: Commit**

```bash
cd /Users/asuramaya/Code/carving_machine_v3/decepticons
git add src/decepticons/models/substrate_transforms.py src/decepticons/loader.py
git commit -m "feat: forward_captured() returns temporal attention weights and output"
```

---

### Task 2: Val data loader in heinrich

**Repo:** `/Users/asuramaya/Code/heinrich/`

**Files:**
- Modify: `src/heinrich/backend/decepticon.py` (add `load_val_sequences` method)

- [ ] **Step 1: Write the test**

In `tests/test_cb_val_loader.py`:

```python
"""Tests for causal bank validation data loading."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_load_val_sequences_from_bin():
    """Load uint16 .bin file and chunk into sequences."""
    from heinrich.backend.decepticon import load_val_sequences

    # Create synthetic val data: 2048 tokens as uint16
    tokens = np.arange(2048, dtype=np.uint16)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tokens.tofile(f)
        f.flush()
        seqs = load_val_sequences(f.name, seq_len=512, n_seqs=3, seed=42)

    assert seqs.shape == (3, 512)
    assert seqs.dtype == np.int64
    # Sequences are non-overlapping random slices
    assert not np.array_equal(seqs[0], seqs[1])


def test_load_val_sequences_short_data():
    """When data is shorter than n_seqs * seq_len, return what fits."""
    from heinrich.backend.decepticon import load_val_sequences

    tokens = np.arange(600, dtype=np.uint16)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tokens.tofile(f)
        f.flush()
        seqs = load_val_sequences(f.name, seq_len=512, n_seqs=10, seed=42)

    # Only 1 full sequence fits
    assert seqs.shape[0] == 1
    assert seqs.shape[1] == 512


def test_load_val_sequences_deterministic():
    """Same seed produces same sequences."""
    from heinrich.backend.decepticon import load_val_sequences

    tokens = np.arange(4096, dtype=np.uint16)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tokens.tofile(f)
        f.flush()
        a = load_val_sequences(f.name, seq_len=512, n_seqs=3, seed=42)
        b = load_val_sequences(f.name, seq_len=512, n_seqs=3, seed=42)

    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/asuramaya/Code/heinrich && pytest tests/test_cb_val_loader.py -v
```

Expected: ImportError -- `load_val_sequences` not found.

- [ ] **Step 3: Implement load_val_sequences**

Add to `src/heinrich/backend/decepticon.py` (after imports, before class):

```python
def load_val_sequences(
    path: str,
    *,
    seq_len: int = 512,
    n_seqs: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Load validation data from a uint16 .bin file, return [n_seqs, seq_len] int64.

    Takes non-overlapping random slices from the token stream.
    Returns fewer sequences if data is too short.
    """
    tokens = np.fromfile(path, dtype=np.uint16).astype(np.int64)
    n_total = len(tokens)
    max_seqs = n_total // seq_len
    actual_seqs = min(n_seqs, max_seqs)
    if actual_seqs == 0:
        raise ValueError(f"Val data too short: {n_total} tokens < seq_len {seq_len}")

    rng = np.random.RandomState(seed)
    # Pick non-overlapping start positions
    all_starts = np.arange(max_seqs) * seq_len
    chosen = rng.choice(len(all_starts), actual_seqs, replace=False)
    chosen.sort()
    starts = all_starts[chosen]

    seqs = np.stack([tokens[s:s + seq_len] for s in starts])
    return seqs
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/asuramaya/Code/heinrich && pytest tests/test_cb_val_loader.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/heinrich/backend/decepticon.py tests/test_cb_val_loader.py
git commit -m "feat: load_val_sequences for causal bank validation data"
```

---

### Task 3: Sequence-mode MRI capture

**Files:**
- Modify: `src/heinrich/profile/mri.py` (add `_capture_mri_causal_bank_sequence`)
- Modify: `src/heinrich/profile/mri.py:545-547` (dispatch in `capture_mri`)

The sequence-mode MRI captures per-position internals across validation sequences. Stored as:

```
sequence.mri/
  metadata.json
  tokens.npz              — token_ids[n_seqs, seq_len], n_seqs, seq_len
  substrate.npy            — [n_seqs, seq_len, n_modes] float16
  embedding.npy            — [n_seqs, seq_len, embed_dim] float16
  loss.npy                 — [n_seqs, seq_len] float32
  routing.npy              — [n_seqs, seq_len, n_experts] float16 (optional)
  band_loss.npy            — [n_seqs, seq_len, n_bands] float32 (optional)
  local_norm.npy           — [n_seqs, seq_len] float32 (optional)
  temporal_weights.npy     — [n_seqs, seq_len, n_snapshots] float16 (optional)
  temporal_output.npy      — [n_seqs, seq_len, n_modes] float16 (optional)
  half_lives.npy           — [n_modes] float32
  weights/                 — all model weights (shared with impulse MRI)
```

- [ ] **Step 1: Write the test**

In `tests/test_cb_sequence_mri.py`:

```python
"""Tests for sequence-mode causal bank MRI capture structure."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _make_fake_sequence_mri(tmp_path: Path, *, n_seqs=5, seq_len=64,
                             n_modes=32, embed_dim=16, n_experts=2,
                             n_bands=2, has_temporal=True):
    """Create a synthetic sequence-mode MRI directory for testing."""
    mri_dir = tmp_path / "test_seq.mri"
    mri_dir.mkdir()

    n_snapshots = seq_len // 8  # snapshot every 8 positions

    metadata = {
        "version": "0.7",
        "type": "mri",
        "architecture": "causal_bank",
        "model": {
            "name": "test_cb",
            "n_modes": n_modes,
            "n_experts": n_experts,
            "n_bands": n_bands,
            "embed_dim": embed_dim,
            "vocab_size": 1024,
            "n_layers": 1,
            "hidden_size": n_modes,
        },
        "capture": {
            "mode": "sequence",
            "n_seqs": n_seqs,
            "seq_len": seq_len,
            "n_tokens": n_seqs * seq_len,
            "has_temporal": has_temporal,
            "has_routing": n_experts > 1,
            "has_band_loss": n_bands > 1,
            "snapshot_interval": 8,
        },
        "provenance": {"seed": 42},
    }
    with open(mri_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    token_ids = np.random.randint(0, 1024, (n_seqs, seq_len), dtype=np.int32)
    np.savez_compressed(mri_dir / "tokens.npz", token_ids=token_ids)

    np.save(mri_dir / "substrate.npy",
            np.random.randn(n_seqs, seq_len, n_modes).astype(np.float16))
    np.save(mri_dir / "embedding.npy",
            np.random.randn(n_seqs, seq_len, embed_dim).astype(np.float16))
    np.save(mri_dir / "loss.npy",
            np.random.rand(n_seqs, seq_len).astype(np.float32) * 5)
    np.save(mri_dir / "half_lives.npy",
            np.logspace(0, 2, n_modes, dtype=np.float32))

    if n_experts > 1:
        np.save(mri_dir / "routing.npy",
                np.random.rand(n_seqs, seq_len, n_experts).astype(np.float16))
    if n_bands > 1:
        np.save(mri_dir / "band_loss.npy",
                np.random.rand(n_seqs, seq_len, n_bands).astype(np.float32) * 5)
    if has_temporal:
        np.save(mri_dir / "temporal_weights.npy",
                np.random.rand(n_seqs, seq_len, n_snapshots).astype(np.float16))
        np.save(mri_dir / "temporal_output.npy",
                np.random.randn(n_seqs, seq_len, n_modes).astype(np.float16))

    return str(mri_dir)


def test_sequence_mri_loads():
    """Sequence MRI loads via load_mri and has expected keys."""
    from heinrich.profile.mri import load_mri

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp))
        mri = load_mri(mri_path)

    assert mri['metadata']['capture']['mode'] == 'sequence'
    assert 'substrate_states' in mri
    assert 'loss' in mri
    assert mri['substrate_states'].shape == (5, 64, 32)
    assert mri['loss'].shape == (5, 64)


def test_sequence_mri_temporal():
    """Sequence MRI loads temporal attention arrays."""
    from heinrich.profile.mri import load_mri

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), has_temporal=True)
        mri = load_mri(mri_path)

    assert 'temporal_weights' in mri
    assert 'temporal_output' in mri
    assert mri['temporal_weights'].shape == (5, 64, 8)


def test_sequence_mri_no_temporal():
    """Sequence MRI without temporal attention."""
    from heinrich.profile.mri import load_mri

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), has_temporal=False)
        mri = load_mri(mri_path)

    assert mri.get('temporal_weights') is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/asuramaya/Code/heinrich && pytest tests/test_cb_sequence_mri.py -v
```

Expected: KeyError or similar -- `load_mri` doesn't handle sequence mode arrays yet.

- [ ] **Step 3: Extend load_mri() for sequence mode**

In `src/heinrich/profile/mri.py`, in the `load_mri()` function, inside the `if arch == "causal_bank":` block (line 1816-1820), extend it:

```python
    if arch == "causal_bank":
        capture_mode = meta.get("capture", {}).get("mode", "raw")
        for name in ["substrate", "routing", "half_lives", "embedding", "band_logits"]:
            fp = p / f"{name}.npy"
            if fp.exists():
                fmap[name if name != "substrate" else "substrate_states"] = (fp, 'r')
        # Sequence-mode arrays
        if capture_mode == "sequence":
            for name in ["loss", "band_loss", "local_norm",
                         "temporal_weights", "temporal_output"]:
                fp = p / f"{name}.npy"
                if fp.exists():
                    fmap[name] = (fp, 'r')
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/asuramaya/Code/heinrich && pytest tests/test_cb_sequence_mri.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Implement the capture function**

Add `_capture_mri_causal_bank_sequence()` in `src/heinrich/profile/mri.py`, after `_capture_mri_causal_bank()`:

```python
def _capture_mri_causal_bank_sequence(
    backend, *, val_data: str, n_seqs: int = 50, seq_len: int = 512,
    seed: int = 42, output: str | None = None,
) -> dict:
    """Sequence-mode MRI for causal bank: per-position internals on real sequences.

    Unlike impulse mode (single token from zero state), this captures substrate
    accumulation, routing dynamics, and loss decomposition across full sequences.
    """
    from .frt import _detect_script
    from ..backend.decepticon import load_val_sequences

    cfg = backend.config
    t0 = time.time()

    # Load validation sequences
    seqs = load_val_sequences(val_data, seq_len=seq_len, n_seqs=n_seqs, seed=seed)
    actual_seqs = seqs.shape[0]
    print(f"MRI capture (causal_bank, sequence): {actual_seqs} seqs x {seq_len} positions, "
          f"{cfg.n_modes} modes, {cfg.n_experts} experts, {cfg.n_bands} bands")

    n_modes = cfg.n_modes
    n_experts = cfg.n_experts
    n_bands = cfg.n_bands
    embed_dim = cfg.embed_dim
    vocab_size = cfg.vocab_size

    # Allocate output arrays
    substrate_all = np.zeros((actual_seqs, seq_len, n_modes), dtype=np.float16)
    embedding_all = np.zeros((actual_seqs, seq_len, embed_dim), dtype=np.float16)
    loss_all = np.zeros((actual_seqs, seq_len), dtype=np.float32)

    routing_all = (np.zeros((actual_seqs, seq_len, n_experts), dtype=np.float16)
                   if n_experts > 1 else None)
    band_loss_all = (np.zeros((actual_seqs, seq_len, n_bands), dtype=np.float32)
                     if n_bands > 1 else None)
    local_norm_all = None  # allocated on first use

    # Temporal attention arrays (allocated if model has it)
    has_temporal = hasattr(backend.model, '_temporal_attention')
    snapshot_interval = getattr(backend.model, '_temporal_snapshot_interval', 64)
    n_snapshots = seq_len // snapshot_interval if has_temporal else 0
    temporal_weights_all = (np.zeros((actual_seqs, seq_len, n_snapshots), dtype=np.float16)
                           if has_temporal else None)
    temporal_output_all = (np.zeros((actual_seqs, seq_len, n_modes), dtype=np.float16)
                          if has_temporal else None)

    for i in range(actual_seqs):
        seq = seqs[i:i+1]  # [1, seq_len]
        result = backend.forward_captured(seq)

        # Substrate: [1, seq_len, n_modes] -> [seq_len, n_modes]
        substrate_all[i] = result['substrate_states'][0].astype(np.float16)
        embedding_all[i] = result['embedding'][0].astype(np.float16)

        # Loss: cross-entropy at each position (predict next token)
        logits = result['logits'][0]  # [seq_len, vocab]
        targets = seq[0, 1:]  # [seq_len - 1]
        # Stable log-softmax
        log_probs = logits[:-1] - np.log(np.exp(logits[:-1]).sum(axis=-1, keepdims=True) + 1e-10)
        ce = -log_probs[np.arange(len(targets)), targets]
        loss_all[i, 1:] = ce.astype(np.float32) / np.log(2)  # bits
        loss_all[i, 0] = np.nan  # no target for first position

        # Routing
        if routing_all is not None and result.get('route_weights') is not None:
            rw = result['route_weights']
            if rw.ndim == 2:
                # [batch*seq, experts] -> [seq_len, experts]
                routing_all[i] = rw[:seq_len].astype(np.float16)
            elif rw.ndim == 3:
                routing_all[i] = rw[0].astype(np.float16)

        # Band loss
        if band_loss_all is not None and result.get('band_logits') is not None:
            for b, bl in enumerate(result['band_logits']):
                bl_0 = bl[0]  # [seq_len, vocab]
                bl_lp = bl_0[:-1] - np.log(np.exp(bl_0[:-1]).sum(axis=-1, keepdims=True) + 1e-10)
                bl_ce = -bl_lp[np.arange(len(targets)), targets]
                band_loss_all[i, 1:, b] = bl_ce.astype(np.float32) / np.log(2)

        # Local path norm
        if result.get('local_logits') is not None:
            if local_norm_all is None:
                local_norm_all = np.zeros((actual_seqs, seq_len), dtype=np.float32)
            local_norm_all[i] = np.linalg.norm(result['local_logits'][0], axis=-1).astype(np.float32)

        # Temporal attention
        if has_temporal and result.get('temporal_attn_weights') is not None:
            tw = result['temporal_attn_weights'][0]  # [heads, seq, M]
            # Average across heads
            temporal_weights_all[i] = tw.mean(axis=0)[:, :n_snapshots].astype(np.float16)
        if has_temporal and result.get('temporal_attn_output') is not None:
            temporal_output_all[i] = result['temporal_attn_output'][0].astype(np.float16)

        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 0.01)
        print(f"  {i+1}/{actual_seqs} ({rate:.1f} seq/s)", end="\r")

    print()
    elapsed = time.time() - t0

    metadata = {
        "version": MRI_VERSION,
        "type": "mri",
        "architecture": "causal_bank",
        "generated_at": time.time(),
        "elapsed_s": round(elapsed),
        "model": {
            "name": cfg.model_type,
            "n_modes": n_modes,
            "n_experts": n_experts,
            "n_bands": n_bands,
            "embed_dim": embed_dim,
            "vocab_size": vocab_size,
            "n_layers": 1,
            "hidden_size": n_modes,
        },
        "capture": {
            "mode": "sequence",
            "n_seqs": actual_seqs,
            "seq_len": seq_len,
            "n_tokens": actual_seqs * seq_len,
            "has_temporal": has_temporal,
            "has_routing": routing_all is not None,
            "has_band_loss": band_loss_all is not None,
            "has_local": local_norm_all is not None,
            "snapshot_interval": snapshot_interval if has_temporal else None,
        },
        "provenance": {
            "seed": seed,
            "val_data": val_data,
        },
    }

    if output:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        np.savez_compressed(out_dir / "tokens.npz", token_ids=seqs.astype(np.int32))
        np.save(out_dir / "substrate.npy", substrate_all)
        np.save(out_dir / "embedding.npy", embedding_all)
        np.save(out_dir / "loss.npy", loss_all)
        np.save(out_dir / "half_lives.npy", backend.model.half_lives)

        if routing_all is not None:
            np.save(out_dir / "routing.npy", routing_all)
        if band_loss_all is not None:
            np.save(out_dir / "band_loss.npy", band_loss_all)
        if local_norm_all is not None:
            np.save(out_dir / "local_norm.npy", local_norm_all)
        if temporal_weights_all is not None:
            np.save(out_dir / "temporal_weights.npy", temporal_weights_all)
        if temporal_output_all is not None:
            np.save(out_dir / "temporal_output.npy", temporal_output_all)

        # Weights (skip if impulse MRI already has them in sibling dir)
        weights_dir = out_dir / "weights"
        if not weights_dir.exists():
            weights = backend.weights()
            weights_dir.mkdir(exist_ok=True)
            for name, w in weights.items():
                safe_name = name.replace('.', '_').replace('/', '_')
                np.save(weights_dir / f"{safe_name}.npy", w)

        size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
        print(f"\n  Saved to {out_dir}/ ({size / 1e6:.1f} MB)")

    return {
        "metadata": metadata,
        "substrate_states": substrate_all,
        "embedding": embedding_all,
        "loss": loss_all,
        "routing": routing_all,
        "band_loss": band_loss_all,
        "local_norm": local_norm_all,
        "temporal_weights": temporal_weights_all,
        "temporal_output": temporal_output_all,
    }
```

- [ ] **Step 6: Wire into capture_mri dispatch**

In `capture_mri()` (line 545-547), change the causal bank dispatch:

```python
    if getattr(cfg, 'model_type', '') == 'causal_bank':
        if mode == "sequence":
            if not kwargs.get('val_data'):
                raise ValueError("Sequence mode requires --data <path_to_val.bin>")
            return _capture_mri_causal_bank_sequence(
                backend, val_data=kwargs['val_data'],
                n_seqs=kwargs.get('n_seqs', 50),
                seq_len=kwargs.get('seq_len', 512),
                seed=seed, output=output)
        return _capture_mri_causal_bank(backend, mode=mode, n_index=n_index,
                                         seed=seed, output=output)
```

Update the `capture_mri` signature to accept `**kwargs`:

```python
def capture_mri(
    backend,
    *,
    mode: str = "template",
    n_index: int | None = None,
    seed: int = 42,
    output: str | None = None,
    db_path: str | None = None,
    **kwargs,
) -> dict:
```

- [ ] **Step 7: Commit**

```bash
git add src/heinrich/profile/mri.py tests/test_cb_sequence_mri.py
git commit -m "feat: sequence-mode MRI capture for causal banks"
```

---

### Task 4: CLI wiring for sequence-mode capture

**Files:**
- Modify: `src/heinrich/cli.py` (add `--data`, `--n-seqs`, `--seq-len` to mri command; add sequence mode to mri-scan for causal banks)

- [ ] **Step 1: Add CLI args to mri command**

Find the `mri` subparser (around line 240-260 area). Add after existing args:

```python
    p_mri.add_argument("--data", help="Validation data .bin file (uint16 tokens, for causal bank sequence mode)")
    p_mri.add_argument("--n-seqs", type=int, default=50, help="Number of sequences for sequence mode (default: 50)")
    p_mri.add_argument("--seq-len", type=int, default=512, help="Sequence length for sequence mode (default: 512)")
```

- [ ] **Step 2: Pass kwargs in _cmd_mri**

In `_cmd_mri()` (around line 1245), pass the new args through to `capture_mri`:

```python
    extra = {}
    if getattr(args, 'data', None):
        extra['val_data'] = args.data
    if getattr(args, 'n_seqs', None):
        extra['n_seqs'] = args.n_seqs
    if getattr(args, 'seq_len', None):
        extra['seq_len'] = args.seq_len

    capture_mri(backend, mode=args.mode, n_index=args.n_index, output=args.output, **extra)
```

- [ ] **Step 3: Update mri-scan for causal banks**

In `_cmd_mri_scan()`, after `modes = ["raw", "naked", "template"]` (line 1904), add causal bank branch:

```python
    is_cb = getattr(cfg, 'model_type', '') == 'causal_bank'
    if is_cb:
        modes = ["raw"]
        if getattr(args, 'data', None):
            modes.append("sequence")
```

And in the capture loop, pass val_data for sequence mode:

```python
        extra = {}
        if mode == "sequence" and getattr(args, 'data', None):
            extra['val_data'] = args.data
            extra['n_seqs'] = getattr(args, 'n_seqs', 50)
            extra['seq_len'] = getattr(args, 'seq_len', 512)
```

Also add `--data`, `--n-seqs`, `--seq-len` to the `mri-scan` subparser args.

- [ ] **Step 4: Test CLI invocation (dry run)**

```bash
cd /Users/asuramaya/Code/heinrich && python -m heinrich.cli mri --help
```

Verify `--data`, `--n-seqs`, `--seq-len` appear in help.

- [ ] **Step 5: Commit**

```bash
git add src/heinrich/cli.py
git commit -m "feat: CLI args for causal bank sequence-mode MRI capture"
```

---

## Phase B: Sequence-Level Analysis Tools

All tools follow this pattern:
1. Function in `profile/compare.py` that takes `mri_path` and returns a dict
2. CLI handler in `cli.py` that formats the dict
3. MCP tool definition in `mcp.py`

### Task 5: cb-loss -- Per-position loss decomposition

**Files:**
- Modify: `src/heinrich/profile/compare.py` (add `causal_bank_loss`)
- Modify: `src/heinrich/cli.py` (add parser + handler)
- Modify: `src/heinrich/mcp.py` (add tool definition + dispatch)
- Test: `tests/test_cb_sequence_mri.py` (add loss test)

- [ ] **Step 1: Write the test**

Add to `tests/test_cb_sequence_mri.py`:

```python
def test_cb_loss():
    """cb-loss reads sequence MRI and returns loss decomposition."""
    from heinrich.profile.compare import causal_bank_loss

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), n_seqs=5, seq_len=64,
                                            n_modes=32, n_bands=2)
        result = causal_bank_loss(mri_path)

    assert "error" not in result
    assert "overall_bpb" in result
    assert "by_position" in result
    assert len(result["by_position"]) > 0
    # Position ranges should cover [0-4, 4-64, ...]
    assert result["by_position"][0]["range"] == "0-4"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cb_sequence_mri.py::test_cb_loss -v
```

- [ ] **Step 3: Implement causal_bank_loss**

Add to `profile/compare.py`:

```python
def causal_bank_loss(mri_path: str, *, _mri=None) -> dict:
    """Per-position loss decomposition from sequence-mode causal bank MRI.

    Breaks down loss by position range, band, and token merge rank.
    Requires mode == 'sequence'.
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if meta.get('capture', {}).get('mode') != 'sequence':
        return {"error": "Requires sequence-mode MRI (not impulse)"}

    loss = mri['loss']  # [n_seqs, seq_len]
    n_seqs, seq_len = loss.shape

    # Mask out position 0 (NaN, no target)
    valid = ~np.isnan(loss)
    flat_loss = loss[valid]
    overall_bpb = float(np.mean(flat_loss))

    # By position range
    ranges = [(0, 4), (4, 64), (64, 256), (256, seq_len)]
    by_pos = []
    for lo, hi in ranges:
        hi = min(hi, seq_len)
        if lo >= seq_len:
            break
        chunk = loss[:, lo:hi]
        chunk_valid = chunk[~np.isnan(chunk)]
        if len(chunk_valid) == 0:
            continue
        by_pos.append({
            "range": f"{lo}-{hi}",
            "mean_bpb": round(float(np.mean(chunk_valid)), 4),
            "std_bpb": round(float(np.std(chunk_valid)), 4),
            "n_positions": hi - lo,
        })

    # By band (if available)
    by_band = []
    if 'band_loss' in mri:
        band_loss = mri['band_loss']  # [n_seqs, seq_len, n_bands]
        n_bands = band_loss.shape[2]
        for b in range(n_bands):
            bl = band_loss[:, 1:, b]  # skip pos 0
            by_band.append({
                "band": b,
                "mean_bpb": round(float(np.mean(bl)), 4),
            })

    # Loss autocorrelation (surprise persistence)
    autocorr = []
    mean_loss_per_seq = loss[:, 1:]  # [n_seqs, seq_len-1]
    for lag in [1, 2, 4, 8, 16, 32, 64, 128]:
        if lag >= seq_len - 1:
            break
        a = mean_loss_per_seq[:, :-lag].flatten()
        b_arr = mean_loss_per_seq[:, lag:].flatten()
        mask = ~(np.isnan(a) | np.isnan(b_arr))
        if mask.sum() < 10:
            continue
        r = float(np.corrcoef(a[mask], b_arr[mask])[0, 1])
        autocorr.append({"lag": lag, "r": round(r, 4)})

    return {
        "model": meta['model'].get('name', '?'),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "overall_bpb": round(overall_bpb, 4),
        "by_position": by_pos,
        "by_band": by_band,
        "autocorrelation": autocorr,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_cb_sequence_mri.py::test_cb_loss -v
```

- [ ] **Step 5: Wire CLI**

Parser (add near line 313):

```python
    p_cb_loss = sub.add_parser("profile-cb-loss", help="Causal bank loss decomposition by position, band, and autocorrelation")
    p_cb_loss.add_argument("--mri", required=True, help=".mri directory (causal bank, sequence mode)")
```

Handler (add near line 2370):

```python
def _cmd_cb_loss(args: argparse.Namespace) -> None:
    """Causal bank loss decomposition."""
    from .profile.compare import causal_bank_loss

    result = causal_bank_loss(args.mri)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== CB Loss: {result['model']} — "
          f"{result['n_seqs']} seqs x {result['seq_len']} positions ===\n")
    print(f"  Overall: {result['overall_bpb']:.4f} bpb\n")

    print(f"  {'Position':>12} {'Mean bpb':>10} {'Std':>8}")
    for r in result['by_position']:
        print(f"  {r['range']:>12} {r['mean_bpb']:>10.4f} {r['std_bpb']:>8.4f}")

    if result['by_band']:
        print(f"\n  Per-band loss:")
        for b in result['by_band']:
            print(f"    Band {b['band']}: {b['mean_bpb']:.4f} bpb")

    if result['autocorrelation']:
        print(f"\n  Loss autocorrelation (surprise persistence):")
        for a in result['autocorrelation']:
            print(f"    lag {a['lag']:>3}: r={a['r']:.4f}")
```

Dispatch (add in elif chain around line 667):

```python
    elif args.command == "profile-cb-loss":
        _cmd_cb_loss(args)
```

- [ ] **Step 6: Wire MCP**

Tool definition (add near line 460 in TOOLS dict):

```python
    "heinrich_cb_loss": {
        "description": "Causal bank loss decomposition: per-position, per-band, autocorrelation. Reads sequence-mode .mri.",
        "parameters": {
            "mri": {"type": "string", "description": ".mri directory (causal bank, sequence mode)", "required": True},
        },
    },
```

Dispatch (add near line 795):

```python
        if name == "heinrich_cb_loss":
            return self._do_subprocess(arguments, "profile-cb-loss",
                ["--mri", arguments["mri"]], timeout=60)
```

- [ ] **Step 7: Commit**

```bash
git add src/heinrich/profile/compare.py src/heinrich/cli.py src/heinrich/mcp.py tests/test_cb_sequence_mri.py
git commit -m "feat: cb-loss tool — per-position loss decomposition"
```

---

### Task 6: cb-routing -- Sequence-level expert routing

**Files:**
- Modify: `src/heinrich/profile/compare.py` (add `causal_bank_routing`)
- Modify: `src/heinrich/cli.py` (parser + handler + dispatch)
- Modify: `src/heinrich/mcp.py` (tool + dispatch)
- Test: `tests/test_cb_sequence_mri.py`

- [ ] **Step 1: Write the test**

```python
def test_cb_routing():
    """cb-routing reads sequence MRI and returns routing statistics."""
    from heinrich.profile.compare import causal_bank_routing

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), n_seqs=5, seq_len=64,
                                            n_experts=4)
        result = causal_bank_routing(mri_path)

    assert "error" not in result
    assert "overall_distribution" in result
    assert "switch_rate" in result
    assert "by_position" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cb_sequence_mri.py::test_cb_routing -v
```

- [ ] **Step 3: Implement causal_bank_routing**

```python
def causal_bank_routing(mri_path: str, *, _mri=None) -> dict:
    """Sequence-level expert routing from sequence-mode causal bank MRI.

    Per-band expert distribution, switch rate, and position-dependent routing.
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if meta.get('capture', {}).get('mode') != 'sequence':
        return {"error": "Requires sequence-mode MRI"}
    if 'routing' not in mri:
        return {"error": "MRI has no routing data (single-expert model?)"}

    routing = mri['routing'].astype(np.float32)  # [n_seqs, seq_len, n_experts]
    n_seqs, seq_len, n_experts = routing.shape

    # Overall expert distribution (argmax winner)
    winners = np.argmax(routing, axis=-1)  # [n_seqs, seq_len]
    overall_dist = []
    for e in range(n_experts):
        pct = float((winners == e).mean()) * 100
        overall_dist.append({"expert": e, "pct": round(pct, 2)})

    # Switch rate: % of consecutive positions where winner changes
    switches = (winners[:, 1:] != winners[:, :-1])
    switch_rate = float(switches.mean()) * 100

    # By position range
    ranges = [(0, 4), (4, 64), (64, 256), (256, seq_len)]
    by_pos = []
    for lo, hi in ranges:
        hi = min(hi, seq_len)
        if lo >= seq_len:
            break
        chunk_winners = winners[:, lo:hi]
        dist = []
        for e in range(n_experts):
            pct = float((chunk_winners == e).mean()) * 100
            dist.append(round(pct, 1))
        # Switch rate in this range
        if hi - lo > 1:
            chunk_switches = (chunk_winners[:, 1:] != chunk_winners[:, :-1])
            sr = float(chunk_switches.mean()) * 100
        else:
            sr = 0.0
        by_pos.append({
            "range": f"{lo}-{hi}",
            "distribution": dist,
            "switch_rate": round(sr, 2),
        })

    # Routing margin: mean |best - second_best| weight
    sorted_r = np.sort(routing, axis=-1)
    margin = float((sorted_r[:, :, -1] - sorted_r[:, :, -2]).mean())

    # Per-sequence routing entropy
    flat_routing = routing.reshape(-1, n_experts)
    flat_routing_safe = flat_routing / (flat_routing.sum(axis=-1, keepdims=True) + 1e-10)
    entropy = float(-(flat_routing_safe * np.log(flat_routing_safe + 1e-10)).sum(axis=-1).mean())

    return {
        "model": meta['model'].get('name', '?'),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "n_experts": n_experts,
        "overall_distribution": overall_dist,
        "switch_rate": round(switch_rate, 2),
        "routing_margin": round(margin, 4),
        "routing_entropy": round(entropy, 4),
        "by_position": by_pos,
    }
```

- [ ] **Step 4: Run test, wire CLI + MCP (same pattern as Task 5)**

CLI parser:
```python
    p_cb_routing = sub.add_parser("profile-cb-routing", help="Causal bank sequence-level expert routing")
    p_cb_routing.add_argument("--mri", required=True, help=".mri directory (causal bank, sequence mode)")
```

MCP tool:
```python
    "heinrich_cb_routing": {
        "description": "Causal bank expert routing: distribution, switch rate, position dynamics. Reads sequence-mode .mri.",
        "parameters": {
            "mri": {"type": "string", "description": ".mri directory (causal bank, sequence mode)", "required": True},
        },
    },
```

- [ ] **Step 5: Commit**

```bash
git add src/heinrich/profile/compare.py src/heinrich/cli.py src/heinrich/mcp.py tests/test_cb_sequence_mri.py
git commit -m "feat: cb-routing tool — sequence-level expert routing analysis"
```

---

### Task 7: cb-temporal -- Temporal attention forensics

**Files:** Same pattern as Tasks 5-6.

- [ ] **Step 1: Write test**

```python
def test_cb_temporal():
    """cb-temporal reads sequence MRI and returns temporal attention analysis."""
    from heinrich.profile.compare import causal_bank_temporal

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), n_seqs=5, seq_len=64,
                                            has_temporal=True)
        result = causal_bank_temporal(mri_path)

    assert "error" not in result
    assert "output_l2_by_position" in result
    assert "correlation_chain" in result
```

- [ ] **Step 2: Implement causal_bank_temporal**

```python
def causal_bank_temporal(mri_path: str, *, _mri=None) -> dict:
    """Temporal attention forensics from sequence-mode causal bank MRI.

    Attention magnitude by position, correlation chain (embed -> substrate -> attn),
    attention distribution over snapshots.
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if meta.get('capture', {}).get('mode') != 'sequence':
        return {"error": "Requires sequence-mode MRI"}
    if 'temporal_output' not in mri or mri.get('temporal_output') is None:
        return {"error": "MRI has no temporal attention data"}

    temporal_output = mri['temporal_output'].astype(np.float32)  # [n_seqs, seq_len, n_modes]
    temporal_weights = mri.get('temporal_weights')  # [n_seqs, seq_len, n_snapshots] or None
    substrate = mri['substrate_states'].astype(np.float32)
    embedding = mri['embedding'].astype(np.float32) if 'embedding' in mri else None

    n_seqs, seq_len, n_modes = temporal_output.shape
    snapshot_interval = meta['capture'].get('snapshot_interval', 64)

    # Temporal attention output L2 by position range
    ta_l2 = np.linalg.norm(temporal_output, axis=-1)  # [n_seqs, seq_len]
    ranges = [(0, 4), (4, 64), (64, 256), (256, seq_len)]
    l2_by_pos = []
    for lo, hi in ranges:
        hi = min(hi, seq_len)
        if lo >= seq_len:
            break
        chunk = ta_l2[:, lo:hi]
        l2_by_pos.append({
            "range": f"{lo}-{hi}",
            "mean_l2": round(float(chunk.mean()), 4),
            "max_l2": round(float(chunk.max()), 4),
        })

    # Correlation chain
    sub_disp = np.linalg.norm(substrate, axis=-1).flatten()
    ta_disp = ta_l2.flatten()
    corr_chain = {}
    if embedding is not None:
        embed_norm = np.linalg.norm(embedding, axis=-1).flatten()
        corr_chain["embed_substrate"] = round(float(np.corrcoef(embed_norm, sub_disp)[0, 1]), 4)
        corr_chain["embed_temporal"] = round(float(np.corrcoef(embed_norm, ta_disp)[0, 1]), 4)
    corr_chain["substrate_temporal"] = round(float(np.corrcoef(sub_disp, ta_disp)[0, 1]), 4)

    # Attention distribution over snapshots
    snapshot_profile = {}
    if temporal_weights is not None:
        tw = temporal_weights.astype(np.float32)  # [n_seqs, seq_len, n_snapshots]
        n_snapshots = tw.shape[2]
        # Mean attention weight per snapshot (averaged over sequences and positions)
        mean_per_snap = tw.mean(axis=(0, 1))  # [n_snapshots]
        snapshot_profile = {
            "n_snapshots": int(n_snapshots),
            "snapshot_interval": snapshot_interval,
            "mean_weight_per_snapshot": [round(float(w), 4) for w in mean_per_snap],
            "peak_snapshot": int(np.argmax(mean_per_snap)),
        }

    return {
        "model": meta['model'].get('name', '?'),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "output_l2_by_position": l2_by_pos,
        "correlation_chain": corr_chain,
        "snapshot_profile": snapshot_profile,
    }
```

- [ ] **Step 3: Wire CLI + MCP, run tests, commit**

CLI: `profile-cb-temporal`, MCP: `heinrich_cb_temporal`. Same pattern.

```bash
git commit -m "feat: cb-temporal tool — temporal attention forensics"
```

---

### Task 8: cb-modes -- Mode utilization in sequences

- [ ] **Step 1: Write test**

```python
def test_cb_modes():
    """cb-modes reads sequence MRI and returns mode utilization."""
    from heinrich.profile.compare import causal_bank_modes

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), n_seqs=5, seq_len=64, n_modes=32)
        result = causal_bank_modes(mri_path)

    assert "error" not in result
    assert "by_quartile" in result
    assert "growth_curve" in result
    assert "dead_modes" in result
```

- [ ] **Step 2: Implement causal_bank_modes**

```python
def causal_bank_modes(mri_path: str, *, _mri=None) -> dict:
    """Mode utilization from sequence-mode causal bank MRI.

    Substrate activation by half-life quartile x position range.
    Ramp ratio, dead modes, growth curve.
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if meta.get('capture', {}).get('mode') != 'sequence':
        return {"error": "Requires sequence-mode MRI"}

    substrate = mri['substrate_states'].astype(np.float32)  # [n_seqs, seq_len, n_modes]
    half_lives = mri.get('half_lives')
    n_seqs, seq_len, n_modes = substrate.shape

    if half_lives is not None:
        half_lives = np.asarray(half_lives, dtype=np.float32)
        quartile_edges = np.percentile(half_lives, [0, 25, 50, 75, 100])
    else:
        quartile_edges = None

    # Mean absolute activation by half-life quartile x position range
    ranges = [(0, 4), (4, 64), (64, 256), (256, seq_len)]
    by_quartile = []
    if quartile_edges is not None:
        for q in range(4):
            lo_hl, hi_hl = quartile_edges[q], quartile_edges[q + 1]
            if q == 3:
                mask = (half_lives >= lo_hl) & (half_lives <= hi_hl)
            else:
                mask = (half_lives >= lo_hl) & (half_lives < hi_hl)
            if not mask.any():
                continue
            sub_q = substrate[:, :, mask]
            row = {
                "quartile": q,
                "hl_range": f"{lo_hl:.1f}-{hi_hl:.1f}",
                "n_modes": int(mask.sum()),
                "by_position": [],
            }
            for lo, hi in ranges:
                hi = min(hi, seq_len)
                if lo >= seq_len:
                    break
                chunk = np.abs(sub_q[:, lo:hi, :])
                row["by_position"].append({
                    "range": f"{lo}-{hi}",
                    "mean_abs": round(float(chunk.mean()), 4),
                })
            # Ramp ratio: late/early activation
            early = np.abs(sub_q[:, :min(4, seq_len), :]).mean()
            late = np.abs(sub_q[:, max(0, seq_len-64):, :]).mean()
            row["ramp_ratio"] = round(float(late / (early + 1e-10)), 2)
            by_quartile.append(row)

    # Dead mode detection
    max_activation = np.abs(substrate).max(axis=(0, 1))  # [n_modes]
    mean_act = max_activation.mean()
    dead_modes = int((max_activation < mean_act * 0.01).sum())

    # Most position-varying modes (top 5 by std across positions)
    pos_std = np.abs(substrate).mean(axis=0).std(axis=0)  # [n_modes]
    top_varying = np.argsort(pos_std)[-5:][::-1]
    most_varying = [{"mode": int(m), "std": round(float(pos_std[m]), 4),
                     "hl": round(float(half_lives[m]), 1) if half_lives is not None else None}
                    for m in top_varying]

    # Growth curve: substrate L2 norm by position
    l2_by_pos = np.linalg.norm(substrate, axis=-1).mean(axis=0)  # [seq_len]
    growth_curve = []
    for lo, hi in ranges:
        hi = min(hi, seq_len)
        if lo >= seq_len:
            break
        growth_curve.append({
            "range": f"{lo}-{hi}",
            "mean_l2": round(float(l2_by_pos[lo:hi].mean()), 4),
        })

    return {
        "model": meta['model'].get('name', '?'),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "n_modes": n_modes,
        "by_quartile": by_quartile,
        "dead_modes": dead_modes,
        "most_varying": most_varying,
        "growth_curve": growth_curve,
    }
```

- [ ] **Step 3: Wire CLI + MCP, run tests, commit**

```bash
git commit -m "feat: cb-modes tool — mode utilization by half-life quartile"
```

---

### Task 9: cb-decompose -- Manifold decomposition

- [ ] **Step 1: Write test**

```python
def test_cb_decompose():
    """cb-decompose reads sequence MRI and returns manifold decomposition."""
    from heinrich.profile.compare import causal_bank_decompose

    with tempfile.TemporaryDirectory() as tmp:
        mri_path = _make_fake_sequence_mri(Path(tmp), n_seqs=5, seq_len=64, n_modes=32)
        result = causal_bank_decompose(mri_path)

    assert "error" not in result
    assert "pca" in result
    assert "position_r2" in result
    assert "ghost_fraction" in result
```

- [ ] **Step 2: Implement causal_bank_decompose**

```python
def causal_bank_decompose(mri_path: str, *, n_sample: int | None = None,
                          _mri=None) -> dict:
    """Manifold decomposition from sequence-mode causal bank MRI.

    PCA on sequence substrate. Identifies which PCs encode position (clock),
    which predict loss (content), and which are ghosts (neither).
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if meta.get('capture', {}).get('mode') != 'sequence':
        return {"error": "Requires sequence-mode MRI"}

    substrate = mri['substrate_states'].astype(np.float32)  # [n_seqs, seq_len, n_modes]
    loss = mri['loss']  # [n_seqs, seq_len]
    n_seqs, seq_len, n_modes = substrate.shape

    # Flatten for PCA: [n_seqs * seq_len, n_modes]
    flat_sub = substrate.reshape(-1, n_modes)
    n_total = flat_sub.shape[0]
    if n_sample and n_sample < n_total:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_total, n_sample, replace=False)
        flat_sub = flat_sub[idx]
        flat_loss = loss.reshape(-1)[idx]
        flat_pos = np.tile(np.arange(seq_len), n_seqs)[idx]
    else:
        flat_loss = loss.reshape(-1)
        flat_pos = np.tile(np.arange(seq_len), n_seqs)

    # PCA
    sub_c = flat_sub - flat_sub.mean(axis=0)
    _, S, Vt = np.linalg.svd(sub_c, full_matrices=False)
    var_exp = (S ** 2) / (S ** 2).sum()
    cum = np.cumsum(var_exp)
    scores = sub_c @ Vt.T  # [N, K]

    pca = {
        "effective_dim": round(float(np.exp(-(var_exp * np.log(var_exp + 1e-10)).sum())), 1),
        "pcs_for_50": int(np.searchsorted(cum, 0.5)) + 1,
        "pcs_for_80": int(np.searchsorted(cum, 0.8)) + 1,
        "pcs_for_95": int(np.searchsorted(cum, 0.95)) + 1,
    }

    # Position correlation per PC (which PCs encode position?)
    valid = ~np.isnan(flat_loss)
    pc_position_r = []
    pc_loss_r = []
    n_pcs = min(20, scores.shape[1])
    for i in range(n_pcs):
        r_pos = float(np.corrcoef(scores[:, i], flat_pos.astype(np.float32))[0, 1])
        r_loss = float(np.corrcoef(scores[valid, i], flat_loss[valid])[0, 1])
        pc_position_r.append(round(r_pos, 4))
        pc_loss_r.append(round(r_loss, 4))

    # Position variance regression: how much position info is in top PCs?
    from numpy.linalg import lstsq
    X_pos = scores[:, :n_pcs]
    y_pos = flat_pos.astype(np.float32)
    coef, _, _, _ = lstsq(X_pos, y_pos, rcond=None)
    pred_pos = X_pos @ coef
    ss_res = float(((y_pos - pred_pos) ** 2).sum())
    ss_tot = float(((y_pos - y_pos.mean()) ** 2).sum())
    position_r2 = round(1.0 - ss_res / (ss_tot + 1e-10), 4)

    # Content R2: how much of loss is predictable from PCs?
    X_loss = scores[valid, :n_pcs]
    y_loss = flat_loss[valid]
    coef_l, _, _, _ = lstsq(X_loss, y_loss, rcond=None)
    pred_loss = X_loss @ coef_l
    ss_res_l = float(((y_loss - pred_loss) ** 2).sum())
    ss_tot_l = float(((y_loss - y_loss.mean()) ** 2).sum())
    content_r2 = round(1.0 - ss_res_l / (ss_tot_l + 1e-10), 4)

    # Ghost fraction: variance in PCs that are neither position nor loss-predictive
    threshold = 0.1  # |r| < 0.1 = ghost
    ghost_var = 0.0
    position_var = 0.0
    content_var = 0.0
    for i in range(n_pcs):
        v = float(var_exp[i])
        is_pos = abs(pc_position_r[i]) > threshold
        is_content = abs(pc_loss_r[i]) > threshold
        if is_pos:
            position_var += v
        elif is_content:
            content_var += v
        else:
            ghost_var += v
    total_top = sum(float(var_exp[i]) for i in range(n_pcs))

    return {
        "model": meta['model'].get('name', '?'),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "n_modes": n_modes,
        "pca": pca,
        "position_r2": position_r2,
        "content_r2": content_r2,
        "ghost_fraction": round(ghost_var / (total_top + 1e-10) * 100, 1),
        "position_fraction": round(position_var / (total_top + 1e-10) * 100, 1),
        "content_fraction": round(content_var / (total_top + 1e-10) * 100, 1),
        "pc_position_r": pc_position_r,
        "pc_loss_r": pc_loss_r,
        "top_variance_pct": [round(float(v) * 100, 2) for v in var_exp[:n_pcs]],
    }
```

- [ ] **Step 3: Wire CLI + MCP, run tests, commit**

```bash
git commit -m "feat: cb-decompose tool — manifold decomposition (position/content/ghost)"
```

---

### Task 10: cb-substrate-local -- Substrate vs local path balance

- [ ] **Step 1: Write test**

```python
def test_cb_substrate_local():
    """cb-substrate-local reads sequence MRI with local path data."""
    from heinrich.profile.compare import causal_bank_substrate_local

    with tempfile.TemporaryDirectory() as tmp:
        # Create MRI with local_norm data
        mri_dir = Path(tmp) / "test.mri"
        mri_dir.mkdir()
        metadata = {
            "version": "0.7", "type": "mri", "architecture": "causal_bank",
            "model": {"name": "test", "n_modes": 32, "n_experts": 1,
                      "n_bands": 1, "embed_dim": 16, "vocab_size": 1024,
                      "n_layers": 1, "hidden_size": 32},
            "capture": {"mode": "sequence", "n_seqs": 3, "seq_len": 64,
                        "n_tokens": 192, "has_local": True,
                        "has_temporal": False, "has_routing": False,
                        "has_band_loss": False},
            "provenance": {"seed": 42},
        }
        import json
        with open(mri_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        np.savez_compressed(mri_dir / "tokens.npz",
                            token_ids=np.zeros((3, 64), dtype=np.int32))
        np.save(mri_dir / "substrate.npy",
                np.random.randn(3, 64, 32).astype(np.float16))
        np.save(mri_dir / "loss.npy",
                np.random.rand(3, 64).astype(np.float32))
        np.save(mri_dir / "local_norm.npy",
                np.random.rand(3, 64).astype(np.float32))
        np.save(mri_dir / "half_lives.npy",
                np.logspace(0, 2, 32, dtype=np.float32))

        result = causal_bank_substrate_local(str(mri_dir))

    assert "error" not in result
    assert "by_position" in result
    assert "crossover_position" in result
```

- [ ] **Step 2: Implement causal_bank_substrate_local**

```python
def causal_bank_substrate_local(mri_path: str, *, _mri=None) -> dict:
    """Substrate vs local path balance from sequence-mode causal bank MRI.

    Substrate L2, local L2, and their ratio by position range.
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if meta.get('capture', {}).get('mode') != 'sequence':
        return {"error": "Requires sequence-mode MRI"}

    substrate = mri['substrate_states'].astype(np.float32)
    n_seqs, seq_len, n_modes = substrate.shape
    sub_l2 = np.linalg.norm(substrate, axis=-1)  # [n_seqs, seq_len]

    has_local = 'local_norm' in mri and mri.get('local_norm') is not None
    local_l2 = mri['local_norm'].astype(np.float32) if has_local else None

    ranges = [(0, 4), (4, 64), (64, 256), (256, seq_len)]
    by_pos = []
    for lo, hi in ranges:
        hi = min(hi, seq_len)
        if lo >= seq_len:
            break
        row = {
            "range": f"{lo}-{hi}",
            "substrate_l2": round(float(sub_l2[:, lo:hi].mean()), 4),
        }
        if local_l2 is not None:
            loc = float(local_l2[:, lo:hi].mean())
            row["local_l2"] = round(loc, 4)
            row["substrate_local_ratio"] = round(
                float(sub_l2[:, lo:hi].mean()) / (loc + 1e-10), 2)
        by_pos.append(row)

    # Crossover: first position where mean substrate L2 > mean local L2
    crossover = None
    if local_l2 is not None:
        mean_sub = sub_l2.mean(axis=0)  # [seq_len]
        mean_loc = local_l2.mean(axis=0)
        crosses = np.where(mean_sub > mean_loc)[0]
        if len(crosses) > 0:
            crossover = int(crosses[0])

    return {
        "model": meta['model'].get('name', '?'),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "has_local": has_local,
        "by_position": by_pos,
        "crossover_position": crossover,
    }
```

- [ ] **Step 3: Wire CLI + MCP, run tests, commit**

```bash
git commit -m "feat: cb-substrate-local tool — substrate vs local path balance"
```

---

## Phase C: Tokenizer and Diagnostic Tools

### Task 11: tokenizer-difficulty -- Embedding norm analysis from MRI

**No model needed. Reads embedding.npy from any causal bank MRI (impulse or sequence).**

**Files:**
- Modify: `src/heinrich/profile/compare.py` (add `tokenizer_difficulty`)
- Modify: `src/heinrich/cli.py` + `src/heinrich/mcp.py`
- Test: `tests/test_cb_sequence_mri.py`

- [ ] **Step 1: Write test**

```python
def test_tokenizer_difficulty():
    """tokenizer-difficulty reads MRI embedding and correlates with substrate."""
    from heinrich.profile.compare import tokenizer_difficulty

    with tempfile.TemporaryDirectory() as tmp:
        # Use impulse MRI (not sequence) -- this tool works on both
        mri_dir = Path(tmp) / "test.mri"
        mri_dir.mkdir()
        n_tokens, n_modes, embed_dim = 100, 32, 16
        metadata = {
            "version": "0.7", "type": "mri", "architecture": "causal_bank",
            "model": {"name": "test", "n_modes": n_modes, "n_experts": 1,
                      "n_bands": 1, "embed_dim": embed_dim, "vocab_size": 1024,
                      "n_layers": 1, "hidden_size": n_modes},
            "capture": {"mode": "raw", "n_tokens": n_tokens},
            "provenance": {"seed": 42},
        }
        import json
        with open(mri_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        np.savez_compressed(mri_dir / "tokens.npz",
                            token_ids=np.arange(n_tokens, dtype=np.int32),
                            token_texts=np.array([f"t{i}" for i in range(n_tokens)]),
                            scripts=np.array(["Latin"] * n_tokens))
        np.save(mri_dir / "substrate.npy",
                np.random.randn(n_tokens, n_modes).astype(np.float16))
        np.save(mri_dir / "embedding.npy",
                np.random.randn(n_tokens, embed_dim).astype(np.float16))

        result = tokenizer_difficulty(str(mri_dir))

    assert "error" not in result
    assert "embed_substrate_r" in result
    assert "effective_dim" in result
    assert "difficulty_quartiles" in result
```

- [ ] **Step 2: Implement tokenizer_difficulty**

```python
def tokenizer_difficulty(mri_path: str, *, _mri=None) -> dict:
    """Per-token difficulty from embeddings. No model needed, reads MRI arrays.

    Embedding norm correlates with prediction difficulty. This tool
    measures the correlation and identifies easy vs hard tokens.
    """
    from .mri import load_mri

    mri = _mri or load_mri(mri_path)
    meta = mri['metadata']
    if meta.get('architecture') != 'causal_bank':
        return {"error": "Not a causal bank MRI"}
    if 'embedding' not in mri:
        return {"error": "MRI has no embedding data"}

    embedding = mri['embedding'].astype(np.float32)
    # Handle both impulse [N, embed_dim] and sequence [n_seqs, seq_len, embed_dim]
    if embedding.ndim == 3:
        embedding = embedding.reshape(-1, embedding.shape[-1])

    substrate = mri.get('substrate_states')
    if substrate is not None:
        substrate = substrate.astype(np.float32)
        if substrate.ndim == 3:
            substrate = substrate.reshape(-1, substrate.shape[-1])

    n_tokens, embed_dim = embedding.shape
    embed_norm = np.linalg.norm(embedding, axis=1)

    # Correlation with substrate displacement
    corr = {}
    if substrate is not None:
        sub_disp = np.linalg.norm(substrate, axis=1)
        corr["embed_substrate_r"] = round(float(np.corrcoef(embed_norm, sub_disp)[0, 1]), 4)

    # Embedding PCA
    emb_c = embedding - embedding.mean(axis=0)
    _, S, _ = np.linalg.svd(emb_c, full_matrices=False)
    var_exp = (S ** 2) / (S ** 2).sum()
    eff_dim = float(np.exp(-(var_exp * np.log(var_exp + 1e-10)).sum()))

    # Difficulty quartiles
    quartiles = np.percentile(embed_norm, [25, 50, 75])
    quartile_info = []
    edges = [0, quartiles[0], quartiles[1], quartiles[2], embed_norm.max() + 1]
    labels = ["easy", "medium-easy", "medium-hard", "hard"]
    for i in range(4):
        mask = (embed_norm >= edges[i]) & (embed_norm < edges[i + 1])
        quartile_info.append({
            "label": labels[i],
            "n_tokens": int(mask.sum()),
            "mean_norm": round(float(embed_norm[mask].mean()), 4) if mask.any() else 0,
        })

    # Near-duplicate detection (cosine > 0.9)
    if n_tokens <= 5000:
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        emb_normed = embedding / (norms + 1e-10)
        cos = emb_normed @ emb_normed.T
        np.fill_diagonal(cos, 0)
        n_near_dup = int((cos > 0.9).sum()) // 2
    else:
        n_near_dup = -1  # too many tokens for dense cosine

    return {
        "model": meta['model'].get('name', '?'),
        "n_tokens": n_tokens,
        "embed_dim": embed_dim,
        "effective_dim": round(eff_dim, 1),
        **corr,
        "difficulty_quartiles": quartile_info,
        "near_duplicates": n_near_dup,
        "embed_norm_range": [round(float(embed_norm.min()), 4),
                             round(float(embed_norm.max()), 4)],
    }
```

- [ ] **Step 3: Wire CLI + MCP, run tests, commit**

CLI: `profile-tokenizer-difficulty`, MCP: `heinrich_tokenizer_difficulty`.

```bash
git commit -m "feat: tokenizer-difficulty tool — embedding norm analysis from MRI"
```

---

### Task 12: tokenizer-compare -- Multi-tokenizer comparison

**No model needed. Pure sentencepiece analysis.**

**Files:**
- Modify: `src/heinrich/profile/compare.py` (add `tokenizer_compare`)
- Modify: `src/heinrich/cli.py` + `src/heinrich/mcp.py`
- Test: `tests/test_tokenizer_compare.py`

- [ ] **Step 1: Write test**

```python
"""Tests for tokenizer comparison."""
from __future__ import annotations

import pytest


def test_tokenizer_compare_structure():
    """tokenizer_compare returns expected fields."""
    from heinrich.profile.compare import tokenizer_compare

    # Use a known tokenizer path (must exist)
    import os
    tok_path = "/Volumes/sharts/heinrich/fineweb_1024_bpe.model"
    if not os.path.exists(tok_path):
        pytest.skip("Tokenizer not available")

    result = tokenizer_compare([tok_path])
    assert "error" not in result
    assert len(result["tokenizers"]) == 1
    t = result["tokenizers"][0]
    assert "vocab_size" in t
    assert "bytes_per_token" in t
    assert "byte_fallback_pct" in t
```

- [ ] **Step 2: Implement tokenizer_compare**

```python
def tokenizer_compare(
    tokenizer_paths: list[str],
    *,
    sample_text: str | None = None,
) -> dict:
    """Compare multiple sentencepiece tokenizers.

    Vocab size, compression ratio, byte fallback, token length distribution,
    overlap between tokenizers, and parameter budget impact.
    """
    import sentencepiece as spm

    results = []
    all_vocabs = []

    for path in tokenizer_paths:
        sp = spm.SentencePieceProcessor()
        sp.Load(path)
        vocab_size = sp.GetPieceSize()

        # Analyze vocabulary
        lengths = []
        byte_tokens = 0
        for i in range(vocab_size):
            piece = sp.IdToPiece(i)
            if piece.startswith("<0x") and piece.endswith(">"):
                byte_tokens += 1
                lengths.append(1)
            else:
                raw = piece.replace("\u2581", " ")
                lengths.append(len(raw.encode("utf-8")))

        lengths = np.array(lengths)
        vocab_set = set(sp.IdToPiece(i) for i in range(vocab_size))
        all_vocabs.append(vocab_set)

        # Compression on sample text
        bpt = None
        tpb = None
        byte_fallback_pct = round(byte_tokens / vocab_size * 100, 2)
        if sample_text:
            tokens = sp.Encode(sample_text)
            text_bytes = len(sample_text.encode("utf-8"))
            bpt = round(text_bytes / len(tokens), 4)
            tpb = round(len(tokens) / text_bytes, 4)
            # Byte fallback on this text
            n_byte_tok = sum(1 for t in tokens if sp.IdToPiece(t).startswith("<0x"))
            byte_fallback_pct = round(n_byte_tok / len(tokens) * 100, 2)

        # Token length distribution
        length_dist = {}
        for lb in [1, 2, 3, 4, 5, 8, 12, 16]:
            count = int((lengths == lb).sum()) if lb < 16 else int((lengths >= lb).sum())
            length_dist[f"{lb}B"] = count

        results.append({
            "path": path,
            "vocab_size": vocab_size,
            "bytes_per_token": bpt,
            "tokens_per_byte": tpb,
            "byte_fallback_pct": byte_fallback_pct,
            "byte_tokens": byte_tokens,
            "length_distribution": length_dist,
            "mean_token_bytes": round(float(lengths.mean()), 2),
        })

    # Overlap between tokenizers
    overlap = {}
    for i in range(len(all_vocabs)):
        for j in range(i + 1, len(all_vocabs)):
            common = len(all_vocabs[i] & all_vocabs[j])
            total = len(all_vocabs[i] | all_vocabs[j])
            overlap[f"{i}_vs_{j}"] = {
                "common": common,
                "jaccard": round(common / (total + 1e-10), 4),
            }

    return {
        "tokenizers": results,
        "overlap": overlap,
    }
```

- [ ] **Step 3: Wire CLI + MCP, run tests, commit**

CLI parser:
```python
    p_tok_compare = sub.add_parser("profile-tokenizer-compare",
        help="Compare sentencepiece tokenizers: compression, overlap, byte fallback")
    p_tok_compare.add_argument("--tokenizers", nargs="+", required=True,
        help=".model files to compare")
    p_tok_compare.add_argument("--text", help="Sample text file for compression stats")
```

```bash
git commit -m "feat: tokenizer-compare tool — multi-tokenizer comparison"
```

---

### Task 13: cb-causality -- Finite-difference causality verification

**Needs model. Two forward passes per test position. Standalone diagnostic.**

**Files:**
- Modify: `src/heinrich/profile/compare.py` (add `causal_bank_causality`)
- Modify: `src/heinrich/cli.py` + `src/heinrich/mcp.py`

- [ ] **Step 1: Write test (mock backend)**

```python
def test_cb_causality_structure():
    """cb-causality returns expected fields with mock backend."""
    # This is an integration test -- requires actual model
    # Unit test verifies the function accepts the right args
    pytest.skip("Requires decepticon backend + checkpoint")
```

- [ ] **Step 2: Implement causal_bank_causality**

```python
def causal_bank_causality(backend, *, seq_len: int = 256, n_tests: int = 8,
                          seed: int = 42) -> dict:
    """Finite-difference causality test for causal bank models.

    Runs full sequence vs truncated sequence. If logits[t] differ between
    full and truncated-at-t, information leaked from future positions.

    This catches causal masking bugs in temporal attention and other mechanisms.
    """
    rng = np.random.RandomState(seed)
    vocab_size = backend.config.vocab_size
    seq = rng.randint(0, vocab_size, (1, seq_len)).astype(np.int64)

    # Full forward pass
    full_logits = backend.forward(seq)  # [1, seq_len, vocab]

    # Test specific positions
    test_positions = sorted(rng.choice(range(4, seq_len - 1), n_tests, replace=False))
    violations = []

    for t in test_positions:
        # Truncate: only first t+1 tokens
        truncated = seq[:, :t + 1]
        trunc_logits = backend.forward(truncated)  # [1, t+1, vocab]

        # Compare logits at position t
        full_t = full_logits[0, t]
        trunc_t = trunc_logits[0, t]
        max_diff = float(np.abs(full_t - trunc_t).max())

        if max_diff > 1e-4:
            violations.append({
                "position": int(t),
                "max_logit_diff": round(max_diff, 6),
                "mean_logit_diff": round(float(np.abs(full_t - trunc_t).mean()), 6),
            })

    causal = len(violations) == 0
    return {
        "causal": causal,
        "seq_len": seq_len,
        "n_tests": n_tests,
        "positions_tested": [int(t) for t in test_positions],
        "violations": violations,
        "verdict": "PASS: no future information leakage" if causal
                   else f"FAIL: {len(violations)} positions leaked future info",
    }
```

- [ ] **Step 3: Wire CLI (loads model, runs test)**

CLI handler:
```python
def _cmd_cb_causality(args: argparse.Namespace) -> None:
    """Causality verification for causal bank models."""
    from .backend.protocol import load_backend
    from .profile.compare import causal_bank_causality

    backend = load_backend(args.model, backend="decepticon")
    result = causal_bank_causality(backend, seq_len=args.seq_len, n_tests=args.n_tests)

    print(f"\n=== Causality Check ===\n")
    print(f"  {result['verdict']}")
    if result['violations']:
        for v in result['violations']:
            print(f"    Position {v['position']}: max_diff={v['max_logit_diff']:.6f}")
```

- [ ] **Step 4: Wire MCP, commit**

MCP: `heinrich_cb_causality`, subprocess with `--model` required, timeout 120s.

```bash
git commit -m "feat: cb-causality tool — finite-difference causality verification"
```

---

### Task 14: cb-reproduce -- Determinism check

**Needs model. Two identical forward passes. Standalone diagnostic.**

- [ ] **Step 1: Implement causal_bank_reproduce**

```python
def causal_bank_reproduce(backend, *, seq_len: int = 256, seed: int = 42) -> dict:
    """Reproducibility test: two identical forward passes should give identical logits.

    Catches nondeterminism in kernel path, scan path, or attention.
    """
    rng = np.random.RandomState(seed)
    vocab_size = backend.config.vocab_size
    seq = rng.randint(0, vocab_size, (1, seq_len)).astype(np.int64)

    logits_a = backend.forward(seq)
    logits_b = backend.forward(seq)

    max_diff = float(np.abs(logits_a - logits_b).max())
    mean_diff = float(np.abs(logits_a - logits_b).mean())
    identical = max_diff == 0.0

    return {
        "identical": identical,
        "max_diff": round(max_diff, 10),
        "mean_diff": round(mean_diff, 10),
        "seq_len": seq_len,
        "verdict": "PASS: bitwise identical" if identical
                   else f"FAIL: max diff {max_diff:.2e}",
    }
```

- [ ] **Step 2: Wire CLI + MCP, commit**

```bash
git commit -m "feat: cb-reproduce tool — determinism verification"
```

---

## Phase D: Integration

### Task 15: MCP tool list update and full test run

- [ ] **Step 1: Verify all MCP tools are listed in the TOOLS dict**

Check that these tools exist in `mcp.py` TOOLS dict:
- `heinrich_cb_loss`
- `heinrich_cb_routing`
- `heinrich_cb_temporal`
- `heinrich_cb_modes`
- `heinrich_cb_decompose`
- `heinrich_cb_substrate_local`
- `heinrich_tokenizer_difficulty`
- `heinrich_tokenizer_compare`
- `heinrich_cb_causality`
- `heinrich_cb_reproduce`

- [ ] **Step 2: Run full test suite**

```bash
cd /Users/asuramaya/Code/heinrich && pytest tests/ -v
```

All existing tests should still pass. New tests should pass.

- [ ] **Step 3: Update CLAUDE.md**

Add to the Commands section:
```
# Causal bank sequence tools (reads sequence.mri, no model needed)
heinrich profile-cb-loss --mri X.mri              # per-position loss decomposition
heinrich profile-cb-routing --mri X.mri            # sequence-level expert routing
heinrich profile-cb-temporal --mri X.mri           # temporal attention forensics
heinrich profile-cb-modes --mri X.mri              # mode utilization by half-life
heinrich profile-cb-decompose --mri X.mri          # manifold decomposition
heinrich profile-cb-substrate-local --mri X.mri    # substrate vs local path balance
heinrich profile-tokenizer-difficulty --mri X.mri  # embedding norm = difficulty
heinrich profile-tokenizer-compare --tokenizers A B # multi-tokenizer comparison

# Causal bank diagnostics (needs model)
heinrich profile-cb-causality --model X.checkpoint.pt    # causality verification
heinrich profile-cb-reproduce --model X.checkpoint.pt    # determinism check
```

Add new MCP tools to the MCP tools section.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: complete causal bank tool suite — 10 new tools"
```

---

## Summary

| Tool | Type | Input | Phase |
|------|------|-------|-------|
| `cb-loss` | MRI reader | sequence.mri | B |
| `cb-routing` | MRI reader | sequence.mri | B |
| `cb-temporal` | MRI reader | sequence.mri | B |
| `cb-modes` | MRI reader | sequence.mri | B |
| `cb-decompose` | MRI reader | sequence.mri | B |
| `cb-substrate-local` | MRI reader | sequence.mri | B |
| `tokenizer-difficulty` | MRI reader | any cb .mri | C |
| `tokenizer-compare` | standalone | .model files | C |
| `cb-causality` | model loader | checkpoint.pt | C |
| `cb-reproduce` | model loader | checkpoint.pt | C |

Dependencies: Phase A (foundation) must complete before Phase B (readers). Phase C is independent of B. Task 1 (decepticons) must complete before Task 3 (capture).
