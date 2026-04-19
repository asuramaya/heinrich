from __future__ import annotations

import json
import struct

import numpy as np

from heinrich import companion as c
from heinrich import companion_serve as cs


def _write_pc_index(path, data: np.ndarray) -> None:
    n_layers, n_tok, full_k = data.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<4sIII", b"PCSC", n_layers, n_tok, full_k))
        for pc in range(full_k):
            f.write(np.ascontiguousarray(data[:, :, pc].astype(np.float16)).tobytes())


def _write_token_index(path, data: np.ndarray) -> None:
    n_layers, n_tok, full_k = data.shape
    token_major = np.transpose(data, (1, 0, 2)).astype(np.float16)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sIII", b"TOKS", n_tok, n_layers, full_k))
        f.write(token_major.tobytes())


def _make_mri(tmp_path, data: np.ndarray) -> None:
    decomp = tmp_path / "decomp"
    decomp.mkdir()
    meta = {
        "n_sample": int(data.shape[1]),
        "n_components": int(data.shape[2]),
        "n_layers": int(data.shape[0]),
        "n_real_layers": max(0, int(data.shape[0]) - 2),
        "layers": [
            {"layer": 0},
            {"layer": "emb"},
            {"layer": "lmh"},
        ][: data.shape[0]],
    }
    (decomp / "meta.json").write_text(json.dumps(meta))
    if data.shape[0] >= 1:
        np.save(decomp / "L00_variance.npy", np.linspace(0.4, 0.1, data.shape[2], dtype=np.float32))
    if data.shape[0] >= 2:
        np.save(decomp / "emb_variance.npy", np.linspace(0.3, 0.05, data.shape[2], dtype=np.float32))
    if data.shape[0] >= 3:
        np.save(decomp / "lmh_variance.npy", np.linspace(0.2, 0.02, data.shape[2], dtype=np.float32))
    _write_pc_index(decomp / "pc_scores.bin", data)
    _write_token_index(decomp / "token_scores.bin", data)


def test_build_serve_artifacts_writes_full_and_step_indexes(tmp_path):
    data = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4)
    _make_mri(tmp_path, data)

    result = cs.build_serve_artifacts(str(tmp_path), steps=(2,), force=True)

    assert "error" not in result
    serve_dir = tmp_path / "serve"
    assert (serve_dir / "pc_scores_full.bin").exists()
    assert (serve_dir / "token_scores_full.bin").exists()
    step_path = serve_dir / "pc_scores_step2.bin"
    assert step_path.exists()

    meta = json.loads((serve_dir / "meta.json").read_text())
    assert meta["source"] == "serve"
    assert meta["n_tokens"] == 5
    assert meta["steps"]["2"]["n_sample"] == 3
    assert len(meta["pc_vars"]) == 3

    n_layers, n_sample, full_k = cs._read_pc_header(step_path)
    assert (n_layers, n_sample, full_k) == (3, 3, 4)
    with open(step_path, "rb") as f:
        f.seek(16)
        slab0 = np.frombuffer(f.read(n_layers * n_sample * 2), dtype=np.float16).reshape(n_layers, n_sample)
    np.testing.assert_allclose(slab0.astype(np.float32), data[:, ::2, 0])


def test_load_serve_meta_synthesizes_from_decomp_when_serve_missing(tmp_path):
    data = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)
    _make_mri(tmp_path, data)
    cs.load_serve_meta.cache_clear()

    meta = cs.load_serve_meta(str(tmp_path))

    assert meta["source"] == "decomp"
    assert meta["n_layers"] == 3
    assert meta["n_tokens"] == 4
    assert meta["full_k"] == 3
    assert len(meta["pc_vars"]) == 3
    assert meta["steps"] == {}


def test_companion_prefers_serve_indexes_for_pc_and_token_reads(tmp_path):
    decomp_data = np.zeros((3, 4, 2), dtype=np.float32)
    serve_data = np.full((3, 4, 2), 7.0, dtype=np.float32)
    _make_mri(tmp_path, decomp_data)

    serve = tmp_path / "serve"
    serve.mkdir()
    _write_pc_index(serve / "pc_scores_full.bin", serve_data)
    _write_token_index(serve / "token_scores_full.bin", serve_data)
    (serve / "meta.json").write_text(json.dumps({
        "version": 1,
        "source": "serve",
        "mri_path": str(tmp_path),
        "n_layers": 3,
        "n_real_layers": 1,
        "n_tokens": 4,
        "full_k": 2,
        "pc_scores": "pc_scores_full.bin",
        "token_scores": "token_scores_full.bin",
        "steps": {},
        "pc_vars": [[], [], []],
    }))
    cs.load_serve_meta.cache_clear()

    pc_blob = c._pc_full(str(tmp_path), 0)
    assert isinstance(pc_blob, (bytes, bytearray))
    n_layers, n_tok, pc = struct.unpack("<III", pc_blob[:12])
    assert (n_layers, n_tok, pc) == (3, 4, 0)
    pc_values = np.frombuffer(pc_blob[12:], dtype=np.float16).reshape(n_layers, n_tok)
    np.testing.assert_allclose(pc_values.astype(np.float32), serve_data[:, :, 0])

    tok_blob = c._token_pca_full(str(tmp_path), 0)
    assert isinstance(tok_blob, (bytes, bytearray))
    n_layers_tok, full_k = struct.unpack("<II", tok_blob[:8])
    assert (n_layers_tok, full_k) == (3, 2)
    tok_values = np.frombuffer(tok_blob[8:], dtype=np.float32).reshape(n_layers_tok, full_k)
    np.testing.assert_allclose(tok_values, serve_data[:, 0, :])


def test_cloud_bundle_returns_full_and_sampled_pc_slabs(tmp_path):
    data = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4)
    _make_mri(tmp_path, data)
    cs.build_serve_artifacts(str(tmp_path), steps=(2,), force=True)

    blob = c._cloud_bundle(str(tmp_path), [0, 3], [1], step=2)

    assert isinstance(blob, (bytes, bytearray))
    magic, version, n_full, n_med, n_layers, n_tok, n_sample, step = struct.unpack("<4sIIIIIII", blob[:32])
    assert (magic, version) == (b"CLDB", 1)
    assert (n_full, n_med, n_layers, n_tok, n_sample, step) == (2, 1, 3, 5, 3, 2)

    off = 32
    full_ids = struct.unpack("<2I", blob[off:off + 8])
    off += 8
    med_ids = struct.unpack("<1I", blob[off:off + 4])
    off += 4
    assert full_ids == (0, 3)
    assert med_ids == (1,)

    full_vals = np.frombuffer(blob, dtype=np.float16, count=n_full * n_layers * n_tok, offset=off).reshape(n_full, n_layers, n_tok)
    off += n_full * n_layers * n_tok * 2
    med_vals = np.frombuffer(blob, dtype=np.float16, count=n_med * n_layers * n_sample, offset=off).reshape(n_med, n_layers, n_sample)

    np.testing.assert_allclose(full_vals[0].astype(np.float32), data[:, :, 0])
    np.testing.assert_allclose(full_vals[1].astype(np.float32), data[:, :, 3])
    np.testing.assert_allclose(med_vals[0].astype(np.float32), data[:, ::2, 1])


def test_token_bundle_returns_exact_full_rows_and_hover_slice(tmp_path):
    data = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)
    _make_mri(tmp_path, data)

    blob = c._token_bundle(str(tmp_path), [1], [1], 2)

    assert isinstance(blob, (bytes, bytearray))
    magic, version, layer, n_entries = struct.unpack("<4sIII", blob[:16])
    assert (magic, version, layer, n_entries) == (b"TKBD", 1, 2, 1)

    token_idx, flags, n_layers, full_k, hover_k, hover_inter = struct.unpack("<IIIIII", blob[16:40])
    assert (token_idx, flags, n_layers, full_k, hover_k, hover_inter) == (1, 3, 3, 3, 3, 0)

    off = 40
    full_vals = np.frombuffer(blob, dtype=np.float32, count=n_layers * full_k, offset=off).reshape(n_layers, full_k)
    off += n_layers * full_k * 4
    hover_vals = np.frombuffer(blob, dtype=np.float16, count=hover_k, offset=off)

    np.testing.assert_allclose(full_vals, data[:, 1, :])
    np.testing.assert_allclose(hover_vals.astype(np.float32), data[2, 1, :])
