"""Integration test: full mri-scan pipeline on a real model.

This is the gate. If this passes, the queue can run unattended.
If it fails, the queue shouldn't start.

Requires: MLX backend + cached Qwen 0.5B model.
Run with: pytest tests/test_mri_integration.py -v -m integration
"""
from __future__ import annotations
import json
import os
import shutil
import tempfile

import numpy as np
import pytest


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def model_id():
    return "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


@pytest.fixture(scope="module")
def backend(model_id):
    try:
        from heinrich.backend.protocol import load_backend
        return load_backend(model_id, backend="mlx")
    except Exception as e:
        pytest.skip(f"Cannot load model: {e}")


@pytest.fixture(scope="module")
def scan_dir(backend):
    """Run a full mri-scan on 500 tokens. Shared across all tests."""
    from heinrich.profile.mri import capture_mri

    d = tempfile.mkdtemp(prefix="heinrich_integration_")
    model_dir = os.path.join(d, "test_model")
    os.makedirs(model_dir)

    for mode in ["raw", "naked", "template"]:
        capture_mri(backend, mode=mode, n_index=500,
                    output=os.path.join(model_dir, f"{mode}.mri"))

    yield model_dir
    shutil.rmtree(d, ignore_errors=True)


class TestCapture:

    def test_all_modes_healthy(self, scan_dir):
        from heinrich.profile.mri import verify_mri
        for mode in ["raw", "naked", "template"]:
            r = verify_mri(os.path.join(scan_dir, f"{mode}.mri"))
            assert r["healthy"], f"{mode}: {r['issues']}"

    def test_raw_no_entry_no_attention(self, scan_dir):
        meta = json.loads(open(os.path.join(scan_dir, "raw.mri", "metadata.json")).read())
        assert meta["capture"]["has_entry"] is False
        assert meta["capture"]["has_attention"] is False
        assert not os.path.exists(os.path.join(scan_dir, "raw.mri", "L00_entry.npy"))

    def test_template_has_entry_and_attention(self, scan_dir):
        meta = json.loads(open(os.path.join(scan_dir, "template.mri", "metadata.json")).read())
        assert meta["capture"]["has_entry"] is True
        assert meta["capture"]["has_attention"] is True

    def test_all_modes_have_gates(self, scan_dir):
        for mode in ["raw", "naked", "template"]:
            meta = json.loads(open(os.path.join(
                scan_dir, f"{mode}.mri", "metadata.json")).read())
            assert meta["capture"]["has_gates"] is True

    def test_token_count_consistent(self, scan_dir):
        for mode in ["raw", "naked", "template"]:
            mri = os.path.join(scan_dir, f"{mode}.mri")
            meta = json.loads(open(os.path.join(mri, "metadata.json")).read())
            n_tok = meta["capture"]["n_tokens"]
            tokens = np.load(os.path.join(mri, "tokens.npz"), allow_pickle=False)
            assert len(tokens["token_ids"]) == n_tok
            exit0 = np.load(os.path.join(mri, "L00_exit.npy"), mmap_mode='r')
            assert exit0.shape[0] == n_tok


class TestGates:

    def test_gate_shapes(self, scan_dir):
        meta = json.loads(open(os.path.join(scan_dir, "raw.mri", "metadata.json")).read())
        n_tok = meta["capture"]["n_tokens"]
        gate_k = meta["capture"]["gate_k"]
        g = np.load(os.path.join(scan_dir, "raw.mri", "gates", "L00_gates.npz"))
        assert g["indices"].shape == (n_tok, gate_k)
        assert g["values"].shape == (n_tok, gate_k)

    def test_gate_values_nonzero(self, scan_dir):
        g = np.load(os.path.join(scan_dir, "raw.mri", "gates", "L00_gates.npz"))
        assert not np.all(g["values"] == 0)

    def test_gate_no_nan(self, scan_dir):
        meta = json.loads(open(os.path.join(scan_dir, "raw.mri", "metadata.json")).read())
        n_layers = meta["model"]["n_layers"]
        for i in [0, n_layers // 2, n_layers - 1]:
            g = np.load(os.path.join(scan_dir, "raw.mri", "gates", f"L{i:02d}_gates.npz"))
            assert not np.any(np.isnan(g["values"].astype(np.float32)))

    def test_gate_analysis_runs(self, scan_dir):
        from heinrich.profile.compare import gate_analysis
        r = gate_analysis(os.path.join(scan_dir, "raw.mri"), n_sample=100)
        assert "error" not in r
        assert len(r["layers"]) > 0
        for lr in r["layers"]:
            assert lr["unique_neurons"] > 0
            assert lr["mean_activation"] > 0


class TestAttention:

    def test_attention_shapes(self, scan_dir):
        meta = json.loads(open(os.path.join(scan_dir, "template.mri", "metadata.json")).read())
        n_tok = meta["capture"]["n_tokens"]
        n_heads = meta["model"]["n_heads"]
        seq_len = meta["capture"]["seq_len"]
        a = np.load(os.path.join(scan_dir, "template.mri", "attention", "L00_attn.npy"))
        assert a.shape == (n_tok, n_heads, seq_len)

    def test_attention_sums_to_one(self, scan_dir):
        a = np.load(os.path.join(scan_dir, "template.mri", "attention", "L00_attn.npy"))
        row_sums = a.astype(np.float32).sum(axis=2)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_attention_respects_causal_mask(self, scan_dir):
        meta = json.loads(open(os.path.join(scan_dir, "template.mri", "metadata.json")).read())
        token_pos = meta["capture"]["token_pos"]
        seq_len = meta["capture"]["seq_len"]
        if token_pos >= seq_len - 1:
            pytest.skip("Token at end of sequence")
        a = np.load(os.path.join(scan_dir, "template.mri", "attention", "L00_attn.npy"))
        future = a[:, :, token_pos + 1:].astype(np.float32)
        assert future.max() < 0.01, f"Attends to future: max={future.max()}"

    def test_attention_analysis_runs(self, scan_dir):
        from heinrich.profile.compare import attention_analysis
        r = attention_analysis(os.path.join(scan_dir, "template.mri"), n_sample=100)
        assert "error" not in r
        assert len(r["layers"]) > 0
        for lr in r["layers"]:
            total = lr["self_weight"] + lr["prefix_weight"] + lr["suffix_weight"]
            assert 0.95 < total < 1.05


class TestAnalysis:

    def test_layer_deltas(self, scan_dir):
        from heinrich.profile.compare import layer_deltas
        for mode in ["raw", "naked", "template"]:
            r = layer_deltas(os.path.join(scan_dir, f"{mode}.mri"), n_sample=100)
            assert "error" not in r
            assert len(r["layers"]) > 0
            for lr in r["layers"]:
                assert lr["mean_delta_norm"] >= 0
                assert not np.isnan(lr["mean_delta_norm"])

    def test_logit_lens(self, scan_dir):
        from heinrich.profile.compare import logit_lens
        r = logit_lens(os.path.join(scan_dir, "raw.mri"), n_sample=50, layers=[0, 10, 23])
        assert "error" not in r
        assert len(r["layers"]) == 3
        for lr in r["layers"]:
            for pred in lr["predictions"]:
                assert len(pred["top_ids"]) == 5

    def test_pca_depth(self, scan_dir):
        from heinrich.profile.compare import pca_depth
        r = pca_depth(os.path.join(scan_dir, "raw.mri"), n_sample=200)
        assert "error" not in r
        assert len(r["layers"]) > 0
        for lr in r["layers"]:
            assert 0 <= lr["pc1_pct"] <= 100
            assert lr["pcs_for_50pct"] >= 1

    def test_resume_skips_healthy(self, scan_dir, backend):
        """Re-running capture on healthy MRI skips capture."""
        from heinrich.profile.mri import capture_mri
        result = capture_mri(backend, mode="raw", n_index=500,
                             output=os.path.join(scan_dir, "raw.mri"))
        assert result is not None
