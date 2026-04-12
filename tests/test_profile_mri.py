"""Unit tests for profile/mri.py: load_mri, verify_mri, backfill_mri.

Uses synthetic MRI directories — no model or GPU needed.
"""
from __future__ import annotations
import json
import numpy as np
import pytest
from heinrich.profile.mri import MRI_VERSION


def _make_mri(tmp_path, *, n_tok=20, n_layers=2, hidden=8, vocab=32,
              n_heads=2, has_entry=True, has_attention=True,
              has_gates=True, intermediate=32, mode="raw", seq_len=1):
    """Create a synthetic valid MRI directory (v0.7 format)."""
    d = tmp_path / "test.mri"
    d.mkdir()

    meta = {
        "version": MRI_VERSION,
        "type": "mri",
        "model": {"n_layers": n_layers, "hidden_size": hidden,
                   "vocab_size": vocab, "name": "test", "n_heads": n_heads},
        "capture": {"mode": mode, "n_tokens": n_tok, "n_layers": n_layers,
                     "token_pos": 3 if mode == "template" else 0,
                     "has_entry": has_entry, "has_pre_mlp": True,
                     "has_attention": has_attention, "has_gates": has_gates,
                     "gate_format": "full", "intermediate_size": intermediate,
                     "seq_len": seq_len},
    }
    (d / "metadata.json").write_text(json.dumps(meta))

    np.savez(d / "tokens.npz",
             token_ids=np.arange(n_tok, dtype=np.int32),
             token_texts=np.array([f"t{i}" for i in range(n_tok)]),
             scripts=np.array(["latin"] * (n_tok // 2) + ["CJK"] * (n_tok - n_tok // 2)))

    bl = {}
    for i in range(n_layers):
        np.save(d / f"L{i:02d}_exit.npy",
                np.random.randn(n_tok, hidden).astype(np.float16))
        np.save(d / f"L{i:02d}_entry.npy",
                np.random.randn(n_tok, hidden).astype(np.float16))
        np.save(d / f"L{i:02d}_pre_mlp.npy",
                np.random.randn(n_tok, hidden).astype(np.float16))
        np.save(d / f"L{i:02d}_attn_out.npy",
                np.random.randn(n_tok, hidden).astype(np.float16))
        bl[f"entry_L{i}"] = np.zeros(hidden, dtype=np.float16)
        bl[f"exit_L{i}"] = np.zeros(hidden, dtype=np.float16)
    np.savez(d / "baselines.npz", **bl)

    np.save(d / "embedding.npy", np.random.randn(vocab, hidden).astype(np.float32))
    np.save(d / "lmhead_raw.npy", np.random.randn(vocab, hidden).astype(np.float32))
    np.save(d / "lmhead.npy", np.random.randn(vocab, hidden).astype(np.float32))
    np.savez(d / "norms.npz", final=np.ones(hidden, dtype=np.float32))

    wd = d / "weights"
    wd.mkdir()
    for i in range(n_layers):
        ld = wd / f"L{i:02d}"
        ld.mkdir()
        for name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj"]:
            np.save(ld / f"{name}.npy", np.random.randn(hidden, hidden).astype(np.float32))

    if has_gates:
        md = d / "mlp"
        md.mkdir()
        for i in range(n_layers):
            np.save(md / f"L{i:02d}_gate.npy",
                    np.random.randn(n_tok, intermediate).astype(np.float16))
            np.save(md / f"L{i:02d}_up.npy",
                    np.random.randn(n_tok, intermediate).astype(np.float16))

    if has_attention:
        ad = d / "attention"
        ad.mkdir()
        for i in range(n_layers):
            aw = np.random.rand(n_tok, n_heads, seq_len).astype(np.float16)
            aw = aw / aw.sum(axis=2, keepdims=True)
            np.save(ad / f"L{i:02d}_weights.npy", aw)
            np.save(ad / f"L{i:02d}_logits.npy",
                    np.random.randn(n_tok, n_heads, seq_len).astype(np.float16))

    return d


class TestLoadMri:

    def test_loads_exit_states(self, tmp_path):
        from heinrich.profile.mri import load_mri
        d = _make_mri(tmp_path)
        m = load_mri(str(d))
        assert "exit_L0" in m
        assert "exit_L1" in m
        assert m["exit_L0"].shape == (20, 8)

    def test_loads_gates(self, tmp_path):
        from heinrich.profile.mri import load_mri
        d = _make_mri(tmp_path, has_gates=True, intermediate=32)
        m = load_mri(str(d))
        assert "gate_L0" in m
        assert "up_L0" in m
        assert m["gate_L0"].shape == (20, 32)

    def test_loads_attention(self, tmp_path):
        from heinrich.profile.mri import load_mri
        d = _make_mri(tmp_path, has_attention=True, mode="template", seq_len=10)
        m = load_mri(str(d))
        assert "attn_weights_L0" in m
        assert m["attn_weights_L0"].shape == (20, 2, 10)

    def test_no_gates_when_absent(self, tmp_path):
        from heinrich.profile.mri import load_mri
        d = _make_mri(tmp_path, has_gates=False)
        m = load_mri(str(d))
        assert not any(k.startswith("gate_indices") for k in m)

    def test_vectors_compatibility_key(self, tmp_path):
        from heinrich.profile.mri import load_mri
        d = _make_mri(tmp_path)
        m = load_mri(str(d))
        assert "vectors" in m
        assert m["vectors"].shape == (20, 8)

    def test_not_a_directory(self):
        from heinrich.profile.mri import load_mri
        with pytest.raises(ValueError, match="Not an MRI directory"):
            load_mri("/nonexistent/path.mri")


class TestVerifyMriExtended:

    def test_valid_raw_mri(self, tmp_path):
        from heinrich.profile.mri import verify_mri
        d = _make_mri(tmp_path, has_gates=True)
        r = verify_mri(str(d))
        assert r["healthy"], r["issues"]

    def test_valid_template_mri(self, tmp_path):
        from heinrich.profile.mri import verify_mri
        d = _make_mri(tmp_path, mode="template", has_entry=True,
                      has_attention=True, has_gates=True, seq_len=10)
        r = verify_mri(str(d))
        assert r["healthy"], r["issues"]

    def test_missing_gate_file(self, tmp_path):
        from heinrich.profile.mri import verify_mri
        d = _make_mri(tmp_path, has_gates=True)
        (d / "mlp" / "L00_gate.npy").unlink()
        r = verify_mri(str(d))
        assert any("gate" in iss.lower() for iss in r["issues"])


class TestLayerDeltas:

    def test_basic_computation(self, tmp_path):
        from heinrich.profile.compare import layer_deltas
        d = _make_mri(tmp_path, n_tok=50, n_layers=3, hidden=16)
        r = layer_deltas(str(d))
        assert "error" not in r
        assert len(r["layers"]) == 3
        # L0 delta is just exit_L0 (no previous layer)
        assert r["layers"][0]["mean_delta_norm"] > 0
        # Amplification for L0 is always 1.0
        assert r["layers"][0]["amplification"] == 1.0

    def test_delta_is_difference(self, tmp_path):
        """Delta at layer i should be exit[i] - exit[i-1]."""
        from heinrich.profile.compare import layer_deltas
        d = _make_mri(tmp_path, n_tok=10, n_layers=2, hidden=4)
        # Overwrite with known values
        exit0 = np.ones((10, 4), dtype=np.float16) * 2.0
        exit1 = np.ones((10, 4), dtype=np.float16) * 5.0
        np.save(d / "L00_exit.npy", exit0)
        np.save(d / "L01_exit.npy", exit1)

        r = layer_deltas(str(d))
        # L1 delta = 5.0 - 2.0 = 3.0 per dim, norm = 3.0 * sqrt(4) = 6.0
        expected_norm = 3.0 * np.sqrt(4)
        assert abs(r["layers"][1]["mean_delta_norm"] - expected_norm) < 0.5


class TestLogitLens:

    def test_basic(self, tmp_path):
        from heinrich.profile.compare import logit_lens
        d = _make_mri(tmp_path, n_tok=30, n_layers=2, hidden=8, vocab=16)
        r = logit_lens(str(d), n_sample=10, layers=[0, 1])
        assert "error" not in r
        assert len(r["layers"]) == 2
        for lr in r["layers"]:
            assert len(lr["predictions"]) == 10
            for pred in lr["predictions"]:
                assert len(pred["top_ids"]) == 5

    def test_top_k_respected(self, tmp_path):
        from heinrich.profile.compare import logit_lens
        d = _make_mri(tmp_path, n_tok=10, n_layers=2, hidden=8, vocab=16)
        r = logit_lens(str(d), top_k=3, n_sample=5)
        for lr in r["layers"]:
            for pred in lr["predictions"]:
                assert len(pred["top_ids"]) == 3


class TestGateAnalysis:

    def test_basic(self, tmp_path):
        from heinrich.profile.compare import gate_analysis
        d = _make_mri(tmp_path, n_tok=100, n_layers=2, has_gates=True, intermediate=64)
        r = gate_analysis(str(d), n_sample=50)
        assert "error" not in r
        assert len(r["layers"]) == 2
        for lr in r["layers"]:
            assert lr["unique_neurons"] > 0
            assert lr["mean_activation"] > 0

    def test_no_gates(self, tmp_path):
        from heinrich.profile.compare import gate_analysis
        d = _make_mri(tmp_path, has_gates=False)
        r = gate_analysis(str(d))
        assert "error" in r


class TestAttentionAnalysis:

    def test_basic(self, tmp_path):
        from heinrich.profile.compare import attention_analysis
        d = _make_mri(tmp_path, n_tok=50, n_layers=2, mode="template",
                      has_entry=True, has_attention=True, seq_len=10, n_heads=2)
        r = attention_analysis(str(d), n_sample=20)
        assert "error" not in r
        assert len(r["layers"]) == 2
        for lr in r["layers"]:
            total = lr["self_weight"] + lr["prefix_weight"] + lr["suffix_weight"]
            assert 0.9 < total < 1.1

    def test_no_attention(self, tmp_path):
        from heinrich.profile.compare import attention_analysis
        d = _make_mri(tmp_path, has_attention=False)
        r = attention_analysis(str(d))
        assert "error" in r
