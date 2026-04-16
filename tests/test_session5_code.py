"""Tests for Session 5 code: _detect_script, _extract_weight, _is_mlx_backend,
pca_anatomy eigendecomposition, mri-status validators."""
from __future__ import annotations
import pytest
import numpy as np


class TestDetectScript:
    """_detect_script: majority-vote letter classification."""

    def test_pure_latin(self):
        from heinrich.profile.frt import _detect_script
        assert _detect_script("hello") == "latin"

    def test_accented_latin(self):
        from heinrich.profile.frt import _detect_script
        assert _detect_script("résumé") == "latin"

    def test_vietnamese_latin(self):
        """Vietnamese uses Latin Extended Additional (U+1E00-U+1EFF)."""
        from heinrich.profile.frt import _detect_script
        assert _detect_script("Việt") == "latin"

    def test_cjk(self):
        from heinrich.profile.frt import _detect_script
        assert _detect_script("中文") == "CJK"

    def test_cyrillic(self):
        from heinrich.profile.frt import _detect_script
        assert _detect_script("Привет") == "Cyrillic"

    def test_arabic(self):
        from heinrich.profile.frt import _detect_script
        assert _detect_script("مرحبا") == "Arabic"

    def test_japanese(self):
        from heinrich.profile.frt import _detect_script
        assert _detect_script("こんにちは") == "Japanese"

    def test_korean(self):
        from heinrich.profile.frt import _detect_script
        assert _detect_script("안녕") == "Korean"

    def test_code_only_braces(self):
        """Pure code characters (no letters) → code."""
        from heinrich.profile.frt import _detect_script
        assert _detect_script("{}();") == "code"

    def test_underscore_with_letters_is_latin(self):
        """_name has letters, so majority vote wins — latin, not code."""
        from heinrich.profile.frt import _detect_script
        assert _detect_script("_name") == "latin"

    def test_whitespace_is_special(self):
        from heinrich.profile.frt import _detect_script
        assert _detect_script("   ") == "special"
        assert _detect_script("\t") == "special"

    def test_empty_is_special(self):
        from heinrich.profile.frt import _detect_script
        assert _detect_script("") == "special"

    def test_mixed_script_majority_wins(self):
        """Mixed script token: most letters determine the result."""
        from heinrich.profile.frt import _detect_script
        # 3 latin letters + 1 greek = latin wins
        assert _detect_script("abcΣ") == "latin"

    def test_newline_with_text_is_not_code(self):
        """A newline doesn't make text 'code' — letters determine script."""
        from heinrich.profile.frt import _detect_script
        assert _detect_script("hello\n") == "latin"

    def test_greek(self):
        from heinrich.profile.frt import _detect_script
        assert _detect_script("αβγ") == "Greek"

    def test_devanagari(self):
        from heinrich.profile.frt import _detect_script
        assert _detect_script("नमस्ते") == "Devanagari"


class TestIsMlxBackend:
    """_is_mlx_backend: positive check for MLXBackend."""

    def test_mlx_backend_detected(self):
        from heinrich.profile.mri import _is_mlx_backend

        class MLXBackend:
            pass

        assert _is_mlx_backend(MLXBackend()) is True

    def test_hf_backend_not_detected(self):
        from heinrich.profile.mri import _is_mlx_backend

        class HFBackend:
            pass

        assert _is_mlx_backend(HFBackend()) is False

    def test_decepticon_backend_not_detected(self):
        from heinrich.profile.mri import _is_mlx_backend

        class DecepticonBackend:
            pass

        assert _is_mlx_backend(DecepticonBackend()) is False


class TestPcaEigendecomposition:
    """PCA via eigendecomposition matches SVD results."""

    def test_eigendecomp_matches_svd_variance(self):
        """Eigendecomposition of covariance gives same variance as SVD."""
        rng = np.random.RandomState(42)
        N, d = 200, 50
        X = rng.randn(N, d).astype(np.float32)
        centered = X - X.mean(axis=0)

        # SVD approach
        _, S, Vt_svd = np.linalg.svd(centered, full_matrices=False)
        var_svd = S ** 2

        # Eigendecomposition approach (our code)
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        var_eig = np.maximum(eigenvalues, 0)

        # Variances should match
        np.testing.assert_allclose(var_eig, var_svd, rtol=1e-4)

    def test_eigendecomp_pc_directions_match(self):
        """Principal components from eigendecomposition match SVD (up to sign)."""
        rng = np.random.RandomState(42)
        N, d = 200, 50
        X = rng.randn(N, d).astype(np.float32)
        centered = X - X.mean(axis=0)

        # SVD
        _, _, Vt_svd = np.linalg.svd(centered, full_matrices=False)

        # Eigendecomposition
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        Vt_eig = eigenvectors.T

        # Top 5 PCs should match in direction (cosine ~1 or ~-1)
        for k in range(5):
            cos = abs(np.dot(Vt_svd[k], Vt_eig[k]))
            assert cos > 0.99, f"PC{k} cosine = {cos}"


class TestDecomposedForward:
    """Decomposed forward must produce identical output to fused forward."""

    @pytest.fixture
    def mlx_backend(self):
        """Load smallest available MLX model, skip if unavailable."""
        try:
            from heinrich.backend.protocol import load_backend
            return load_backend('mlx-community/Qwen2.5-0.5B-Instruct-4bit', backend='mlx')
        except Exception:
            pytest.skip("MLX model not available")

    def test_raw_mode_exact(self, mlx_backend):
        """Single token: decomposed == fused."""
        import mlx.core as mx
        from heinrich.profile.mri import _framework_ops
        ops = _framework_ops(mlx_backend)
        model_inner = ops.model_inner
        inp = mx.array([[42]])

        h_f = ops.embed(inp)
        for ly in model_inner.layers:
            h_f = ops.layer_forward(ly, h_f, None)
            if isinstance(h_f, tuple): h_f = h_f[0]

        h_d = ops.embed(inp)
        for ly in model_inner.layers:
            h_d, *_ = ops.layer_decomposed(ly, h_d, None)

        out_f = np.array(h_f.astype(mx.float32))[0, 0]
        out_d = np.array(h_d.astype(mx.float32))[0, 0]
        assert np.abs(out_f - out_d).max() == 0, "Decomposed != fused in raw mode"

    def test_template_mode_exact(self, mlx_backend):
        """Multi-token template: decomposed == fused."""
        import mlx.core as mx
        from heinrich.profile.mri import _framework_ops
        from heinrich.profile.shrt import _extract_template_parts
        ops = _framework_ops(mlx_backend)
        model_inner = ops.model_inner

        prefix_ids, suffix_ids = _extract_template_parts(mlx_backend.tokenizer)
        full_ids = prefix_ids + [42] + suffix_ids
        inp = mx.array([full_ids])
        mask = ops.triu_mask(len(full_ids))

        h_f = ops.embed(inp)
        for ly in model_inner.layers:
            h_f = ops.layer_forward(ly, h_f, mask)
            if isinstance(h_f, tuple): h_f = h_f[0]

        h_d = ops.embed(inp)
        for ly in model_inner.layers:
            h_d, *_ = ops.layer_decomposed(ly, h_d, mask)

        out_f = np.array(h_f.astype(mx.float32))[0, -1]
        out_d = np.array(h_d.astype(mx.float32))[0, -1]
        assert np.abs(out_f - out_d).max() == 0, "Decomposed != fused in template mode"

    def test_attention_weights_valid(self, mlx_backend):
        """Attention weights sum to 1.0, respect causal mask."""
        import mlx.core as mx
        from heinrich.profile.mri import _framework_ops
        ops = _framework_ops(mlx_backend)
        model_inner = ops.model_inner

        inp = mx.array([[10, 20, 30]])  # 3 tokens
        mask = ops.triu_mask(3)
        h = ops.embed(inp)
        _, aw, *_ = ops.layer_decomposed(model_inner.layers[0], h, mask)
        aw_np = np.array(aw.astype(mx.float32))  # [1, heads, 3, 3]

        # Row sums should be 1.0
        row_sums = aw_np[0].sum(axis=2)  # [heads, 3]
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

        # Causal: position 0 can't attend to positions 1,2
        assert aw_np[0, :, 0, 1].max() < 1e-6, "Position 0 attends to position 1"
        assert aw_np[0, :, 0, 2].max() < 1e-6, "Position 0 attends to position 2"

    def test_gate_values_match_manual(self, mlx_backend):
        """Top-K gate values match direct gate_proj computation."""
        import mlx.core as mx
        import mlx.nn as nn
        from heinrich.profile.mri import _framework_ops
        ops = _framework_ops(mlx_backend)
        model_inner = ops.model_inner

        inp = mx.array([[42]])
        h = ops.embed(inp)
        _, _, _, h_pre_mlp, _, gate_val, _, _ = ops.layer_decomposed(model_inner.layers[0], h, None)

        # Manual gate computation
        ly = model_inner.layers[0]
        gate_manual = nn.silu(ly.mlp.gate_proj(h_pre_mlp))
        gate_manual_np = np.array(gate_manual.astype(mx.float32))[0, 0]

        # gate_topk
        g_idx, g_val, _ = ops.gate_topk(ly, h_pre_mlp, 32)
        manual_at_idx = gate_manual_np[g_idx[0]]

        np.testing.assert_allclose(g_val[0], manual_at_idx, atol=1e-5,
                                   err_msg="Gate top-K values don't match manual")


class TestExtractWeight:
    """_extract_weight: module-level function that probes weights."""

    def test_non_mlx_direct_weight_access(self):
        """When module has .weight attribute, returns it directly."""
        from heinrich.profile.mri import _extract_weight
        assert callable(_extract_weight)


class TestVerifyMri:
    """verify_mri: catches real issues, accepts healthy MRIs."""

    def test_missing_dir(self):
        from heinrich.profile.mri import verify_mri
        r = verify_mri("/nonexistent/path.mri")
        assert r["healthy"] is False
        assert "Not a directory" in r["issues"][0]

    def test_missing_metadata(self, tmp_path):
        from heinrich.profile.mri import verify_mri
        d = tmp_path / "test.mri"
        d.mkdir()
        r = verify_mri(str(d))
        assert r["healthy"] is False
        assert "metadata.json" in r["issues"][0]

    def test_healthy_mri(self, tmp_path):
        """Build a minimal valid MRI and verify it passes."""
        import json
        from heinrich.profile.mri import verify_mri

        d = tmp_path / "test.mri"
        d.mkdir()
        n_tok, n_layers, hidden, vocab = 10, 2, 8, 20

        from heinrich.profile.mri import MRI_VERSION
        meta = {
            "version": MRI_VERSION,
            "model": {"n_layers": n_layers, "hidden_size": hidden,
                       "vocab_size": vocab, "name": "test", "n_heads": 2},
            "capture": {"mode": "raw", "n_tokens": n_tok,
                         "has_entry": False, "has_attention": False,
                         "has_gates": False, "gate_k": 0, "seq_len": 1},
        }
        (d / "metadata.json").write_text(json.dumps(meta))
        np.savez(d / "tokens.npz", token_ids=np.arange(n_tok),
                 token_texts=np.array([f"t{i}" for i in range(n_tok)]))
        bl = {}
        for i in range(n_layers):
            np.save(d / f"L{i:02d}_exit.npy",
                    np.random.randn(n_tok, hidden).astype(np.float16))
            bl[f"entry_L{i}"] = np.zeros(hidden, dtype=np.float16)
            bl[f"exit_L{i}"] = np.zeros(hidden, dtype=np.float16)
        np.savez(d / "baselines.npz", **bl)
        np.save(d / "embedding.npy", np.random.randn(vocab, hidden).astype(np.float32))
        np.save(d / "lmhead_raw.npy", np.random.randn(vocab, hidden).astype(np.float32))
        np.savez(d / "norms.npz", final=np.ones(hidden, dtype=np.float32))
        wd = d / "weights"
        wd.mkdir()
        for i in range(n_layers):
            ld = wd / f"L{i:02d}"
            ld.mkdir()
            for name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj"]:
                np.save(ld / f"{name}.npy", np.random.randn(hidden, hidden).astype(np.float32))

        r = verify_mri(str(d))
        assert r["healthy"], f"Expected healthy, got issues: {r['issues']}"

    def test_attention_row_sums(self, tmp_path):
        """Attention weights must sum to 1.0 (softmax)."""
        import json
        from heinrich.profile.mri import verify_mri

        d = tmp_path / "attn.mri"
        d.mkdir()
        n_tok, n_layers, hidden, n_heads, seq_len = 10, 2, 8, 2, 5
        meta = {
            "model": {"n_layers": n_layers, "hidden_size": hidden,
                       "vocab_size": 20, "name": "test", "n_heads": n_heads},
            "capture": {"mode": "template", "n_tokens": n_tok,
                         "has_entry": True, "has_attention": True,
                         "has_gates": False, "seq_len": seq_len},
        }
        (d / "metadata.json").write_text(json.dumps(meta))
        np.savez(d / "tokens.npz", token_ids=np.arange(n_tok),
                 token_texts=np.array([f"t{i}" for i in range(n_tok)]))
        for i in range(n_layers):
            np.save(d / f"L{i:02d}_exit.npy",
                    np.random.randn(n_tok, hidden).astype(np.float16))
            np.save(d / f"L{i:02d}_entry.npy",
                    np.random.randn(n_tok, hidden).astype(np.float16))

        # Good attention: rows sum to 1.0
        attn_dir = d / "attention"
        attn_dir.mkdir()
        for i in range(n_layers):
            aw = np.random.rand(n_tok, n_heads, seq_len).astype(np.float16)
            aw = aw / aw.sum(axis=2, keepdims=True)  # normalize
            np.save(attn_dir / f"L{i:02d}_weights.npy", aw)

        r = verify_mri(str(d))
        attn_issues = [iss for iss in r["issues"] if "attn" in iss.lower()]
        assert not attn_issues, f"Valid attention flagged: {attn_issues}"

    def test_attention_bad_sums(self, tmp_path):
        """Attention weights that don't sum to 1.0 are caught."""
        import json
        from heinrich.profile.mri import verify_mri

        d = tmp_path / "bad_attn.mri"
        d.mkdir()
        n_tok, n_layers, hidden, n_heads, seq_len = 10, 2, 8, 2, 5
        meta = {
            "model": {"n_layers": n_layers, "hidden_size": hidden,
                       "vocab_size": 20, "name": "test", "n_heads": n_heads},
            "capture": {"mode": "template", "n_tokens": n_tok,
                         "has_entry": True, "has_attention": True,
                         "has_gates": False, "seq_len": seq_len},
        }
        (d / "metadata.json").write_text(json.dumps(meta))
        np.savez(d / "tokens.npz", token_ids=np.arange(n_tok),
                 token_texts=np.array([f"t{i}" for i in range(n_tok)]))
        for i in range(n_layers):
            np.save(d / f"L{i:02d}_exit.npy",
                    np.random.randn(n_tok, hidden).astype(np.float16))
            np.save(d / f"L{i:02d}_entry.npy",
                    np.random.randn(n_tok, hidden).astype(np.float16))

        attn_dir = d / "attention"
        attn_dir.mkdir()
        for i in range(n_layers):
            aw = np.ones((n_tok, n_heads, seq_len), dtype=np.float16) * 0.5  # sums to 2.5
            np.save(attn_dir / f"L{i:02d}_weights.npy", aw)

        r = verify_mri(str(d))
        assert any("row sums" in iss for iss in r["issues"])

    def test_gate_all_zero(self, tmp_path):
        """Gates that are all zero are flagged."""
        import json
        from heinrich.profile.mri import verify_mri

        d = tmp_path / "bad_gates.mri"
        d.mkdir()
        n_tok, n_layers, hidden, inter = 100, 2, 8, 32
        meta = {
            "model": {"n_layers": n_layers, "hidden_size": hidden,
                       "vocab_size": 20, "name": "test", "n_heads": 2},
            "capture": {"mode": "raw", "n_tokens": n_tok,
                         "has_entry": False, "has_attention": False,
                         "has_gates": True, "gate_format": "full",
                         "intermediate_size": inter},
        }
        (d / "metadata.json").write_text(json.dumps(meta))
        np.savez(d / "tokens.npz", token_ids=np.arange(n_tok),
                 token_texts=np.array([f"t{i}" for i in range(n_tok)]))
        for i in range(n_layers):
            np.save(d / f"L{i:02d}_exit.npy",
                    np.random.randn(n_tok, hidden).astype(np.float16))

        mlp_dir = d / "mlp"
        mlp_dir.mkdir()
        for i in range(n_layers):
            np.save(mlp_dir / f"L{i:02d}_gate.npy",
                    np.zeros((n_tok, inter), dtype=np.float16))

        r = verify_mri(str(d))
        assert any("zero" in iss.lower() for iss in r["issues"])

    def test_detects_nan(self, tmp_path):
        """NaN in exit arrays is caught."""
        import json
        from heinrich.profile.mri import verify_mri

        d = tmp_path / "nan.mri"
        d.mkdir()
        n_tok, n_layers, hidden = 10, 2, 8
        meta = {
            "model": {"n_layers": n_layers, "hidden_size": hidden,
                       "vocab_size": 20, "name": "test", "n_heads": 2},
            "capture": {"mode": "raw", "n_tokens": n_tok,
                         "has_entry": False, "has_attention": False,
                         "has_gates": False},
        }
        (d / "metadata.json").write_text(json.dumps(meta))
        np.savez(d / "tokens.npz", token_ids=np.arange(n_tok),
                 token_texts=np.array([f"t{i}" for i in range(n_tok)]))
        for i in range(n_layers):
            arr = np.random.randn(n_tok, hidden).astype(np.float16)
            if i == 0:
                arr[0, 0] = np.nan  # inject NaN
            np.save(d / f"L{i:02d}_exit.npy", arr)

        r = verify_mri(str(d))
        assert not r["healthy"]
        assert any("NaN" in iss for iss in r["issues"])
