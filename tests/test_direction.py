"""Tests for full-K direction analysis."""
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
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


def test_direction_nonlinear_linear_data():
    """A perfectly linear split should have knn ~= linear accuracy."""
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


def test_steer_test_returns_structure():
    """Steer test should return clean/steered outputs and a change metric."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scores = _make_fake_scores(100, 50, tmpdir)

        # Create fake components file (needed to map PC->hidden space)
        decomp = Path(tmpdir) / "decomp"
        hidden_dim = 128
        components = np.random.randn(50, hidden_dim).astype(np.float32)
        np.save(str(decomp / "L00_components.npy"), components)

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
            assert result["clean"] == "The king was powerful"
            assert result["steered"] == "The queen was beautiful"
            # Verify backend.generate was called twice (clean + steered)
            assert mock_backend.generate.call_count == 2
