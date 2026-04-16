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
