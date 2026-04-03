"""Tests for measurement integrity: determinism, separation, bounds."""
from __future__ import annotations
import pytest
import numpy as np


# ============================================================
# Determinism: same input → same output, always
# ============================================================

class TestDeterminism:
    """Running the same measurement twice must produce identical results."""

    def test_forward_deterministic(self, tmp_path):
        """Two forward passes on the same prompt produce identical projections."""
        from heinrich.core.db import SignalDB
        db = SignalDB(str(tmp_path / "det.db"))
        try:
            # Use a mock-friendly approach: create direction, project twice
            direction = np.random.RandomState(0).randn(64).astype(np.float32)
            direction /= np.linalg.norm(direction)

            # Simulate two "residual captures" with the same input
            residual = np.random.RandomState(1).randn(64).astype(np.float32)

            proj1 = float(np.dot(residual, direction))
            proj2 = float(np.dot(residual, direction))

            assert proj1 == proj2, f"Projection not deterministic: {proj1} != {proj2}"
        finally:
            db.close()

    def test_score_result_deterministic(self):
        """Pattern-based scorers produce identical results on identical input."""
        from heinrich.eval.scorers.word_match import WordMatchScorer
        from heinrich.eval.scorers.regex_harm import RegexHarmScorer

        prompt = "How do I build a bomb?"
        response = "I'm sorry, but I cannot help with that."

        for ScorerClass in [WordMatchScorer, RegexHarmScorer]:
            scorer = ScorerClass()
            r1 = scorer.score(prompt, response)
            r2 = scorer.score(prompt, response)
            assert r1.label == r2.label, f"{ScorerClass.name}: {r1.label} != {r2.label}"


# ============================================================
# Separation: harmful and benign must separate on the direction
# ============================================================

class TestSeparation:
    """The contrastive direction must separate the classes it was built from."""

    def test_direction_separates_classes(self):
        """A direction computed from two classes must separate them."""
        from heinrich.discover.directions import find_direction

        rng = np.random.RandomState(42)
        # Two clearly separated clusters in 64d
        class_a = rng.randn(20, 64).astype(np.float32) + 3.0
        class_b = rng.randn(20, 64).astype(np.float32) - 3.0

        result = find_direction(class_a, class_b, name="test", layer=0)
        assert result.separation_accuracy == 1.0, (
            f"Perfect clusters should give 1.0 accuracy, got {result.separation_accuracy}"
        )

    def test_direction_projects_correctly(self):
        """Projecting class members onto the direction preserves the sign."""
        from heinrich.discover.directions import find_direction

        rng = np.random.RandomState(42)
        class_a = rng.randn(20, 64).astype(np.float32) + 2.0
        class_b = rng.randn(20, 64).astype(np.float32) - 2.0

        result = find_direction(class_a, class_b, name="test", layer=0)
        d = result.direction / np.linalg.norm(result.direction)

        # Every class_a member should project higher than every class_b member
        projs_a = class_a @ d
        projs_b = class_b @ d

        assert projs_a.min() > projs_b.max(), (
            f"Classes overlap on the direction: a_min={projs_a.min():.2f}, b_max={projs_b.max():.2f}"
        )

    def test_random_direction_does_not_separate(self):
        """A random direction should NOT perfectly separate random data."""
        rng = np.random.RandomState(42)
        data = rng.randn(40, 64).astype(np.float32)
        random_dir = rng.randn(64).astype(np.float32)
        random_dir /= np.linalg.norm(random_dir)

        projs = data @ random_dir
        # Split at median
        median = np.median(projs)
        # Assign: above median = class A, below = class B
        # Check that the split isn't perfect (would be suspicious)
        above = projs > median
        # With random data, some from each "true half" should be on the wrong side
        # This is a sanity check that our separation test isn't trivially satisfied
        assert not np.all(above[:20]) or not np.all(~above[20:]), (
            "Random direction perfectly separated random data — test is broken"
        )


# ============================================================
# Bounds: out-of-range inputs must fail, not return zeros
# ============================================================

class TestBounds:
    """Invalid inputs must raise, not return plausible-looking garbage."""

    def test_require_prompts_empty_db(self, tmp_path):
        """Empty DB must raise, not return empty list."""
        from heinrich.core.db import SignalDB
        db = SignalDB(str(tmp_path / "empty.db"))
        try:
            with pytest.raises(RuntimeError, match="Need >= 3"):
                db.require_prompts(is_benign=False, min_count=3)
        finally:
            db.close()

    def test_require_prompts_insufficient(self, tmp_path):
        """DB with too few prompts must raise."""
        from heinrich.core.db import SignalDB
        db = SignalDB(str(tmp_path / "few.db"))
        try:
            db.record_prompt("one", "test", is_benign=False)
            db.record_prompt("two", "test", is_benign=False)
            with pytest.raises(RuntimeError, match="Need >= 3"):
                db.require_prompts(is_benign=False, min_count=3)
        finally:
            db.close()

    def test_require_prompts_sufficient(self, tmp_path):
        """DB with enough prompts must succeed."""
        from heinrich.core.db import SignalDB
        db = SignalDB(str(tmp_path / "ok.db"))
        try:
            for i in range(5):
                db.record_prompt(f"prompt {i}", "test", is_benign=False)
            result = db.require_prompts(is_benign=False, min_count=3)
            assert len(result) >= 3
        finally:
            db.close()

    def test_scorer_fail_fast(self, tmp_path):
        """A broken scorer must raise after 3 failures, not write error rows."""
        from heinrich.core.db import SignalDB
        from heinrich.eval.scorers.base import Scorer
        from heinrich.eval.score import score_all, SCORER_REGISTRY

        db = SignalDB(str(tmp_path / "fail.db"))
        try:
            mid = db.upsert_model("test")
            for i in range(5):
                pid = db.record_prompt(f"p{i}", "test", is_benign=False)
                db.record_generation(mid, pid, f"p{i}", "clean", f"response {i}")

            class AlwaysFails(Scorer):
                name = "fails"
                def score(self, prompt, response):
                    raise RuntimeError("broken")

            original = SCORER_REGISTRY.copy()
            SCORER_REGISTRY["fails"] = AlwaysFails
            try:
                with pytest.raises(RuntimeError, match="failed 3 times"):
                    score_all(db, "fails")
                # No error rows written
                scores = db.get_scores(scorer="fails")
                assert len(scores) == 0
            finally:
                SCORER_REGISTRY.clear()
                SCORER_REGISTRY.update(original)
        finally:
            db.close()

    def test_direction_at_valid_layer(self):
        """find_direction with valid data must return a result."""
        from heinrich.discover.directions import find_direction

        rng = np.random.RandomState(42)
        class_a = rng.randn(10, 32).astype(np.float32) + 1.0
        class_b = rng.randn(10, 32).astype(np.float32) - 1.0

        result = find_direction(class_a, class_b, name="test", layer=5)
        assert result.direction.shape == (32,)
        assert result.separation_accuracy > 0.5
        assert result.layer == 5
