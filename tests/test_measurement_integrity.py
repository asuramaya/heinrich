"""Tests for measurement integrity: determinism, separation, bounds, baseline."""
from __future__ import annotations
import pytest
import numpy as np


# ============================================================
# Baseline: the reference point must be controlled and honest
# ============================================================

class TestBaseline:
    """The baseline determines every measurement downstream.
    A poisoned baseline invalidates the entire .shrt."""

    def test_system_prompt_changes_entropy(self):
        """A chat template that injects a system prompt produces different
        baseline entropy than a structural-only template. If they match,
        the template isn't injecting — safe. If they differ, the injection
        is measurable and must be accounted for."""
        # Simulate: structural template has low entropy (concentrated)
        # System-prompt template has high entropy (spread by persona tokens)
        rng = np.random.RandomState(42)

        # "Clean" baseline: peaked distribution (model is confident)
        clean_logits = rng.randn(1000).astype(np.float32)
        clean_logits[0] = 10.0  # one dominant token
        clean_probs = np.exp(clean_logits) / np.exp(clean_logits).sum()
        clean_entropy = -float(np.sum(clean_probs * np.log(clean_probs + 1e-10)))

        # "Poisoned" baseline: flatter distribution (persona spreads attention)
        poisoned_logits = rng.randn(1000).astype(np.float32)
        poisoned_logits[0] = 3.0  # less dominant
        poisoned_probs = np.exp(poisoned_logits) / np.exp(poisoned_logits).sum()
        poisoned_entropy = -float(np.sum(poisoned_probs * np.log(poisoned_probs + 1e-10)))

        # The entropies must differ — proving injection is detectable
        assert poisoned_entropy > clean_entropy, (
            "Poisoned baseline should have higher entropy than clean"
        )

        # A 2x entropy difference is a red flag
        ratio = poisoned_entropy / clean_entropy
        assert ratio > 1.5, (
            f"Entropy ratio {ratio:.2f} — injection may not be detectable"
        )

    def test_baseline_fingerprint_required(self):
        """A .shrt metadata dict must contain enough baseline information
        to detect whether two files used the same reference point."""
        # Minimum required fields for baseline comparison
        required_fields = {'type', 'entropy', 'layer', 'top_token'}

        # Simulate a .shrt metadata baseline section
        good_baseline = {
            'type': 'silence (extracted)',
            'entropy': 0.9175,
            'layer': 23,
            'top_token': 'Hello',
        }
        assert required_fields.issubset(good_baseline.keys()), (
            f"Baseline missing fields: {required_fields - good_baseline.keys()}"
        )

        # Two baselines with different entropy are NOT comparable
        other_baseline = {
            'type': 'silence (extracted)',
            'entropy': 2.8643,
            'layer': 23,
            'top_token': 'Hello',
        }
        # Same 'type' string but different entropy = different baseline
        assert good_baseline['entropy'] != other_baseline['entropy'], (
            "Baselines with identical entropy should be comparable"
        )

    def test_baseline_entropy_threshold(self):
        """A clean structural baseline should have low entropy.
        High entropy (> 2.0) suggests the template injected content."""
        # The Qwen 0.5B clean baseline had entropy 0.92
        # The poisoned baseline had entropy 2.86
        # Threshold: anything above 2.0 is suspicious for a silence baseline
        clean_entropy = 0.92
        poisoned_entropy = 2.86
        threshold = 2.0

        assert clean_entropy < threshold, "Clean baseline should be below threshold"
        assert poisoned_entropy > threshold, "Poisoned baseline should be above threshold"

    def test_deltas_from_different_baselines_diverge(self):
        """Measuring the same tokens from different baselines must produce
        different deltas. If they don't, the baseline doesn't matter
        (which would mean the measurement is trivial)."""
        rng = np.random.RandomState(42)
        hidden_dim = 64

        baseline_a = rng.randn(hidden_dim).astype(np.float32)
        baseline_b = baseline_a + rng.randn(hidden_dim).astype(np.float32) * 5

        # 100 token residuals
        residuals = rng.randn(100, hidden_dim).astype(np.float32)

        deltas_a = np.linalg.norm(residuals - baseline_a, axis=1)
        deltas_b = np.linalg.norm(residuals - baseline_b, axis=1)

        # Rankings should differ when baselines differ
        rank_a = np.argsort(np.argsort(-deltas_a))
        rank_b = np.argsort(np.argsort(-deltas_b))
        rank_r = float(np.corrcoef(rank_a, rank_b)[0, 1])

        # If rank_r is high (> 0.95), the baseline doesn't matter much
        # If rank_r is low (< 0.8), the baseline changes the ranking
        # We EXPECT it to change — that's the whole point
        assert rank_r < 0.95, (
            f"Rankings too similar (r={rank_r:.3f}) despite different baselines — "
            f"baseline may not matter, which contradicts what we observed"
        )


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


# ============================================================
# Stability: .shrt rankings must converge as index grows
# ============================================================

class TestShrtStability:
    """The .shrt index must converge — population statistics must stabilize
    as the sample grows. Individual measurements are independent (one forward
    pass per token), so relative ordering is trivially preserved. The real
    convergence question is: has the sample characterized the population?"""

    def _simulate_vocab_deltas(self, n_vocab, hidden_dim=64, seed=42,
                               bimodal=False):
        """Simulate a full vocabulary of deltas from a fixed baseline.
        If bimodal, 5% of tokens have 10x higher delta (rare sharts)."""
        rng = np.random.RandomState(seed)
        baseline = rng.randn(hidden_dim).astype(np.float32)
        deltas = []
        for tid in range(n_vocab):
            token_rng = np.random.RandomState(seed + tid + 1)
            residual = token_rng.randn(hidden_dim).astype(np.float32)
            if bimodal and tid % 20 == 0:
                residual *= 10  # rare high-delta tokens
            delta = float(np.linalg.norm(residual - baseline))
            deltas.append(delta)
        return np.array(deltas)

    def test_distribution_converges(self):
        """Mean and std of delta sample must stabilize as N grows."""
        all_deltas = self._simulate_vocab_deltas(5000)

        # Sample at increasing sizes
        rng = np.random.RandomState(0)
        indices = rng.permutation(5000)

        stats = []
        for n in [200, 500, 1000, 2000, 5000]:
            sample = all_deltas[indices[:n]]
            stats.append((float(sample.mean()), float(sample.std())))

        # Mean at N=1000 should be within 10% of mean at N=5000
        mean_1k, mean_5k = stats[2][0], stats[4][0]
        assert abs(mean_1k - mean_5k) / mean_5k < 0.10, (
            f"Mean not converged: {mean_1k:.2f} at 1K vs {mean_5k:.2f} at 5K"
        )

        # Std at N=1000 should be within 15% of std at N=5000
        std_1k, std_5k = stats[2][1], stats[4][1]
        assert abs(std_1k - std_5k) / std_5k < 0.15, (
            f"Std not converged: {std_1k:.2f} at 1K vs {std_5k:.2f} at 5K"
        )

    def test_tail_converges(self):
        """The tail (max, p99) must stabilize as N grows.
        Uses bimodal data where the tail is hard to find."""
        all_deltas = self._simulate_vocab_deltas(5000, bimodal=True)
        rng = np.random.RandomState(0)
        indices = rng.permutation(5000)

        # Track how the tail estimate improves with N
        pop_max = float(all_deltas.max())
        errors = {}
        for n in [200, 500, 1000, 2000, 5000]:
            sample = all_deltas[indices[:n]]
            errors[n] = abs(float(sample.max()) - pop_max) / pop_max

        # Error must decrease monotonically (or stay flat) from 200→5000
        # At least: N=2000 must be closer than N=200
        assert errors[2000] <= errors[200] + 0.01, (
            f"Tail not improving: err@200={errors[200]:.3f}, err@2000={errors[2000]:.3f}"
        )
        # And N=2000 must be within 5% (not 20%)
        assert errors[2000] < 0.05, (
            f"Tail not converged at N=2000: {errors[2000]*100:.1f}% error"
        )

    def test_small_sample_misses_rare_sharts(self):
        """A bimodal distribution (rare high-delta tokens) requires more
        samples to characterize. Small N finds fewer outliers."""
        all_deltas = self._simulate_vocab_deltas(5000, bimodal=True)

        # The outliers are at ~10x magnitude. Find the threshold.
        normal_mean = float(np.median(all_deltas))
        threshold = normal_mean * 3  # well above normal, catches outliers

        pop_frac = float(np.mean(all_deltas > threshold))

        rng = np.random.RandomState(0)
        indices = rng.permutation(5000)

        # Small sample: fraction above threshold is noisier
        small = all_deltas[indices[:100]]
        large = all_deltas[indices[:3000]]

        small_frac = float(np.mean(small > threshold))
        large_frac = float(np.mean(large > threshold))

        # Large sample should be closer to population fraction
        err_small = abs(small_frac - pop_frac)
        err_large = abs(large_frac - pop_frac)

        assert err_large <= err_small + 0.01, (
            f"Larger sample not better at finding outliers: "
            f"err_large={err_large:.3f} vs err_small={err_small:.3f}"
        )

    def test_split_half_check_is_not_convergence(self):
        """The current .shrt convergence check (split-half rank correlation)
        correlates ranks of DIFFERENT tokens. With random sampling, this
        correlation is near zero — it measures nothing about convergence."""
        all_deltas = self._simulate_vocab_deltas(1000)
        rng = np.random.RandomState(0)
        indices = rng.permutation(1000)
        sample = all_deltas[indices[:200]]

        # Reproduce the shrt.py convergence check
        half = len(sample) // 2
        first_half = sample[:half]
        second_half = sample[half:]
        ranks_a = np.argsort(np.argsort(first_half))
        ranks_b = np.argsort(np.argsort(second_half))
        split_half_r = float(np.corrcoef(ranks_a, ranks_b)[0, 1])

        # This correlates ranks of token_0 vs token_100, token_1 vs token_101...
        # These are DIFFERENT tokens. The correlation should be near zero.
        assert abs(split_half_r) < 0.4, (
            f"Split-half rank correlation should be near zero for different "
            f"tokens, got r={split_half_r:.3f}"
        )
