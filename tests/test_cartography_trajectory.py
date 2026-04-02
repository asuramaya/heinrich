"""Tests for heinrich.cartography.trajectory."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from heinrich.cartography.trajectory import (
    TrajectoryPoint,
    find_compliance_emergence,
    trajectory_trend,
    self_feed_trajectory,
)


# ---------------------------------------------------------------------------
# TrajectoryPoint dataclass
# ---------------------------------------------------------------------------

class TestTrajectoryPoint:
    def test_construction(self):
        pt = TrajectoryPoint(
            round=0,
            cos_target=0.85,
            cos_baseline=0.12,
            top_token="Sure",
            response="Sure, here is how...",
            has_refuse=False,
            has_comply=True,
            n_tokens=42,
        )
        assert pt.round == 0
        assert pt.cos_target == 0.85
        assert pt.cos_baseline == 0.12
        assert pt.top_token == "Sure"
        assert pt.response == "Sure, here is how..."
        assert pt.has_refuse is False
        assert pt.has_comply is True
        assert pt.n_tokens == 42

    def test_equality(self):
        kwargs = dict(
            round=1, cos_target=0.5, cos_baseline=0.3,
            top_token="I", response="I cannot", has_refuse=True,
            has_comply=False, n_tokens=10,
        )
        assert TrajectoryPoint(**kwargs) == TrajectoryPoint(**kwargs)

    def test_different_points_not_equal(self):
        base = dict(
            round=0, cos_target=0.5, cos_baseline=0.3,
            top_token="I", response="I cannot", has_refuse=True,
            has_comply=False, n_tokens=10,
        )
        a = TrajectoryPoint(**base)
        b = TrajectoryPoint(**{**base, "round": 1})
        assert a != b


# ---------------------------------------------------------------------------
# find_compliance_emergence
# ---------------------------------------------------------------------------

class TestFindComplianceEmergence:
    def _pt(self, round: int, has_comply: bool, has_refuse: bool) -> TrajectoryPoint:
        return TrajectoryPoint(
            round=round, cos_target=0.0, cos_baseline=0.0,
            top_token="", response="", has_refuse=has_refuse,
            has_comply=has_comply, n_tokens=0,
        )

    def test_returns_none_when_no_compliance(self):
        trajectory = [self._pt(i, False, True) for i in range(5)]
        assert find_compliance_emergence(trajectory) is None

    def test_returns_none_for_empty_trajectory(self):
        assert find_compliance_emergence([]) is None

    def test_returns_correct_round_when_compliance_found(self):
        trajectory = [
            self._pt(0, False, True),   # refuse only
            self._pt(1, True, True),    # comply + refuse (not pure compliance)
            self._pt(2, True, False),   # comply without refuse -- this is the one
            self._pt(3, True, False),
        ]
        assert find_compliance_emergence(trajectory) == 2

    def test_returns_first_round_if_immediate_compliance(self):
        trajectory = [
            self._pt(0, True, False),
            self._pt(1, True, False),
        ]
        assert find_compliance_emergence(trajectory) == 0

    def test_comply_plus_refuse_is_not_emergence(self):
        """A point with both compliance and refusal should not count."""
        trajectory = [self._pt(0, True, True)]
        assert find_compliance_emergence(trajectory) is None


# ---------------------------------------------------------------------------
# trajectory_trend
# ---------------------------------------------------------------------------

class TestTrajectoryTrend:
    def _pt(self, round: int, cos_target: float) -> TrajectoryPoint:
        return TrajectoryPoint(
            round=round, cos_target=cos_target, cos_baseline=0.0,
            top_token="", response="", has_refuse=False,
            has_comply=False, n_tokens=0,
        )

    def test_upward_trend(self):
        trajectory = [self._pt(0, 0.3), self._pt(1, 0.5), self._pt(2, 0.7)]
        assert trajectory_trend(trajectory) == "\u2191"

    def test_downward_trend(self):
        trajectory = [self._pt(0, 0.7), self._pt(1, 0.5), self._pt(2, 0.3)]
        assert trajectory_trend(trajectory) == "\u2193"

    def test_flat_trend(self):
        trajectory = [self._pt(0, 0.5), self._pt(1, 0.505)]
        assert trajectory_trend(trajectory) == "\u2192"

    def test_flat_at_boundary(self):
        """Exactly 0.01 above should still be flat."""
        trajectory = [self._pt(0, 0.5), self._pt(1, 0.51)]
        assert trajectory_trend(trajectory) == "\u2192"

    def test_just_above_threshold_is_upward(self):
        trajectory = [self._pt(0, 0.5), self._pt(1, 0.5101)]
        assert trajectory_trend(trajectory) == "\u2191"

    def test_just_below_threshold_is_downward(self):
        trajectory = [self._pt(0, 0.5), self._pt(1, 0.4899)]
        assert trajectory_trend(trajectory) == "\u2193"

    def test_single_point_is_flat(self):
        assert trajectory_trend([self._pt(0, 0.5)]) == "\u2192"

    def test_empty_trajectory_is_flat(self):
        assert trajectory_trend([]) == "\u2192"


# ---------------------------------------------------------------------------
# self_feed_trajectory (requires mocking runtime)
# ---------------------------------------------------------------------------

class TestSelfFeedTrajectory:
    """Test self_feed_trajectory with mocked runtime.forward_pass and runtime.generate."""

    def _make_forward_result(self, residual=None):
        return {
            "residual": residual if residual is not None else np.zeros(8),
            "top_token": "Sure",
            "n_tokens": 20,
            "logits": np.zeros(100),
            "probs": np.ones(100) / 100,
            "top_id": 0,
            "entropy": 3.0,
        }

    def _make_gen_result(self, text="Sure, here is the information."):
        return {"generated": text}

    @patch("heinrich.cartography.runtime.generate")
    @patch("heinrich.cartography.runtime.forward_pass")
    def test_returns_correct_number_of_rounds(self, mock_forward, mock_generate):
        mock_forward.return_value = self._make_forward_result()
        mock_generate.return_value = self._make_gen_result()

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(50))
        tokenizer.decode.return_value = "decoded"

        trajectory = self_feed_trajectory(
            model, tokenizer, "test prompt", n_rounds=4,
        )
        assert len(trajectory) == 4
        assert [pt.round for pt in trajectory] == [0, 1, 2, 3]

    @patch("heinrich.cartography.runtime.generate")
    @patch("heinrich.cartography.runtime.forward_pass")
    def test_cosine_with_target_residual(self, mock_forward, mock_generate):
        target = np.array([1.0, 0.0, 0.0, 0.0])
        residual = np.array([0.8, 0.1, 0.0, 0.0])

        mock_forward.return_value = self._make_forward_result(residual=residual)
        mock_generate.return_value = self._make_gen_result()

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(10))

        trajectory = self_feed_trajectory(
            model, tokenizer, "test",
            n_rounds=1, target_residual=target,
        )
        assert len(trajectory) == 1
        assert trajectory[0].cos_target > 0.9  # very similar vectors

    @patch("heinrich.cartography.runtime.generate")
    @patch("heinrich.cartography.runtime.forward_pass")
    def test_cosine_zero_when_no_reference(self, mock_forward, mock_generate):
        mock_forward.return_value = self._make_forward_result()
        mock_generate.return_value = self._make_gen_result()

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(10))

        trajectory = self_feed_trajectory(
            model, tokenizer, "test", n_rounds=1,
        )
        assert trajectory[0].cos_target == 0.0
        assert trajectory[0].cos_baseline == 0.0

    @patch("heinrich.cartography.runtime.generate")
    @patch("heinrich.cartography.runtime.forward_pass")
    def test_classification_flows_through(self, mock_forward, mock_generate):
        mock_forward.return_value = self._make_forward_result()
        # "sorry" triggers refuse; "certainly" triggers comply
        mock_generate.side_effect = [
            self._make_gen_result("I'm sorry, I cannot help."),
            self._make_gen_result("Certainly, here are the details."),
        ]

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(10))

        trajectory = self_feed_trajectory(
            model, tokenizer, "test", n_rounds=2,
        )
        assert trajectory[0].has_refuse is True
        assert trajectory[1].has_comply is True

    @patch("heinrich.cartography.runtime.generate")
    @patch("heinrich.cartography.runtime.forward_pass")
    def test_context_truncation(self, mock_forward, mock_generate):
        mock_forward.return_value = self._make_forward_result()
        mock_generate.return_value = self._make_gen_result("response text")

        model = MagicMock()
        tokenizer = MagicMock()
        # Return a token list longer than max_context_tokens
        tokenizer.encode.return_value = list(range(200))
        tokenizer.decode.return_value = "truncated context"

        trajectory = self_feed_trajectory(
            model, tokenizer, "test prompt",
            n_rounds=2, max_context_tokens=50,
        )
        assert len(trajectory) == 2
        # On second round, decode should be called for truncation
        assert tokenizer.decode.called

    @patch("heinrich.cartography.runtime.generate")
    @patch("heinrich.cartography.runtime.forward_pass")
    def test_custom_followups(self, mock_forward, mock_generate):
        mock_forward.return_value = self._make_forward_result()
        mock_generate.return_value = self._make_gen_result("ok")

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(10))

        followups = ["Tell me more.", "Go on."]
        trajectory = self_feed_trajectory(
            model, tokenizer, "initial",
            n_rounds=3, followups=followups,
        )
        assert len(trajectory) == 3
        # Round 0 uses initial_prompt, rounds 1+ use followups[round_idx % len]
        calls = mock_forward.call_args_list
        # Round 1: followups[1 % 2] = "Go on."
        prompt_r1 = calls[1][0][2]  # third positional arg is prompt
        assert "Go on." in prompt_r1
        # Round 2: followups[2 % 2] = "Tell me more."
        prompt_r2 = calls[2][0][2]
        assert "Tell me more." in prompt_r2

    @patch("heinrich.cartography.runtime.generate")
    @patch("heinrich.cartography.runtime.forward_pass")
    def test_response_truncated_to_80_chars(self, mock_forward, mock_generate):
        mock_forward.return_value = self._make_forward_result()
        long_response = "A" * 200
        mock_generate.return_value = self._make_gen_result(long_response)

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(10))

        trajectory = self_feed_trajectory(
            model, tokenizer, "test", n_rounds=1,
        )
        assert len(trajectory[0].response) == 80
