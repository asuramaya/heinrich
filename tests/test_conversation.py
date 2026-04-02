"""Tests for heinrich.cartography.conversation."""
import numpy as np
from heinrich.cartography.conversation import (
    TurnMeasurement, ConversationTrace,
)


class TestTurnMeasurement:
    def test_creation(self):
        t = TurnMeasurement(
            turn=0, role="user", text="hello",
            refuse_prob=0.8, comply_prob=0.1,
            residual_projection=0.5, entropy=3.0,
            top_token="I", drift_from_baseline=0.0,
        )
        assert t.role == "user"
        assert t.refuse_prob == 0.8


class TestConversationTrace:
    def _make_trace(self, refuse_probs, comply_probs):
        turns = []
        for i, (rp, cp) in enumerate(zip(refuse_probs, comply_probs)):
            turns.append(TurnMeasurement(
                turn=i // 2, role="user" if i % 2 == 0 else "assistant",
                text="test", refuse_prob=rp, comply_prob=cp,
                residual_projection=0.0, entropy=3.0,
                top_token="t", drift_from_baseline=float(i) * 0.1,
            ))
        return ConversationTrace(
            turns=turns,
            first_compliance_turn=None,
            safety_recovered=False,
            max_drift=max(t.drift_from_baseline for t in turns),
            final_refuse_prob=turns[-1].refuse_prob if turns else 0,
        )

    def test_compliance_rate_all_refuse(self):
        trace = self._make_trace(
            [0.8, 0.8, 0.7, 0.7],  # user, asst, user, asst
            [0.1, 0.1, 0.2, 0.2],
        )
        assert trace.compliance_rate() == 0.0

    def test_compliance_rate_all_comply(self):
        trace = self._make_trace(
            [0.1, 0.1, 0.1, 0.1],
            [0.8, 0.8, 0.8, 0.8],
        )
        assert trace.compliance_rate() == 1.0

    def test_drift_trajectory(self):
        trace = self._make_trace([0.5, 0.5, 0.5, 0.5], [0.3, 0.3, 0.3, 0.3])
        drift = trace.drift_trajectory()
        assert len(drift) == 4
        assert drift[0] == 0.0
        assert drift[-1] > 0
