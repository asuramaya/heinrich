"""Reform 4: Preregistration -- record hypotheses before running experiments.

Usage:
    from heinrich.cartography.preregister import preregister
    reg = preregister(
        db,
        hypothesis="Forensic framing bypasses violence refusal",
        test="Compare direct vs forensic refusal rates on violence prompts",
        confirm_if="forensic_refuse < direct_refuse * 0.5",
        falsify_if="forensic_refuse >= direct_refuse * 0.5",
        chance_level=0.5,
        n_samples=100,
    )
    # ... run experiment ...
    update_preregistration(db, reg.id, result="confirmed", verified=True)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class Preregistration:
    """A preregistered hypothesis."""
    id: int
    hypothesis: str
    test: str
    confirm_if: str
    falsify_if: str
    chance_level: float
    n_samples: int


def preregister(
    db: Any,
    hypothesis: str,
    test: str,
    confirm_if: str,
    falsify_if: str,
    chance_level: float = 0.5,
    n_samples: int = 100,
) -> Preregistration:
    """Record a preregistration before running an experiment.

    Args:
        db: SignalDB instance
        hypothesis: The hypothesis to test
        test: Description of the test
        confirm_if: Condition for confirmation
        falsify_if: Condition for falsification
        chance_level: Baseline probability
        n_samples: Number of samples needed

    Returns:
        Preregistration dataclass with id
    """
    prereg_id = db.record_preregistration(
        hypothesis=hypothesis,
        test=test,
        confirm_if=confirm_if,
        falsify_if=falsify_if,
        chance_level=chance_level,
        n_samples=n_samples,
    )
    return Preregistration(
        id=prereg_id,
        hypothesis=hypothesis,
        test=test,
        confirm_if=confirm_if,
        falsify_if=falsify_if,
        chance_level=chance_level,
        n_samples=n_samples,
    )


def update_preregistration(
    db: Any,
    prereg_id: int,
    result: str,
    verified: bool,
) -> None:
    """Update a preregistration with experimental results."""
    db.update_preregistration(prereg_id, result=result, verified=verified)
