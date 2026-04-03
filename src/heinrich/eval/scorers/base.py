"""Base scorer protocol."""
from dataclasses import dataclass


@dataclass
class ScoreResult:
    label: str                    # 'safe', 'unsafe', 'ambiguous', 'error'
    confidence: float | None      # None until calibration computes real reliability
    raw_output: str               # full explanation for audit


class Scorer:
    name: str = "base"
    requires_model: bool = False

    def score(self, prompt: str, response: str) -> ScoreResult:
        raise NotImplementedError
