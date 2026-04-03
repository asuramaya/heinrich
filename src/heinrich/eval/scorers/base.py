"""Base scorer protocol."""
from dataclasses import dataclass


@dataclass
class ScoreResult:
    label: str                    # measurement scorers: raw label (REFUSES, STRUCTURAL, refuse_prob=0.85, etc.)
                                  # external model scorers: prefixed judgment (qwen3guard:safe, llamaguard:unsafe)
                                  # error: 'error'
    confidence: float | None      # None until calibration computes real reliability
    raw_output: str               # full explanation for audit


class Scorer:
    name: str = "base"
    requires_model: bool = False

    def score(self, prompt: str, response: str) -> ScoreResult:
        raise NotImplementedError
