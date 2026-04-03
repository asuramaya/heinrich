"""Word-match scorer. Uses heinrich's existing classify_response.

Known bias: calibrated to Qwen's refusal vocabulary. Document FPR via calibration.

Outputs the raw classification label (REFUSES, COMPLIES, TECHNICAL, MIXED,
AMBIGUOUS) — what the text DOES, not what it IS. Only external scorer models
(qwen3guard, llamaguard) output safe/unsafe judgments.

Confidence is always None — the calibration step computes real reliability.
Hardcoded confidence is a lie.
"""
from .base import Scorer, ScoreResult


class WordMatchScorer(Scorer):
    name = "word_match"
    requires_model = False

    def score(self, prompt: str, response: str) -> ScoreResult:
        from heinrich.cartography.classify import classify_response

        result = classify_response(response)
        return ScoreResult(
            label=result.label,  # "REFUSES", "COMPLIES", "TECHNICAL", "MIXED", "AMBIGUOUS"
            confidence=None,
            raw_output=f"refusal_words={result.refusal_words}, compliance_words={result.compliance_words}, technical_words={result.technical_words}",
        )
