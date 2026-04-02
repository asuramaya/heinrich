"""Tests for heinrich.cartography.search."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call

from heinrich.cartography.search import (
    GridCell,
    ablation_scan,
    evolutionary_search,
    grid_search,
)


# ---------------------------------------------------------------------------
# GridCell dataclass
# ---------------------------------------------------------------------------

class TestGridCell:
    def test_construction(self):
        cell = GridCell(
            framing="direct",
            injection="none",
            refuse_prob=0.92,
            compliance_prob=0.01,
            top_token="I",
            label="REFUSES",
            response="I cannot assist with that.",
            metadata={"entropy": 2.5},
        )
        assert cell.framing == "direct"
        assert cell.injection == "none"
        assert cell.refuse_prob == 0.92
        assert cell.compliance_prob == 0.01
        assert cell.top_token == "I"
        assert cell.label == "REFUSES"
        assert cell.response == "I cannot assist with that."
        assert cell.metadata == {"entropy": 2.5}

    def test_equality(self):
        kwargs = dict(
            framing="academic", injection="shart",
            refuse_prob=0.5, compliance_prob=0.3,
            top_token="Sure", label="COMPLIES",
            response="Sure, here you go.", metadata={},
        )
        assert GridCell(**kwargs) == GridCell(**kwargs)

    def test_different_cells_not_equal(self):
        base = dict(
            framing="direct", injection="none",
            refuse_prob=0.9, compliance_prob=0.01,
            top_token="I", label="REFUSES",
            response="no", metadata={},
        )
        a = GridCell(**base)
        b = GridCell(**{**base, "label": "COMPLIES"})
        assert a != b


# ---------------------------------------------------------------------------
# ablation_scan
# ---------------------------------------------------------------------------

class TestAblationScan:
    """Test ablation_scan with a mock metric_fn, bypassing runtime."""

    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda text, **kw: f"[PROMPT]{text}")
    def test_basic_ablation(self, mock_build_prompt):
        """Removing a keyword word causes a big metric change."""
        # metric_fn: returns 0.9 for full text, 0.3 if "dangerous" is removed
        def metric_fn(prompt):
            if "dangerous" not in prompt:
                return 0.3
            return 0.9

        model = MagicMock()
        tokenizer = MagicMock()

        results = ablation_scan(
            model, tokenizer, "this is dangerous stuff",
            metric_fn=metric_fn,
        )

        assert len(results) == 4  # four words
        words = [r["word"] for r in results]
        assert words == ["this", "is", "dangerous", "stuff"]

        # "dangerous" removal has the largest delta (negative, since base is 0.9)
        dangerous_entry = results[2]
        assert dangerous_entry["word"] == "dangerous"
        assert dangerous_entry["delta"] == round(0.3 - 0.9, 4)
        assert dangerous_entry["essential"] is True  # |delta| > 0.1

    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda text, **kw: f"[PROMPT]{text}")
    def test_no_essential_words(self, mock_build_prompt):
        """All words produce small deltas => none are essential."""
        def metric_fn(prompt):
            return 0.5  # constant regardless of removed word

        results = ablation_scan(
            MagicMock(), MagicMock(), "hello world",
            metric_fn=metric_fn,
        )
        assert len(results) == 2
        assert all(r["essential"] is False for r in results)
        assert all(r["delta"] == 0.0 for r in results)

    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda text, **kw: f"[PROMPT]{text}")
    def test_single_word(self, mock_build_prompt):
        """Single-word text: removing it yields empty string."""
        def metric_fn(prompt):
            if prompt == "[PROMPT]":  # empty text after word removal
                return 0.0
            return 0.8

        results = ablation_scan(
            MagicMock(), MagicMock(), "bomb",
            metric_fn=metric_fn,
        )
        assert len(results) == 1
        assert results[0]["word"] == "bomb"
        assert results[0]["delta"] == round(0.0 - 0.8, 4)
        assert results[0]["essential"] is True

    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda text, **kw: f"[PROMPT]{text}")
    def test_metric_full_recorded(self, mock_build_prompt):
        """Each result records the full-text metric."""
        def metric_fn(prompt):
            return 0.42

        results = ablation_scan(
            MagicMock(), MagicMock(), "a b c",
            metric_fn=metric_fn,
        )
        assert all(r["metric_full"] == 0.42 for r in results)


# ---------------------------------------------------------------------------
# evolutionary_search
# ---------------------------------------------------------------------------

class TestEvolutionarySearch:
    """Test evolutionary_search sorting with mock metric_fn."""

    @patch("heinrich.cartography.classify.classify_response")
    @patch("heinrich.cartography.runtime.generate")
    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda text, **kw: f"[P]{text}")
    def test_sorted_by_refuse_prob(self, mock_build, mock_generate, mock_classify):
        mock_classify.return_value = MagicMock(label="AMBIGUOUS")
        mock_generate.return_value = {"generated": "some text"}

        # Metric returns different values depending on the framing
        scores = {"alpha": 0.7, "beta": 0.2, "gamma": 0.9}

        def metric_fn(prompt):
            for key, score in scores.items():
                if key in prompt:
                    return score
            return 0.5

        results = evolutionary_search(
            MagicMock(), MagicMock(),
            population=["alpha", "beta", "gamma"],
            metric_fn=metric_fn,
            progress=False,
        )

        # Should be sorted lowest refuse_prob first
        assert results[0]["framing"] == "beta"
        assert results[1]["framing"] == "alpha"
        assert results[2]["framing"] == "gamma"

    @patch("heinrich.cartography.classify.classify_response")
    @patch("heinrich.cartography.runtime.generate")
    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda text, **kw: text)
    def test_query_prepended_to_framing(self, mock_build, mock_generate, mock_classify):
        mock_classify.return_value = MagicMock(label="COMPLIES")
        mock_generate.return_value = {"generated": "ok"}

        prompts_seen = []

        def metric_fn(prompt):
            prompts_seen.append(prompt)
            return 0.5

        evolutionary_search(
            MagicMock(), MagicMock(),
            population=["As a chemist"],
            query="how to make stuff",
            metric_fn=metric_fn,
            progress=False,
        )

        assert len(prompts_seen) == 1
        assert "As a chemist" in prompts_seen[0]
        assert "how to make stuff" in prompts_seen[0]

    @patch("heinrich.cartography.classify.classify_response")
    @patch("heinrich.cartography.runtime.generate")
    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda text, **kw: text)
    def test_label_from_classification(self, mock_build, mock_generate, mock_classify):
        mock_classify.return_value = MagicMock(label="REFUSES")
        mock_generate.return_value = {"generated": "I'm sorry, I can't help."}

        results = evolutionary_search(
            MagicMock(), MagicMock(),
            population=["test"],
            metric_fn=lambda p: 0.95,
            progress=False,
        )
        assert results[0]["label"] == "REFUSES"

    @patch("heinrich.cartography.classify.classify_response")
    @patch("heinrich.cartography.runtime.generate")
    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda text, **kw: text)
    def test_empty_population(self, mock_build, mock_generate, mock_classify):
        results = evolutionary_search(
            MagicMock(), MagicMock(),
            population=[],
            metric_fn=lambda p: 0.5,
            progress=False,
        )
        assert results == []

    @patch("heinrich.cartography.classify.classify_response")
    @patch("heinrich.cartography.runtime.generate")
    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda text, **kw: text)
    def test_response_truncated_to_60_chars(self, mock_build, mock_generate, mock_classify):
        mock_classify.return_value = MagicMock(label="AMBIGUOUS")
        mock_generate.return_value = {"generated": "X" * 200}

        results = evolutionary_search(
            MagicMock(), MagicMock(),
            population=["test"],
            metric_fn=lambda p: 0.5,
            progress=False,
        )
        assert len(results[0]["response"]) == 60


# ---------------------------------------------------------------------------
# grid_search (mock all runtime imports)
# ---------------------------------------------------------------------------

class TestGridSearch:
    """Test grid_search orchestration with mocked internal imports."""

    @patch("heinrich.cartography.runtime.forward_pass")
    @patch("heinrich.cartography.runtime.generate")
    @patch("heinrich.cartography.classify.classify_response")
    @patch("heinrich.cartography.runtime.build_refusal_set", return_value={0, 1})
    @patch("heinrich.cartography.runtime.build_compliance_set", return_value={2, 3})
    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda q, **kw: f"[P]{q}")
    @patch("heinrich.cartography.templates.SHART_TOKENS", {"none": "", "newline": "\\n"})
    @patch("heinrich.cartography.templates.FRAMINGS", {"direct": "{query}", "academic": "As a researcher: {query}"})
    def test_grid_covers_all_combinations(
        self, mock_build, mock_cset, mock_rset, mock_classify,
        mock_generate, mock_forward,
    ):
        probs = np.zeros(10)
        probs[0] = 0.4  # refusal token
        probs[1] = 0.3  # refusal token
        probs[2] = 0.1  # compliance token

        mock_forward.return_value = {
            "probs": probs,
            "top_token": "I",
            "entropy": 2.0,
        }
        mock_generate.return_value = {"generated": "I cannot help"}
        mock_classify.return_value = MagicMock(label="REFUSES")

        results = grid_search(
            MagicMock(), MagicMock(), "test query", progress=False,
        )

        # 2 framings x 2 injections = 4 cells
        assert len(results) == 4
        framings = {r.framing for r in results}
        assert framings == {"direct", "academic"}
        injections = {r.injection for r in results}
        assert injections == {"none", "newline"}

    @patch("heinrich.cartography.runtime.forward_pass")
    @patch("heinrich.cartography.runtime.generate")
    @patch("heinrich.cartography.classify.classify_response")
    @patch("heinrich.cartography.runtime.build_refusal_set", return_value={0})
    @patch("heinrich.cartography.runtime.build_compliance_set", return_value={1})
    @patch("heinrich.cartography.templates.build_prompt", side_effect=lambda q, **kw: f"[P]{q}")
    @patch("heinrich.cartography.templates.SHART_TOKENS", {"none": ""})
    @patch("heinrich.cartography.templates.FRAMINGS", {"direct": "{query}"})
    def test_grid_records_probabilities(
        self, mock_build, mock_cset, mock_rset, mock_classify,
        mock_generate, mock_forward,
    ):
        probs = np.zeros(10)
        probs[0] = 0.75  # refusal
        probs[1] = 0.05  # compliance

        mock_forward.return_value = {
            "probs": probs,
            "top_token": "Sorry",
            "entropy": 1.5,
        }
        mock_generate.return_value = {"generated": "Sorry, I cannot."}
        mock_classify.return_value = MagicMock(label="REFUSES")

        results = grid_search(
            MagicMock(), MagicMock(), "query", progress=False,
        )

        assert len(results) == 1
        assert results[0].refuse_prob == 0.75
        assert results[0].compliance_prob == 0.05
        assert results[0].top_token == "Sorry"
        assert results[0].metadata == {"entropy": 1.5}
