"""Tests for the full pipeline integration: discover -> attack -> eval.

Tests the conditions table, discover __main__, attack run, and
the auto-conditions resolution in the eval run module.
"""
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from heinrich.core.db import SignalDB


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def tmp_db(tmp_path):
    """Create a fresh SignalDB in a temp directory."""
    db_path = tmp_path / "test_pipeline.db"
    db = SignalDB(str(db_path))
    yield db
    db.close()


# ============================================================
# Step 1: Conditions table
# ============================================================

class TestConditionsTable:
    def test_record_condition(self, tmp_db):
        mid = tmp_db.upsert_model("test-model")
        cid = tmp_db.record_condition(mid, "clean", kind="baseline", source="test")
        assert cid == 1

    def test_record_condition_with_params(self, tmp_db):
        mid = tmp_db.upsert_model("test-model")
        cid = tmp_db.record_condition(
            mid, "steer_-0.10",
            kind="steer",
            params_dict={"alpha": -0.10, "layer": 27},
            source="attack",
        )
        assert cid >= 1

        conds = tmp_db.get_conditions(mid)
        assert len(conds) == 1
        assert conds[0]["name"] == "steer_-0.10"
        params = json.loads(conds[0]["params"])
        assert params["alpha"] == -0.10
        assert params["layer"] == 27

    def test_get_conditions_empty(self, tmp_db):
        mid = tmp_db.upsert_model("test-model")
        conds = tmp_db.get_conditions(mid)
        assert conds == []

    def test_get_conditions_multiple(self, tmp_db):
        mid = tmp_db.upsert_model("test-model")
        tmp_db.record_condition(mid, "clean", kind="baseline")
        tmp_db.record_condition(mid, "steer_-0.10", kind="steer")
        tmp_db.record_condition(mid, "steer_-0.15", kind="steer")

        conds = tmp_db.get_conditions(mid)
        assert len(conds) == 3
        names = [c["name"] for c in conds]
        assert "clean" in names
        assert "steer_-0.10" in names

    def test_get_conditions_filter_by_model(self, tmp_db):
        mid1 = tmp_db.upsert_model("model-a")
        mid2 = tmp_db.upsert_model("model-b")
        tmp_db.record_condition(mid1, "clean", kind="baseline")
        tmp_db.record_condition(mid2, "clean", kind="baseline")
        tmp_db.record_condition(mid2, "steer_-0.10", kind="steer")

        conds1 = tmp_db.get_conditions(mid1)
        conds2 = tmp_db.get_conditions(mid2)
        assert len(conds1) == 1
        assert len(conds2) == 2

    def test_get_conditions_all(self, tmp_db):
        mid1 = tmp_db.upsert_model("model-a")
        mid2 = tmp_db.upsert_model("model-b")
        tmp_db.record_condition(mid1, "clean")
        tmp_db.record_condition(mid2, "clean")
        all_conds = tmp_db.get_conditions()
        assert len(all_conds) == 2

    def test_upsert_condition(self, tmp_db):
        """Upserting a condition with the same (model_id, name) should update."""
        mid = tmp_db.upsert_model("test-model")
        cid1 = tmp_db.record_condition(mid, "clean", kind="baseline", source="discover")
        cid2 = tmp_db.record_condition(mid, "clean", kind="baseline", source="attack")
        # Same row should be updated
        assert cid1 == cid2

    def test_conditions_in_summary(self, tmp_db):
        mid = tmp_db.upsert_model("test-model")
        tmp_db.record_condition(mid, "clean")
        summary = tmp_db.summary()
        assert "conditions" in summary["normalized_tables"]
        assert summary["normalized_tables"]["conditions"] == 1


# ============================================================
# Step 2: discover __main__
# ============================================================

class TestDiscoverMain:
    def test_import(self):
        """discover.__main__ should be importable."""
        from heinrich.discover.__main__ import discover_to_db
        assert callable(discover_to_db)


# ============================================================
# Step 3: attack run
# ============================================================

class TestAttackRun:
    def test_import(self):
        """attack.run should be importable."""
        from heinrich.attack.run import attack_to_db
        assert callable(attack_to_db)

    def test_attack_no_direction(self, tmp_db):
        """Attack with no direction in DB should produce clean-only condition."""
        from heinrich.attack.run import attack_to_db

        mid = tmp_db.upsert_model("test-model")
        db_file = str(tmp_db.path)
        tmp_db.close()

        # Reopen via attack_to_db which creates its own SignalDB
        # No directions in DB -> should just write clean
        result = attack_to_db("test-model", db_path=db_file, progress=False)
        assert "clean" in result["conditions"]
        assert result["cliff_alpha"] is None


# ============================================================
# Step 4: Auto conditions resolution
# ============================================================

class TestAutoConditions:
    def test_resolve_no_auto(self, tmp_db):
        from heinrich.eval.run import _resolve_conditions
        result = _resolve_conditions(tmp_db, "test-model", ["clean", "steer_-0.10"])
        assert result == ["clean", "steer_-0.10"]

    def test_resolve_auto_no_model(self, tmp_db):
        from heinrich.eval.run import _resolve_conditions
        result = _resolve_conditions(tmp_db, "nonexistent-model", ["auto"])
        assert result == ["clean"]

    def test_resolve_auto_with_conditions(self, tmp_db):
        from heinrich.eval.run import _resolve_conditions
        mid = tmp_db.upsert_model("test-model")
        tmp_db.record_condition(mid, "clean", kind="baseline")
        tmp_db.record_condition(mid, "steer_-0.10", kind="steer")

        result = _resolve_conditions(tmp_db, "test-model", ["auto"])
        assert "clean" in result
        assert "steer_-0.10" in result

    def test_resolve_auto_empty_conditions(self, tmp_db):
        from heinrich.eval.run import _resolve_conditions
        mid = tmp_db.upsert_model("test-model")
        result = _resolve_conditions(tmp_db, "test-model", ["auto"])
        assert result == ["clean"]


# ============================================================
# Step 5: Top-level orchestrator
# ============================================================

class TestRunOrchestrator:
    def test_import(self):
        """run module should be importable."""
        from heinrich.run import run_full_pipeline
        assert callable(run_full_pipeline)


# ============================================================
# Step 6: CLI integration
# ============================================================

class TestCLI:
    def test_run_command_in_parser(self):
        from heinrich.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--model", "test-model",
            "--prompts", "simple_safety",
            "--scorers", "word_match",
        ])
        assert args.command == "run"
        assert args.model == "test-model"
        assert args.prompts == "simple_safety"

    def test_run_command_with_all_options(self):
        from heinrich.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--model", "test-model",
            "--prompts", "simple_safety,harmbench",
            "--scorers", "word_match,regex_harm",
            "--output", "/tmp/test.json",
            "--max-prompts", "5",
            "--skip-discover",
            "--skip-attack",
        ])
        assert args.skip_discover is True
        assert args.skip_attack is True
        assert args.max_prompts == 5


# ============================================================
# Integration: conditions flow through the pipeline
# ============================================================

class TestPipelineFlow:
    def test_conditions_written_by_discover_read_by_eval(self, tmp_db):
        """Conditions written by discover/attack should be readable by the pipeline."""
        mid = tmp_db.upsert_model("test-model")

        # Simulate discover step
        tmp_db.record_direction(
            mid, "safety", 27,
            stability=0.95, effect_size=2.5,
            provenance="discover",
        )
        tmp_db.record_condition(mid, "clean", kind="baseline", source="discover")

        # Simulate attack step
        tmp_db.record_condition(
            mid, "steer_-0.10",
            kind="steer",
            params_dict={"alpha": -0.10, "layer": 27, "direction": "safety"},
            source="attack",
        )
        tmp_db.record_condition(
            mid, "steer_-0.15",
            kind="steer",
            params_dict={"alpha": -0.15, "layer": 27, "direction": "safety"},
            source="attack",
        )

        # Resolve auto conditions
        from heinrich.eval.run import _resolve_conditions
        conditions = _resolve_conditions(tmp_db, "test-model", ["auto"])
        assert len(conditions) == 3
        assert "clean" in conditions
        assert "steer_-0.10" in conditions
        assert "steer_-0.15" in conditions

    def test_generate_validates_steer_conditions(self):
        """Steer conditions should pass validation."""
        from heinrich.eval.generate import _validate_condition
        _validate_condition("clean")
        _validate_condition("steer_-0.10")
        _validate_condition("steer_-0.15")
        _validate_condition("steer_0.05")

        with pytest.raises(ValueError):
            _validate_condition("invalid_condition")

    def test_db_migration_creates_conditions(self, tmp_path):
        """Fresh DB should have conditions table after migration."""
        db = SignalDB(str(tmp_path / "fresh.db"))
        # Check table exists by querying
        row = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conditions'"
        ).fetchone()
        assert row is not None
        db.close()

    def test_conditions_in_clear_normalized(self, tmp_db):
        """clear_normalized should clear conditions table."""
        mid = tmp_db.upsert_model("test-model")
        tmp_db.record_condition(mid, "clean")
        assert len(tmp_db.get_conditions()) == 1
        tmp_db.clear_normalized()
        assert len(tmp_db.get_conditions()) == 0
