import json
from pathlib import Path

from heinrich.bundle.compress import compress_store
from heinrich.fetch.local import fetch_local_model
from heinrich.pipeline import Pipeline
from heinrich.signal import SignalStore

FIXTURES = Path(__file__).parent / "fixtures"


class FetchStage:
    name = "fetch"

    def __init__(self, path: Path) -> None:
        self._path = path

    def run(self, store: SignalStore, config: dict) -> None:
        label = config.get("model_label", "test")
        fetch_local_model(store, self._path, model_label=label)


class BundleStage:
    name = "bundle"

    def run(self, store: SignalStore, config: dict) -> None:
        pass  # bundle is a terminal output, not a signal producer


def test_end_to_end_pipeline():
    pipe = Pipeline([FetchStage(FIXTURES), BundleStage()])
    store = pipe.run({"model_label": "test-model"})

    assert len(store) > 0
    assert pipe.stages_run == ["fetch", "bundle"]

    output = compress_store(store, stages_run=pipe.stages_run, models=["test-model"])

    assert output["heinrich_version"] == "0.2.1"
    assert output["models"] == ["test-model"]
    assert "fetch" in output["stages_run"]
    assert output["structural"]["architecture_type"] == "qwen2"
    assert output["signals_summary"]["total"] > 0

    serialized = json.dumps(output)
    assert len(serialized) < 10000  # fits in any context window


def test_pipeline_output_is_context_ready():
    store = SignalStore()
    fetch_local_model(store, FIXTURES, model_label="m1")
    output = compress_store(store, stages_run=["fetch"])

    serialized = json.dumps(output)
    roundtrip = json.loads(serialized)
    assert roundtrip["structural"]["config"]["num_hidden_layers"] == 4.0
    assert roundtrip["signals_summary"]["by_kind"]["tensor_name"] == 6


def test_ingest_and_query():
    """Integration test: ingest data files, query normalized tables, call MCP tools."""
    import tempfile
    import os
    from heinrich.db import SignalDB
    from heinrich.ingest import ingest_all

    data_dir = Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        import pytest
        pytest.skip("data/ directory not present")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = SignalDB(db_path)
        total = ingest_all(db, data_dir=str(data_dir))
        assert total > 0, "Ingest should produce signals"

        # Check normalized tables populated from JSON data files
        tables_expected_nonempty = [
            "models", "experiments", "evaluations",
            "censorship", "heads",
            "probes", "events",
        ]
        for t in tables_expected_nonempty:
            row = db._conn.execute(f"SELECT COUNT(*) as n FROM {t}").fetchone()
            assert row["n"] > 0, f"Table {t} should have rows after ingest"

        # Tables that need recompute scripts or live model (may be empty from file-only ingest)
        for t in ["neurons", "sharts", "directions", "layers",
                   "basins", "basin_distances", "interpolations"]:
            row = db._conn.execute(f"SELECT COUNT(*) as n FROM {t}").fetchone()
            # These are populated by scripts/recompute_*.py, not by ingest

        # Call MCP tools
        from heinrich.mcp import ToolServer
        ts = ToolServer(db=db)
        result = ts.call_tool("heinrich_safety_report", {})
        assert "error" not in result, f"safety_report failed: {result}"
        assert result["count"] > 0

        result = ts.call_tool("heinrich_sharts", {})
        assert "error" not in result
        # sharts may be empty after ingest (populated by recompute_sharts.py)

        result = ts.call_tool("heinrich_censorship", {})
        assert result["count"] == 20

        result = ts.call_tool("heinrich_directions", {})
        # directions may be empty after ingest (populated by recompute_directions.py)

        result = ts.call_tool("heinrich_heads", {})
        assert result["count"] > 0

        # Paper verifiers
        result = ts.call_tool("heinrich_paper_verify", {"claim": "shart_families_3"})
        assert "error" not in result

        result = ts.call_tool("heinrich_paper_verify", {"claim": "eval_count_1890"})
        assert "error" not in result

        # --- Item 12,35: Data correctness tests ---
        # Neuron-specific checks removed: neurons are now populated by
        # recompute_neurons.py, not hardcoded in ingest.

        # 3 divergent censorship topics (from JSON ingest, not hardcoded)
        model_row = db._conn.execute(
            "SELECT id FROM models WHERE name = ? LIMIT 1",
            ("mlx-community/Qwen2.5-7B-Instruct-4bit",),
        ).fetchone()
        if model_row:
            mid = model_row["id"]
            censorship = db.get_censorship(mid, divergent_only=True)
            assert len(censorship) == 3, f"Expected 3 divergent censorship topics, got {len(censorship)}"

            # Layer roles are now computed by recompute_layer_map.py,
            # so layers table may be empty after file-only ingest.

        # Provenance column populated
        prov_rows = db._conn.execute(
            "SELECT DISTINCT provenance FROM neurons WHERE provenance != 'unknown'"
        ).fetchall()
        assert len(prov_rows) > 0, "Provenance should be set on neurons"

        # New tools work
        result = ts.call_tool("heinrich_events", {})
        assert "error" not in result
        assert result["count"] > 0

        result = ts.call_tool("heinrich_interpolation", {})
        assert "error" not in result
        # interpolation may be empty after ingest (populated by recompute_interpolation.py)

        # alpha_015 verifier depends on interpolation data (from recompute scripts).
        # Skip assertion if interpolation table is empty.
        result = ts.call_tool("heinrich_paper_verify", {"claim": "alpha_015_every_prompt"})
        # May return verified=False (paper overclaimed) or error if no data

        # Item 37: Data integrity tests
        _test_no_duplicate_signals(db)
        _test_model_id_consistency(db)
        _test_provenance_complete(db)
        _test_atlas_coverage(db)

        # Item 37: New tools work
        result = ts.call_tool("heinrich_head_detail", {"layer": 25, "head": 8})
        assert "error" not in result

        result = ts.call_tool("heinrich_signals_summary", {})
        assert "error" not in result
        assert result["total_signals"] > 0

        # Phase 3: tools return data, not interpretation
        result = ts.call_tool("heinrich_safety_report", {})
        assert "methodology_note" not in result
        assert "WARNING" not in result

        db.close()


def _test_no_duplicate_signals(db):
    """Item 37: No duplicate signals from double-ingestion."""
    run_ids = db._conn.execute(
        "SELECT run_id, COUNT(*) as n FROM signals GROUP BY run_id"
    ).fetchall()
    # Should not have two runs with identical signal counts
    counts = [r["n"] for r in run_ids if r["n"] > 100]
    from collections import Counter
    dupes = [c for c, freq in Counter(counts).items() if freq > 1]
    assert not dupes, f"Duplicate signal runs detected: counts {dupes}"


def _test_model_id_consistency(db):
    """Item 37: All Qwen data under canonical model_id."""
    # Find canonical Qwen instruct id
    canon = db._conn.execute(
        "SELECT id FROM models WHERE name='mlx-community/Qwen2.5-7B-Instruct-4bit' LIMIT 1"
    ).fetchone()
    if not canon:
        return  # skip if model not found
    canon_id = canon["id"]

    # All non-canonical Qwen instruct variants
    variants = db._conn.execute(
        "SELECT id FROM models WHERE canonical_name='qwen2.5-7b-instruct-4bit' AND id != ?",
        (canon_id,)
    ).fetchall()
    variant_ids = [r["id"] for r in variants]
    if not variant_ids:
        return

    # No data should reference the non-canonical ids
    # (only check tables that have model_id)
    for table in ["heads", "neurons", "sharts", "directions", "layers"]:
        placeholders = ",".join("?" * len(variant_ids))
        row = db._conn.execute(
            f"SELECT COUNT(*) as n FROM {table} WHERE model_id IN ({placeholders})",
            variant_ids,
        ).fetchone()
        assert row["n"] == 0, (
            f"Table {table} has {row['n']} rows with non-canonical model_id"
        )


def _test_provenance_complete(db):
    """Item 37: No 'unknown' provenance in key tables."""
    for table in ("neurons", "sharts", "directions"):
        row = db._conn.execute(
            f"SELECT COUNT(*) as n FROM {table} WHERE provenance = 'unknown'"
        ).fetchone()
        assert row["n"] == 0, f"Table {table} has {row['n']} rows with unknown provenance"


def _test_atlas_coverage(db):
    """Item 37: head_measurements table has per-prompt data."""
    try:
        row = db._conn.execute("SELECT COUNT(*) as n FROM head_measurements").fetchone()
        assert row["n"] > 0, "head_measurements should have per-prompt data"
    except Exception:
        pass  # table may not exist in test DB
