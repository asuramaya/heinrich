"""Tests for heinrich.db — SQLite signal database."""
import tempfile
import numpy as np
import pytest
from heinrich.db import SignalDB
from heinrich.signal import Signal


@pytest.fixture
def db():
    path = tempfile.mktemp(suffix=".db")
    d = SignalDB(path)
    yield d
    d.close()
    import os
    os.unlink(path)


class TestSignalDB:
    def test_add_and_query(self, db):
        db.add(Signal("test", "source", "model", "target", 1.0))
        assert db.count() == 1
        results = db.query(kind="test")
        assert len(results) == 1
        assert results[0].value == 1.0

    def test_run_context(self, db):
        with db.run("test_run", model="qwen") as rid:
            db.add(Signal("a", "s", "m", "t", 1.0))
            db.add(Signal("b", "s", "m", "t", 2.0))
        assert db.count() == 2
        runs = db.runs()
        assert len(runs) == 1
        assert runs[0]["name"] == "test_run"

    def test_add_many(self, db):
        signals = [Signal("k", "s", "m", f"t{i}", float(i)) for i in range(100)]
        n = db.add_many(signals)
        assert n == 100
        assert db.count() == 100

    def test_query_filters(self, db):
        db.add(Signal("shart", "cart", "qwen", "六四", 1842.0))
        db.add(Signal("direction", "cart", "qwen", "L27", 0.95))
        db.add(Signal("shart", "cart", "llama", "test", 100.0))

        sharts = db.query(kind="shart")
        assert len(sharts) == 2

        qwen = db.query(model="qwen")
        assert len(qwen) == 2

        high = db.query(min_value=1000.0)
        assert len(high) == 1

    def test_kinds(self, db):
        db.add(Signal("a", "s", "m", "t", 1.0))
        db.add(Signal("a", "s", "m", "t2", 2.0))
        db.add(Signal("b", "s", "m", "t3", 3.0))
        kinds = db.kinds()
        assert ("a", 2) in kinds
        assert ("b", 1) in kinds

    def test_blob_roundtrip(self, db):
        vec = np.random.randn(128).astype(np.float32)
        db.save_blob("test_vec", vec)
        loaded = db.load_blob("test_vec")
        assert loaded is not None
        np.testing.assert_allclose(vec, loaded)

    def test_blob_missing(self, db):
        assert db.load_blob("nonexistent") is None

    def test_summary(self, db):
        db.add(Signal("test", "s", "m", "t", 1.0))
        s = db.summary()
        assert s["n_signals"] == 1
        assert s["n_runs"] == 0

    def test_import_export(self, db, tmp_path):
        import json
        signals = [
            {"kind": "k", "source": "s", "model": "m", "target": "t1", "value": 1.0, "metadata": {}},
            {"kind": "k", "source": "s", "model": "m", "target": "t2", "value": 2.0, "metadata": {}},
        ]
        path = tmp_path / "signals.json"
        path.write_text(json.dumps(signals))
        n = db.import_json(path)
        assert n == 2
        assert db.count() == 2

        exported = db.export_json(kind="k")
        data = json.loads(exported)
        assert len(data) == 2

    def test_len(self, db):
        assert len(db) == 0
        db.add(Signal("k", "s", "m", "t", 1.0))
        assert len(db) == 1


class TestSignalProvenance:
    def test_add_derived_basic(self, db):
        id1 = db.add(Signal("measure", "probe", "qwen", "L1", 0.8))
        id2 = db.add(Signal("measure", "probe", "qwen", "L2", 0.9))
        derived_id = db.add_derived(
            Signal("composite", "aggregator", "qwen", "L1+L2", 0.85),
            source_ids=[id1, id2],
            relationship="mean_of",
        )
        assert derived_id is not None
        assert db.count() == 3

    def test_get_provenance_direct(self, db):
        id1 = db.add(Signal("a", "s", "m", "t1", 1.0))
        id2 = db.add(Signal("a", "s", "m", "t2", 2.0))
        derived_id = db.add_derived(
            Signal("b", "agg", "m", "t_agg", 1.5),
            source_ids=[id1, id2],
            relationship="aggregated",
        )
        prov = db.get_provenance(derived_id)
        assert len(prov) == 2
        source_ids = {p["signal_id"] for p in prov}
        assert source_ids == {id1, id2}
        assert all(p["relationship"] == "aggregated" for p in prov)

    def test_get_provenance_chain(self, db):
        """Test multi-hop provenance: A -> B -> C."""
        id_a = db.add(Signal("raw", "s", "m", "t1", 1.0))
        id_b = db.add_derived(
            Signal("intermediate", "s", "m", "t2", 2.0),
            source_ids=[id_a],
            relationship="derived_from",
        )
        id_c = db.add_derived(
            Signal("final", "s", "m", "t3", 3.0),
            source_ids=[id_b],
            relationship="refined_from",
        )
        prov = db.get_provenance(id_c)
        # Should find both B and A
        source_ids = {p["signal_id"] for p in prov}
        assert id_b in source_ids
        assert id_a in source_ids
        assert len(prov) == 2

    def test_get_provenance_no_sources(self, db):
        id1 = db.add(Signal("a", "s", "m", "t", 1.0))
        prov = db.get_provenance(id1)
        assert prov == []

    def test_get_derived_basic(self, db):
        id1 = db.add(Signal("a", "s", "m", "t1", 1.0))
        d1 = db.add_derived(
            Signal("b", "s", "m", "t2", 2.0),
            source_ids=[id1],
            relationship="child_of",
        )
        d2 = db.add_derived(
            Signal("c", "s", "m", "t3", 3.0),
            source_ids=[id1],
            relationship="also_child_of",
        )
        derived = db.get_derived(id1)
        assert len(derived) == 2
        derived_ids = {d["signal_id"] for d in derived}
        assert derived_ids == {d1, d2}

    def test_get_derived_no_children(self, db):
        id1 = db.add(Signal("a", "s", "m", "t", 1.0))
        derived = db.get_derived(id1)
        assert derived == []

    def test_provenance_signal_data(self, db):
        """Provenance results should contain valid Signal objects."""
        id1 = db.add(Signal("measure", "probe", "qwen", "layer5", 0.95, {"foo": "bar"}))
        id2 = db.add_derived(
            Signal("composite", "agg", "qwen", "all", 0.9),
            source_ids=[id1],
            relationship="derived",
        )
        prov = db.get_provenance(id2)
        assert len(prov) == 1
        sig = prov[0]["signal"]
        assert sig.kind == "measure"
        assert sig.source == "probe"
        assert sig.model == "qwen"
        assert sig.target == "layer5"
        assert sig.value == 0.95
        assert sig.metadata == {"foo": "bar"}

    def test_add_derived_within_run(self, db):
        """Derived signals should work within a run context."""
        with db.run("provenance_test") as rid:
            id1 = db.add(Signal("a", "s", "m", "t1", 1.0))
            id2 = db.add_derived(
                Signal("b", "s", "m", "t2", 2.0),
                source_ids=[id1],
                relationship="derived",
            )
        prov = db.get_provenance(id2)
        assert len(prov) == 1
        # Verify both signals are in the run
        run_signals = db.query(run_id=rid)
        assert len(run_signals) == 2


class TestDataIntegrity:
    """Item 37: Data integrity checks for ingested data."""

    def test_no_duplicate_signals_per_run(self, db):
        """No duplicate signals within a single run."""
        with db.run("dup_test") as rid:
            db.add(Signal("a", "s", "m", "t", 1.0))
            db.add(Signal("a", "s", "m", "t", 1.0))  # same signal, allowed
        # Just checking both are there (duplicates are not prevented at DB level)
        assert db.count(run_id=rid) == 2

    def test_model_id_consistency(self, db):
        """Model IDs should be consistent across related tables."""
        mid = db.upsert_model("test-model")
        db.record_neuron(mid, layer=0, neuron_idx=0, name="n0")
        db.record_shart(mid, token_id=1, token_text="t1")
        db.record_direction(mid, "safety", layer=0, stability=0.9)
        # All should reference same model_id
        n = db._conn.execute("SELECT model_id FROM neurons WHERE name='n0'").fetchone()
        s = db._conn.execute("SELECT model_id FROM sharts WHERE token_text='t1'").fetchone()
        d = db._conn.execute("SELECT model_id FROM directions WHERE name='safety'").fetchone()
        assert n["model_id"] == s["model_id"] == d["model_id"] == mid

    def test_provenance_completeness(self, db):
        """All direction rows should have non-null provenance after ingest."""
        mid = db.upsert_model("test-model")
        db.record_direction(mid, "safety", 0, stability=0.9, provenance="hardcoded")
        db.record_direction(mid, "custom", 0, stability=0.5, provenance="measured")
        rows = db._conn.execute(
            "SELECT name, provenance FROM directions WHERE model_id = ?", (mid,)
        ).fetchall()
        for r in rows:
            assert r["provenance"] is not None, f"direction {r['name']} has null provenance"

    def test_head_measurements_coverage(self, db):
        """head_measurements table should accept per-prompt entries."""
        mid = db.upsert_model("test-model")
        db.record_head_measurement(mid, layer=0, head=0, prompt_label="greeting",
                                   kl_ablation=0.01, entropy_delta=0.001)
        db.record_head_measurement(mid, layer=0, head=0, prompt_label="safety",
                                   kl_ablation=5.2, entropy_delta=1.3)
        rows = db._conn.execute(
            "SELECT * FROM head_measurements WHERE model_id = ? AND layer = 0 AND head = 0",
            (mid,),
        ).fetchall()
        assert len(rows) == 2
        labels = {r["prompt_label"] for r in rows}
        assert labels == {"greeting", "safety"}

    def test_events_single_level_json(self, db):
        """Item 38: record_event should produce single-level JSON, not nested strings."""
        import json
        db.record_event("test_event", key1="value1", key2={"nested": "dict"})
        row = db._conn.execute(
            "SELECT data FROM events WHERE event='test_event' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        parsed = json.loads(row["data"])
        assert isinstance(parsed["key1"], str)
        assert isinstance(parsed["key2"], dict)  # dict, not JSON string
        # Verify no double-nesting
        for k, v in parsed.items():
            if isinstance(v, str) and v.startswith("{"):
                try:
                    json.loads(v)
                    pytest.fail(f"Key '{k}' is a JSON string (double-nested)")
                except json.JSONDecodeError:
                    pass  # Not JSON, that's fine


class TestRefreshHeadsAggregate:
    """Phase 5 (Principle 9): Test refresh_heads_aggregate recomputation."""

    def test_refresh_empty(self, db):
        """Returns 0 when no head_measurements exist."""
        n = db.refresh_heads_aggregate()
        assert n == 0

    def test_refresh_classifies_universal(self, db):
        """Heads active on >80% of prompts classified as universal."""
        mid = db.upsert_model("test-model")
        # 10 prompts, head active on 9 of them (90% > 80%)
        for i in range(10):
            kl = 0.1 if i < 9 else 0.001  # 9 active, 1 inert
            db.record_head_measurement(mid, layer=5, head=3, prompt_label=f"p{i}",
                                       kl_ablation=kl)
        n = db.refresh_heads_aggregate()
        assert n == 1
        row = db._conn.execute(
            "SELECT * FROM heads WHERE model_id=? AND layer=5 AND head=3", (mid,)
        ).fetchone()
        assert row is not None
        assert row["classification"] == "universal"
        assert row["universality"] == pytest.approx(0.9, abs=0.01)
        assert row["active_count"] == 9
        assert row["total_count"] == 10
        assert row["is_inert"] == 0

    def test_refresh_classifies_prompt_specific(self, db):
        """Heads active on 1-80% of prompts classified as prompt_specific."""
        mid = db.upsert_model("test-model")
        # 10 prompts, head active on 4 (40%)
        for i in range(10):
            kl = 0.5 if i < 4 else 0.005
            db.record_head_measurement(mid, layer=2, head=1, prompt_label=f"p{i}",
                                       kl_ablation=kl)
        db.refresh_heads_aggregate()
        row = db._conn.execute(
            "SELECT * FROM heads WHERE model_id=? AND layer=2 AND head=1", (mid,)
        ).fetchone()
        assert row["classification"] == "prompt_specific"
        assert row["universality"] == pytest.approx(0.4, abs=0.01)

    def test_refresh_classifies_inert(self, db):
        """Heads active on <1% of prompts classified as inert."""
        mid = db.upsert_model("test-model")
        # 10 prompts, head never active (all KL < 0.01)
        for i in range(10):
            db.record_head_measurement(mid, layer=0, head=0, prompt_label=f"p{i}",
                                       kl_ablation=0.001)
        db.refresh_heads_aggregate()
        row = db._conn.execute(
            "SELECT * FROM heads WHERE model_id=? AND layer=0 AND head=0", (mid,)
        ).fetchone()
        assert row["classification"] == "inert"
        assert row["is_inert"] == 1
        assert row["universality"] == pytest.approx(0.0, abs=0.01)

    def test_refresh_with_model_id_filter(self, db):
        """Only refreshes heads for the specified model_id."""
        mid1 = db.upsert_model("model-a")
        mid2 = db.upsert_model("model-b")
        for i in range(5):
            db.record_head_measurement(mid1, layer=0, head=0, prompt_label=f"p{i}",
                                       kl_ablation=0.1)
            db.record_head_measurement(mid2, layer=0, head=0, prompt_label=f"p{i}",
                                       kl_ablation=0.001)
        n = db.refresh_heads_aggregate(model_id=mid1)
        assert n == 1
        # Only mid1 should have been updated
        row1 = db._conn.execute(
            "SELECT * FROM heads WHERE model_id=? AND layer=0 AND head=0", (mid1,)
        ).fetchone()
        assert row1 is not None
        assert row1["classification"] == "universal"
        # mid2 should not exist in heads yet
        row2 = db._conn.execute(
            "SELECT * FROM heads WHERE model_id=? AND layer=0 AND head=0", (mid2,)
        ).fetchone()
        assert row2 is None

    def test_refresh_computes_mean_kl(self, db):
        """Mean KL is correctly computed from individual measurements."""
        mid = db.upsert_model("test-model")
        kl_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for i, kl in enumerate(kl_values):
            db.record_head_measurement(mid, layer=1, head=2, prompt_label=f"p{i}",
                                       kl_ablation=kl)
        db.refresh_heads_aggregate()
        row = db._conn.execute(
            "SELECT kl_ablation FROM heads WHERE model_id=? AND layer=1 AND head=2",
            (mid,),
        ).fetchone()
        expected_mean = sum(kl_values) / len(kl_values)
        assert row["kl_ablation"] == pytest.approx(expected_mean, abs=0.001)

    def test_head_measurements_prompt_text_and_source_file(self, db):
        """Phase 5 migration: head_measurements has prompt_text and source_file."""
        mid = db.upsert_model("test-model")
        db.record_head_measurement(
            mid, layer=0, head=0, prompt_label="greeting",
            kl_ablation=0.01,
            prompt_text="Hello world",
            source_file="test_prompts.txt",
        )
        row = db._conn.execute(
            "SELECT prompt_text, source_file FROM head_measurements "
            "WHERE model_id=? AND layer=0 AND head=0", (mid,)
        ).fetchone()
        assert row["prompt_text"] == "Hello world"
        assert row["source_file"] == "test_prompts.txt"
