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
