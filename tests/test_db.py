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
