"""SQLite signal database — persistent, queryable core for the heinrich pipeline.

Replaces in-memory SignalStore with a real database. Every measurement,
every experiment, every shart scan — indexed and queryable.

Schema:
- runs: experiment sessions with timestamps and metadata
- signals: individual measurements with foreign key to run
- blobs: large numpy arrays (residuals, directions) stored as binary

The MCP server reads from this. Scripts write to this. Everything is
queryable from the CLI or programmatically.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Sequence

from .signal import Signal


DEFAULT_DB_PATH = Path("~/.heinrich/signals.db").expanduser()


class SignalDB:
    """SQLite-backed signal store with run tracking."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        self._current_run_id: int | None = None

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                model TEXT,
                script TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER REFERENCES runs(id),
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                model TEXT NOT NULL,
                target TEXT NOT NULL,
                value REAL NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_signals_kind ON signals(kind);
            CREATE INDEX IF NOT EXISTS idx_signals_source ON signals(source);
            CREATE INDEX IF NOT EXISTS idx_signals_model ON signals(model);
            CREATE INDEX IF NOT EXISTS idx_signals_target ON signals(target);
            CREATE INDEX IF NOT EXISTS idx_signals_run ON signals(run_id);
            CREATE INDEX IF NOT EXISTS idx_signals_value ON signals(value DESC);

            CREATE TABLE IF NOT EXISTS blobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER REFERENCES runs(id),
                name TEXT NOT NULL,
                dtype TEXT NOT NULL,
                shape TEXT NOT NULL,
                data BLOB NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_blobs_name ON blobs(name);
            CREATE INDEX IF NOT EXISTS idx_blobs_run ON blobs(run_id);
        """)
        self._conn.commit()

    # === Run management ===

    @contextmanager
    def run(self, name: str, *, model: str = "", script: str = "",
            metadata: dict | None = None) -> Iterator[int]:
        """Context manager for an experiment run. Yields run_id."""
        meta = json.dumps(metadata or {})
        cur = self._conn.execute(
            "INSERT INTO runs (name, model, script, started_at, metadata) VALUES (?, ?, ?, ?, ?)",
            (name, model, script, time.time(), meta),
        )
        run_id = cur.lastrowid
        self._current_run_id = run_id
        self._conn.commit()
        try:
            yield run_id
        finally:
            self._conn.execute(
                "UPDATE runs SET ended_at = ? WHERE id = ?",
                (time.time(), run_id),
            )
            self._conn.commit()
            self._current_run_id = None

    # === Signal operations ===

    def add(self, signal: Signal, *, run_id: int | None = None) -> int:
        """Add a signal. Returns signal id."""
        rid = run_id or self._current_run_id
        meta = json.dumps(signal.metadata) if signal.metadata else "{}"
        cur = self._conn.execute(
            "INSERT INTO signals (run_id, kind, source, model, target, value, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rid, signal.kind, signal.source, signal.model, signal.target,
             signal.value, meta, time.time()),
        )
        self._conn.commit()
        return cur.lastrowid

    def add_many(self, signals: Sequence[Signal], *, run_id: int | None = None) -> int:
        """Bulk insert signals. Returns count inserted."""
        rid = run_id or self._current_run_id
        now = time.time()
        rows = [
            (rid, s.kind, s.source, s.model, s.target, s.value,
             json.dumps(s.metadata) if s.metadata else "{}", now)
            for s in signals
        ]
        self._conn.executemany(
            "INSERT INTO signals (run_id, kind, source, model, target, value, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def query(
        self,
        *,
        kind: str | None = None,
        source: str | None = None,
        model: str | None = None,
        target: str | None = None,
        run_id: int | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        limit: int = 1000,
        order_by: str = "value DESC",
    ) -> list[Signal]:
        """Query signals with filters."""
        clauses = []
        params: list = []

        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if model:
            clauses.append("model = ?")
            params.append(model)
        if target:
            clauses.append("target = ?")
            params.append(target)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if min_value is not None:
            clauses.append("value >= ?")
            params.append(min_value)
        if max_value is not None:
            clauses.append("value <= ?")
            params.append(max_value)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT kind, source, model, target, value, metadata FROM signals WHERE {where} ORDER BY {order_by} LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [
            Signal(
                kind=r["kind"], source=r["source"], model=r["model"],
                target=r["target"], value=r["value"],
                metadata=json.loads(r["metadata"]),
            )
            for r in rows
        ]

    def count(self, *, kind: str | None = None, run_id: int | None = None) -> int:
        """Count signals matching filters."""
        clauses, params = [], []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = " AND ".join(clauses) if clauses else "1=1"
        row = self._conn.execute(f"SELECT COUNT(*) as n FROM signals WHERE {where}", params).fetchone()
        return row["n"]

    def kinds(self) -> list[tuple[str, int]]:
        """List all signal kinds with counts."""
        rows = self._conn.execute(
            "SELECT kind, COUNT(*) as n FROM signals GROUP BY kind ORDER BY n DESC"
        ).fetchall()
        return [(r["kind"], r["n"]) for r in rows]

    def runs(self, limit: int = 50) -> list[dict]:
        """List recent runs."""
        rows = self._conn.execute(
            "SELECT r.id, r.name, r.model, r.script, r.started_at, r.ended_at, r.metadata, "
            "(SELECT COUNT(*) FROM signals s WHERE s.run_id = r.id) as n_signals "
            "FROM runs r ORDER BY r.started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # === Blob operations (numpy arrays) ===

    def save_blob(self, name: str, array, *, run_id: int | None = None) -> int:
        """Save a numpy array as a named blob."""
        import numpy as np
        rid = run_id or self._current_run_id
        data = array.tobytes()
        dtype = str(array.dtype)
        shape = json.dumps(list(array.shape))
        cur = self._conn.execute(
            "INSERT INTO blobs (run_id, name, dtype, shape, data, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (rid, name, dtype, shape, data, time.time()),
        )
        self._conn.commit()
        return cur.lastrowid

    def load_blob(self, name: str, *, run_id: int | None = None):
        """Load a numpy array by name. Returns most recent if no run_id."""
        import numpy as np
        clauses = ["name = ?"]
        params: list = [name]
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = " AND ".join(clauses)
        row = self._conn.execute(
            f"SELECT dtype, shape, data FROM blobs WHERE {where} ORDER BY created_at DESC LIMIT 1",
            params,
        ).fetchone()
        if row is None:
            return None
        dtype = np.dtype(row["dtype"])
        shape = tuple(json.loads(row["shape"]))
        return np.frombuffer(row["data"], dtype=dtype).reshape(shape)

    # === Import/Export ===

    def import_json(self, path: str | Path, *, run_name: str = "import") -> int:
        """Import signals from a JSON file (old SignalStore format)."""
        data = json.loads(Path(path).read_text())
        if isinstance(data, list):
            signals = [Signal(**row) for row in data]
        elif isinstance(data, dict) and "signals" in data:
            signals = [Signal(**row) for row in data["signals"]]
        else:
            return 0
        with self.run(run_name) as rid:
            return self.add_many(signals, run_id=rid)

    def export_json(self, **filters) -> str:
        """Export matching signals as JSON."""
        signals = self.query(**filters)
        return json.dumps([asdict(s) for s in signals], indent=2)

    # === Stats ===

    def summary(self) -> dict[str, Any]:
        """Database summary statistics."""
        n_signals = self._conn.execute("SELECT COUNT(*) as n FROM signals").fetchone()["n"]
        n_runs = self._conn.execute("SELECT COUNT(*) as n FROM runs").fetchone()["n"]
        n_blobs = self._conn.execute("SELECT COUNT(*) as n FROM blobs").fetchone()["n"]
        kinds = self.kinds()
        return {
            "n_signals": n_signals,
            "n_runs": n_runs,
            "n_blobs": n_blobs,
            "kinds": dict(kinds[:20]),
            "db_path": str(self.path),
            "db_size_mb": round(self.path.stat().st_size / 1024 / 1024, 2) if self.path.exists() else 0,
        }

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self):
        return self.count()

    def __repr__(self):
        return f"SignalDB({self.path}, {self.count()} signals)"
