"""SQLite signal database — persistent, queryable core for the heinrich pipeline.

Schema v2: normalized investigation tables (models, experiments, neurons,
sharts, layers, censorship, basins, evaluations, probes, directions, heads,
interpolations, events) plus the legacy signal/blob/derivation tables for
backward compatibility.

Single-writer discipline (ChronoHorn pattern):
  - All writes go through a dedicated writer thread via queue.
  - Reads use a separate autocommit connection (WAL mode).
  - This eliminates lock contention and "database is locked" errors when
    reads and writes overlap.
"""
from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Sequence

from .signal import Signal


DEFAULT_DB_PATH = Path("~/.heinrich/signals.db").expanduser()

_SENTINEL = object()  # poison pill for writer thread shutdown


class SignalDB:
    """SQLite-backed signal store with run tracking and investigation tables.

    Uses ChronoHorn-style single-writer discipline:
    - ``_writer_conn`` — dedicated connection used only by the writer thread.
    - ``_conn``        — read-only autocommit connection for queries.
    - ``_write_queue`` — all mutations are serialized through this queue.
    """

    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # --- writer connection (only touched by _writer_loop) ---
        self._writer_conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._writer_conn.execute("PRAGMA journal_mode=WAL")
        self._writer_conn.execute("PRAGMA synchronous=NORMAL")
        self._writer_conn.row_factory = sqlite3.Row

        # --- reader connection (autocommit via isolation_level=None) ---
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row

        # --- write queue & thread ---
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="signaldb-writer",
        )
        self._writer_thread.start()

        # Create tables synchronously (wait for writer thread to finish)
        self._create_tables()

        self._current_run_id: int | None = None

    # ------------------------------------------------------------------
    # Writer thread
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        """Process write operations from the queue sequentially."""
        conn = self._writer_conn
        while True:
            item = self._write_queue.get()
            if item is _SENTINEL:
                self._write_queue.task_done()
                break
            sql, params, result_event, result_box, is_script = item
            try:
                if is_script:
                    # Multi-statement DDL — use executescript (ignores params)
                    conn.executescript(sql)
                    if result_box is not None:
                        result_box.append(None)
                elif isinstance(sql, list):
                    # batch of (sql, params) pairs
                    cur = None
                    for s, p in sql:
                        cur = conn.execute(s, p)
                    conn.commit()
                    if result_box is not None:
                        result_box.append(cur.lastrowid if cur else None)
                else:
                    cur = conn.execute(sql, params)
                    conn.commit()
                    if result_box is not None:
                        result_box.append(cur.lastrowid)
            except Exception as exc:
                if result_box is not None:
                    result_box.append(exc)
            finally:
                if result_event is not None:
                    result_event.set()
                self._write_queue.task_done()

    def _write(self, sql: str, params: tuple | list = (), *, wait: bool = False) -> int | None:
        """Enqueue a single SQL write.  If *wait*, block until committed and
        return ``lastrowid``."""
        if wait:
            ev = threading.Event()
            box: list = []
            self._write_queue.put((sql, params, ev, box, False))
            ev.wait()
            result = box[0]
            if isinstance(result, Exception):
                raise result
            return result
        else:
            self._write_queue.put((sql, params, None, None, False))
            return None

    def _write_script(self, sql: str, *, wait: bool = False) -> None:
        """Enqueue a multi-statement DDL script via executescript."""
        if wait:
            ev = threading.Event()
            box: list = []
            self._write_queue.put((sql, (), ev, box, True))
            ev.wait()
            result = box[0]
            if isinstance(result, Exception):
                raise result
        else:
            self._write_queue.put((sql, (), None, None, True))

    def _write_many(self, operations: list[tuple[str, tuple | list]], *, wait: bool = False) -> int | None:
        """Enqueue a batch of SQL writes executed in a single transaction.
        If *wait*, block until committed and return last ``lastrowid``."""
        if wait:
            ev = threading.Event()
            box: list = []
            self._write_queue.put((operations, (), ev, box, False))
            ev.wait()
            result = box[0]
            if isinstance(result, Exception):
                raise result
            return result
        else:
            self._write_queue.put((operations, (), None, None, False))
            return None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        ddl = """
            -- === Legacy tables (backward-compatible) ===

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

            CREATE TABLE IF NOT EXISTS derivations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                derived_signal_id INTEGER REFERENCES signals(id),
                source_signal_id INTEGER REFERENCES signals(id),
                relationship TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_derivations_derived ON derivations(derived_signal_id);
            CREATE INDEX IF NOT EXISTS idx_derivations_source ON derivations(source_signal_id);

            -- === Investigation tables ===

            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                family TEXT, params REAL, n_layers INTEGER, n_heads INTEGER,
                d_model INTEGER, n_vocab INTEGER, quantization TEXT, json_blob TEXT
            );

            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER REFERENCES models(id),
                name TEXT NOT NULL, script TEXT, kind TEXT,
                started_at REAL, completed_at REAL, status TEXT DEFAULT 'running',
                n_evaluations INTEGER, elapsed_sec REAL, json_archive TEXT
            );

            CREATE TABLE IF NOT EXISTS neurons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER REFERENCES models(id),
                layer INTEGER NOT NULL, neuron_idx INTEGER NOT NULL,
                name TEXT, category TEXT, max_z REAL,
                delta_chat REAL, delta_safety REAL, causal_effect REAL, json_blob TEXT,
                UNIQUE(model_id, layer, neuron_idx)
            );

            CREATE TABLE IF NOT EXISTS sharts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER REFERENCES models(id),
                token_id INTEGER NOT NULL, token_text TEXT, category TEXT,
                max_z REAL, n_anomalous_neurons INTEGER, top_neuron INTEGER,
                refuse_prob REAL, basin TEXT, manifold_region TEXT, json_blob TEXT,
                UNIQUE(model_id, token_id)
            );

            CREATE TABLE IF NOT EXISTS layers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER REFERENCES models(id),
                layer INTEGER NOT NULL, role TEXT,
                n_chat_neurons INTEGER, top_delta REAL,
                mean_delta_build REAL, mean_delta_explain REAL, dampening REAL,
                effective_rank REAL, direction_gap_safety REAL, direction_gap_language REAL,
                attention_invariance REAL,
                UNIQUE(model_id, layer)
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER REFERENCES experiments(id),
                dataset TEXT NOT NULL, prompt_text TEXT, category TEXT,
                attack TEXT NOT NULL, alpha REAL, framing TEXT,
                refuse_prob REAL, comply_prob REAL, top_token TEXT,
                generation_text TEXT, quality TEXT, is_false_refusal BOOLEAN
            );

            CREATE TABLE IF NOT EXISTS censorship (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER REFERENCES models(id),
                topic TEXT NOT NULL, en_status TEXT, zh_status TEXT,
                en_text TEXT, zh_text TEXT, divergent BOOLEAN,
                UNIQUE(model_id, topic)
            );

            CREATE TABLE IF NOT EXISTS basins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER REFERENCES models(id),
                name TEXT NOT NULL, layer INTEGER,
                pc0 REAL, pc4 REAL, centroid_blob BLOB,
                UNIQUE(model_id, name, layer)
            );

            CREATE TABLE IF NOT EXISTS basin_distances (
                model_id INTEGER REFERENCES models(id),
                basin_a TEXT, basin_b TEXT, layer INTEGER, distance REAL,
                PRIMARY KEY (model_id, basin_a, basin_b, layer)
            );

            CREATE TABLE IF NOT EXISTS directions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER REFERENCES models(id),
                name TEXT NOT NULL, layer INTEGER NOT NULL,
                stability REAL, effect_size REAL, n_dims_90pct INTEGER,
                vector_blob BLOB,
                UNIQUE(model_id, name, layer)
            );

            CREATE TABLE IF NOT EXISTS heads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER REFERENCES models(id),
                layer INTEGER NOT NULL, head INTEGER NOT NULL,
                kl_ablation REAL, is_inert BOOLEAN, oproj_norm REAL,
                oproj_cluster INTEGER, safety_specific BOOLEAN,
                chat_attention_weight REAL, json_blob TEXT,
                UNIQUE(model_id, layer, head)
            );

            CREATE TABLE IF NOT EXISTS probes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER REFERENCES experiments(id),
                step INTEGER, layer INTEGER,
                residual_norm REAL, delta_norm REAL,
                safety_proj REAL, language_proj REAL, chat_proj REAL,
                dampening REAL, pca_pc0 REAL, pca_pc4 REAL,
                basin TEXT, top_neurons TEXT, attention_pos2_weight REAL
            );

            CREATE TABLE IF NOT EXISTS interpolations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER REFERENCES models(id),
                alpha REAL, top_token TEXT, behavior TEXT,
                output_text TEXT, refuse_prob REAL, off_line_distance REAL,
                config_label TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                event TEXT NOT NULL,
                data TEXT DEFAULT '{}'
            );

            -- === Indexes for new tables ===

            CREATE INDEX IF NOT EXISTS idx_evaluations_dataset ON evaluations(dataset);
            CREATE INDEX IF NOT EXISTS idx_evaluations_attack ON evaluations(attack);
            CREATE INDEX IF NOT EXISTS idx_evaluations_category ON evaluations(category);
            CREATE INDEX IF NOT EXISTS idx_sharts_category ON sharts(category);
            CREATE INDEX IF NOT EXISTS idx_sharts_max_z ON sharts(max_z);
            CREATE INDEX IF NOT EXISTS idx_probes_step ON probes(step);
            CREATE INDEX IF NOT EXISTS idx_probes_layer ON probes(layer);
            CREATE INDEX IF NOT EXISTS idx_neurons_category ON neurons(category);
            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
        """
        # DDL is executed synchronously via the writer thread so tables exist
        # before any other method is called.
        self._write_script(ddl, wait=True)

    # ------------------------------------------------------------------
    # Legacy: Run management
    # ------------------------------------------------------------------

    @contextmanager
    def run(self, name: str, *, model: str = "", script: str = "",
            metadata: dict | None = None) -> Iterator[int]:
        """Context manager for an experiment run. Yields run_id."""
        meta = json.dumps(metadata or {})
        run_id = self._write(
            "INSERT INTO runs (name, model, script, started_at, metadata) VALUES (?, ?, ?, ?, ?)",
            (name, model, script, time.time(), meta),
            wait=True,
        )
        self._current_run_id = run_id
        try:
            yield run_id
        finally:
            self._write(
                "UPDATE runs SET ended_at = ? WHERE id = ?",
                (time.time(), run_id),
                wait=True,
            )
            self._current_run_id = None

    # ------------------------------------------------------------------
    # Legacy: Signal operations
    # ------------------------------------------------------------------

    def add(self, signal: Signal, *, run_id: int | None = None) -> int:
        """Add a signal. Returns signal id."""
        rid = run_id or self._current_run_id
        meta = json.dumps(signal.metadata) if signal.metadata else "{}"
        return self._write(
            "INSERT INTO signals (run_id, kind, source, model, target, value, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rid, signal.kind, signal.source, signal.model, signal.target,
             signal.value, meta, time.time()),
            wait=True,
        )

    def add_many(self, signals: Sequence[Signal], *, run_id: int | None = None) -> int:
        """Bulk insert signals. Returns count inserted."""
        rid = run_id or self._current_run_id
        now = time.time()
        ops = [
            (
                "INSERT INTO signals (run_id, kind, source, model, target, value, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (rid, s.kind, s.source, s.model, s.target, s.value,
                 json.dumps(s.metadata) if s.metadata else "{}", now),
            )
            for s in signals
        ]
        self._write_many(ops, wait=True)
        return len(ops)

    # ------------------------------------------------------------------
    # Legacy: Signal provenance
    # ------------------------------------------------------------------

    def add_derived(
        self,
        signal: Signal,
        source_ids: Sequence[int],
        relationship: str,
        *,
        run_id: int | None = None,
    ) -> int:
        """Add a signal that was derived from other signals.

        Inserts the signal, then records derivation edges from each source_id.
        Returns the id of the newly inserted derived signal.
        """
        derived_id = self.add(signal, run_id=run_id)
        now = time.time()
        ops = [
            (
                "INSERT INTO derivations (derived_signal_id, source_signal_id, relationship, created_at) "
                "VALUES (?, ?, ?, ?)",
                (derived_id, src_id, relationship, now),
            )
            for src_id in source_ids
        ]
        self._write_many(ops, wait=True)
        return derived_id

    def get_provenance(self, signal_id: int) -> list[dict]:
        """Return the full provenance chain for a signal.

        Walks derivation edges backwards from derived to sources, recursively.
        Returns a list of dicts with keys: signal_id, relationship, sources.
        The list is ordered from immediate sources to deepest ancestors.
        """
        result = []
        visited: set[int] = set()
        bfs_queue = [signal_id]

        while bfs_queue:
            current_id = bfs_queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            rows = self._conn.execute(
                "SELECT d.source_signal_id, d.relationship, "
                "s.kind, s.source, s.model, s.target, s.value, s.metadata "
                "FROM derivations d "
                "JOIN signals s ON s.id = d.source_signal_id "
                "WHERE d.derived_signal_id = ?",
                (current_id,),
            ).fetchall()

            for r in rows:
                src_id = r["source_signal_id"]
                result.append({
                    "signal_id": src_id,
                    "relationship": r["relationship"],
                    "signal": Signal(
                        kind=r["kind"], source=r["source"], model=r["model"],
                        target=r["target"], value=r["value"],
                        metadata=json.loads(r["metadata"]),
                    ),
                })
                if src_id not in visited:
                    bfs_queue.append(src_id)

        return result

    def get_derived(self, signal_id: int) -> list[dict]:
        """Return signals that were derived from this signal (direct children only).

        Returns list of dicts with keys: signal_id, relationship, signal.
        """
        rows = self._conn.execute(
            "SELECT d.derived_signal_id, d.relationship, "
            "s.kind, s.source, s.model, s.target, s.value, s.metadata "
            "FROM derivations d "
            "JOIN signals s ON s.id = d.derived_signal_id "
            "WHERE d.source_signal_id = ?",
            (signal_id,),
        ).fetchall()

        return [
            {
                "signal_id": r["derived_signal_id"],
                "relationship": r["relationship"],
                "signal": Signal(
                    kind=r["kind"], source=r["source"], model=r["model"],
                    target=r["target"], value=r["value"],
                    metadata=json.loads(r["metadata"]),
                ),
            }
            for r in rows
        ]

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

    # ------------------------------------------------------------------
    # Legacy: Blob operations (numpy arrays)
    # ------------------------------------------------------------------

    def save_blob(self, name: str, array, *, run_id: int | None = None) -> int:
        """Save a numpy array as a named blob."""
        import numpy as np
        rid = run_id or self._current_run_id
        data = array.tobytes()
        dtype = str(array.dtype)
        shape = json.dumps(list(array.shape))
        return self._write(
            "INSERT INTO blobs (run_id, name, dtype, shape, data, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (rid, name, dtype, shape, data, time.time()),
            wait=True,
        )

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

    # ------------------------------------------------------------------
    # Legacy: Import/Export
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Legacy: Stats
    # ------------------------------------------------------------------

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

    # ==================================================================
    # New: investigation write methods
    # ==================================================================

    def _upsert_sql(self, table: str, conflict_cols: list[str],
                    data: dict[str, Any]) -> tuple[str, tuple]:
        """Build an INSERT … ON CONFLICT … DO UPDATE statement.

        Returns (sql, params) tuple.
        """
        cols = list(data.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        conflict = ", ".join(conflict_cols)
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c not in conflict_cols)
        if not updates:
            updates = f"{cols[0]} = excluded.{cols[0]}"
        sql = (
            f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) "
            f"ON CONFLICT({conflict}) DO UPDATE SET {updates}"
        )
        return sql, tuple(data[c] for c in cols)

    # --- models ---

    def upsert_model(self, name: str, **kwargs: Any) -> int:
        """Insert or update a model by name.  Returns model id."""
        data: dict[str, Any] = {"name": name}
        for k in ("family", "params", "n_layers", "n_heads", "d_model",
                   "n_vocab", "quantization", "json_blob"):
            if k in kwargs:
                data[k] = kwargs[k]
        sql, params = self._upsert_sql("models", ["name"], data)
        self._write(sql, params, wait=True)
        row = self._conn.execute(
            "SELECT id FROM models WHERE name = ?", (name,)
        ).fetchone()
        return row["id"]

    # --- experiments ---

    def record_experiment(self, name: str, model_id: int, **kwargs: Any) -> int:
        """Record a new experiment.  Returns experiment id."""
        data: dict[str, Any] = {
            "name": name,
            "model_id": model_id,
            "started_at": kwargs.get("started_at", time.time()),
        }
        for k in ("script", "kind", "completed_at", "status",
                   "n_evaluations", "elapsed_sec", "json_archive"):
            if k in kwargs:
                data[k] = kwargs[k]
        cols = list(data.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        sql = f"INSERT INTO experiments ({col_names}) VALUES ({placeholders})"
        return self._write(sql, tuple(data[c] for c in cols), wait=True)

    # --- evaluations ---

    def record_evaluation(self, experiment_id: int, **kwargs: Any) -> int:
        """Record a single evaluation row.  Returns evaluation id."""
        data: dict[str, Any] = {"experiment_id": experiment_id}
        for k in ("dataset", "prompt_text", "category", "attack", "alpha",
                   "framing", "refuse_prob", "comply_prob", "top_token",
                   "generation_text", "quality", "is_false_refusal"):
            if k in kwargs:
                data[k] = kwargs[k]
        cols = list(data.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        sql = f"INSERT INTO evaluations ({col_names}) VALUES ({placeholders})"
        return self._write(sql, tuple(data[c] for c in cols), wait=True)

    # --- neurons ---

    def record_neuron(self, model_id: int, layer: int, neuron_idx: int, **kwargs: Any) -> int:
        """Upsert a neuron row.  Returns neuron id."""
        data: dict[str, Any] = {
            "model_id": model_id, "layer": layer, "neuron_idx": neuron_idx,
        }
        for k in ("name", "category", "max_z", "delta_chat", "delta_safety",
                   "causal_effect", "json_blob"):
            if k in kwargs:
                data[k] = kwargs[k]
        sql, params = self._upsert_sql("neurons", ["model_id", "layer", "neuron_idx"], data)
        self._write(sql, params, wait=True)
        row = self._conn.execute(
            "SELECT id FROM neurons WHERE model_id = ? AND layer = ? AND neuron_idx = ?",
            (model_id, layer, neuron_idx),
        ).fetchone()
        return row["id"]

    # --- sharts ---

    def record_shart(self, model_id: int, token_id: int, **kwargs: Any) -> int:
        """Upsert a shart row.  Returns shart id."""
        data: dict[str, Any] = {"model_id": model_id, "token_id": token_id}
        for k in ("token_text", "category", "max_z", "n_anomalous_neurons",
                   "top_neuron", "refuse_prob", "basin", "manifold_region", "json_blob"):
            if k in kwargs:
                data[k] = kwargs[k]
        sql, params = self._upsert_sql("sharts", ["model_id", "token_id"], data)
        self._write(sql, params, wait=True)
        row = self._conn.execute(
            "SELECT id FROM sharts WHERE model_id = ? AND token_id = ?",
            (model_id, token_id),
        ).fetchone()
        return row["id"]

    # --- layers ---

    def record_layer(self, model_id: int, layer: int, **kwargs: Any) -> int:
        """Upsert a layer row.  Returns layer id."""
        data: dict[str, Any] = {"model_id": model_id, "layer": layer}
        for k in ("role", "n_chat_neurons", "top_delta", "mean_delta_build",
                   "mean_delta_explain", "dampening", "effective_rank",
                   "direction_gap_safety", "direction_gap_language",
                   "attention_invariance"):
            if k in kwargs:
                data[k] = kwargs[k]
        sql, params = self._upsert_sql("layers", ["model_id", "layer"], data)
        self._write(sql, params, wait=True)
        row = self._conn.execute(
            "SELECT id FROM layers WHERE model_id = ? AND layer = ?",
            (model_id, layer),
        ).fetchone()
        return row["id"]

    # --- censorship ---

    def record_censorship(self, model_id: int, topic: str, **kwargs: Any) -> int:
        """Upsert a censorship row.  Returns censorship id."""
        data: dict[str, Any] = {"model_id": model_id, "topic": topic}
        for k in ("en_status", "zh_status", "en_text", "zh_text", "divergent"):
            if k in kwargs:
                data[k] = kwargs[k]
        sql, params = self._upsert_sql("censorship", ["model_id", "topic"], data)
        self._write(sql, params, wait=True)
        row = self._conn.execute(
            "SELECT id FROM censorship WHERE model_id = ? AND topic = ?",
            (model_id, topic),
        ).fetchone()
        return row["id"]

    # --- basins ---

    def record_basin(self, model_id: int, name: str, **kwargs: Any) -> int:
        """Upsert a basin row.  Returns basin id."""
        data: dict[str, Any] = {"model_id": model_id, "name": name}
        for k in ("layer", "pc0", "pc4", "centroid_blob"):
            if k in kwargs:
                data[k] = kwargs[k]
        conflict_cols = ["model_id", "name", "layer"]
        # layer must be present for the conflict clause
        if "layer" not in data:
            data["layer"] = None
        sql, params = self._upsert_sql("basins", conflict_cols, data)
        self._write(sql, params, wait=True)
        layer = data.get("layer")
        if layer is not None:
            row = self._conn.execute(
                "SELECT id FROM basins WHERE model_id = ? AND name = ? AND layer = ?",
                (model_id, name, layer),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT id FROM basins WHERE model_id = ? AND name = ? AND layer IS NULL",
                (model_id, name),
            ).fetchone()
        return row["id"]

    # --- basin_distances ---

    def record_basin_distance(self, model_id: int, a: str, b: str, layer: int, distance: float) -> None:
        """Upsert a basin-distance measurement."""
        sql, params = self._upsert_sql(
            "basin_distances",
            ["model_id", "basin_a", "basin_b", "layer"],
            {"model_id": model_id, "basin_a": a, "basin_b": b,
             "layer": layer, "distance": distance},
        )
        self._write(sql, params, wait=True)

    # --- directions ---

    def record_direction(self, model_id: int, name: str, layer: int, **kwargs: Any) -> int:
        """Upsert a direction row.  Returns direction id."""
        data: dict[str, Any] = {"model_id": model_id, "name": name, "layer": layer}
        for k in ("stability", "effect_size", "n_dims_90pct", "vector_blob"):
            if k in kwargs:
                data[k] = kwargs[k]
        sql, params = self._upsert_sql("directions", ["model_id", "name", "layer"], data)
        self._write(sql, params, wait=True)
        row = self._conn.execute(
            "SELECT id FROM directions WHERE model_id = ? AND name = ? AND layer = ?",
            (model_id, name, layer),
        ).fetchone()
        return row["id"]

    # --- heads ---

    def record_head(self, model_id: int, layer: int, head: int, **kwargs: Any) -> int:
        """Upsert an attention head row.  Returns head id."""
        data: dict[str, Any] = {"model_id": model_id, "layer": layer, "head": head}
        for k in ("kl_ablation", "is_inert", "oproj_norm", "oproj_cluster",
                   "safety_specific", "chat_attention_weight", "json_blob"):
            if k in kwargs:
                data[k] = kwargs[k]
        sql, params = self._upsert_sql("heads", ["model_id", "layer", "head"], data)
        self._write(sql, params, wait=True)
        row = self._conn.execute(
            "SELECT id FROM heads WHERE model_id = ? AND layer = ? AND head = ?",
            (model_id, layer, head),
        ).fetchone()
        return row["id"]

    # --- probes ---

    def record_probe(self, experiment_id: int, step: int, layer: int, **kwargs: Any) -> int:
        """Insert a probe measurement.  Returns probe id."""
        data: dict[str, Any] = {
            "experiment_id": experiment_id, "step": step, "layer": layer,
        }
        for k in ("residual_norm", "delta_norm", "safety_proj", "language_proj",
                   "chat_proj", "dampening", "pca_pc0", "pca_pc4",
                   "basin", "top_neurons", "attention_pos2_weight"):
            if k in kwargs:
                data[k] = kwargs[k]
        cols = list(data.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        sql = f"INSERT INTO probes ({col_names}) VALUES ({placeholders})"
        return self._write(sql, tuple(data[c] for c in cols), wait=True)

    # --- interpolations ---

    def record_interpolation(self, model_id: int, alpha: float, **kwargs: Any) -> int:
        """Insert an interpolation row.  Returns interpolation id."""
        data: dict[str, Any] = {"model_id": model_id, "alpha": alpha}
        for k in ("top_token", "behavior", "output_text", "refuse_prob",
                   "off_line_distance", "config_label"):
            if k in kwargs:
                data[k] = kwargs[k]
        cols = list(data.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        sql = f"INSERT INTO interpolations ({col_names}) VALUES ({placeholders})"
        return self._write(sql, tuple(data[c] for c in cols), wait=True)

    # --- events ---

    def record_event(self, event: str, **data: Any) -> int:
        """Record a timestamped event with optional JSON data."""
        return self._write(
            "INSERT INTO events (ts, event, data) VALUES (?, ?, ?)",
            (time.time(), event, json.dumps(data)),
            wait=True,
        )

    # ==================================================================
    # New: investigation read methods
    # ==================================================================

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
        return dict(row) if row is not None else None

    @staticmethod
    def _rows_to_dicts(rows: list) -> list[dict]:
        return [dict(r) for r in rows]

    # --- models ---

    def get_model(self, name: str) -> dict | None:
        """Look up a model by name."""
        row = self._conn.execute(
            "SELECT * FROM models WHERE name = ?", (name,)
        ).fetchone()
        return self._row_to_dict(row)

    # --- sharts ---

    def get_sharts(
        self,
        model_id: int | None = None,
        category: str | None = None,
        min_z: float | None = None,
        top_k: int = 50,
    ) -> list[dict]:
        """Query sharts with optional filters, ordered by max_z descending."""
        clauses: list[str] = []
        params: list = []
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if min_z is not None:
            clauses.append("max_z >= ?")
            params.append(min_z)
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = self._conn.execute(
            f"SELECT * FROM sharts WHERE {where} ORDER BY max_z DESC LIMIT ?",
            (*params, top_k),
        ).fetchall()
        return self._rows_to_dicts(rows)

    # --- neurons ---

    def get_neurons(
        self,
        model_id: int | None = None,
        category: str | None = None,
        layer: int | None = None,
        top_k: int = 50,
    ) -> list[dict]:
        """Query neurons with optional filters, ordered by max_z descending."""
        clauses: list[str] = []
        params: list = []
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if layer is not None:
            clauses.append("layer = ?")
            params.append(layer)
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = self._conn.execute(
            f"SELECT * FROM neurons WHERE {where} ORDER BY max_z DESC LIMIT ?",
            (*params, top_k),
        ).fetchall()
        return self._rows_to_dicts(rows)

    # --- evaluations ---

    def get_evaluations(
        self,
        experiment_id: int | None = None,
        dataset: str | None = None,
        attack: str | None = None,
        category: str | None = None,
        top_k: int = 100,
    ) -> list[dict]:
        """Query evaluations with optional filters."""
        clauses: list[str] = []
        params: list = []
        if experiment_id is not None:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if dataset is not None:
            clauses.append("dataset = ?")
            params.append(dataset)
        if attack is not None:
            clauses.append("attack = ?")
            params.append(attack)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = self._conn.execute(
            f"SELECT * FROM evaluations WHERE {where} ORDER BY id DESC LIMIT ?",
            (*params, top_k),
        ).fetchall()
        return self._rows_to_dicts(rows)

    # --- layer map ---

    def get_layer_map(self, model_id: int) -> list[dict]:
        """Return all layers for a model, ordered by layer index."""
        rows = self._conn.execute(
            "SELECT * FROM layers WHERE model_id = ? ORDER BY layer",
            (model_id,),
        ).fetchall()
        return self._rows_to_dicts(rows)

    # --- censorship ---

    def get_censorship(self, model_id: int, divergent_only: bool = False) -> list[dict]:
        """Return censorship rows for a model."""
        if divergent_only:
            rows = self._conn.execute(
                "SELECT * FROM censorship WHERE model_id = ? AND divergent = 1 ORDER BY topic",
                (model_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM censorship WHERE model_id = ? ORDER BY topic",
                (model_id,),
            ).fetchall()
        return self._rows_to_dicts(rows)

    # --- basin geometry ---

    def get_basin_geometry(self, model_id: int) -> dict:
        """Return basins and inter-basin distances for a model.

        Returns ``{"basins": [...], "distances": [...]}``.
        """
        basins = self._conn.execute(
            "SELECT * FROM basins WHERE model_id = ? ORDER BY name, layer",
            (model_id,),
        ).fetchall()
        distances = self._conn.execute(
            "SELECT * FROM basin_distances WHERE model_id = ? ORDER BY layer, basin_a, basin_b",
            (model_id,),
        ).fetchall()
        return {
            "basins": self._rows_to_dicts(basins),
            "distances": self._rows_to_dicts(distances),
        }

    # --- safety report ---

    def safety_report(
        self,
        model_id: int | None = None,
        dataset: str | None = None,
    ) -> dict:
        """Aggregate refusal rates by category and attack.

        Returns ``{"by_category": {cat: {"n": ..., "mean_refuse": ...}},
                   "by_attack":   {atk: {"n": ..., "mean_refuse": ...}},
                   "overall":     {"n": ..., "mean_refuse": ...}}``.
        """
        join = ""
        clauses: list[str] = []
        params: list = []
        if model_id is not None:
            join = " JOIN experiments e ON e.id = ev.experiment_id"
            clauses.append("e.model_id = ?")
            params.append(model_id)
        if dataset is not None:
            clauses.append("ev.dataset = ?")
            params.append(dataset)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        base = f"FROM evaluations ev{join}{where}"

        # by category
        rows_cat = self._conn.execute(
            f"SELECT ev.category, COUNT(*) as n, AVG(ev.refuse_prob) as mean_refuse "
            f"{base} GROUP BY ev.category ORDER BY n DESC",
            params,
        ).fetchall()
        by_category = {
            r["category"]: {"n": r["n"], "mean_refuse": r["mean_refuse"]}
            for r in rows_cat
        }

        # by attack
        rows_atk = self._conn.execute(
            f"SELECT ev.attack, COUNT(*) as n, AVG(ev.refuse_prob) as mean_refuse "
            f"{base} GROUP BY ev.attack ORDER BY n DESC",
            params,
        ).fetchall()
        by_attack = {
            r["attack"]: {"n": r["n"], "mean_refuse": r["mean_refuse"]}
            for r in rows_atk
        }

        # overall
        row_all = self._conn.execute(
            f"SELECT COUNT(*) as n, AVG(ev.refuse_prob) as mean_refuse {base}",
            params,
        ).fetchone()
        overall = {"n": row_all["n"], "mean_refuse": row_all["mean_refuse"]}

        return {
            "by_category": by_category,
            "by_attack": by_attack,
            "overall": overall,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Shut down the writer thread and close both connections."""
        # Drain the queue, then send poison pill
        self._write_queue.put(_SENTINEL)
        self._writer_thread.join(timeout=5.0)
        self._writer_conn.close()
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self):
        return self.count()

    def __repr__(self):
        return f"SignalDB({self.path}, {self.count()} signals)"
