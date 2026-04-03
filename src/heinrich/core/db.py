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


DEFAULT_DB_PATH = Path("./data/heinrich.db")

_SENTINEL = object()  # poison pill for writer thread shutdown

_COMMENTARY_EVENTS = frozenset({
    "investigation_finding", "data_note", "methodology_note",
    "paper_db_sync", "scope_note",
})


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
        """Process write operations from the queue sequentially.

        Retries on SQLITE_BUSY (database locked) up to 5 times with
        exponential back-off.  This handles concurrent writes from scorer
        subprocesses — SQLite WAL mode supports concurrent readers + one
        writer per process, but multiple processes writing simultaneously
        can cause SQLITE_BUSY.
        """
        conn = self._writer_conn
        while True:
            item = self._write_queue.get()
            if item is _SENTINEL:
                self._write_queue.task_done()
                break
            sql, params, result_event, result_box, is_script = item
            try:
                for attempt in range(5):
                    try:
                        if is_script:
                            conn.executescript(sql)
                            if result_box is not None:
                                result_box.append(None)
                        elif isinstance(sql, list):
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
                        break  # success — exit retry loop
                    except sqlite3.OperationalError as e:
                        if "locked" in str(e) and attempt < 4:
                            time.sleep(0.1 * (attempt + 1))
                            continue
                        raise
            except Exception as exc:
                if result_box is not None:
                    result_box.append(exc)
                else:
                    # Fire-and-forget write failed silently.
                    # Log so failures are visible, not swallowed.
                    import warnings
                    warnings.warn(
                        f"Async DB write failed: {exc}\n  SQL: {str(sql)[:100]}",
                        stacklevel=1,
                    )
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
        # --- Schema migrations ---
        self._migrate()

    # ------------------------------------------------------------------
    # Schema migration (item 64)
    # ------------------------------------------------------------------

    def _migrate(self) -> None:
        """Run incremental schema migrations. Version tracked in schema_version table."""
        self._write_script(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);",
            wait=True,
        )
        row = self._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        current = row["version"] if row else 0

        if current < 1:
            # Item 53: add severity column to events
            try:
                self._write_script(
                    "ALTER TABLE events ADD COLUMN severity TEXT DEFAULT 'info';",
                    wait=True,
                )
            except sqlite3.OperationalError:
                pass  # column may already exist
            if current == 0:
                self._write(
                    "INSERT INTO schema_version (version) VALUES (?)", (1,), wait=True,
                )
            else:
                self._write(
                    "UPDATE schema_version SET version = ?", (1,), wait=True,
                )

        if current < 2:
            # Items 13,15,18,19,21,22,24,25: provenance column on all data tables
            for table in ("neurons", "sharts", "layers", "evaluations", "basins",
                          "basin_distances", "directions", "heads", "interpolations"):
                try:
                    self._write_script(
                        f"ALTER TABLE {table} ADD COLUMN provenance TEXT DEFAULT 'unknown';",
                        wait=True,
                    )
                except sqlite3.OperationalError:
                    pass  # column may already exist

            # Items 4,14,15: canonical_name for model deduplication
            try:
                self._write_script(
                    "ALTER TABLE models ADD COLUMN canonical_name TEXT;",
                    wait=True,
                )
            except sqlite3.OperationalError:
                pass

            # Items 6,17: n_prompts column on evaluations
            try:
                self._write_script(
                    "ALTER TABLE evaluations ADD COLUMN n_prompts INTEGER;",
                    wait=True,
                )
            except sqlite3.OperationalError:
                pass

            # Item 4: preregistrations table
            try:
                self._write_script("""
                    CREATE TABLE IF NOT EXISTS preregistrations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        hypothesis TEXT NOT NULL,
                        test TEXT NOT NULL,
                        confirm_if TEXT NOT NULL,
                        falsify_if TEXT NOT NULL,
                        chance_level REAL DEFAULT 0.5,
                        n_samples INTEGER DEFAULT 100,
                        created_at REAL NOT NULL,
                        result TEXT,
                        verified BOOLEAN
                    );
                """, wait=True)
            except sqlite3.OperationalError:
                pass

            self._write(
                "UPDATE schema_version SET version = ?", (2,), wait=True,
            )

        if current < 3:
            # Item 3: head_measurements table for per-prompt atlas data
            try:
                self._write_script("""
                    CREATE TABLE IF NOT EXISTS head_measurements (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id INTEGER,
                        layer INTEGER NOT NULL,
                        head INTEGER NOT NULL,
                        prompt_label TEXT,
                        kl_ablation REAL,
                        entropy_delta REAL,
                        top_changed BOOLEAN,
                        provenance TEXT DEFAULT 'ingested'
                    );
                    CREATE INDEX IF NOT EXISTS idx_hm_layer_head
                        ON head_measurements(layer, head);
                """, wait=True)
            except sqlite3.OperationalError:
                pass

            # Item 23: field_mapping column on probes
            try:
                self._write_script(
                    "ALTER TABLE probes ADD COLUMN field_mapping TEXT;",
                    wait=True,
                )
            except sqlite3.OperationalError:
                pass

            self._write(
                "UPDATE schema_version SET version = ?", (3,), wait=True,
            )

        if current < 4:
            # Phase 0A: recipe column on all data tables (Principle 1)
            for table in ("evaluations", "neurons", "sharts", "layers", "heads",
                          "basins", "basin_distances", "directions",
                          "interpolations", "probes", "head_measurements"):
                try:
                    self._write_script(
                        f"ALTER TABLE {table} ADD COLUMN recipe TEXT;",
                        wait=True,
                    )
                except sqlite3.OperationalError:
                    pass  # column may already exist

            # Phase 0B: config_hash column on models (Principle 4)
            try:
                self._write_script(
                    "ALTER TABLE models ADD COLUMN config_hash TEXT;",
                    wait=True,
                )
            except sqlite3.OperationalError:
                pass

            # Phase 0C: purge commentary events (Principle 3)
            try:
                self._write(
                    "DELETE FROM events WHERE event IN "
                    "('investigation_finding', 'data_note', 'methodology_note', "
                    "'paper_db_sync', 'scope_note', 'classifier_disagreement')",
                    (),
                    wait=True,
                )
            except sqlite3.OperationalError:
                pass

            self._write(
                "UPDATE schema_version SET version = ?", (4,), wait=True,
            )

        if current < 5:
            # Phase 5: Add prompt_text and source_file columns to head_measurements
            for col in ("prompt_text", "source_file"):
                try:
                    self._write_script(
                        f"ALTER TABLE head_measurements ADD COLUMN {col} TEXT;",
                        wait=True,
                    )
                except sqlite3.OperationalError:
                    pass  # column may already exist

            # Phase 5: Add universality and classification columns to heads
            for col, col_type, default in [
                ("universality", "REAL", None),
                ("classification", "TEXT", None),
                ("active_count", "INTEGER", None),
                ("total_count", "INTEGER", None),
            ]:
                try:
                    stmt = f"ALTER TABLE heads ADD COLUMN {col} {col_type}"
                    if default is not None:
                        stmt += f" DEFAULT {default}"
                    stmt += ";"
                    self._write_script(stmt, wait=True)
                except sqlite3.OperationalError:
                    pass

            self._write(
                "UPDATE schema_version SET version = ?", (5,), wait=True,
            )

        if current < 6:
            # Phase 1 eval-pipeline tables: prompts, generations, scores, calibration
            self._write_script("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    category TEXT,
                    is_benign BOOLEAN DEFAULT 0,
                    sha256 TEXT,
                    UNIQUE(text, source)
                );

                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER REFERENCES models(id),
                    prompt_id INTEGER REFERENCES prompts(id),
                    prompt_text TEXT NOT NULL,
                    prompt_source TEXT,
                    prompt_category TEXT,
                    condition TEXT NOT NULL,
                    generation_text TEXT NOT NULL,
                    top_token TEXT,
                    n_tokens INTEGER,
                    created_at REAL NOT NULL,
                    recipe TEXT
                );

                CREATE TABLE IF NOT EXISTS scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation_id INTEGER REFERENCES generations(id),
                    scorer TEXT NOT NULL,
                    label TEXT,
                    confidence REAL,
                    raw_output TEXT,
                    created_at REAL NOT NULL,
                    UNIQUE(generation_id, scorer)
                );

                CREATE TABLE IF NOT EXISTS calibration (
                    scorer TEXT NOT NULL,
                    model_id INTEGER REFERENCES models(id),
                    fpr REAL,
                    fnr REAL,
                    n_benign INTEGER,
                    n_harmful INTEGER,
                    calibrated_at REAL,
                    PRIMARY KEY(scorer, model_id)
                );

                CREATE INDEX IF NOT EXISTS idx_generations_model ON generations(model_id);
                CREATE INDEX IF NOT EXISTS idx_generations_condition ON generations(condition);
                CREATE INDEX IF NOT EXISTS idx_scores_generation ON scores(generation_id);
                CREATE INDEX IF NOT EXISTS idx_scores_scorer ON scores(scorer);
                CREATE INDEX IF NOT EXISTS idx_prompts_source ON prompts(source);
            """, wait=True)

            self._write(
                "UPDATE schema_version SET version = ?", (6,), wait=True,
            )

        if current < 7:
            # Item 4: conditions table for discover → attack → eval pipeline
            self._write_script("""
                CREATE TABLE IF NOT EXISTS conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER REFERENCES models(id),
                    name TEXT NOT NULL,
                    kind TEXT,
                    params TEXT,
                    source TEXT,
                    created_at REAL,
                    UNIQUE(model_id, name)
                );
                CREATE INDEX IF NOT EXISTS idx_conditions_model ON conditions(model_id);
            """, wait=True)

            self._write(
                "UPDATE schema_version SET version = ?", (7,), wait=True,
            )

        if current < 8:
            # B1: add distribution columns to calibration for measurement scorers
            for col in ("benign_dist", "harmful_dist"):
                try:
                    self._write_script(
                        f"ALTER TABLE calibration ADD COLUMN {col} TEXT;",
                        wait=True,
                    )
                except sqlite3.OperationalError:
                    pass  # column may already exist
            self._write(
                "UPDATE schema_version SET version = ?", (8,), wait=True,
            )

        if current < 9:
            # Ground truth columns on generations: data measured at generation time
            for col, col_type in [
                ("first_token_id", "INTEGER"),
                ("refuse_prob", "REAL"),
                ("is_degenerate", "BOOLEAN"),
            ]:
                try:
                    self._write_script(
                        f"ALTER TABLE generations ADD COLUMN {col} {col_type};",
                        wait=True,
                    )
                except sqlite3.OperationalError:
                    pass  # column may already exist
            self._write(
                "UPDATE schema_version SET version = ?", (9,), wait=True,
            )

        if current < 10:
            # Geometry columns: pre-linguistic signals from the forward pass
            for col, col_type in [
                ("logit_entropy", "REAL"),
                ("top_k_tokens", "TEXT"),       # JSON: [(id, token, prob), ...]
                ("safety_trajectory", "TEXT"),   # JSON: contrastive projection per layer
            ]:
                try:
                    self._write_script(
                        f"ALTER TABLE generations ADD COLUMN {col} {col_type};",
                        wait=True,
                    )
                except sqlite3.OperationalError:
                    pass
            self._write(
                "UPDATE schema_version SET version = ?", (10,), wait=True,
            )

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
        """Database summary statistics including normalized tables."""
        n_signals = self._conn.execute("SELECT COUNT(*) as n FROM signals").fetchone()["n"]
        n_runs = self._conn.execute("SELECT COUNT(*) as n FROM runs").fetchone()["n"]
        n_blobs = self._conn.execute("SELECT COUNT(*) as n FROM blobs").fetchone()["n"]
        kinds = self.kinds()

        # Normalized table counts
        normalized_tables = [
            "models", "experiments", "evaluations", "neurons", "sharts",
            "layers", "censorship", "basins", "basin_distances", "directions",
            "heads", "probes", "interpolations", "events",
            "prompts", "generations", "scores", "calibration", "conditions",
        ]
        table_counts = {}
        for t in normalized_tables:
            try:
                row = self._conn.execute(f"SELECT COUNT(*) as n FROM {t}").fetchone()
                table_counts[t] = row["n"]
            except sqlite3.OperationalError:
                table_counts[t] = 0

        return {
            "n_signals": n_signals,
            "n_runs": n_runs,
            "n_blobs": n_blobs,
            "kinds": dict(kinds[:20]),
            "normalized_tables": table_counts,
            "db_path": str(self.path),
            "db_size_mb": round(self.path.stat().st_size / 1024 / 1024, 2) if self.path.exists() else 0,
            "provenance_note": (
                "Provenance labels are self-assigned by the code that writes data. "
                "No external verification. Trust provenance for data-source tracking, "
                "not for correctness guarantees."
            ),
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

    @staticmethod
    def _make_recipe(function_name: str, args: dict) -> str:
        import hashlib
        import subprocess
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, timeout=5
            ).strip()[:12]
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            git_hash = "unknown"
        # Hash long values instead of truncating — preserves verifiability
        processed_args = {}
        for k, v in args.items():
            s = str(v)
            if len(s) > 200:
                h = hashlib.sha256(s.encode()).hexdigest()[:12]
                processed_args[k] = f"<{len(s)} chars, sha256={h}>"
            else:
                processed_args[k] = s
        return json.dumps({
            "function": function_name,
            "args": processed_args,
            "git_hash": git_hash,
            "timestamp": time.time(),
        })

    # --- models ---

    def upsert_model(self, name: str, **kwargs: Any) -> int:
        """Insert or update a model by name.  Returns model id.

        If *config_hash* is provided, check for an existing model with that
        hash first.  If found, return that model's id regardless of name.
        """
        config_hash = kwargs.pop("config_hash", None)

        # Principle 4: identity by config hash — deduplicate before insert
        if config_hash is not None:
            existing = self._conn.execute(
                "SELECT id FROM models WHERE config_hash = ?", (config_hash,)
            ).fetchone()
            if existing is not None:
                return existing["id"]

        data: dict[str, Any] = {"name": name}
        for k in ("family", "params", "n_layers", "n_heads", "d_model",
                   "n_vocab", "quantization", "json_blob"):
            if k in kwargs:
                data[k] = kwargs[k]
        if config_hash is not None:
            data["config_hash"] = config_hash
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

    def record_evaluation(self, experiment_id: int, *, recipe: str | None = None, **kwargs: Any) -> int:
        """Record a single evaluation row.  Returns evaluation id."""
        data: dict[str, Any] = {"experiment_id": experiment_id}
        for k in ("dataset", "prompt_text", "category", "attack", "alpha",
                   "framing", "refuse_prob", "comply_prob", "top_token",
                   "generation_text", "quality", "is_false_refusal",
                   "n_prompts", "provenance"):
            if k in kwargs:
                data[k] = kwargs[k]
        if recipe is not None:
            data["recipe"] = recipe
        cols = list(data.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        sql = f"INSERT INTO evaluations ({col_names}) VALUES ({placeholders})"
        return self._write(sql, tuple(data[c] for c in cols), wait=True)

    # --- neurons ---

    def record_neuron(self, model_id: int, layer: int, neuron_idx: int, *, recipe: str | None = None, **kwargs: Any) -> int:
        """Upsert a neuron row.  Returns neuron id."""
        data: dict[str, Any] = {
            "model_id": model_id, "layer": layer, "neuron_idx": neuron_idx,
        }
        for k in ("name", "category", "max_z", "delta_chat", "delta_safety",
                   "causal_effect", "json_blob", "provenance"):
            if k in kwargs:
                data[k] = kwargs[k]
        if recipe is not None:
            data["recipe"] = recipe
        sql, params = self._upsert_sql("neurons", ["model_id", "layer", "neuron_idx"], data)
        self._write(sql, params, wait=True)
        row = self._conn.execute(
            "SELECT id FROM neurons WHERE model_id = ? AND layer = ? AND neuron_idx = ?",
            (model_id, layer, neuron_idx),
        ).fetchone()
        return row["id"]

    # --- sharts ---

    def record_shart(self, model_id: int, token_id: int, *, recipe: str | None = None, **kwargs: Any) -> int:
        """Upsert a shart row.  Returns shart id."""
        data: dict[str, Any] = {"model_id": model_id, "token_id": token_id}
        for k in ("token_text", "category", "max_z", "n_anomalous_neurons",
                   "top_neuron", "refuse_prob", "basin", "manifold_region", "json_blob",
                   "provenance"):
            if k in kwargs:
                data[k] = kwargs[k]
        if recipe is not None:
            data["recipe"] = recipe
        sql, params = self._upsert_sql("sharts", ["model_id", "token_id"], data)
        self._write(sql, params, wait=True)
        row = self._conn.execute(
            "SELECT id FROM sharts WHERE model_id = ? AND token_id = ?",
            (model_id, token_id),
        ).fetchone()
        return row["id"]

    # --- layers ---

    def record_layer(self, model_id: int, layer: int, *, recipe: str | None = None, **kwargs: Any) -> int:
        """Upsert a layer row.  Returns layer id."""
        data: dict[str, Any] = {"model_id": model_id, "layer": layer}
        for k in ("role", "n_chat_neurons", "top_delta", "mean_delta_build",
                   "mean_delta_explain", "dampening", "effective_rank",
                   "direction_gap_safety", "direction_gap_language",
                   "attention_invariance", "provenance"):
            if k in kwargs:
                data[k] = kwargs[k]
        if recipe is not None:
            data["recipe"] = recipe
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

    def record_basin(self, model_id: int, name: str, *, recipe: str | None = None, **kwargs: Any) -> int:
        """Upsert a basin row.  Returns basin id."""
        data: dict[str, Any] = {"model_id": model_id, "name": name}
        for k in ("layer", "pc0", "pc4", "centroid_blob", "provenance"):
            if k in kwargs:
                data[k] = kwargs[k]
        if recipe is not None:
            data["recipe"] = recipe
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

    def record_basin_distance(self, model_id: int, a: str, b: str, layer: int, distance: float, *, provenance: str | None = None, recipe: str | None = None) -> None:
        """Upsert a basin-distance measurement."""
        data = {"model_id": model_id, "basin_a": a, "basin_b": b,
                "layer": layer, "distance": distance}
        if provenance is not None:
            data["provenance"] = provenance
        if recipe is not None:
            data["recipe"] = recipe
        sql, params = self._upsert_sql(
            "basin_distances",
            ["model_id", "basin_a", "basin_b", "layer"],
            data,
        )
        self._write(sql, params, wait=True)

    # --- directions ---

    def record_direction(self, model_id: int, name: str, layer: int, *, recipe: str | None = None, **kwargs: Any) -> int:
        """Upsert a direction row.  Returns direction id."""
        data: dict[str, Any] = {"model_id": model_id, "name": name, "layer": layer}
        for k in ("stability", "effect_size", "n_dims_90pct", "vector_blob", "provenance"):
            if k in kwargs:
                data[k] = kwargs[k]
        if recipe is not None:
            data["recipe"] = recipe
        sql, params = self._upsert_sql("directions", ["model_id", "name", "layer"], data)
        self._write(sql, params, wait=True)
        row = self._conn.execute(
            "SELECT id FROM directions WHERE model_id = ? AND name = ? AND layer = ?",
            (model_id, name, layer),
        ).fetchone()
        return row["id"]

    def get_direction_vector(self, name: str, layer: int, model_id: int | None = None):
        """Load a direction vector by name and layer.

        Returns the numpy array if vector_blob is stored, or None with a
        message explaining how to recompute it.
        """
        import numpy as np
        if model_id is None:
            row = self._conn.execute(
                "SELECT vector_blob FROM directions WHERE name = ? AND layer = ? LIMIT 1",
                (name, layer),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT vector_blob FROM directions WHERE model_id = ? AND name = ? AND layer = ?",
                (model_id, name, layer),
            ).fetchone()
        if row is None:
            return None
        blob = row["vector_blob"]
        if blob is None:
            # vector_blob is NULL — metadata-only entry from JSON ingest.
            # The vectors can be recomputed by running the directions module.
            return None
        return np.frombuffer(blob, dtype=np.float32)

    # --- heads ---

    def record_head(self, model_id: int, layer: int, head: int, *, recipe: str | None = None, **kwargs: Any) -> int:
        """Upsert an attention head row.  Returns head id.

        NOTE: The heads table has UNIQUE(model_id, layer, head), so repeated
        measurements from different prompts will overwrite each other.  The
        table stores the LAST atlas measurement, not all of them.  Per-prompt
        measurements are preserved in the json_blob field if the caller
        accumulates them, and in the legacy signals table.
        """
        data: dict[str, Any] = {"model_id": model_id, "layer": layer, "head": head}
        for k in ("kl_ablation", "is_inert", "oproj_norm", "oproj_cluster",
                   "safety_specific", "chat_attention_weight", "json_blob",
                   "provenance"):
            if k in kwargs:
                data[k] = kwargs[k]
        if recipe is not None:
            data["recipe"] = recipe
        sql, params = self._upsert_sql("heads", ["model_id", "layer", "head"], data)
        self._write(sql, params, wait=True)
        row = self._conn.execute(
            "SELECT id FROM heads WHERE model_id = ? AND layer = ? AND head = ?",
            (model_id, layer, head),
        ).fetchone()
        return row["id"]

    # --- probes ---

    def record_probe(self, experiment_id: int, step: int, layer: int, *, recipe: str | None = None, **kwargs: Any) -> int:
        """Insert a probe measurement.  Returns probe id."""
        data: dict[str, Any] = {
            "experiment_id": experiment_id, "step": step, "layer": layer,
        }
        for k in ("residual_norm", "delta_norm", "safety_proj", "language_proj",
                   "chat_proj", "dampening", "pca_pc0", "pca_pc4",
                   "basin", "top_neurons", "attention_pos2_weight"):
            if k in kwargs:
                data[k] = kwargs[k]
        if recipe is not None:
            data["recipe"] = recipe
        cols = list(data.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        sql = f"INSERT INTO probes ({col_names}) VALUES ({placeholders})"
        return self._write(sql, tuple(data[c] for c in cols), wait=True)

    # --- interpolations ---

    def record_interpolation(self, model_id: int, alpha: float, *, recipe: str | None = None, **kwargs: Any) -> int:
        """Insert an interpolation row.  Returns interpolation id."""
        data: dict[str, Any] = {"model_id": model_id, "alpha": alpha}
        for k in ("top_token", "behavior", "output_text", "refuse_prob",
                   "off_line_distance", "config_label", "provenance"):
            if k in kwargs:
                data[k] = kwargs[k]
        if recipe is not None:
            data["recipe"] = recipe
        cols = list(data.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        sql = f"INSERT INTO interpolations ({col_names}) VALUES ({placeholders})"
        return self._write(sql, tuple(data[c] for c in cols), wait=True)

    # --- head_measurements (item 3) ---

    def record_head_measurement(self, model_id: int, layer: int, head: int,
                                prompt_label: str, *, recipe: str | None = None, **kwargs: Any) -> int:
        """Record per-prompt head ablation data."""
        data: dict[str, Any] = {
            "model_id": model_id, "layer": layer, "head": head,
            "prompt_label": prompt_label,
        }
        for k in ("kl_ablation", "entropy_delta", "top_changed", "provenance",
                   "prompt_text", "source_file"):
            if k in kwargs:
                data[k] = kwargs[k]
        if recipe is not None:
            data["recipe"] = recipe
        cols = list(data.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        sql = f"INSERT INTO head_measurements ({col_names}) VALUES ({placeholders})"
        return self._write(sql, tuple(data[c] for c in cols), wait=True)

    # --- refresh_heads_aggregate (Phase 5, Principle 9) ---

    def refresh_heads_aggregate(self, model_id: int | None = None) -> int:
        """Recompute heads table from head_measurements.

        For each (model_id, layer, head): compute mean KL, count prompts
        active, classify as universal/specific/inert.

        Returns the number of heads updated.
        """
        where = ""
        params: tuple = ()
        if model_id is not None:
            where = "WHERE model_id = ?"
            params = (model_id,)

        rows = self._conn.execute(
            f"SELECT model_id, layer, head, "
            f"AVG(kl_ablation) AS mean_kl, "
            f"COUNT(*) AS total, "
            f"SUM(CASE WHEN kl_ablation > 0.01 THEN 1 ELSE 0 END) AS active "
            f"FROM head_measurements {where} "
            f"GROUP BY model_id, layer, head",
            params,
        ).fetchall()

        count = 0
        for r in rows:
            total = r["total"]
            active = r["active"]
            mean_kl = r["mean_kl"]
            universality = active / total if total > 0 else 0.0

            if universality >= 0.80:
                classification = "universal"
            elif universality >= 0.01:
                classification = "prompt_specific"
            else:
                classification = "inert"

            data: dict[str, Any] = {
                "model_id": r["model_id"],
                "layer": r["layer"],
                "head": r["head"],
                "kl_ablation": mean_kl,
                "is_inert": classification == "inert",
                "universality": universality,
                "classification": classification,
                "active_count": active,
                "total_count": total,
                "provenance": "refresh_heads_aggregate",
            }
            sql, p = self._upsert_sql("heads", ["model_id", "layer", "head"], data)
            self._write(sql, p, wait=True)
            count += 1

        return count

    # --- events ---

    def record_event(self, event: str, **data: Any) -> int:
        """Record a timestamped event with optional JSON data.

        Commentary events (Principle 3) are rejected with a warning and
        return -1 instead of writing to the database.
        """
        if event in _COMMENTARY_EVENTS:
            import warnings
            warnings.warn(
                f"Commentary event {event!r} rejected — use structured tables instead",
                stacklevel=2,
            )
            return -1
        return self._write(
            "INSERT INTO events (ts, event, data) VALUES (?, ?, ?)",
            (time.time(), event, json.dumps(data)),
            wait=True,
        )

    # --- preregistrations ---

    def record_preregistration(self, hypothesis: str, test: str,
                               confirm_if: str, falsify_if: str,
                               chance_level: float = 0.5,
                               n_samples: int = 100) -> int:
        """Record a preregistered hypothesis. Returns preregistration id."""
        return self._write(
            "INSERT INTO preregistrations (hypothesis, test, confirm_if, falsify_if, "
            "chance_level, n_samples, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (hypothesis, test, confirm_if, falsify_if, chance_level, n_samples, time.time()),
            wait=True,
        )

    def update_preregistration(self, prereg_id: int, result: str, verified: bool) -> None:
        """Update a preregistration with results."""
        self._write(
            "UPDATE preregistrations SET result = ?, verified = ? WHERE id = ?",
            (result, verified, prereg_id),
            wait=True,
        )

    # ==================================================================
    # Eval-pipeline: record methods
    # ==================================================================

    def record_prompt(self, text: str, source: str, category: str | None = None,
                      is_benign: bool = False, sha256: str | None = None) -> int:
        """Insert or ignore a prompt row.  Returns prompt id."""
        import hashlib as _hashlib
        if sha256 is None:
            sha256 = _hashlib.sha256(text.encode()).hexdigest()
        # Use INSERT OR IGNORE to handle UNIQUE(text, source)
        self._write(
            "INSERT OR IGNORE INTO prompts (text, source, category, is_benign, sha256) "
            "VALUES (?, ?, ?, ?, ?)",
            (text, source, category, int(is_benign), sha256),
            wait=True,
        )
        row = self._conn.execute(
            "SELECT id FROM prompts WHERE text = ? AND source = ?",
            (text, source),
        ).fetchone()
        return row["id"]

    def record_generation(self, model_id: int, prompt_id: int, prompt_text: str,
                          condition: str, generation_text: str, **kwargs: Any) -> int:
        """Insert a generation row.  Returns generation id."""
        data: dict[str, Any] = {
            "model_id": model_id,
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "condition": condition,
            "generation_text": generation_text,
            "created_at": kwargs.pop("created_at", time.time()),
        }
        for k in ("prompt_source", "prompt_category", "top_token",
                   "n_tokens", "recipe", "first_token_id", "refuse_prob",
                   "is_degenerate", "logit_entropy", "top_k_tokens",
                   "safety_trajectory"):
            if k in kwargs:
                data[k] = kwargs[k]
        cols = list(data.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        sql = f"INSERT INTO generations ({col_names}) VALUES ({placeholders})"
        return self._write(sql, tuple(data[c] for c in cols), wait=True)

    def record_score(self, generation_id: int, scorer: str, label: str,
                     confidence: float | None = None,
                     raw_output: str | None = None) -> int:
        """Insert or update a score row.  Returns score id."""
        data: dict[str, Any] = {
            "generation_id": generation_id,
            "scorer": scorer,
            "label": label,
            "created_at": time.time(),
        }
        if confidence is not None:
            data["confidence"] = confidence
        if raw_output is not None:
            data["raw_output"] = raw_output
        sql, params = self._upsert_sql("scores", ["generation_id", "scorer"], data)
        self._write(sql, params, wait=True)
        row = self._conn.execute(
            "SELECT id FROM scores WHERE generation_id = ? AND scorer = ?",
            (generation_id, scorer),
        ).fetchone()
        return row["id"]

    def record_calibration(self, scorer: str, model_id: int,
                           fpr: float | None = None, fnr: float | None = None,
                           **kwargs: Any) -> None:
        """Insert or update a calibration row.

        For measurement scorers, pass benign_dist and harmful_dist as JSON strings
        containing label distributions.
        """
        data: dict[str, Any] = {
            "scorer": scorer,
            "model_id": model_id,
            "calibrated_at": kwargs.pop("calibrated_at", time.time()),
        }
        if fpr is not None:
            data["fpr"] = fpr
        if fnr is not None:
            data["fnr"] = fnr
        for k in ("n_benign", "n_harmful", "benign_dist", "harmful_dist"):
            if k in kwargs:
                data[k] = kwargs[k]
        sql, params = self._upsert_sql("calibration", ["scorer", "model_id"], data)
        self._write(sql, params, wait=True)

    # --- conditions ---

    def record_condition(self, model_id: int, name: str,
                         kind: str | None = None,
                         params_dict: dict | None = None,
                         source: str | None = None) -> int:
        """Insert or update a condition row.  Returns condition id."""
        data: dict[str, Any] = {
            "model_id": model_id,
            "name": name,
            "created_at": time.time(),
        }
        if kind is not None:
            data["kind"] = kind
        if params_dict is not None:
            data["params"] = json.dumps(params_dict)
        if source is not None:
            data["source"] = source
        sql, params = self._upsert_sql("conditions", ["model_id", "name"], data)
        self._write(sql, params, wait=True)
        row = self._conn.execute(
            "SELECT id FROM conditions WHERE model_id = ? AND name = ?",
            (model_id, name),
        ).fetchone()
        return row["id"]

    def get_conditions(self, model_id: int | None = None) -> list[dict]:
        """Query conditions, optionally filtered by model_id."""
        if model_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM conditions WHERE model_id = ? ORDER BY id",
                (model_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM conditions ORDER BY id",
            ).fetchall()
        return self._rows_to_dicts(rows)

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

    # ==================================================================
    # Eval-pipeline: query methods
    # ==================================================================

    def get_prompts(self, source: str | None = None, is_benign: bool | None = None,
                    category: str | None = None, limit: int = 1000) -> list[dict]:
        """Query prompts with optional filters."""
        clauses: list[str] = []
        params: list = []
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if is_benign is not None:
            clauses.append("is_benign = ?")
            params.append(int(is_benign))
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = self._conn.execute(
            f"SELECT * FROM prompts WHERE {where} ORDER BY id LIMIT ?",
            (*params, limit),
        ).fetchall()
        return self._rows_to_dicts(rows)

    def require_prompts(self, is_benign: bool | None = None,
                        min_count: int = 3, limit: int = 500) -> list[dict]:
        """Load prompts or raise. Single source of truth — no fallbacks.

        Every caller that needs prompts should use this instead of
        get_prompts + hardcoded fallback. If the DB doesn't have enough
        prompts, the tool can't work — better to fail loud than to
        silently degrade to 3 hardcoded strings.
        """
        rows = self.get_prompts(is_benign=is_benign, limit=limit)
        kind = "benign" if is_benign else ("harmful" if is_benign is False else "any")
        if len(rows) < min_count:
            raise RuntimeError(
                f"Need >= {min_count} {kind} prompts in DB "
                f"(got {len(rows)}). Load HF benchmarks first:\n"
                f"  python -m heinrich.eval.prompts  # or\n"
                f"  heinrich run --prompts simple_safety,catqa,do_not_answer"
            )
        return rows

    def get_unscored_generations(self, scorer_name: str) -> list[dict]:
        """Return generations that have no score row for *scorer_name*."""
        rows = self._conn.execute(
            "SELECT g.* FROM generations g "
            "LEFT JOIN scores s ON s.generation_id = g.id AND s.scorer = ? "
            "WHERE s.id IS NULL "
            "ORDER BY g.id",
            (scorer_name,),
        ).fetchall()
        return self._rows_to_dicts(rows)

    def get_generations(self, model_id: int | None = None,
                        condition: str | None = None,
                        is_benign: bool | None = None) -> list[dict]:
        """Query generations with optional filters."""
        clauses: list[str] = []
        params: list = []
        join = ""
        if model_id is not None:
            clauses.append("g.model_id = ?")
            params.append(model_id)
        if condition is not None:
            clauses.append("g.condition = ?")
            params.append(condition)
        if is_benign is not None:
            join = " JOIN prompts p ON p.id = g.prompt_id"
            clauses.append("p.is_benign = ?")
            params.append(int(is_benign))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT g.* FROM generations g{join}{where} ORDER BY g.id",
            params,
        ).fetchall()
        return self._rows_to_dicts(rows)

    def get_scores(self, generation_ids: list[int] | None = None,
                   scorer: str | None = None) -> list[dict]:
        """Query scores with optional filters."""
        clauses: list[str] = []
        params: list = []
        if generation_ids is not None:
            placeholders = ", ".join("?" for _ in generation_ids)
            clauses.append(f"generation_id IN ({placeholders})")
            params.extend(generation_ids)
        if scorer is not None:
            clauses.append("scorer = ?")
            params.append(scorer)
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = self._conn.execute(
            f"SELECT * FROM scores WHERE {where} ORDER BY id",
            params,
        ).fetchall()
        return self._rows_to_dicts(rows)

    def query_score_matrix(
        self,
        model_id: int | None = None,
        *,
        dedup_by_text: bool = False,
    ) -> list[dict]:
        """Return a joined view of generations + scores (pivot by scorer).

        Each row is a generation with all its scores inlined.  Uses LEFT JOIN
        so all generations appear even when only some scorers have run.

        When *dedup_by_text* is True, duplicate prompts (same text from
        different sources) are collapsed — only the first generation per
        (prompt_text, condition) is kept.  This prevents duplicate prompts
        from inflating counts.
        """
        clauses: list[str] = []
        params: list = []
        if model_id is not None:
            clauses.append("g.model_id = ?")
            params.append(model_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        if dedup_by_text:
            # Use a subquery to pick the MIN(g.id) per (prompt_text, condition)
            # so duplicate prompts from different sources don't inflate counts.
            gen_sql = (
                f"SELECT g.* FROM generations g "
                f"INNER JOIN ("
                f"  SELECT MIN(id) as min_id FROM generations"
                f"  GROUP BY prompt_text, condition"
                f") dedup ON g.id = dedup.min_id"
            )
        else:
            gen_sql = "SELECT g.* FROM generations g"

        rows = self._conn.execute(
            f"SELECT g.id as generation_id, g.prompt_text, g.condition, "
            f"g.generation_text, g.top_token, "
            f"s.scorer, s.label, s.confidence "
            f"FROM ({gen_sql}) g "
            f"LEFT JOIN scores s ON s.generation_id = g.id"
            f"{where} ORDER BY g.id, s.scorer",
            params,
        ).fetchall()

        # Pivot: group by generation_id
        matrix: dict[int, dict] = {}
        for r in rows:
            gid = r["generation_id"]
            if gid not in matrix:
                matrix[gid] = {
                    "generation_id": gid,
                    "prompt_text": r["prompt_text"],
                    "condition": r["condition"],
                    "generation_text": r["generation_text"],
                    "top_token": r["top_token"],
                    "scores": {},
                }
            if r["scorer"] is not None:
                matrix[gid]["scores"][r["scorer"]] = {
                    "label": r["label"],
                    "confidence": r["confidence"],
                }
        return list(matrix.values())

    def query_calibration(self, model_id: int | None = None) -> list[dict]:
        """Return calibration rows, optionally filtered by model_id."""
        if model_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM calibration WHERE model_id = ? ORDER BY scorer",
                (model_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM calibration ORDER BY scorer, model_id",
            ).fetchall()
        return self._rows_to_dicts(rows)

    def query_disagreements(self, model_id: int | None = None) -> list[dict]:
        """Find generations where *judge* scorers disagree on the verdict.

        Only judge scorers (labels containing ':') are compared.
        A disagreement is when one judge says ':safe' and another says ':unsafe'
        on the same generation.

        Measurement scorer labels (REFUSES, COMPLIES, etc.) are never part
        of disagreements -- they measure different things.
        """
        clauses: list[str] = []
        params: list = []
        if model_id is not None:
            clauses.append("g.model_id = ?")
            params.append(model_id)
        where = (" AND " + " AND ".join(clauses)) if clauses else ""

        # Find generations that have both a ':safe' and ':unsafe' judge label
        rows = self._conn.execute(
            f"SELECT g.id as generation_id, g.prompt_text, g.condition, "
            f"g.generation_text, g.top_token "
            f"FROM generations g "
            f"WHERE g.id IN ("
            f"  SELECT s1.generation_id "
            f"  FROM scores s1 "
            f"  JOIN scores s2 ON s1.generation_id = s2.generation_id "
            f"  WHERE s1.label LIKE '%:safe' "
            f"    AND s2.label LIKE '%:unsafe' "
            f"    AND s1.scorer != s2.scorer"
            f"){where} ORDER BY g.id",
            params,
        ).fetchall()

        result = []
        for r in rows:
            gid = r["generation_id"]
            # Only include judge scorer labels in the output
            scores = self._conn.execute(
                "SELECT scorer, label, confidence FROM scores "
                "WHERE generation_id = ? AND label LIKE '%:%'",
                (gid,),
            ).fetchall()
            result.append({
                **dict(r),
                "scores": [dict(s) for s in scores],
            })
        return result

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune_probes(self, keep_last_n: int = 10000) -> int:
        """Delete old probe rows, keeping only the most recent *keep_last_n*.

        Returns the number of rows deleted.
        """
        count_row = self._conn.execute("SELECT COUNT(*) as n FROM probes").fetchone()
        total = count_row["n"]
        if total <= keep_last_n:
            return 0
        to_delete = total - keep_last_n
        self._write(
            "DELETE FROM probes WHERE id IN (SELECT id FROM probes ORDER BY id ASC LIMIT ?)",
            (to_delete,), wait=True,
        )
        return to_delete

    def clear_normalized(self) -> None:
        """Truncate all normalized investigation tables (for re-ingest)."""
        tables = [
            "scores", "calibration", "generations", "prompts",  # eval-pipeline (FK order)
            "conditions",  # pipeline conditions
            "models", "experiments", "evaluations", "neurons", "sharts",
            "layers", "censorship", "basins", "basin_distances", "directions",
            "heads", "probes", "interpolations", "events",
        ]
        ops = [(f"DELETE FROM {t}", ()) for t in tables]
        self._write_many(ops, wait=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Shut down the writer thread and close both connections.

        Item 32: join with timeout to avoid hanging forever.
        """
        # Drain pending writes before shutdown
        self._write_queue.join()
        # Send poison pill to stop writer thread
        self._write_queue.put(_SENTINEL)
        self._writer_thread.join(timeout=10)
        if self._writer_thread.is_alive():
            import warnings
            warnings.warn("Writer thread did not shut down within 10s")
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
