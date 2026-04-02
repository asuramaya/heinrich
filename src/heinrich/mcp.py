"""MCP tool server — exposes heinrich pipeline stages as callable tools."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .signal import Signal, SignalStore
from .bundle.compress import compress_store
from .fetch.github import is_github_url


# Tool registry
TOOLS = {
    "heinrich_fetch": {
        "description": "Fetch model metadata, configs, and shard hashes from a local path or HuggingFace repo.",
        "parameters": {
            "source": {"type": "string", "description": "Local path or HuggingFace repo ID", "required": True},
            "label": {"type": "string", "description": "Model label for signals"},
        },
    },
    "heinrich_inspect": {
        "description": "Run spectral and structural analysis on weight tensors (.npz file).",
        "parameters": {
            "source": {"type": "string", "description": "Path to .npz weight bundle", "required": True},
            "label": {"type": "string", "description": "Model label for signals"},
        },
    },
    "heinrich_diff": {
        "description": "Compare two weight bundles and compute deltas, ranks, circuit scores.",
        "parameters": {
            "lhs": {"type": "string", "description": "Path to first .npz bundle", "required": True},
            "rhs": {"type": "string", "description": "Path to second .npz bundle", "required": True},
            "lhs_label": {"type": "string", "description": "Label for first model"},
            "rhs_label": {"type": "string", "description": "Label for second model"},
        },
    },
    "heinrich_probe": {
        "description": "Run behavioral testing with prompts against a model provider.",
        "parameters": {
            "prompts": {"type": "array", "description": "List of prompts to test", "required": True},
            "control": {"type": "string", "description": "Control prompt for comparison"},
            "model": {"type": "string", "description": "Model name"},
        },
    },
    "heinrich_bundle": {
        "description": "Compress a signal store into context-optimized JSON output.",
        "parameters": {
            "signals_path": {"type": "string", "description": "Path to signals JSONL file"},
            "top_k": {"type": "integer", "description": "Number of top signals to include"},
        },
    },
    "heinrich_signals": {
        "description": "Query and filter the current signal store.",
        "parameters": {
            "kind": {"type": "string", "description": "Filter by signal kind"},
            "source": {"type": "string", "description": "Filter by source stage"},
            "model": {"type": "string", "description": "Filter by model label"},
            "top_k": {"type": "integer", "description": "Return top-k by value"},
        },
    },
    "heinrich_status": {
        "description": "Show what stages have been run and current signal count.",
        "parameters": {},
    },
    "heinrich_pipeline": {
        "description": "Run fetch+inspect+diff+bundle as a single pipeline.",
        "parameters": {
            "models": {"type": "array", "description": "List of model paths or repo IDs", "required": True},
            "base": {"type": "string", "description": "Base model path or repo ID"},
        },
    },
    "heinrich_validate": {
        "description": "Validate a submission directory or GitHub PR — structural checks, consistency, claim level.",
        "parameters": {
            "source": {"type": "string", "description": "Local directory path or GitHub PR URL", "required": True},
            "label": {"type": "string", "description": "Label for signals"},
        },
    },
    "heinrich_compete": {
        "description": "Validate a source against a rule profile (parameter-golf, anti-bad, or custom JSON).",
        "parameters": {
            "source": {"type": "string", "description": "Local directory path or GitHub PR URL", "required": True},
            "profile": {"type": "string", "description": "Profile name or path to JSON rules file", "required": True},
            "label": {"type": "string", "description": "Label for signals"},
        },
    },
    "heinrich_observe": {
        "description": "Analyze a grid/matrix or logit distribution and emit signals.",
        "parameters": {
            "grid": {"type": "array", "description": "2D grid as nested array"},
            "logits": {"type": "array", "description": "Logit array"},
            "label": {"type": "string"},
        },
    },
    "heinrich_loop": {
        "description": "Run observe-analyze-act loop on a sequence of states.",
        "parameters": {
            "states": {"type": "array", "description": "List of 2D grid states", "required": True},
            "max_turns": {"type": "integer"},
            "label": {"type": "string"},
        },
    },
    "heinrich_self_analyze": {
        "description": "Run text through a model and capture internal signals (entropy, attention, hidden states).",
        "parameters": {
            "text": {"type": "string", "description": "Text to process", "required": True},
            "label": {"type": "string"},
        },
    },
    "heinrich_configure": {
        "description": "Configure the session — set provider, model, or other settings.",
        "parameters": {
            "provider": {"type": "string", "description": "Provider type: mock, hf-local"},
            "model": {"type": "string", "description": "Model path or HF repo ID"},
            "device": {"type": "string", "description": "Device: cpu, cuda, mps"},
            "max_new_tokens": {"type": "integer"},
        },
    },
    "heinrich_cartography": {
        "description": "Map the model's control surface — discover knobs and their behavioral effects.",
        "parameters": {
            "prompt": {"type": "string", "description": "Prompt to use for perturbation testing", "required": True},
            "sweep": {"type": "string", "description": "Sweep type: coarse (heads only) or full"},
            "label": {"type": "string"},
        },
    },
    "heinrich_audit": {
        "description": "Run a complete behavioral security audit on a model. Auto-detects architecture, chat format, and runs safety probes, direction finding, shart scanning, and framing bypass tests. Stores all results to SQLite database.",
        "parameters": {
            "model_id": {"type": "string", "description": "HuggingFace model ID or local path", "required": True},
            "backend": {"type": "string", "description": "Backend: mlx, hf, or auto (default)"},
        },
    },
    "heinrich_db_query": {
        "description": "Query the persistent signal database. Returns matching signals sorted by value.",
        "parameters": {
            "kind": {"type": "string", "description": "Filter by signal kind (e.g. shart, direction, probe)"},
            "model": {"type": "string", "description": "Filter by model"},
            "source": {"type": "string", "description": "Filter by source stage"},
            "min_value": {"type": "number", "description": "Minimum signal value"},
            "limit": {"type": "integer", "description": "Max results (default 50)"},
        },
    },
    "heinrich_db_runs": {
        "description": "List recent experiment runs from the signal database.",
        "parameters": {
            "limit": {"type": "integer", "description": "Max runs to list (default 20)"},
        },
    },
    "heinrich_db_summary": {
        "description": "Get database summary — signal counts, kinds, size.",
        "parameters": {},
    },
    # --- Qwen safety investigation tools ---
    "heinrich_safety_report": {
        "description": "Return per-category, per-attack safety breakdown from benchmark evaluations.",
        "parameters": {
            "dataset": {"type": "string", "description": "Filter by dataset name"},
            "category": {"type": "string", "description": "Filter by safety category (e.g. violence, discrimination)"},
            "attack": {"type": "string", "description": "Filter by attack type (e.g. direct, forensic)"},
        },
    },
    "heinrich_sharts": {
        "description": "Query anomalous tokens by category, z-score, or top neuron.",
        "parameters": {
            "category": {"type": "string", "description": "Filter by shart category"},
            "min_z": {"type": "number", "description": "Minimum z-score threshold"},
            "top_neuron": {"type": "integer", "description": "Filter by top activating neuron index"},
            "top_k": {"type": "integer", "description": "Number of results to return (default 30)"},
        },
    },
    "heinrich_neurons": {
        "description": "Query named neurons by category, layer, or causal effect.",
        "parameters": {
            "category": {"type": "string", "description": "Filter by neuron category (e.g. political, sexual)"},
            "layer": {"type": "integer", "description": "Filter by layer number"},
            "causal_only": {"type": "boolean", "description": "Only return neurons with confirmed causal effects"},
            "top_k": {"type": "integer", "description": "Number of results to return (default 30)"},
        },
    },
    "heinrich_censorship": {
        "description": "Return bilingual censorship map. Optionally filter to divergent topics only.",
        "parameters": {
            "divergent_only": {"type": "boolean", "description": "Only return topics where EN/ZH censorship diverges"},
            "topic": {"type": "string", "description": "Filter by topic name"},
        },
    },
    "heinrich_layer_map": {
        "description": "Return the L0-L27 layer profile with roles, dampening, neuron counts.",
        "parameters": {
            "model": {"type": "string", "description": "Filter by model name"},
        },
    },
    "heinrich_basin_geometry": {
        "description": "Return compliance basin distances and interpolation data.",
        "parameters": {
            "model": {"type": "string", "description": "Filter by model name"},
        },
    },
    "heinrich_directions": {
        "description": "List known behavioral directions with stability and effect size.",
        "parameters": {
            "layer": {"type": "integer", "description": "Filter by layer number"},
            "min_stability": {"type": "number", "description": "Minimum stability threshold"},
        },
    },
    "heinrich_benchmark_compare": {
        "description": "Compare refusal rates across attack configurations.",
        "parameters": {
            "dataset": {"type": "string", "description": "Filter by dataset name"},
            "attacks": {"type": "array", "description": "List of attack types to compare"},
        },
    },
    "heinrich_paper_verify": {
        "description": "Verify a specific claim from the paper against DB data.",
        "parameters": {
            "claim": {
                "type": "string",
                "description": (
                    "Claim ID to verify. One of: alpha_015_every_prompt, shart_families_3, "
                    "eval_count_1890, neuron_1934_political, discrimination_unprotected, "
                    "violence_collapses, monitor_paradox, basin_asymmetry"
                ),
                "required": True,
            },
        },
    },
    "heinrich_heads": {
        "description": "Query attention head importance and clustering.",
        "parameters": {
            "layer": {"type": "integer", "description": "Filter by layer number"},
            "inert_only": {"type": "boolean", "description": "Only return inert (low-importance) heads"},
            "safety_only": {"type": "boolean", "description": "Only return safety-critical heads"},
        },
    },
}


class ToolServer:
    """Stateful tool server that maintains a signal store across calls."""

    def __init__(self, *, db=None) -> None:
        self._store = SignalStore()
        self._stages_run: list[str] = []
        from .probe.provider import MockProvider
        self._provider: Any = MockProvider()
        from .db import SignalDB
        self._db: SignalDB = db or SignalDB()

    @property
    def store(self) -> SignalStore:
        return self._store

    def list_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions for MCP registration."""
        return [{"name": name, **defn} for name, defn in TOOLS.items()]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call and return the result."""
        if name == "heinrich_fetch":
            return self._do_fetch(arguments)
        if name == "heinrich_inspect":
            return self._do_inspect(arguments)
        if name == "heinrich_diff":
            return self._do_diff(arguments)
        if name == "heinrich_probe":
            return self._do_probe(arguments)
        if name == "heinrich_bundle":
            return self._do_bundle(arguments)
        if name == "heinrich_signals":
            return self._do_signals(arguments)
        if name == "heinrich_status":
            return self._do_status(arguments)
        if name == "heinrich_pipeline":
            return self._do_pipeline(arguments)
        if name == "heinrich_validate":
            return self._do_validate(arguments)
        if name == "heinrich_compete":
            return self._do_compete(arguments)
        if name == "heinrich_observe":
            return self._do_observe(arguments)
        if name == "heinrich_loop":
            return self._do_loop(arguments)
        if name == "heinrich_self_analyze":
            return self._do_self_analyze(arguments)
        if name == "heinrich_configure":
            return self._do_configure(arguments)
        if name == "heinrich_cartography":
            return self._do_cartography(arguments)
        if name == "heinrich_audit":
            return self._do_audit(arguments)
        if name == "heinrich_db_query":
            return self._do_db_query(arguments)
        if name == "heinrich_db_runs":
            return self._do_db_runs(arguments)
        if name == "heinrich_db_summary":
            return self._do_db_summary(arguments)
        if name == "heinrich_safety_report":
            return self._do_safety_report(arguments)
        if name == "heinrich_sharts":
            return self._do_sharts(arguments)
        if name == "heinrich_neurons":
            return self._do_neurons(arguments)
        if name == "heinrich_censorship":
            return self._do_censorship(arguments)
        if name == "heinrich_layer_map":
            return self._do_layer_map(arguments)
        if name == "heinrich_basin_geometry":
            return self._do_basin_geometry(arguments)
        if name == "heinrich_directions":
            return self._do_directions(arguments)
        if name == "heinrich_benchmark_compare":
            return self._do_benchmark_compare(arguments)
        if name == "heinrich_paper_verify":
            return self._do_paper_verify(arguments)
        if name == "heinrich_heads":
            return self._do_heads(arguments)
        return {"error": f"Unknown tool: {name}"}

    def _do_fetch(self, args: dict[str, Any]) -> dict[str, Any]:
        source = args["source"]
        label = args.get("label", source.split("/")[-1])
        source_path = Path(source)

        if source_path.is_dir():
            from .fetch.local import fetch_local_model
            fetch_local_model(self._store, source_path, model_label=label)
            # Also inspect directory contents
            from .inspect.directory import inspect_directory
            inspect_directory(self._store, source_path, label=label)
        elif is_github_url(source):
            from .fetch.github import fetch_github_path
            fetch_github_path(self._store, source, label=label)
        else:
            from .fetch.hf import fetch_hf_model
            fetch_hf_model(self._store, source, model_label=label)
        self._stages_run.append("fetch")
        return compress_store(self._store, stages_run=self._stages_run)

    def _do_inspect(self, args: dict[str, Any]) -> dict[str, Any]:
        source = Path(args["source"])
        label = args.get("label", source.name)
        if source.is_dir():
            from .inspect.directory import inspect_directory
            inspect_directory(self._store, source, label=label)
        else:
            from .inspect import InspectStage
            stage = InspectStage()
            stage.run(self._store, {"weights_path": str(source), "model_label": label})
        self._stages_run.append("inspect")
        return compress_store(self._store, stages_run=self._stages_run)

    def _do_diff(self, args: dict[str, Any]) -> dict[str, Any]:
        from .diff import DiffStage
        stage = DiffStage()
        stage.run(self._store, {
            "lhs_weights": args["lhs"], "rhs_weights": args["rhs"],
            "lhs_label": args.get("lhs_label", "lhs"), "rhs_label": args.get("rhs_label", "rhs"),
        })
        self._stages_run.append("diff")
        return compress_store(self._store, stages_run=self._stages_run)

    def _do_probe(self, args: dict[str, Any]) -> dict[str, Any]:
        from .probe import ProbeStage
        stage = ProbeStage()
        stage.run(self._store, {
            "provider": self._provider,
            "model": args.get("model", "model"),
            "prompts": args.get("prompts", []),
            "control_prompt": args.get("control"),
        })
        self._stages_run.append("probe")
        return compress_store(self._store, stages_run=self._stages_run)

    def _do_bundle(self, args: dict[str, Any]) -> dict[str, Any]:
        top_k = args.get("top_k", 10)
        return compress_store(self._store, stages_run=self._stages_run, top_k=top_k)

    def _do_signals(self, args: dict[str, Any]) -> dict[str, Any]:
        filtered = self._store.filter(
            kind=args.get("kind"),
            source=args.get("source"),
            model=args.get("model"),
        )
        top_k = args.get("top_k", 50)
        sorted_signals = sorted(filtered, key=lambda s: s.value, reverse=True)[:top_k]
        return {
            "count": len(filtered),
            "signals": [
                {"kind": s.kind, "source": s.source, "model": s.model,
                 "target": s.target, "value": s.value, "metadata": s.metadata}
                for s in sorted_signals
            ],
        }

    def _do_status(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "stages_run": self._stages_run,
            "signal_count": len(self._store),
            "summary": self._store.summary(),
        }

    def _do_validate(self, args: dict[str, Any]) -> dict[str, Any]:
        self._do_fetch(args)
        source = args["source"]
        source_path = Path(source)
        if source_path.is_dir():
            # Also check for .npz artifacts and inspect them
            for npz in sorted(source_path.rglob("*.npz")):
                from .inspect import InspectStage
                stage = InspectStage()
                stage.run(self._store, {"weights_path": str(npz), "model_label": args.get("label", source_path.name)})
        self._stages_run.append("validate")
        # Add claim level if audits bundle exists
        bundle_dir = source_path / "audits_bundle" if source_path.is_dir() else None
        if bundle_dir and bundle_dir.is_dir():
            from .bundle.ledger import infer_claim_level
            claim = json.loads((bundle_dir / "claim.json").read_text()) if (bundle_dir / "claim.json").exists() else {}
            metrics = json.loads((bundle_dir / "metrics.json").read_text()) if (bundle_dir / "metrics.json").exists() else {}
            audits = json.loads((bundle_dir / "audits.json").read_text()) if (bundle_dir / "audits.json").exists() else {}
            result = infer_claim_level(claim, metrics, audits)
            self._store.add(Signal("claim_level", "validate", args.get("label", ""), "claim_level",
                                   float(result["level"]),
                                   {"label": result["label"], "notes": result["notes"]}))
        return compress_store(self._store, stages_run=self._stages_run)

    def _do_pipeline(self, args: dict[str, Any]) -> dict[str, Any]:
        models = args.get("models", [])
        for model_ref in models:
            self._do_fetch({"source": model_ref, "label": Path(model_ref).name})
        return compress_store(self._store, stages_run=self._stages_run)

    def _do_observe(self, args: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        label = args.get("label", "observe")
        grid = args.get("grid")
        logits = args.get("logits")
        if grid is not None:
            from .inspect.matrix import analyze_matrix
            self._store.extend(analyze_matrix(np.array(grid, dtype=np.float64), label=label, name="observed"))
        if logits is not None:
            from .inspect.self_analysis import analyze_logits
            self._store.extend(analyze_logits(np.array(logits, dtype=np.float64), label=label))
        self._stages_run.append("observe")
        return compress_store(self._store, stages_run=self._stages_run)

    def _do_loop(self, args: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        from .pipeline import Loop
        from .probe.environment import MockEnvironment, ObserveStage, ActStage
        states = [np.array(s, dtype=np.float64) for s in args.get("states", [])]
        if not states: return {"error": "No states provided"}
        env = MockEnvironment(states)
        loop = Loop([ObserveStage()], act=ActStage(), max_iterations=args.get("max_turns", 10))
        self._store = loop.run({"environment": env, "model_label": args.get("label", "loop"), "next_action": 1}, store=self._store)
        self._stages_run.extend([s for s in loop.stages_run if s not in self._stages_run])
        return compress_store(self._store, stages_run=self._stages_run)

    def _do_configure(self, args: dict[str, Any]) -> dict[str, Any]:
        provider_type = args.get("provider", "mock")
        if provider_type == "hf-local":
            from .probe.provider import HuggingFaceLocalProvider
            config = {k: v for k, v in args.items() if k != "provider" and v is not None}
            self._provider = HuggingFaceLocalProvider(config)
        elif provider_type == "mlx":
            from .probe.mlx_provider import MLXProvider
            config = {k: v for k, v in args.items() if k != "provider" and v is not None}
            self._provider = MLXProvider(config)
        else:
            from .probe.provider import MockProvider
            self._provider = MockProvider()
        return {"configured": True, "provider": self._provider.describe()}

    def _do_self_analyze(self, args: dict[str, Any]) -> dict[str, Any]:
        from .probe.self_analyze import SelfAnalyzeStage
        stage = SelfAnalyzeStage()
        config = {
            "provider": self._provider,
            "text": args.get("text", ""),
            "model_label": args.get("label", "self"),
            "_iteration": len([s for s in self._stages_run if s == "self_analyze"]),
        }
        stage.run(self._store, config)
        self._stages_run.append("self_analyze")
        return compress_store(self._store, stages_run=self._stages_run)

    def _do_compete(self, args: dict[str, Any]) -> dict[str, Any]:
        source = args["source"]
        profile_name = args["profile"]
        label = args.get("label", profile_name)

        from .bundle.profiles import get_profile, apply_profile
        from .inspect.directory import inspect_directory
        from .inspect.codescan import scan_directory

        source_path = Path(source)
        profile = get_profile(profile_name)

        # Step 1: Apply rule profile
        apply_profile(self._store, source_path, profile, label=label)

        # Step 2: Inspect directory contents
        if source_path.is_dir():
            inspect_directory(self._store, source_path, label=label)

        # Step 3: Scan code for risks
        if source_path.is_dir():
            signals = scan_directory(source_path, label=label)
            self._store.extend(signals)

        # Step 4: Inspect any .npz artifacts
        if source_path.is_dir():
            from .inspect import InspectStage
            for npz in sorted(source_path.rglob("*.npz")):
                try:
                    InspectStage().run(self._store, {"weights_path": str(npz), "model_label": label})
                except Exception:
                    pass

        self._stages_run.append("compete")
        return compress_store(self._store, stages_run=self._stages_run)

    def _do_cartography(self, args: dict[str, Any]) -> dict[str, Any]:
        prompt = args["prompt"]
        sweep_type = args.get("sweep", "coarse")
        label = args.get("label", "cartography")

        from .cartography.surface import ControlSurface
        from .cartography.sweep import coarse_head_sweep, find_sensitive_layers
        from .cartography.atlas import Atlas
        from .cartography.manifold import cluster_by_layer, cluster_by_effect
        from .cartography.controls import ControlPanel

        try:
            provider = self._provider
            provider._ensure_loaded()
            model = provider._model
            tokenizer = provider._tokenizer
        except Exception as exc:
            return {"error": f"Provider not ready: {exc}. Use heinrich_configure with provider=mlx first."}

        try:
            surface = ControlSurface.from_mlx_model(model)
        except Exception:
            # Fallback to Qwen 7B defaults
            surface = ControlSurface.from_config(
                n_layers=28, n_heads=28, head_dim=128, intermediate_size=18944, hidden_size=3584)

        results = coarse_head_sweep(model, tokenizer, prompt, surface, store=self._store)
        atlas = Atlas()
        atlas.add_all(results)
        atlas.to_signals(self._store, label=label)

        clusters = cluster_by_effect(atlas, n_clusters=4)
        panel = ControlPanel.from_clusters(clusters)
        sensitive_layers = find_sensitive_layers(results, top_k=5)
        top_knobs = atlas.top_by_kl(k=10)

        self._stages_run.append("cartography")
        return {
            "surface_summary": surface.summary(),
            "sweep_results": len(results),
            "sensitive_layers": sensitive_layers,
            "top_knobs": [
                {"id": r.knob.id, "kl": round(r.kl_divergence, 4),
                 "entropy_delta": round(r.entropy_delta, 4),
                 "top_changed": r.top_token_changed}
                for r in top_knobs
            ],
            "clusters": len(clusters),
            "panel_summary": panel.summary(),
        }

    def _do_audit(self, args: dict[str, Any]) -> dict[str, Any]:
        from .cartography.audit import full_audit
        model_id = args["model_id"]
        backend = args.get("backend", "auto")
        try:
            report = full_audit(model_id, backend=backend, progress=True)
            self._stages_run.append("audit")
            return report.to_dict()
        except Exception as e:
            return {"error": str(e)}

    def _do_db_query(self, args: dict[str, Any]) -> dict[str, Any]:
        from .db import SignalDB
        db = SignalDB()
        signals = db.query(
            kind=args.get("kind"),
            model=args.get("model"),
            source=args.get("source"),
            min_value=args.get("min_value"),
            limit=args.get("limit", 50),
        )
        db.close()
        return {
            "count": len(signals),
            "signals": [
                {"kind": s.kind, "source": s.source, "model": s.model,
                 "target": s.target, "value": s.value, "metadata": s.metadata}
                for s in signals
            ],
        }

    def _do_db_runs(self, args: dict[str, Any]) -> dict[str, Any]:
        from .db import SignalDB
        db = SignalDB()
        runs = db.runs(limit=args.get("limit", 20))
        db.close()
        return {"runs": runs}

    def _do_db_summary(self, args: dict[str, Any]) -> dict[str, Any]:
        from .db import SignalDB
        db = SignalDB()
        summary = db.summary()
        db.close()
        return summary

    # ------------------------------------------------------------------
    # Helpers for investigation table queries
    # ------------------------------------------------------------------

    def _table_exists(self, table: str) -> bool:
        """Check if a table exists in the DB."""
        row = self._db._conn.execute(
            "SELECT COUNT(*) as n FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row["n"] > 0

    def _query_table(self, sql: str, params: tuple = ()) -> list[dict]:
        """Run a raw SQL query and return rows as dicts."""
        rows = self._db._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    _INGEST_MSG = "Table not found. Run heinrich_ingest first to populate investigation tables."

    # ------------------------------------------------------------------
    # Qwen safety investigation tool handlers
    # ------------------------------------------------------------------

    def _do_safety_report(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._table_exists("evaluations"):
            return {"error": self._INGEST_MSG}
        clauses: list[str] = []
        params: list = []
        if args.get("dataset"):
            clauses.append("dataset = ?")
            params.append(args["dataset"])
        if args.get("category"):
            clauses.append("category = ?")
            params.append(args["category"])
        if args.get("attack"):
            clauses.append("attack = ?")
            params.append(args["attack"])
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = (
            f"SELECT category, attack, "
            f"COUNT(*) as n, "
            f"AVG(refuse_prob) as avg_refuse_prob, "
            f"AVG(comply_prob) as avg_comply_prob "
            f"FROM evaluations WHERE {where} "
            f"GROUP BY category, attack ORDER BY category, attack"
        )
        rows = self._query_table(sql, tuple(params))
        for row in rows:
            row["refusal_rate"] = round(row["avg_refuse_prob"] or 0.0, 4)
            row["compliance_rate"] = round(row["avg_comply_prob"] or 0.0, 4)
        return {"count": len(rows), "breakdown": rows}

    def _do_sharts(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._table_exists("sharts"):
            return {"error": self._INGEST_MSG}
        clauses: list[str] = []
        params: list = []
        if args.get("category"):
            clauses.append("category = ?")
            params.append(args["category"])
        if args.get("min_z") is not None:
            clauses.append("max_z >= ?")
            params.append(args["min_z"])
        if args.get("top_neuron") is not None:
            clauses.append("top_neuron = ?")
            params.append(args["top_neuron"])
        top_k = args.get("top_k", 30)
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM sharts WHERE {where} ORDER BY max_z DESC LIMIT ?"
        params.append(top_k)
        rows = self._query_table(sql, tuple(params))
        return {"count": len(rows), "sharts": rows}

    def _do_neurons(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._table_exists("neurons"):
            return {"error": self._INGEST_MSG}
        clauses: list[str] = []
        params: list = []
        if args.get("category"):
            clauses.append("category = ?")
            params.append(args["category"])
        if args.get("layer") is not None:
            clauses.append("layer = ?")
            params.append(args["layer"])
        if args.get("causal_only"):
            clauses.append("causal_effect IS NOT NULL AND causal_effect > 0")
        top_k = args.get("top_k", 30)
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM neurons WHERE {where} ORDER BY layer, neuron_idx LIMIT ?"
        params.append(top_k)
        rows = self._query_table(sql, tuple(params))
        return {"count": len(rows), "neurons": rows}

    def _do_censorship(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._table_exists("censorship"):
            return {"error": self._INGEST_MSG}
        clauses: list[str] = []
        params: list = []
        if args.get("divergent_only"):
            clauses.append("divergent = 1")
        if args.get("topic"):
            clauses.append("topic = ?")
            params.append(args["topic"])
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM censorship WHERE {where} ORDER BY topic"
        rows = self._query_table(sql, tuple(params))
        return {"count": len(rows), "censorship": rows}

    def _do_layer_map(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._table_exists("layers"):
            return {"error": self._INGEST_MSG}
        clauses: list[str] = []
        params: list = []
        if args.get("model"):
            # Look up model_id by name
            model_row = self._db._conn.execute(
                "SELECT id FROM models WHERE name = ?", (args["model"],)
            ).fetchone()
            if model_row:
                clauses.append("model_id = ?")
                params.append(model_row["id"])
            else:
                return {"count": 0, "layers": [], "note": f"Model {args['model']!r} not found"}
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM layers WHERE {where} ORDER BY layer"
        rows = self._query_table(sql, tuple(params))
        if not rows:
            return {"count": 0, "layers": [], "note": "Layer map is empty. Run scripts/recompute_layer_map.py to populate."}
        return {"count": len(rows), "layers": rows}

    def _do_basin_geometry(self, args: dict[str, Any]) -> dict[str, Any]:
        has_basins = self._table_exists("basins")
        has_distances = self._table_exists("basin_distances")
        has_interp = self._table_exists("interpolations")
        if not (has_basins or has_distances or has_interp):
            return {"error": self._INGEST_MSG}
        result: dict[str, Any] = {}
        clauses: list[str] = []
        params: list = []
        if args.get("model"):
            model_row = self._db._conn.execute(
                "SELECT id FROM models WHERE name = ?", (args["model"],)
            ).fetchone()
            if model_row:
                clauses.append("model_id = ?")
                params.append(model_row["id"])
            else:
                return {"note": f"Model {args['model']!r} not found", "basins": [], "distances": [], "interpolations": []}
        where = " AND ".join(clauses) if clauses else "1=1"
        if has_basins:
            result["basins"] = self._query_table(
                f"SELECT * FROM basins WHERE {where}", tuple(params)
            )
        if has_distances:
            result["distances"] = self._query_table(
                f"SELECT * FROM basin_distances WHERE {where}", tuple(params)
            )
        if has_interp:
            result["interpolations"] = self._query_table(
                f"SELECT * FROM interpolations WHERE {where}", tuple(params)
            )
        if not any(result.get(k) for k in ("basins", "distances", "interpolations")):
            result["note"] = "Basin geometry tables are empty. Run scripts/recompute_basins.py to populate."
        return result

    def _do_directions(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._table_exists("directions"):
            return {"error": self._INGEST_MSG}
        clauses: list[str] = []
        params: list = []
        if args.get("layer") is not None:
            clauses.append("layer = ?")
            params.append(args["layer"])
        if args.get("min_stability") is not None:
            clauses.append("stability >= ?")
            params.append(args["min_stability"])
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM directions WHERE {where} ORDER BY effect_size DESC"
        rows = self._query_table(sql, tuple(params))
        return {"count": len(rows), "directions": rows}

    def _do_benchmark_compare(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._table_exists("evaluations"):
            return {"error": self._INGEST_MSG}
        clauses: list[str] = []
        params: list = []
        if args.get("dataset"):
            clauses.append("dataset = ?")
            params.append(args["dataset"])
        attacks = args.get("attacks")
        if attacks:
            placeholders = ",".join("?" for _ in attacks)
            clauses.append(f"attack IN ({placeholders})")
            params.extend(attacks)
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = (
            f"SELECT attack, "
            f"COUNT(*) as n, "
            f"AVG(refuse_prob) as avg_refuse_prob, "
            f"AVG(comply_prob) as avg_comply_prob "
            f"FROM evaluations WHERE {where} "
            f"GROUP BY attack ORDER BY attack"
        )
        rows = self._query_table(sql, tuple(params))
        for row in rows:
            row["refusal_rate"] = round(row["avg_refuse_prob"] or 0.0, 4)
            row["compliance_rate"] = round(row["avg_comply_prob"] or 0.0, 4)
        return {"count": len(rows), "comparison": rows}

    def _do_paper_verify(self, args: dict[str, Any]) -> dict[str, Any]:
        claim = args.get("claim", "")
        verifiers = {
            "alpha_015_every_prompt": self._verify_alpha_015,
            "shart_families_3": self._verify_shart_families,
            "eval_count_1890": self._verify_eval_count,
            "neuron_1934_political": self._verify_neuron_1934,
            "discrimination_unprotected": self._verify_discrimination,
            "violence_collapses": self._verify_violence_collapses,
            "monitor_paradox": self._verify_monitor_paradox,
            "basin_asymmetry": self._verify_basin_asymmetry,
        }
        if claim not in verifiers:
            return {
                "error": f"Unknown claim: {claim!r}",
                "valid_claims": list(verifiers.keys()),
            }
        return verifiers[claim]()

    # --- Paper claim verification sub-handlers ---

    def _verify_alpha_015(self) -> dict[str, Any]:
        """Paper: 'every prompt complies at alpha=-0.15'."""
        if not self._table_exists("evaluations"):
            return {"claim": "alpha_015_every_prompt", "error": self._INGEST_MSG}
        rows = self._query_table(
            "SELECT COUNT(*) as n, "
            "AVG(refuse_prob) as avg_refuse, "
            "AVG(comply_prob) as avg_comply "
            "FROM evaluations WHERE alpha = -0.15"
        )
        r = rows[0] if rows else {"n": 0, "avg_refuse": None, "avg_comply": None}
        avg_refuse = round(r["avg_refuse"], 4) if r["avg_refuse"] is not None else None
        return {
            "claim": "alpha_015_every_prompt",
            "paper_says": "every prompt complies at alpha=-0.15",
            "db_total": r["n"],
            "db_avg_refuse_prob": avg_refuse,
            "db_avg_comply_prob": round(r["avg_comply"], 4) if r["avg_comply"] is not None else None,
            "verified": avg_refuse == 0.0 if avg_refuse is not None else None,
        }

    def _verify_shart_families(self) -> dict[str, Any]:
        """Paper: '~3 real families'."""
        if not self._table_exists("sharts"):
            return {"claim": "shart_families_3", "error": self._INGEST_MSG}
        rows = self._query_table(
            "SELECT category, COUNT(*) as n FROM sharts GROUP BY category ORDER BY n DESC"
        )
        return {
            "claim": "shart_families_3",
            "paper_says": "~3 real families",
            "db_families": len(rows),
            "db_breakdown": rows,
            "verified": len(rows) >= 2 and len(rows) <= 5,
        }

    def _verify_eval_count(self) -> dict[str, Any]:
        """Paper: '1,890 evaluations'."""
        if not self._table_exists("evaluations"):
            return {"claim": "eval_count_1890", "error": self._INGEST_MSG}
        rows = self._query_table(
            "SELECT COUNT(*) as n FROM evaluations "
            "WHERE dataset IS NOT NULL"
        )
        n = rows[0]["n"] if rows else 0
        return {
            "claim": "eval_count_1890",
            "paper_says": "1,890 evaluations",
            "db_count": n,
            "verified": n == 1890,
        }

    def _verify_neuron_1934(self) -> dict[str, Any]:
        """Paper: 'neuron 1934 is political detector'."""
        if not self._table_exists("neurons"):
            return {"claim": "neuron_1934_political", "error": self._INGEST_MSG}
        rows = self._query_table(
            "SELECT * FROM neurons WHERE neuron_idx = 1934"
        )
        is_political = any(
            r.get("category", "").lower() in ("political", "politics")
            for r in rows
        )
        return {
            "claim": "neuron_1934_political",
            "paper_says": "neuron 1934 is political detector",
            "db_rows": rows,
            "verified": is_political,
        }

    def _verify_discrimination(self) -> dict[str, Any]:
        """Paper: (not in v1). Check discrimination + direct attack data."""
        if not self._table_exists("evaluations"):
            return {"claim": "discrimination_unprotected", "error": self._INGEST_MSG}
        rows = self._query_table(
            "SELECT COUNT(*) as n, "
            "AVG(refuse_prob) as avg_refuse, "
            "AVG(comply_prob) as avg_comply "
            "FROM evaluations WHERE category = 'discrimination' AND attack = 'direct'"
        )
        r = rows[0] if rows else {"n": 0, "avg_refuse": None, "avg_comply": None}
        avg_refuse = round(r["avg_refuse"], 4) if r["avg_refuse"] is not None else None
        return {
            "claim": "discrimination_unprotected",
            "paper_says": "(not in v1)",
            "db_total": r["n"],
            "db_avg_refuse_prob": avg_refuse,
            "db_avg_comply_prob": round(r["avg_comply"], 4) if r["avg_comply"] is not None else None,
            "data_exists": r["n"] > 0,
        }

    def _verify_violence_collapses(self) -> dict[str, Any]:
        """Paper: (not in v1). Compare violence: direct vs forensic."""
        if not self._table_exists("evaluations"):
            return {"claim": "violence_collapses", "error": self._INGEST_MSG}
        rows = self._query_table(
            "SELECT attack, "
            "COUNT(*) as n, "
            "AVG(refuse_prob) as avg_refuse "
            "FROM evaluations WHERE category = 'violence' "
            "GROUP BY attack ORDER BY attack"
        )
        by_attack = {}
        for r in rows:
            rate = round(r["avg_refuse"], 4) if r["avg_refuse"] is not None else None
            by_attack[r["attack"]] = {"n": r["n"], "refusal_rate": rate}
        direct_rate = by_attack.get("direct", {}).get("refusal_rate")
        forensic_rate = by_attack.get("forensic", {}).get("refusal_rate")
        collapses = (
            direct_rate is not None
            and forensic_rate is not None
            and forensic_rate < direct_rate * 0.5
        )
        return {
            "claim": "violence_collapses",
            "paper_says": "(not in v1)",
            "by_attack": by_attack,
            "collapses": collapses,
            "data_exists": len(rows) > 0,
        }

    def _verify_monitor_paradox(self) -> dict[str, Any]:
        """Paper: (v2 only). Check if defense_wave data exists."""
        has_defense = self._table_exists("defense_waves")
        if not has_defense:
            # Also check alternate table name
            has_defense = self._table_exists("defense_wave")
        return {
            "claim": "monitor_paradox",
            "paper_says": "(v2 only)",
            "defense_wave_data_exists": has_defense,
            "verified": None,
            "note": "Defense wave data exists" if has_defense else "No defense wave data found. This is a v2-only claim.",
        }

    def _verify_basin_asymmetry(self) -> dict[str, Any]:
        """Paper: '85/15'. Check interpolation REFUSE vs COMPLY counts."""
        if not self._table_exists("interpolations"):
            return {"claim": "basin_asymmetry", "error": self._INGEST_MSG}
        rows = self._query_table(
            "SELECT behavior, COUNT(*) as n FROM interpolations GROUP BY behavior"
        )
        by_behavior = {r["behavior"]: r["n"] for r in rows if r["behavior"]}
        total = sum(by_behavior.values())
        refuse_pct = round(100 * by_behavior.get("REFUSE", 0) / total, 1) if total else None
        comply_pct = round(100 * by_behavior.get("COMPLY", 0) / total, 1) if total else None
        verified = (
            refuse_pct is not None
            and comply_pct is not None
            and abs(refuse_pct - 85) < 5
            and abs(comply_pct - 15) < 5
        )
        if not total:
            return {
                "claim": "basin_asymmetry",
                "paper_says": "85/15",
                "db_total": 0,
                "note": "Interpolations table is empty. Run scripts/recompute_interpolation.py to populate.",
                "verified": None,
            }
        return {
            "claim": "basin_asymmetry",
            "paper_says": "85/15",
            "db_breakdown": by_behavior,
            "db_total": total,
            "refuse_pct": refuse_pct,
            "comply_pct": comply_pct,
            "verified": verified,
        }

    def _do_heads(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._table_exists("heads"):
            return {"error": self._INGEST_MSG}
        clauses: list[str] = []
        params: list = []
        if args.get("layer") is not None:
            clauses.append("layer = ?")
            params.append(args["layer"])
        if args.get("inert_only"):
            clauses.append("is_inert = 1")
        if args.get("safety_only"):
            clauses.append("safety_specific = 1")
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM heads WHERE {where} ORDER BY layer, head"
        rows = self._query_table(sql, tuple(params))
        return {"count": len(rows), "heads": rows}
