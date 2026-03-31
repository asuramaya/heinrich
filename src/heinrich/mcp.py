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
}


class ToolServer:
    """Stateful tool server that maintains a signal store across calls."""

    def __init__(self) -> None:
        self._store = SignalStore()
        self._stages_run: list[str] = []

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
        from .probe import ProbeStage, MockProvider
        stage = ProbeStage()
        stage.run(self._store, {
            "provider": MockProvider(),
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
