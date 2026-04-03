"""Heinrich CLI — unified entry point for the forensics pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bundle.compress import compress_store
from .fetch.local import fetch_local_model
from .signal import SignalStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="heinrich",
        description="Model forensics and signal-mixing pipeline.",
    )
    sub = parser.add_subparsers(dest="command")

    p_fetch = sub.add_parser("fetch", help="Fetch model metadata and signals")
    p_fetch.add_argument("source", help="Local path or HuggingFace repo ID")
    p_fetch.add_argument("--label", help="Model label for signals")
    p_fetch.add_argument("--json", action="store_true", dest="json_output", help="JSON output")

    sub.add_parser("status", help="Show current session status")

    p_inspect = sub.add_parser("inspect", help="Spectral and structural analysis of weights")
    p_inspect.add_argument("source", help="Path to .npz weight bundle")
    p_inspect.add_argument("--label", help="Model label for signals")

    p_diff = sub.add_parser("diff", help="Compare two weight bundles")
    p_diff.add_argument("lhs", help="Path to first .npz weight bundle")
    p_diff.add_argument("rhs", help="Path to second .npz weight bundle")
    p_diff.add_argument("--lhs-label", default="lhs")
    p_diff.add_argument("--rhs-label", default="rhs")

    p_probe = sub.add_parser("probe", help="Behavioral testing with mock provider")
    p_probe.add_argument("--prompt", action="append", dest="prompts", required=True)
    p_probe.add_argument("--control", help="Control prompt for comparison")
    p_probe.add_argument("--model", default="model")

    p_report = sub.add_parser("report", help="Generate a text report from signals")
    p_report.add_argument("source", help="Path to signals JSONL or directory to scan")
    p_report.add_argument("--top", type=int, default=20)

    p_bundle = sub.add_parser("bundle", help="Assemble a validity bundle from manifest")
    p_bundle.add_argument("manifest", help="Path to manifest JSON file")
    p_bundle.add_argument("out_dir", help="Output directory for the bundle")

    sub.add_parser("serve", help="Run MCP stdio server")

    p_compete = sub.add_parser("compete", help="Validate against a rule profile")
    p_compete.add_argument("source", help="Directory to validate")
    p_compete.add_argument("--profile", required=True, help="Profile name or JSON file")
    p_compete.add_argument("--label", help="Label for signals")

    p_observe = sub.add_parser("observe", help="Analyze a grid/matrix or logit distribution")
    p_observe.add_argument("source", help="JSON file with grid or logits array")
    p_observe.add_argument("--label", default="observe")

    p_loop = sub.add_parser("loop", help="Run observe-analyze-act loop on states")
    p_loop.add_argument("source", help="JSON file with list of grid states")
    p_loop.add_argument("--max-turns", type=int, default=10)
    p_loop.add_argument("--label", default="loop")

    # Cartography: audit
    p_audit = sub.add_parser("audit", help="Run full behavioral security audit on a model")
    p_audit.add_argument("model_id", help="HuggingFace model ID or local path")
    p_audit.add_argument("--backend", choices=["mlx", "hf", "auto"], default="auto")
    p_audit.add_argument("--output", help="Path to save JSON report")
    p_audit.add_argument("--depth", choices=["quick", "standard", "deep"], default="standard",
                         help="Audit depth (quick=phases 1-6, standard=+heads, deep=+PCA)")
    p_audit.add_argument("--force", action="store_true", help="Bypass cached results")

    # Database: db subcommand with sub-subcommands
    p_db = sub.add_parser("db", help="Query the signal database")
    db_sub = p_db.add_subparsers(dest="db_command")

    p_db_query = db_sub.add_parser("query", help="Query signals")
    p_db_query.add_argument("--kind", help="Filter by signal kind")
    p_db_query.add_argument("--model", help="Filter by model name")
    p_db_query.add_argument("--limit", type=int, default=20)

    p_db_runs = db_sub.add_parser("runs", help="List recent runs")
    p_db_runs.add_argument("--limit", type=int, default=20)

    db_sub.add_parser("summary", help="Show database stats")

    # Item 66: ingest subcommand
    p_ingest = sub.add_parser("ingest", help="Ingest all JSON data files into the signal database")
    p_ingest.add_argument("--data-dir", default="data", help="Path to data directory (default: data)")
    p_ingest.add_argument("--model", default=None, help="Model name override")

    # Item 67: recompute subcommand
    sub.add_parser("recompute", help="Run all recompute scripts (layer_map, basins, interpolation)")

    # Item 70: reset subcommand
    sub.add_parser("reset", help="Clear all normalized tables and reingest")

    # Full pipeline: discover -> attack -> eval -> report
    p_run = sub.add_parser("run", help="Run full pipeline: discover -> attack -> eval -> report")
    p_run.add_argument("--model", required=True, help="Model ID")
    p_run.add_argument("--prompts", required=True, help="Comma-separated prompt set names")
    p_run.add_argument("--scorers", required=True, help="Comma-separated scorer names")
    p_run.add_argument("--output", "-o", default=None, help="Output JSON file path")
    p_run.add_argument("--db", default=None, help="Database path for pipeline")
    p_run.add_argument("--max-prompts", type=int, default=None, help="Max prompts per set")
    p_run.add_argument("--timeout", type=int, default=1800, help="Subprocess timeout")
    p_run.add_argument("--skip-discover", action="store_true", help="Skip discover step")
    p_run.add_argument("--skip-attack", action="store_true", help="Skip attack step")

    # Eval pipeline
    p_eval = sub.add_parser("eval", help="Run the full eval pipeline (generate, score, calibrate, report)")
    p_eval.add_argument("--model", required=True, help="Model ID (e.g. mlx-community/Qwen2.5-7B-Instruct-4bit)")
    p_eval.add_argument("--prompts", required=True, help="Comma-separated prompt set names")
    p_eval.add_argument("--scorers", required=True, help="Comma-separated scorer names")
    p_eval.add_argument("--conditions", default="clean", help="Comma-separated conditions (default: clean)")
    p_eval.add_argument("--output", "-o", default=None, help="Output JSON file path")
    p_eval.add_argument("--db", default=None, help="Database path for eval (default: ./data/heinrich.db)")
    p_eval.add_argument("--max-prompts", type=int, default=None, help="Max prompts per set")

    # Item 69: shared DB path
    parser.add_argument("--db-path", default=None,
                        help="Path to SQLite database (default: ./data/heinrich.db)")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fetch":
        _cmd_fetch(args)
    elif args.command == "status":
        _cmd_status(args)
    elif args.command == "inspect":
        _cmd_inspect(args)
    elif args.command == "diff":
        _cmd_diff(args)
    elif args.command == "probe":
        _cmd_probe(args)
    elif args.command == "report":
        _cmd_report(args)
    elif args.command == "bundle":
        _cmd_bundle(args)
    elif args.command == "serve":
        from .mcp_transport import run_stdio_server
        run_stdio_server()
    elif args.command == "compete":
        _cmd_compete(args)
    elif args.command == "observe":
        _cmd_observe(args)
    elif args.command == "loop":
        _cmd_loop(args)
    elif args.command == "audit":
        _cmd_audit(args)
    elif args.command == "db":
        _cmd_db(args)
    elif args.command == "ingest":
        _cmd_ingest(args)
    elif args.command == "recompute":
        _cmd_recompute(args)
    elif args.command == "reset":
        _cmd_reset(args)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "eval":
        _cmd_eval(args)
    else:
        parser.print_help()


def _cmd_fetch(args: argparse.Namespace) -> None:
    store = SignalStore()
    source = args.source
    label = args.label or Path(source).name

    source_path = Path(source)
    if source_path.is_dir():
        fetch_local_model(store, source_path, model_label=label)
    else:
        try:
            from .fetch.hf import fetch_hf_model
            fetch_hf_model(store, source, model_label=label)
        except ImportError:
            print("Install huggingface_hub for remote fetching: pip install heinrich[fetch]", file=sys.stderr)
            sys.exit(1)

    output = compress_store(store, stages_run=["fetch"])
    print(json.dumps(output, indent=2))


def _cmd_status(args: argparse.Namespace) -> None:
    print(json.dumps({"status": "no active session", "stages_run": []}, indent=2))


def _cmd_inspect(args: argparse.Namespace) -> None:
    from .inspect import InspectStage
    store = SignalStore()
    label = args.label or Path(args.source).stem
    stage = InspectStage()
    stage.run(store, {"weights_path": args.source, "model_label": label})
    output = compress_store(store, stages_run=["inspect"])
    print(json.dumps(output, indent=2))


def _cmd_diff(args: argparse.Namespace) -> None:
    from .diff import DiffStage
    store = SignalStore()
    stage = DiffStage()
    stage.run(store, {"lhs_weights": args.lhs, "rhs_weights": args.rhs,
                      "lhs_label": args.lhs_label, "rhs_label": args.rhs_label})
    output = compress_store(store, stages_run=["diff"])
    print(json.dumps(output, indent=2))


def _cmd_probe(args: argparse.Namespace) -> None:
    from .probe import ProbeStage, MockProvider
    store = SignalStore()
    stage = ProbeStage()
    stage.run(store, {
        "provider": MockProvider(),
        "model": args.model,
        "prompts": args.prompts,
        "control_prompt": args.control,
    })
    output = compress_store(store, stages_run=["probe"])
    print(json.dumps(output, indent=2))


def _cmd_report(args: argparse.Namespace) -> None:
    from .bundle.ledger import scan_directory
    from .bundle.scoring import rank_signals
    source = Path(args.source)
    store = SignalStore()
    if source.is_dir():
        scan_directory(source, store=store, model_label=source.name)
    elif source.suffix == ".jsonl":
        for line in source.read_text().splitlines():
            if line.strip():
                from .signal import Signal
                store.add(Signal(**json.loads(line)))
    output = compress_store(store, stages_run=["report"])
    output["ranked_signals"] = rank_signals(store, top_k=args.top)
    print(json.dumps(output, indent=2))


def _cmd_bundle(args: argparse.Namespace) -> None:
    from .bundle.validity import write_validity_bundle
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    result = write_validity_bundle(manifest, Path(args.out_dir))
    print(json.dumps(result, indent=2))


def _cmd_compete(args: argparse.Namespace) -> None:
    from .bundle.profiles import get_profile, apply_profile
    from .inspect.directory import inspect_directory
    from .inspect.codescan import scan_directory
    store = SignalStore()
    source = Path(args.source)
    profile = get_profile(args.profile)
    label = args.label or args.profile
    apply_profile(store, source, profile, label=label)
    if source.is_dir():
        inspect_directory(store, source, label=label)
        code_signals = scan_directory(source, label=label)
        store.extend(code_signals)
    output = compress_store(store, stages_run=["compete"])
    print(json.dumps(output, indent=2))


def _cmd_observe(args: argparse.Namespace) -> None:
    import numpy as np
    from .inspect.matrix import analyze_matrix
    from .inspect.self_analysis import analyze_logits
    data = json.loads(Path(args.source).read_text(encoding="utf-8"))
    store = SignalStore()
    label = args.label
    if "grid" in data:
        store.extend(analyze_matrix(np.array(data["grid"], dtype=np.float64), label=label, name="observed"))
    if "logits" in data:
        store.extend(analyze_logits(np.array(data["logits"], dtype=np.float64), label=label))
    output = compress_store(store, stages_run=["observe"])
    print(json.dumps(output, indent=2))


def _cmd_loop(args: argparse.Namespace) -> None:
    import numpy as np
    from .pipeline import Loop
    from .probe.environment import MockEnvironment, ObserveStage, ActStage
    data = json.loads(Path(args.source).read_text(encoding="utf-8"))
    states = [np.array(s, dtype=np.float64) for s in data.get("states", data if isinstance(data, list) else [])]
    if not states:
        print(json.dumps({"error": "No states found in source file"}))
        return
    env = MockEnvironment(states)
    loop = Loop([ObserveStage()], act=ActStage(), max_iterations=args.max_turns)
    store = loop.run({"environment": env, "model_label": args.label, "next_action": 1})
    output = compress_store(store, stages_run=loop.stages_run)
    print(json.dumps(output, indent=2))


def _cmd_audit(args: argparse.Namespace) -> None:
    from .cartography.audit import full_audit
    report = full_audit(
        args.model_id,
        backend=args.backend,
        depth=args.depth,
        force=args.force,
    )
    output = report.to_dict()
    if args.output:
        report.save(args.output)
        print(f"Report saved to {args.output}", file=sys.stderr)
    print(json.dumps(output, indent=2, default=str))


def _get_db(args: argparse.Namespace):
    """Get SignalDB with optional --db-path."""
    from .db import SignalDB
    if hasattr(args, 'db_path') and args.db_path:
        return SignalDB(args.db_path)
    return SignalDB()


def _cmd_db(args: argparse.Namespace) -> None:
    db = _get_db(args)

    if args.db_command == "query":
        kwargs = {}
        if args.kind:
            kwargs["kind"] = args.kind
        if args.model:
            kwargs["model"] = args.model
        kwargs["limit"] = args.limit
        signals = db.query(**kwargs)
        output = [
            {"kind": s.kind, "source": s.source, "model": s.model,
             "target": s.target, "value": s.value, "metadata": s.metadata}
            for s in signals
        ]
        print(json.dumps(output, indent=2, default=str))

    elif args.db_command == "runs":
        runs = db.runs(limit=args.limit)
        print(json.dumps(runs, indent=2, default=str))

    elif args.db_command == "summary":
        summary = db.summary()
        print(json.dumps(summary, indent=2, default=str))

    else:
        print(json.dumps({"error": "Use: heinrich db query|runs|summary"}))

    db.close()


def _cmd_ingest(args: argparse.Namespace) -> None:
    from .ingest import ingest_all, DEFAULT_MODEL
    db = _get_db(args)
    data_dir = args.data_dir
    model = args.model or DEFAULT_MODEL
    total = ingest_all(db, data_dir=data_dir, model_name=model)
    print(f"\nTotal signals ingested: {total}")
    summary = db.summary()
    print(json.dumps(summary["normalized_tables"], indent=2))
    db.close()


def _cmd_recompute(args: argparse.Namespace) -> None:
    import subprocess
    scripts = [
        "scripts/recompute_layer_map.py",
        "scripts/recompute_basins.py",
        "scripts/recompute_interpolation.py",
    ]
    for script in scripts:
        script_path = Path(__file__).resolve().parent.parent.parent / script
        if script_path.exists():
            print(f"\n--- Running {script} ---")
            subprocess.run([sys.executable, str(script_path)], check=False)
        else:
            print(f"Script not found: {script_path}", file=sys.stderr)


def _cmd_reset(args: argparse.Namespace) -> None:
    db = _get_db(args)
    print("Clearing all normalized tables...")
    db.clear_normalized()
    print("Done. Run 'heinrich ingest' to repopulate.")
    db.close()


def _cmd_run(args: argparse.Namespace) -> None:
    from .run import run_full_pipeline
    run_full_pipeline(
        model=args.model,
        prompts=args.prompts.split(","),
        scorers=args.scorers.split(","),
        output=args.output,
        db_path=args.db,
        max_prompts=args.max_prompts,
        timeout=args.timeout,
        skip_discover=args.skip_discover,
        skip_attack=args.skip_attack,
    )


def _cmd_eval(args: argparse.Namespace) -> None:
    from .eval.run import run_pipeline
    run_pipeline(
        model=args.model,
        prompts=args.prompts.split(","),
        scorers=args.scorers.split(","),
        conditions=args.conditions.split(","),
        output=args.output,
        db_path=args.db,
        max_prompts=args.max_prompts,
    )


if __name__ == "__main__":
    main()
