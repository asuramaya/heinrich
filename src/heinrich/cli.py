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

    # Visualizer
    p_viz = sub.add_parser("viz", help="Start the web visualizer sidecar")
    p_viz.add_argument("--port", type=int, default=8377, help="Port (default: 8377)")
    p_viz.add_argument("--db", default=None, help="Database path")

    # Tokenizer profile (.frt)
    p_frt = sub.add_parser("frt-profile", help="Generate a .frt tokenizer profile")
    p_frt.add_argument("--tokenizer", required=True, help="Tokenizer name or model ID")
    p_frt.add_argument("--output", "-o", default=None, help="Output .frt.npz file path")

    # Shart profile (.shrt)
    p_shrt = sub.add_parser("shart-profile", help="Generate a .shrt shart profile for a model")
    p_shrt.add_argument("--model", required=True, help="Model ID")
    p_shrt.add_argument("--n-index", type=int, default=15000, help="Index size (default 15K)")
    p_shrt.add_argument("--layers", default=None, help="Layers to measure (comma-separated, or 'all')")
    p_shrt.add_argument("--output", "-o", default=None, help="Output .shrt.npz file path")
    p_shrt.add_argument("--db", default=None, help="Database path for prompts/directions")

    # Output profile (.sht)
    p_sht = sub.add_parser("sht-profile", help="Generate a .sht output profile for a model")
    p_sht.add_argument("--model", required=True, help="Model ID")
    p_sht.add_argument("--n-index", type=int, default=15000, help="Index size (default 15K)")
    p_sht.add_argument("--output", "-o", default=None, help="Output .sht.npz file path")

    # Profile analysis
    p_chain = sub.add_parser("profile-chain", help="Connect .frt → .shrt → .sht for one model")
    p_chain.add_argument("--frt", required=True, help=".frt.npz file")
    p_chain.add_argument("--shrt", required=True, help=".shrt.npz file")
    p_chain.add_argument("--sht", required=True, help=".sht.npz file")

    p_cross = sub.add_parser("profile-cross", help="Compare two .shrt profiles across models")
    p_cross.add_argument("--a", required=True, help="First .shrt.npz file")
    p_cross.add_argument("--b", required=True, help="Second .shrt.npz file")
    p_cross.add_argument("--frt", default=None, help=".frt.npz for script breakdown (optional)")

    p_survey = sub.add_parser("profile-survey", help="Compare all .shrt profiles (baseline-independent)")
    p_survey.add_argument("--shrt", nargs="+", required=True, help=".shrt.npz files")
    p_survey.add_argument("--frt", nargs="*", default=None, help="Matching .frt.npz files (optional)")

    p_mismatch = sub.add_parser("profile-mismatch", help="Tokenizer-weight mismatch for one model")
    p_mismatch.add_argument("--shrt", required=True, help=".shrt.npz file")
    p_mismatch.add_argument("--frt", required=True, help=".frt.npz file")

    p_depth = sub.add_parser("profile-depth", help="Compare models at relative depth (needs --layers all)")
    p_depth.add_argument("--shrt", nargs="+", required=True, help=".shrt.npz files (run with --layers all)")
    p_depth.add_argument("--frt", nargs="*", default=None, help="Matching .frt.npz files for script breakdown")

    p_lscript = sub.add_parser("profile-layer-scripts", help="Script rankings at every layer (needs --layers all)")
    p_lscript.add_argument("--shrt", required=True, help=".shrt.npz file (run with --layers all)")
    p_lscript.add_argument("--frt", required=True, help=".frt.npz file")

    p_health = sub.add_parser("profile-tokenizer-health", help="Where does the tokenizer break? Data first, no hypothesis.")
    p_health.add_argument("--frt", required=True, help=".frt.npz file")
    p_health.add_argument("--shrt", default=None, help=".shrt.npz file (optional, adds displacement context)")

    p_embed = sub.add_parser("profile-embedding", help="Examine the embedding table — the link between .frt and .shrt")
    p_embed.add_argument("--model", required=True, help="Model ID")
    p_embed.add_argument("--frt", required=True, help=".frt.npz file")
    p_embed.add_argument("--shrt", default=None, help=".shrt.npz file (optional, adds displacement correlation)")

    p_scatter = sub.add_parser("profile-scatter", help="Displacement × output scatter — correlational, not causal")
    p_scatter.add_argument("--shrt", required=True, help=".shrt.npz file")
    p_scatter.add_argument("--sht", required=True, help=".sht.npz file")
    p_scatter.add_argument("--frt", default=None, help=".frt.npz file (optional, adds script breakdown)")

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
    elif args.command == "viz":
        from .viz import run_server
        run_server(port=args.port, db_path=args.db or "data/heinrich.db")
    elif args.command == "frt-profile":
        _cmd_frt(args)
    elif args.command == "shart-profile":
        _cmd_shrt(args)
    elif args.command == "sht-profile":
        _cmd_sht(args)
    elif args.command == "profile-chain":
        _cmd_profile_chain(args)
    elif args.command == "profile-cross":
        _cmd_profile_cross(args)
    elif args.command == "profile-survey":
        _cmd_profile_survey(args)
    elif args.command == "profile-mismatch":
        _cmd_profile_mismatch(args)
    elif args.command == "profile-depth":
        _cmd_profile_depth(args)
    elif args.command == "profile-layer-scripts":
        _cmd_layer_scripts(args)
    elif args.command == "profile-tokenizer-health":
        _cmd_tokenizer_health(args)
    elif args.command == "profile-embedding":
        _cmd_embedding(args)
    elif args.command == "profile-scatter":
        _cmd_scatter(args)
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


def _cmd_frt(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from .profile.frt import generate_frt

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    output = args.output or f"data/runs/{args.tokenizer.split('/')[-1]}.frt.npz"
    meta = generate_frt(tokenizer, output=output)

    print(f"\n=== .frt: {args.tokenizer} ===")
    print(f"  vocab: {meta['tokenizer']['vocab_size']} ({meta['tokenizer']['n_real']} real)")
    print(f"  hash: {meta['tokenizer']['vocab_hash']}")
    print(f"  bytes/token: {meta['byte_stats']['mean']:.1f} +/- {meta['byte_stats']['std']:.1f}")
    print(f"  scripts: {meta['scripts']}")
    print(f"  backend: {meta['tokenizer'].get('backend', '?')}, roundtrip failures: {meta['tokenizer'].get('roundtrip_failures', '?')}")
    if meta.get("warnings"):
        print(f"\n  WARNINGS:")
        for w in meta["warnings"]:
            print(f"    ! {w}")
    print(f"\n  Saved to {output}")


def _cmd_shrt(args: argparse.Namespace) -> None:
    from .backend.protocol import load_backend
    from .profile.shrt import generate_shrt

    backend = load_backend(args.model)
    db = None
    if args.db:
        from .core.db import SignalDB
        db = SignalDB(args.db)

    layers = None
    if args.layers:
        if args.layers == 'all':
            layers = [-1]
        else:
            layers = [int(x) for x in args.layers.split(',')]

    output = args.output or f"data/runs/{args.model.split('/')[-1]}.shrt.npz"
    shrt = generate_shrt(backend, db=db, n_index=args.n_index, layers=layers, output=output)

    print(f"\n=== {shrt['model']['name']} ===")
    tp = shrt['index'].get('throughput', {})
    print(f"  {shrt['index']['n_sampled']} tokens indexed in {shrt['index']['elapsed_s']}s ({tp.get('tokens_per_sec', 0):.0f} tok/s)")
    print(f"  cold={tp.get('cold_ms', 0):.0f}ms warm={tp.get('warm_ms', 0):.0f}ms cv={tp.get('cv', 0):.2f}")
    conv = shrt['index'].get('converged_at_pct', 100)
    conv_n = shrt['index'].get('converged_at_n', shrt['index']['n_sampled'])
    print(f"  converged at N={conv_n} ({conv}% of sample)")
    print(f"  mean delta: {shrt['distribution']['mean']} +/- {shrt['distribution']['std']}")
    for typ, stats in shrt["by_type"].items():
        print(f"    {typ:<15} n={stats['n']:>5}  mean={stats['mean']:>6.1f}")
    if shrt.get("warnings"):
        print(f"\n  WARNINGS:")
        for w in shrt["warnings"]:
            print(f"    ! {w}")
    print(f"\n  Saved to {output}")

    if db:
        db.close()


def _cmd_sht(args: argparse.Namespace) -> None:
    from .backend.protocol import load_backend
    from .profile.sht import generate_sht

    backend = load_backend(args.model)
    output = args.output or f"data/runs/{args.model.split('/')[-1]}.sht.npz"
    meta = generate_sht(backend, n_index=args.n_index, output=output)

    print(f"\n=== .sht: {meta['model']['name']} ===")
    print(f"  {meta['index']['n_sampled']} tokens")
    print(f"  KL: mean={meta['distribution']['kl_mean']:.4f} max={meta['distribution']['kl_max']:.4f}")
    print(f"  top changed: {meta['distribution']['pct_top_changed']}%")
    print(f"  Saved to {output}")


def _cmd_profile_chain(args: argparse.Namespace) -> None:
    from .profile.compare import chain
    result = chain(args.frt, args.shrt, args.sht)
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    print(f"\n=== Three-stage chain (N={result['n_shared']}) ===")
    print(f"  .frt → .shrt (bytes→delta):  r = {result['correlations']['bytes_to_delta']:+.4f}")
    print(f"  .shrt → .sht (delta→KL):     r = {result['correlations']['delta_to_kl']:+.4f}")
    print(f"  .frt → .sht  (bytes→KL):     r = {result['correlations']['bytes_to_kl']:+.4f}")
    print(f"\n  {'script':<12} {'n':>5} {'delta':>7} {'±':>5} {'KL':>7} {'±':>5}")
    for s, st in result['scripts'].items():
        print(f"  {s:<12} {st['n']:>5} {st['mean_delta']:>7.1f} {st['delta_ci']:>5.1f} {st['mean_kl']:>7.2f} {st['kl_ci']:>5.2f}")


def _cmd_profile_cross(args: argparse.Namespace) -> None:
    from .profile.compare import cross
    result = cross(args.a, args.b, frt_path=args.frt)
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    print(f"\n=== Cross-model: {result['model_a']} vs {result['model_b']} ===")
    print(f"  Shared tokens: {result['n_shared']}")
    print(f"  Delta correlation: r = {result['delta_correlation']:+.4f}")
    print(f"  Rank correlation:  r = {result['rank_correlation']:+.4f}")
    print(f"  Sensitivity A: {result['sensitivity_a']:.4f}  B: {result['sensitivity_b']:.4f}  (ratio: {result['sensitivity_ratio']:.1f}x)")
    print(f"  Baseline match: {result['baseline_match']}")
    if not result['baseline_match']:
        print(f"    WARNING: baselines differ significantly")
        print(f"    A: entropy={result['baseline_a']['entropy']:.4f} top={result['baseline_a']['top_token']}")
        print(f"    B: entropy={result['baseline_b']['entropy']:.4f} top={result['baseline_b']['top_token']}")
    if 'scripts' in result:
        print(f"\n  {'script':<12} {'n':>4} {'mean_A':>7} {'mean_B':>7} {'ratio':>6}")
        for s, st in result['scripts'].items():
            print(f"  {s:<12} {st['n']:>4} {st['mean_a']:>7.1f} {st['mean_b']:>7.1f} {st['ratio']:>6.2f}x")


def _cmd_scatter(args: argparse.Namespace) -> None:
    from .profile.compare import displacement_output_scatter
    result = displacement_output_scatter(args.shrt, args.sht, frt_path=args.frt)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Displacement x Output: Two Measurements (N={result['n_shared']}) ===")
    print(f"  r(delta, KL) = {result['r_delta_kl']}  (both are effects of the token, neither causes the other)")
    print(f"  median delta = {result['median_delta']}, median KL = {result['median_kl']}")

    for cat in ['high_both', 'high_kl', 'high_delta', 'low_both']:
        c = result[cat]
        if c['n'] == 0:
            continue
        print(f"\n  {cat.upper()} ({c['n']} tokens, {c['pct']}%): delta={c['mean_delta']:.1f} KL={c['mean_kl']:.4f}")
        if 'scripts' in c:
            top_s = list(c['scripts'].items())[:5]
            print(f"    scripts: {', '.join(f'{s}={n}' for s, n in top_s)}")
        if 'top' in c:
            for t in c['top'][:3]:
                tok_text = f"script={t.get('script', '?')}"
                print(f"    id={t['id']:>6} delta={t['delta']:>6.1f} KL={t['kl']:>7.4f} {tok_text}")


def _cmd_embedding(args: argparse.Namespace) -> None:
    from .profile.compare import embedding_profile
    result = embedding_profile(args.model, args.frt, shrt_path=args.shrt)

    print(f"\n=== Embedding: {result['model']} ===")
    print(f"  shape: {result['embedding_shape']}")
    print(f"  norm: mean={result['norm_mean']:.4f} std={result['norm_std']:.4f} range=[{result['norm_min']:.4f}, {result['norm_max']:.4f}]")

    if 'r_norm_delta' in result:
        print(f"  r(embedding_norm, delta) = {result['r_norm_delta']}")

    print(f"\n  {'script':<12} {'n':>5} {'mean_norm':>9} {'std':>7}", end='')
    if 'script_norm_delta_r' in result:
        print(f" {'r(norm,d)':>9}", end='')
    print()

    for s, data in sorted(result['script_norms'].items(), key=lambda x: -x[1]['mean_norm']):
        print(f"  {s:<12} {data['n']:>5} {data['mean_norm']:>9.4f} {data['std_norm']:>7.4f}", end='')
        if 'script_norm_delta_r' in result and s in result['script_norm_delta_r']:
            print(f" {result['script_norm_delta_r'][s]:>+9.4f}", end='')
        print()


def _cmd_tokenizer_health(args: argparse.Namespace) -> None:
    from .profile.compare import tokenizer_health
    result = tokenizer_health(args.frt, shrt_path=args.shrt)

    print(f"\n=== Tokenizer Health ({result['vocab_size']} tokens) ===\n")
    print(f"  Clean (round-trip OK):    {result['n_clean']}")
    print(f"  Collapsed (decode collision): {result['n_collapsed']} ({result['collision_groups']} groups)")
    print(f"  Silent (empty decode):    {result['n_silent']}")

    for group_name in ['clean', 'collapsed', 'silent']:
        g = result.get(group_name)
        if not g:
            continue
        print(f"\n  {group_name} (n={g['n']}):")
        print(f"    mean ID: {g['mean_id']:.0f}  mean bytes: {g['mean_bytes']:.1f}")
        if g.get('mean_delta') is not None:
            print(f"    mean delta: {g['mean_delta']:.2f} (n={g['n_with_delta']})")
        top_scripts = list(g['scripts'].items())[:5]
        print(f"    scripts: {', '.join(f'{s}={n}' for s, n in top_scripts)}")

    if result.get('largest_collisions'):
        print(f"\n  Largest collisions:")
        for c in result['largest_collisions'][:5]:
            print(f"    {c['text']:<25} {c['n_ids']} IDs")


def _cmd_layer_scripts(args: argparse.Namespace) -> None:
    from .profile.compare import layer_scripts
    result = layer_scripts(args.shrt, args.frt)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Layer-Script Trajectories: {result['model']} ({result['n_layers']}L) ===\n")

    # Show trajectory summary: how each script's relative displacement changes
    print(f"{'script':<12} {'early':>6} {'mid':>6} {'late':>6} {'range':>6}  trajectory")
    for sc in sorted(result['script_trajectories'],
                     key=lambda x: -result['script_trajectories'][x]['range']):
        t = result['script_trajectories'][sc]
        # Simple trajectory description
        if t['late'] > t['early'] + 0.1:
            arrow = "rises"
        elif t['late'] < t['early'] - 0.1:
            arrow = "falls"
        else:
            arrow = "flat"
        print(f"{sc:<12} {t['early']:>6.3f} {t['mid']:>6.3f} {t['late']:>6.3f} {t['range']:>6.3f}  {arrow}")


def _cmd_profile_mismatch(args: argparse.Namespace) -> None:
    from .profile.compare import mismatch
    result = mismatch(args.shrt, args.frt)
    print(f"\n=== Mismatch: {result['model']} (sens={result['sensitivity']:.4f}, d/byte={result['grand_delta_per_byte']:.2f}) ===\n")
    print(f"{'script':<12} {'vocab%':>6} {'n':>5} {'bytes':>5} {'d/byte':>6} {'rel_raw':>7} {'rel_d/b':>7}")
    for s in sorted(result['scripts'], key=lambda x: -result['scripts'][x]['relative_per_byte']):
        st = result['scripts'][s]
        print(f"{s:<12} {st['allocation_pct']:>5.1f}% {st['n_measured']:>5} {st['mean_bytes']:>5.1f} {st['delta_per_byte']:>6.2f} {st['relative_raw']:>7.3f} {st['relative_per_byte']:>7.3f}")


def _cmd_profile_depth(args: argparse.Namespace) -> None:
    from .profile.compare import depth_compare
    result = depth_compare(args.shrt, frt_paths=args.frt)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    profiles = result['profiles']
    header = f"{'depth':>6}"
    for p in profiles:
        label = f"{p['model']}({p['n_layers']}L)"
        header += f"  {label:>16} {'cv':>6}"
    print(f"\n{header}")

    for pct in result['checkpoints']:
        row = f"{pct*100:>5.0f}%"
        for p in profiles:
            d = p['depths'].get(pct, {})
            nd = d.get('norm_delta', 0)
            cv = d.get('cv', 0)
            row += f"  {nd:>16.4f} {cv:>6.4f}"
        print(row)

    print()
    for p in profiles:
        last = p['depths'].get(1.0, {})
        first = p['depths'].get(0.1, {})
        if first and last and first.get('norm_delta', 0) > 0:
            amp = last['norm_delta'] / first['norm_delta']
            print(f"  {p['model']:>10} ({p['n_layers']}L, h={p['hidden_size']}): amplification {amp:.0f}x, final cv={last.get('cv', 0):.4f}")

        if p.get('jumps'):
            jump_strs = [f"L{j['layer']}(+{j['jump_pct']:.0f}%)" for j in p['jumps']]
            print(f"    jumps: {', '.join(jump_strs)}")

        if p.get('final_layer_explosion'):
            exp = p['final_layer_explosion']
            print(f"    final layer (L{exp['from_layer']}→L{exp['to_layer']}): {exp['overall_ratio']:.1f}x overall")
            for sc, st in sorted(exp['by_script'].items(), key=lambda x: -x[1]['mean_ratio']):
                flags = []
                if st.get('provisional'):
                    flags.append("provisional")
                if st.get('exhaustible'):
                    flags.append(f"max={st['vocab_ceiling']}")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                pct = st.get('pct_sampled', 0)
                print(f"      {sc:<12} n={st['n']:>3}/{st['vocab_ceiling']:>5} ({pct:.0f}%)  ratio={st['mean_ratio']:.1f}x ±{st['ci_95']:.2f}{flag_str}")


def _cmd_profile_survey(args: argparse.Namespace) -> None:
    from .profile.compare import survey
    result = survey(args.shrt, frt_paths=args.frt)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Profile Survey ({result['n_models']} models) ===\n")
    print(f"{'model':<10} {'hidden':>6} {'sensitivity':>11} {'baseline_ent':>12} {'top':>8}")
    for p in sorted(result['profiles'], key=lambda x: -x['sensitivity']):
        print(f"{p['model']:<10} {p['hidden_size']:>6} {p['sensitivity']:>11.4f} {p['baseline_entropy']:>12.2f} {p['baseline_top']:>8}")

    if result['concordance']:
        print(f"\n{'script':<12} {'n':>3} {'relative':>8} {'std':>5} {'range':>12} {'status':>10}")
        for sc in sorted(result['concordance'], key=lambda x: -result['concordance'][x]['mean_relative']):
            c = result['concordance'][sc]
            status = "UNIVERSAL" if c['consistent'] else "variable"
            rng = f"{c['min']:.2f}-{c['max']:.2f}"
            print(f"{sc:<12} {c['n_models']:>3} {c['mean_relative']:>8.3f} {c['std_relative']:>5.3f} {rng:>12} {status:>10}")

    if result['kendalls_w'] is not None:
        print(f"\nKendall's W: {result['kendalls_w']:.4f}  (1.0=perfect agreement, 0.0=none)")


if __name__ == "__main__":
    main()
