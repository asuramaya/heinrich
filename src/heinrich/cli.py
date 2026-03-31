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


if __name__ == "__main__":
    main()
