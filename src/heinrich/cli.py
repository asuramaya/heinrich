"""Heinrich CLI — unified entry point for the forensics pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bundle.compress import compress_store
from .fetch.local import fetch_local_model
from .signal import SignalStore


def _json_or(args, result, formatter, *, analysis_name: str | None = None):
    """If --json, dump result dict to stdout. Otherwise call formatter.

    If analysis_name is provided and --no-db is not set, also emits
    signals to the DB from the analysis result.
    """
    if getattr(args, 'json_output', False):
        json.dump(result, sys.stdout, default=str)
        print()
    else:
        formatter(result)
    # Emit signals to DB
    if analysis_name and not getattr(args, 'no_db', False) and "error" not in result:
        try:
            from .profile.signals import emit_signals
            from .core.db import SignalDB
            db = SignalDB()
            mri_path = getattr(args, 'mri', None)
            n = emit_signals(analysis_name, result, db, mri_path=mri_path)
            if n > 0 and not getattr(args, 'json_output', False):
                print(f"  ({n} signals → DB)", file=sys.stderr)
            db.close()
        except Exception as e:
            print(f"  WARNING: DB write failed: {e}", file=sys.stderr)
            print(f"  Results above are correct but were NOT recorded.", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="heinrich",
        description="Model forensics and signal-mixing pipeline.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output JSON to stdout (for MCP/companion)")
    parser.add_argument("--no-db", action="store_true", dest="no_db",
                        help="Skip writing signals to DB")
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

    # Visualizer (companion is primary, viz is alias)
    p_comp = sub.add_parser("companion", aliases=["viz"], help="Live 3D MRI viewer (http://localhost:8377)")
    p_comp.add_argument("--port", type=int, default=8377, help="Port (default: 8377)")

    # Crystal inspector — finds single-token crystals in raw-mode MRIs.
    p_crystal = sub.add_parser("crystal-inspect",
        help="Scan an MRI for raw-mode crystals (isolated high-|z| tokens) per layer")
    p_crystal.add_argument("--mri", required=True, help="Path to .mri directory")
    p_crystal.add_argument("--layer", type=int, default=None, help="Layer to scan (default: all real layers)")
    p_crystal.add_argument("--z-threshold", type=float, default=6.0,
        help="|z| cutoff on any PC (default: 6)")
    p_crystal.add_argument("--top", type=int, default=20, help="Top-N crystals to report (default: 20)")

    p_vizdeprecated = sub.add_parser("viz-legacy", help="[deprecated] Old DB-based visualizer")
    p_vizdeprecated.add_argument("--port", type=int, default=8378, help="Port (default: 8378)")
    p_vizdeprecated.add_argument("--db", default=None, help="Database path")

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

    # Direction audit: 5-test pipeline on a mean-diff direction
    p_audit = sub.add_parser("audit-direction", help="Run 5-test falsification on a mean-diff direction")
    p_audit.add_argument("--model", required=True, help="Model ID (MLX)")
    p_audit.add_argument("--datasets", nargs="+", required=True,
                         help="CSV paths; first is train-on, rest are transfer targets. Columns: statement,label")
    p_audit.add_argument("--layer", type=int, required=True, help="Layer to capture residual at")
    p_audit.add_argument("--n-per-class", type=int, default=80)
    p_audit.add_argument("--train-frac", type=float, default=0.8)
    p_audit.add_argument("--seed", type=int, default=42)
    p_audit.add_argument("--n-bootstrap", type=int, default=50)
    p_audit.add_argument("--n-permutation", type=int, default=200)
    p_audit.add_argument("--truth-tokens", default=None,
                         help="Comma-separated expected-truth single-tokens (for vocab projection test)")
    p_audit.add_argument("--false-tokens", default=None,
                         help="Comma-separated expected-false single-tokens")
    p_audit.add_argument("--output", "-o", default=None, help="Output JSON path")

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
    p_lscript.add_argument("--safety-shrt", default=None, help=".shrt.npz with discovered safety direction (for crossing overlap test)")

    p_health = sub.add_parser("profile-tokenizer-health", help="Where does the tokenizer break? Data first, no hypothesis.")
    p_health.add_argument("--frt", required=True, help=".frt.npz file")
    p_health.add_argument("--shrt", default=None, help=".shrt.npz file (optional, adds displacement context)")

    p_decompose = sub.add_parser("profile-decompose", help="Decompose displacement into attention vs MLP per layer per script")
    p_decompose.add_argument("--model", required=True, help="Model ID")
    p_decompose.add_argument("--frt", required=True, help=".frt.npz file")
    p_decompose.add_argument("--n-index", type=int, default=500, help="Tokens to measure")
    p_decompose.add_argument("--layers", default="all", help="Layers (comma-separated or 'all')")
    p_decompose.add_argument("--output", "-o", default=None, help="Output .npz file")

    p_validate = sub.add_parser("profile-validate-ablation", help="Validate attention/MLP decomposition: does attn + mlp = full?")
    p_validate.add_argument("--model", required=True, help="Model ID")
    p_validate.add_argument("--layer", type=int, default=12, help="Layer to test")

    p_embed = sub.add_parser("profile-embedding", help="Examine the embedding table — the link between .frt and .shrt")
    p_embed.add_argument("--model", required=True, help="Model ID")
    p_embed.add_argument("--frt", required=True, help=".frt.npz file")
    p_embed.add_argument("--shrt", default=None, help=".shrt.npz file (optional, adds displacement correlation)")

    p_scatter = sub.add_parser("profile-scatter", help="Displacement × output scatter — correlational, not causal")
    p_scatter.add_argument("--shrt", required=True, help=".shrt.npz file")
    p_scatter.add_argument("--sht", required=True, help=".sht.npz file")
    p_scatter.add_argument("--frt", default=None, help=".frt.npz file (optional, adds script breakdown)")

    p_within = sub.add_parser("profile-within-script", help="Within-script dispersion: tests partial knowledge hypothesis at token level")
    p_within.add_argument("--shrt", required=True, help=".shrt.npz file")
    p_within.add_argument("--frt", required=True, help=".frt.npz file")

    p_dirs = sub.add_parser("profile-directions", help="Directional analysis of displacement vectors: coherence, separation, safety projection")
    p_dirs.add_argument("--shrt", required=True, help=".shrt.npz file (must contain vectors)")
    p_dirs.add_argument("--frt", required=True, help=".frt.npz file")
    p_dirs.add_argument("--safety-shrt", default=None, help=".shrt.npz with discovered safety direction")

    p_pca = sub.add_parser("profile-pca-anatomy", help="Name the unnamed axes: what each PC separates, extreme tokens, direction cosines")
    p_pca.add_argument("--shrt", required=True, help=".shrt.npz or .mri directory")
    p_pca.add_argument("--frt", required=True, help=".frt.npz or .mri directory")
    p_pca.add_argument("--directions", nargs="*", default=None, help="Named direction files: name=path.npy")
    p_pca.add_argument("--n-components", type=int, default=20, help="Number of PCs to analyze (default: 20)")

    p_pcasurvey = sub.add_parser("profile-pca-survey", help="Compare PCA structure across models: which axes are shared, which are unique")
    p_pcasurvey.add_argument("--pairs", nargs="+", required=True, help="shrt:frt pairs (e.g. model1.shrt.npz:model1.frt.npz)")
    p_pcasurvey.add_argument("--n-components", type=int, default=10, help="Number of PCs per model (default: 10)")

    p_pcadepth = sub.add_parser("profile-pca-depth", help="PCA structure at every layer: how dimensionality evolves through the network")
    p_pcadepth.add_argument("--mri", required=True, help=".mri directory")
    p_pcadepth.add_argument("--n-sample", type=int, default=5000, help="Max tokens to sample (default: 5000)")
    p_pcadepth.add_argument("--layers", nargs="*", type=int, default=None, help="Specific layers to analyze (default: all)")

    p_mriverify = sub.add_parser("mri-verify", help="Smoke-test a model: 5-token MRI capture to verify architecture compatibility")
    p_mriverify.add_argument("--model", required=True, help="Model ID or checkpoint path")
    p_mriverify.add_argument("--backend", choices=["auto", "mlx", "hf", "decepticon"], default="auto")
    p_mriverify.add_argument("--result-json", default=None, help="Decepticon: path to result.json")
    p_mriverify.add_argument("--tokenizer-path", default=None, help="Decepticon: path to tokenizer model")

    p_mristatus = sub.add_parser("mri-status", help="Show all MRIs: what's complete, what's missing, what's running")
    p_mristatus.add_argument("--dir", default="/Volumes/sharts", help="MRI directory (default: /Volumes/sharts)")

    p_mrifixups = sub.add_parser("mri-check-fixups",
                                  help="Scan a directory for MRIs with stale fix_level (captured before recent bug fixes)")
    p_mrifixups.add_argument("--dir", default="/Volumes/sharts", help="MRI directory to scan")
    p_mrifixups.add_argument("--min-fix-level", type=int, default=None,
                              help="Minimum required fix_level (default: current)")

    p_mrirecap = sub.add_parser("mri-recapture",
                                 help="Re-run stale MRIs using provenance in their metadata.json")
    p_mrirecap.add_argument("--dir", default="/Volumes/sharts", help="MRI directory to scan")
    p_mrirecap.add_argument("--min-fix-level", type=int, default=None,
                             help="Minimum required fix_level (default: current)")
    p_mrirecap.add_argument("--execute", action="store_true",
                             help="Actually run captures (default: dry-run plan)")
    p_mrirecap.add_argument("--only", nargs="*", default=None,
                             help="Only recapture MRIs whose name contains one of these substrings")

    p_mriscan = sub.add_parser("mri-scan", help="Full MRI workup: capture all modes, health check, layer deltas, logit lens, PCA depth")
    p_mriscan.add_argument("--model", required=True, help="Model ID or checkpoint path")
    p_mriscan.add_argument("--output", "-o", required=True, help="Model output directory (e.g. /Volumes/sharts/qwen-0.5b)")
    p_mriscan.add_argument("--backend", choices=["auto", "mlx", "hf", "decepticon"], default="auto")
    p_mriscan.add_argument("--n-index", type=int, default=None, help="Number of tokens (default: full vocabulary)")
    p_mriscan.add_argument("--result-json", default=None, help="Decepticon: path to result.json")
    p_mriscan.add_argument("--tokenizer-path", default=None, help="Decepticon: path to tokenizer model")
    p_mriscan.add_argument("--data", default=None, help="Validation data .bin (causal bank sequence mode)")
    p_mriscan.add_argument("--n-seqs", type=int, default=50, help="Sequences for sequence mode (default: 50)")
    p_mriscan.add_argument("--seq-len", type=int, default=512, help="Sequence length (default: 512)")

    p_mrihealth = sub.add_parser("mri-health", help="Deep health check: verify shapes, NaN, weights, consistency for every MRI")
    p_mrihealth.add_argument("--dir", default="/Volumes/sharts", help="MRI directory (default: /Volumes/sharts)")
    p_mrihealth.add_argument("--mri", nargs="*", default=None, help="Specific .mri directories to check (default: all)")

    p_decompose = sub.add_parser("mri-decompose", help="PCA decompose an MRI for the companion viewer. Writes decomp/ directory with scores, variance, and binary blob")
    p_decompose.add_argument("--mri", required=True, help=".mri directory")
    p_decompose.add_argument("--n-sample", type=int, default=0, help="Tokens to sample (0 = full vocabulary)")
    p_decompose.add_argument("--n-components", type=int, default=0, help="PCA components (0 = all directions = hidden_size)")

    p_serve = sub.add_parser("mri-serve", help="Build query-shaped serve/ artifacts for the companion viewer from an existing decomposition")
    p_serve.add_argument("--mri", required=True, help=".mri directory")
    p_serve.add_argument("--steps", default="10,25,50", help="Precompute sampled PC indexes for these token strides")
    p_serve.add_argument("--force", action="store_true", help="Rebuild serve artifacts even if they already exist")

    p_logitlens = sub.add_parser("profile-logit-lens", help="What would the model predict at each layer? Applies norm+lmhead to exit states")
    p_logitlens.add_argument("--mri", required=True, help=".mri directory")
    p_logitlens.add_argument("--top-k", type=int, default=5, help="Top K predictions per layer (default: 5)")
    p_logitlens.add_argument("--layers", nargs="*", type=int, default=None, help="Specific layers (default: all)")
    p_logitlens.add_argument("--n-sample", type=int, default=100, help="Tokens to sample (default: 100)")

    p_layerdeltas = sub.add_parser("profile-layer-deltas", help="What each layer computes: exit[i] - exit[i-1] norms and amplification")
    p_layerdeltas.add_argument("--mri", required=True, help=".mri directory")
    p_layerdeltas.add_argument("--n-sample", type=int, default=5000, help="Tokens to sample (default: 5000)")

    p_gates = sub.add_parser("profile-gates", help="MLP gate analysis: which neurons fire, diversity, per-script specialization")
    p_gates.add_argument("--mri", required=True, help=".mri directory")
    p_gates.add_argument("--n-sample", type=int, default=5000, help="Tokens to sample (default: 5000)")

    p_attn = sub.add_parser("profile-attention", help="Attention analysis: where does the token attend (template mode)")
    p_attn.add_argument("--mri", required=True, help=".mri directory")
    p_attn.add_argument("--n-sample", type=int, default=5000, help="Tokens to sample (default: 5000)")

    p_cb_manifold = sub.add_parser("profile-cb-manifold", help="Causal bank manifold: PCA, band loadings, readout alignment, routing, gates")
    p_cb_manifold.add_argument("--mri", required=True, help=".mri directory (causal bank)")
    p_cb_manifold.add_argument("--n-sample", type=int, default=5000, help="Tokens to sample (default: 5000)")

    p_cb_compare = sub.add_parser("profile-cb-compare", help="Compare two causal bank MRIs: CKA, displacement, routing")
    p_cb_compare.add_argument("--a", required=True, help="First .mri directory")
    p_cb_compare.add_argument("--b", required=True, help="Second .mri directory")
    p_cb_compare.add_argument("--n-sample", type=int, default=1000, help="Tokens to sample (default: 1000)")

    p_cb_health = sub.add_parser("profile-cb-health", help="Validate causal bank MRI: shapes, NaN, consistency")
    p_cb_health.add_argument("--mri", required=True, help=".mri directory (causal bank)")

    p_cb_loss = sub.add_parser("profile-cb-loss", help="Causal bank loss decomposition by position, band, and autocorrelation")
    p_cb_loss.add_argument("--mri", required=True, help=".mri directory (causal bank, sequence mode)")

    p_cb_routing = sub.add_parser("profile-cb-routing", help="Causal bank sequence-level expert routing")
    p_cb_routing.add_argument("--mri", required=True, help=".mri directory (causal bank, sequence mode)")

    p_cb_temporal = sub.add_parser("profile-cb-temporal", help="Causal bank temporal attention forensics")
    p_cb_temporal.add_argument("--mri", required=True, help=".mri directory (causal bank, sequence mode)")

    p_cb_modes = sub.add_parser("profile-cb-modes", help="Causal bank mode utilization by half-life quartile")
    p_cb_modes.add_argument("--mri", required=True, help=".mri directory (causal bank, sequence mode)")

    p_cb_decompose = sub.add_parser("profile-cb-decompose", help="Causal bank manifold decomposition: position/content/ghost")
    p_cb_decompose.add_argument("--mri", required=True, help=".mri directory (causal bank, sequence mode)")
    p_cb_decompose.add_argument("--n-sample", type=int, default=None, help="Tokens to sample (default: all)")

    p_cb_omega = sub.add_parser("profile-cb-omega-forensics", help="Omega projection forensics: band allocation, Fourier survival, weight rank")
    p_cb_omega.add_argument("--checkpoint", required=True, help="Checkpoint .pt file (not MRI)")

    p_cb_rot = sub.add_parser("profile-cb-rotation-forensics",
                              help="Rotation/gate forensics for non-adaptive substrates (gated_delta, lasso, SO(3)/SO(5))")
    p_cb_rot.add_argument("--checkpoint", required=True, help="Checkpoint .pt file (not MRI)")

    p_cb_add = sub.add_parser("profile-cb-additivity",
                              help="Check whether a combined mutation's measurements match the solo-mutation additive prediction (bpb and/or geometry metrics)")
    p_cb_add.add_argument("--baseline", required=True, help="Baseline MRI (no mutations)")
    p_cb_add.add_argument("--mutations", nargs="+", required=True,
                          help="One solo-mutation MRI per axis being combined")
    p_cb_add.add_argument("--combination", required=True,
                          help="MRI with all mutations applied simultaneously")
    p_cb_add.add_argument("--noise-floor", type=float, default=0.004,
                          help="bpb noise floor (default 0.004; session-11 3-seed spread)")
    p_cb_add.add_argument("--metrics", nargs="+", default=["bpb"],
                          choices=["bpb", "eff_dim", "pos_r2", "cont_r2", "active_frac"],
                          help="Metrics to check additivity on (default: bpb)")
    p_cb_add.add_argument("--eff-dim-floor", type=float, default=2.0)
    p_cb_add.add_argument("--pos-r2-floor", type=float, default=0.005)
    p_cb_add.add_argument("--cont-r2-floor", type=float, default=0.010)
    p_cb_add.add_argument("--active-frac-floor", type=float, default=0.010)
    p_cb_add.add_argument("--svd-samples", type=int, default=5000,
                           help="Rows sampled per SVD for tail-PC position R² (default: 5000)")

    p_cb_pcb = sub.add_parser("profile-cb-pc-bands",
                               help="PC-band decomposition: reports variance %, position R², byte R² per PC band on a CB substrate. Detects two-band (content + position) partitions that EffDim undercounts.")
    p_cb_pcb.add_argument("--mris", nargs="+", required=True, help="One or more .seq.mri directories")
    p_cb_pcb.add_argument("--n-bootstrap", type=int, default=0,
                           help="Bootstrap K train/test splits for per-band pos_r2 ± SEM (default: 0 = off)")

    p_cb_traj = sub.add_parser("profile-cb-trajectory", help="Trajectory analysis across multiple checkpoints: EffDim, R², mode utilization derivatives")
    p_cb_traj.add_argument("--mris", nargs="+", required=True, help="Ordered MRI paths (by training step)")

    p_cb_invert = sub.add_parser("profile-cb-invertibility", help="How far back can the substrate reconstruct past bytes?")
    p_cb_invert.add_argument("--mri", required=True, help=".mri directory (sequence mode)")
    p_cb_invert.add_argument("--max-lookback", type=int, default=512, help="Maximum lookback distance")

    p_cb_rot = sub.add_parser("profile-cb-rotation-probe", help="Nonlinear + rotational information probes: MLP vs linear, angular decomposition")
    p_cb_rot.add_argument("--mri", required=True, help=".mri directory (causal bank, sequence mode)")
    p_cb_rot.add_argument("--n-sample", type=int, default=None, help="Tokens to sample (default: all)")

    p_cb_gate = sub.add_parser("profile-cb-gate-forensics", help="Causal bank write gate forensics: position dependence, difficulty correlation, effective rank")
    p_cb_gate.add_argument("--mri", required=True, help=".mri directory (causal bank, sequence mode)")

    p_cb_substrate = sub.add_parser("profile-cb-substrate-local", help="Causal bank substrate vs local path balance")
    p_cb_substrate.add_argument("--mri", required=True, help=".mri directory (causal bank, sequence mode)")

    p_tok_diff = sub.add_parser("profile-tokenizer-difficulty", help="Per-token difficulty from embedding norms (reads MRI)")
    p_tok_diff.add_argument("--mri", required=True, help=".mri directory (causal bank)")

    p_tok_compare = sub.add_parser("profile-tokenizer-compare", help="Compare sentencepiece tokenizers: compression, overlap, byte fallback")
    p_tok_compare.add_argument("--tokenizers", nargs="+", required=True, help=".model files to compare")
    p_tok_compare.add_argument("--text", default=None, help="Sample text file for compression stats")

    p_cb_causality = sub.add_parser("profile-cb-causality", help="Finite-difference causality verification for causal bank models")
    p_cb_causality.add_argument("--model", required=True, help="Checkpoint path (.checkpoint.pt)")
    p_cb_causality.add_argument("--seq-len", type=int, default=256, help="Sequence length (default: 256)")
    p_cb_causality.add_argument("--n-tests", type=int, default=8, help="Number of test positions (default: 8)")
    p_cb_causality.add_argument("--result-json", default=None, help="Path to result.json")
    p_cb_causality.add_argument("--tokenizer-path", default=None, help="Path to tokenizer model")

    p_cb_reproduce = sub.add_parser("profile-cb-reproduce", help="Determinism check for causal bank models")
    p_cb_reproduce.add_argument("--model", required=True, help="Checkpoint path (.checkpoint.pt)")
    p_cb_reproduce.add_argument("--seq-len", type=int, default=256, help="Sequence length (default: 256)")
    p_cb_reproduce.add_argument("--result-json", default=None, help="Path to result.json")
    p_cb_reproduce.add_argument("--tokenizer-path", default=None, help="Path to tokenizer model")

    p_cb_effctx = sub.add_parser("profile-cb-effective-context",
                                  help="Context-knee test: per-position bpb on random-prefix sequences, identifies effective context length")
    p_cb_effctx.add_argument("--model", required=True, help="Checkpoint path (.checkpoint.pt)")
    p_cb_effctx.add_argument("--val", default=None, help="Val bytes file (optional; random tokens used if omitted)")
    p_cb_effctx.add_argument("--seqlen", type=int, default=512, help="Sequence length (default: 512)")
    p_cb_effctx.add_argument("--n-trials", type=int, default=30, help="Number of trial sequences (default: 30)")
    p_cb_effctx.add_argument("--buckets", type=str,
                              default="1,2,4,8,16,32,64,128,256,512",
                              help="Comma-separated bucket bounds (default: 1,2,4,8,16,32,64,128,256,512)")
    p_cb_effctx.add_argument("--knee-threshold", type=float, default=0.01,
                              help="bpb delta below which adjacent buckets count as saturated (default: 0.01)")
    p_cb_effctx.add_argument("--result-json", default=None, help="Path to result.json for model config")
    p_cb_effctx.add_argument("--tokenizer-path", default=None, help="Path to tokenizer model")

    p_cb_abl = sub.add_parser("profile-cb-ablations",
                              help="Per-path bpb contribution: substrate/local/truncate ablations")
    p_cb_abl.add_argument("--model", required=True, help="Checkpoint path (.checkpoint.pt)")
    p_cb_abl.add_argument("--ablate", required=True,
                          help="Ablation mode: substrate | local | truncate:K")
    p_cb_abl.add_argument("--val", default=None, help="Val bytes file (optional)")
    p_cb_abl.add_argument("--n-tokens", type=int, default=50000,
                          help="Number of token predictions to measure over (default: 50000)")
    p_cb_abl.add_argument("--result-json", default=None, help="Path to result.json")
    p_cb_abl.add_argument("--tokenizer-path", default=None, help="Path to tokenizer model")

    p_lookup = sub.add_parser("profile-lookup-fraction", help="How much of language modeling is a table lookup vs computation?")
    p_lookup.add_argument("--mri", required=True, help=".mri directory")
    p_lookup.add_argument("--n-sample", type=int, default=5000, help="Tokens to sample (default: 5000)")

    p_drift = sub.add_parser("profile-distribution-drift", help="What do the frozen zone whispers change? Distribution shift per layer")
    p_drift.add_argument("--mri", required=True, help=".mri directory")
    p_drift.add_argument("--n-sample", type=int, default=1000, help="Tokens to sample (default: 1000)")

    p_horizon = sub.add_parser("profile-retrieval-horizon", help="How far back does the token look? Per-layer attention reach (template only)")
    p_horizon.add_argument("--mri", required=True, help=".mri directory")
    p_horizon.add_argument("--n-sample", type=int, default=1000, help="Tokens to sample (default: 1000)")

    p_opp = sub.add_parser("profile-layer-opposition", help="Do MLP and attention oppose? Direct MLP output from stored gate*up*down_proj")
    p_opp.add_argument("--mri", required=True, help=".mri directory")
    p_opp.add_argument("--n-sample", type=int, default=1000, help="Tokens to sample (default: 1000)")

    p_shart = sub.add_parser("profile-shart-anatomy", help="What makes a shart: crystal neuron, gradient sensitivity, frozen zone, bandwidth")
    p_shart.add_argument("--mri", required=True, help=".mri directory")
    p_shart.add_argument("--n-sample", type=int, default=None, help="Tokens to sample (default: all)")
    p_shart.add_argument("--top-n", type=int, default=20, help="Top/bottom N tokens to show (default: 20)")

    p_cross = sub.add_parser("profile-cross-model", help="Compare models on shared vocabulary: displacement, gradient, shart overlap")
    p_cross.add_argument("--mri", nargs="+", required=True, help="Two or more .mri directories to compare")
    p_cross.add_argument("--n-sample", type=int, default=None, help="Max shared tokens (default: all)")

    p_regrad = sub.add_parser("mri-regrad", help="Recompute embedding gradients for existing MRIs (fixes tied-weight bug)")
    p_regrad.add_argument("--model", required=True, help="Model ID")
    p_regrad.add_argument("--mri", nargs="+", required=True, help="MRI directories to fix")

    p_bw = sub.add_parser("profile-bandwidth", help="Bandwidth efficiency: what fraction of model bytes do useful work per token?")
    p_bw.add_argument("--mri", required=True, help=".mri directory")
    p_bw.add_argument("--n-sample", type=int, default=5000, help="Tokens to sample (default: 5000)")

    p_codeanat = sub.add_parser("profile-code-anatomy", help="Decompose 'code' tokens: do structural, keywords, operators fall the same?")
    p_codeanat.add_argument("--shrt", required=True, help=".shrt.npz file")
    p_codeanat.add_argument("--frt", required=True, help=".frt.npz file")
    p_codeanat.add_argument("--all-layers", default=None, help="All-layers .shrt.npz file for per-layer trajectories")

    p_trd = sub.add_parser("trd-profile", help="Generate a .trd per-head thread map from .shrt vectors + o_proj weights")
    p_trd.add_argument("--model", required=True, help="Model ID")
    p_trd.add_argument("--shrt", required=True, help=".shrt.npz file with displacement vectors")
    p_trd.add_argument("--frt", required=True, help=".frt.npz file")
    p_trd.add_argument("--layers", default=None, help="Layers to decompose (comma-separated, default: 6 evenly spaced)")
    p_trd.add_argument("--output", "-o", default=None, help="Output .trd.npz file path")

    p_srank = sub.add_parser("profile-safety-rank", help="Project all tokens onto safety direction, rank by safety projection")
    p_srank.add_argument("--shrt", required=True, help="Full-vocab .shrt.npz")
    p_srank.add_argument("--direction", required=True, help="Safety direction .npy file")
    p_srank.add_argument("--frt", default=None, help=".frt.npz (for script labels if .shrt is v0.2)")
    p_srank.add_argument("--trd", default=None, help=".trd.npz (for per-head safety correlation)")

    p_discover_dir = sub.add_parser("profile-discover-direction", help="Discover safety direction natively on a model using DB prompts")
    p_discover_dir.add_argument("--model", required=True, help="Model ID")
    p_discover_dir.add_argument("--db", default="data/heinrich.db", help="Database path")
    p_discover_dir.add_argument("--n-harmful", type=int, default=100, help="Number of harmful prompts")
    p_discover_dir.add_argument("--n-benign", type=int, default=100, help="Number of benign prompts")
    p_discover_dir.add_argument("--output", "-o", default=None, help="Output .npy file for direction vector")

    p_layerimp = sub.add_parser("profile-layer-importance", help="Rank layers by contribution to output state (reads MRI, no model needed)")
    p_layerimp.add_argument("--mri", required=True, help=".mri directory")

    p_earlyexit = sub.add_parser("profile-early-exit", help="Find tokens resolved before the final layer (reads MRI)")
    p_earlyexit.add_argument("--mri", required=True, help=".mri directory")
    p_earlyexit.add_argument("--threshold", type=float, default=95.0, help="Cosine similarity threshold (%)")

    p_tmplover = sub.add_parser("profile-template-overhead", help="Measure template vs content contribution per layer")
    p_tmplover.add_argument("--template-mri", required=True, help="Template mode .mri directory")
    p_tmplover.add_argument("--raw-mri", required=True, help="Raw mode .mri directory")

    p_backfill = sub.add_parser("mri-backfill", help="Fill missing data (embedding, norms, weights, lmhead_raw) in existing MRI directories")
    p_backfill.add_argument("--model", required=True, help="Model ID")
    p_backfill.add_argument("--mri", nargs="+", required=True, help="MRI directories to backfill")

    p_mri = sub.add_parser("mri", help="Complete model residual image: tokenizer + state + baselines + directions in one file")
    p_mri.add_argument("--model", required=True, help="Model ID or checkpoint path")
    p_mri.add_argument("--backend", choices=["auto", "mlx", "hf", "decepticon"], default="auto", help="Backend (default: auto)")
    p_mri.add_argument("--mode", choices=["template", "naked", "raw", "sequence"], default="template")
    p_mri.add_argument("--n-index", type=int, default=None, help="Number of tokens (default: full vocabulary)")
    p_mri.add_argument("--output", "-o", required=True, help="Output .mri directory")
    p_mri.add_argument("--db", default=None, help="Database path for direction discovery")
    p_mri.add_argument("--result-json", default=None, help="Decepticon: path to result.json for model config")
    p_mri.add_argument("--tokenizer-path", default=None, help="Decepticon: path to tokenizer model file")
    p_mri.add_argument("--data", default=None, help="Validation data .bin file (uint16 tokens, for causal bank sequence mode)")
    p_mri.add_argument("--n-seqs", type=int, default=50, help="Number of sequences for sequence mode (default: 50)")
    p_mri.add_argument("--seq-len", type=int, default=512, help="Sequence length for sequence mode (default: 512)")
    p_mri.add_argument("--byte-level", action="store_true", help="Read val data as uint8 bytes (for byte-level models)")
    p_mri.add_argument("--warmup", type=int, default=0, help="Warmup bytes before capture window (warm-state MRI)")

    p_capture = sub.add_parser("total-capture", help="[legacy] Use 'mri' instead. Capture every token, every layer.")
    p_capture.add_argument("--model", required=True, help="Model ID")
    p_capture.add_argument("--n-index", type=int, default=None, help="Number of tokens (default: full vocabulary)")
    p_capture.add_argument("--output", "-o", required=True, help="Output .shrt.npz file")
    p_capture.add_argument("--naked", action="store_true", help="Naked mode: single token, BOS baseline, no template")
    p_capture.add_argument("--raw", action="store_true", help="Raw mode: token alone, no BOS, no baseline. Absolute state.")

    p_basin = sub.add_parser("profile-basin", help="Map basin structure along a direction: where are attractors, where is void?")
    p_basin.add_argument("--model", required=True, help="Model ID")
    p_basin.add_argument("--direction", required=True, help="Direction .npy file")
    p_basin.add_argument("--layer", type=int, required=True, help="Layer to steer at")
    p_basin.add_argument("--mean-gap", type=float, default=1.0, help="Direction scaling factor")
    p_basin.add_argument("--n-prompts", type=int, default=20, help="Number of test prompts")

    p_ftok = sub.add_parser("profile-first-token", help="First-token logit gap for a direction (no generation needed)")
    p_ftok.add_argument("--model", required=True, help="Model ID")
    p_ftok.add_argument("--direction", required=True, help="Direction .npy file")

    p_lmh = sub.add_parser("profile-lmhead", help="Output matrix geometry: SVD, condition number, direction amplification")
    p_lmh.add_argument("--model", required=True, help="Model ID")
    p_lmh.add_argument("--directions", nargs="*", default=[], help="Direction .npy files to project")

    p_silence = sub.add_parser("profile-silence", help="Measure the silence: where does the baseline sit in displacement and safety space?")
    p_silence.add_argument("--model", required=True, help="Model ID")
    p_silence.add_argument("--shrt", required=True, help="Full-vocab .shrt.npz file")
    p_silence.add_argument("--frt", required=True, help=".frt.npz file")
    p_silence.add_argument("--db", default="data/heinrich.db", help="Database path for safety direction")
    p_silence.add_argument("--direction", default=None, help="Safety direction .npy file (overrides DB lookup)")

    p_matrix = sub.add_parser("profile-matrix", help="Data coverage matrix: what measurements exist per model")
    p_matrix.add_argument("--data-dir", default="data/runs", help="Directory containing .npz files")

    # Item 69: shared DB path
    parser.add_argument("--db-path", default=None,
                        help="Path to SQLite database (default: ./data/heinrich.db)")

    # === discover ===
    p = sub.add_parser("discover-directions", help="Find contrastive directions from prompt pairs")
    p.add_argument("--model", required=True)
    p.add_argument("--name", default="safety", help="Direction name")
    p.add_argument("--layer", type=int, default=None, help="Target layer (default: auto)")
    p.add_argument("--n-sample", type=int, default=100, help="Prompts per class")

    p = sub.add_parser("discover-neurons", help="Scan MLP neurons for safety-relevant activations")
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--top-k", type=int, default=20)

    p = sub.add_parser("discover-axes", help="Discover orthogonal behavioral axes (safety, truth, creativity, ...)")
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=None)

    p = sub.add_parser("discover-dimensionality", help="Estimate intrinsic dimensionality of behavioral manifold")
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--n-prompts", type=int, default=200)

    # === attack ===
    p = sub.add_parser("attack-cliff", help="Binary search for steering cliff per layer")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--direction", required=True, help="Direction .npy file")
    p.add_argument("--layer", type=int, required=True)

    p = sub.add_parser("attack-steer", help="Generate with direction vector applied")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--direction", required=True, help="Direction .npy file")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--alpha", type=float, default=1.0, help="Steering magnitude")
    p.add_argument("--max-tokens", type=int, default=100)

    p = sub.add_parser("attack-surface", help="Map vulnerability surface across directions and prompts")
    p.add_argument("--model", required=True)
    p.add_argument("--directions", required=True, help="Directory of .npy direction files")
    p.add_argument("--prompts", required=True, help="File with one prompt per line")
    p.add_argument("--layer", type=int, required=True)

    # === trace ===
    p = sub.add_parser("trace-causal", help="Position-aware causal tracing: where and when does the model decide?")
    p.add_argument("--model", required=True)
    p.add_argument("--clean", required=True, help="Clean prompt")
    p.add_argument("--corrupt", required=True, help="Corrupt prompt")

    p = sub.add_parser("trace-conversation", help="Track safety across multi-turn conversation")
    p.add_argument("--model", required=True)
    p.add_argument("--turns", required=True, help="JSON file with conversation turns")
    p.add_argument("--direction", default=None, help="Safety direction .npy file")
    p.add_argument("--layer", type=int, default=None)

    p = sub.add_parser("trace-generation", help="Monitor residual stream during autoregressive generation")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--directions", default=None, help="Directory of .npy direction files to track")

    # === inspect (weight forensics) ===
    p = sub.add_parser("inspect-safetensors", help="Catalog tensors in a .safetensors file")
    p.add_argument("source", help="Path to .safetensors file")
    p.add_argument("--name-regex", default=None, help="Filter tensor names")

    p = sub.add_parser("inspect-spectral", help="Spectral stats of a weight matrix")
    p.add_argument("source", help="Path to .npy matrix file")
    p.add_argument("--topk", type=int, default=16)

    p = sub.add_parser("inspect-bundle", help="Full audit of a weight bundle (.npz or .safetensors)")
    p.add_argument("source", help="Path to weight bundle")
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--only-square", action="store_true")

    # === embed (token-direction analysis) ===
    p = sub.add_parser("embed-direction", help="Find tokens most aligned/opposed to a direction vector")
    p.add_argument("--model", required=True)
    p.add_argument("--direction", required=True, help="Direction .npy file")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--space", choices=["embedding", "unembedding"], default="unembedding")

    # === probe ===
    p = sub.add_parser("probe-battery", help="Full behavioral probe battery: framing, multi-turn, special tokens")
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=100)

    p = sub.add_parser("probe-safetybench", help="Safety benchmark with optional steering attacks")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="simple_safety")
    p.add_argument("--alpha", type=float, default=0.0, help="Attack steering magnitude")
    p.add_argument("--max-tokens", type=int, default=100)

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
    elif args.command in ("companion", "viz"):
        from .companion import run_companion
        run_companion(port=args.port)
    elif args.command == "crystal-inspect":
        _cmd_crystal_inspect(args)
    elif args.command == "viz-legacy":
        from .viz import run_server
        run_server(port=args.port, db_path=args.db)
    elif args.command == "frt-profile":
        _cmd_frt(args)
    elif args.command == "shart-profile":
        _cmd_shrt(args)
    elif args.command == "audit-direction":
        _cmd_audit_direction(args)
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
    elif args.command == "profile-decompose":
        _cmd_decompose(args)
    elif args.command == "profile-validate-ablation":
        _cmd_validate_ablation(args)
    elif args.command == "profile-embedding":
        _cmd_embedding(args)
    elif args.command == "profile-scatter":
        _cmd_scatter(args)
    elif args.command == "profile-within-script":
        _cmd_within_script(args)
    elif args.command == "profile-layer-importance":
        _cmd_layer_importance(args)
    elif args.command == "profile-early-exit":
        _cmd_early_exit(args)
    elif args.command == "profile-template-overhead":
        _cmd_template_overhead(args)
    elif args.command == "mri-backfill":
        _cmd_backfill(args)
    elif args.command == "mri":
        _cmd_mri(args)
    elif args.command == "total-capture":
        _cmd_total_capture(args)
    elif args.command == "profile-basin":
        _cmd_basin(args)
    elif args.command == "profile-first-token":
        _cmd_first_token(args)
    elif args.command == "profile-lmhead":
        _cmd_lmhead(args)
    elif args.command == "profile-safety-rank":
        _cmd_safety_rank(args)
    elif args.command == "profile-discover-direction":
        _cmd_discover_direction(args)
    elif args.command == "profile-silence":
        _cmd_silence(args)
    elif args.command == "trd-profile":
        _cmd_trd(args)
    elif args.command == "profile-matrix":
        _cmd_matrix(args)
    elif args.command == "profile-directions":
        _cmd_directions(args)
    elif args.command == "profile-code-anatomy":
        _cmd_code_anatomy(args)
    elif args.command == "profile-pca-anatomy":
        _cmd_pca_anatomy(args)
    elif args.command == "profile-pca-survey":
        _cmd_pca_survey(args)
    elif args.command == "profile-pca-depth":
        _cmd_pca_depth(args)
    elif args.command == "mri-verify":
        _cmd_mri_verify(args)
    elif args.command == "mri-status":
        _cmd_mri_status(args)
    elif args.command == "mri-check-fixups":
        _cmd_mri_check_fixups(args)
    elif args.command == "mri-recapture":
        _cmd_mri_recapture(args)
    elif args.command == "mri-scan":
        _cmd_mri_scan(args)
    elif args.command == "mri-health":
        _cmd_mri_health(args)
    elif args.command == "mri-decompose":
        _cmd_mri_decompose(args)
    elif args.command == "mri-serve":
        _cmd_mri_serve(args)
    elif args.command == "profile-logit-lens":
        _cmd_logit_lens(args)
    elif args.command == "profile-layer-deltas":
        _cmd_layer_deltas(args)
    elif args.command == "profile-gates":
        _cmd_gates(args)
    elif args.command == "profile-attention":
        _cmd_attention(args)
    elif args.command == "profile-cb-manifold":
        _cmd_cb_manifold(args)
    elif args.command == "profile-cb-compare":
        _cmd_cb_compare(args)
    elif args.command == "profile-cb-health":
        _cmd_cb_health(args)
    elif args.command == "profile-cb-loss":
        _cmd_cb_loss(args)
    elif args.command == "profile-cb-routing":
        _cmd_cb_routing(args)
    elif args.command == "profile-cb-temporal":
        _cmd_cb_temporal(args)
    elif args.command == "profile-cb-modes":
        _cmd_cb_modes(args)
    elif args.command == "profile-cb-decompose":
        _cmd_cb_decompose(args)
    elif args.command == "profile-cb-omega-forensics":
        _cmd_cb_omega_forensics(args)
    elif args.command == "profile-cb-rotation-forensics":
        _cmd_cb_rotation_forensics(args)
    elif args.command == "profile-cb-additivity":
        _cmd_cb_additivity(args)
    elif args.command == "profile-cb-pc-bands":
        _cmd_cb_pc_bands(args)
    elif args.command == "profile-cb-trajectory":
        _cmd_cb_trajectory(args)
    elif args.command == "profile-cb-invertibility":
        _cmd_cb_invertibility(args)
    elif args.command == "profile-cb-rotation-probe":
        _cmd_cb_rotation_probe(args)
    elif args.command == "profile-cb-gate-forensics":
        _cmd_cb_gate_forensics(args)
    elif args.command == "profile-cb-substrate-local":
        _cmd_cb_substrate_local(args)
    elif args.command == "profile-tokenizer-difficulty":
        _cmd_tokenizer_difficulty(args)
    elif args.command == "profile-tokenizer-compare":
        _cmd_tokenizer_compare(args)
    elif args.command == "profile-cb-causality":
        _cmd_cb_causality(args)
    elif args.command == "profile-cb-reproduce":
        _cmd_cb_reproduce(args)
    elif args.command == "profile-cb-effective-context":
        _cmd_cb_effective_context(args)
    elif args.command == "profile-cb-ablations":
        _cmd_cb_ablations(args)
    elif args.command == "profile-lookup-fraction":
        _cmd_lookup_fraction(args)
    elif args.command == "profile-distribution-drift":
        _cmd_distribution_drift(args)
    elif args.command == "profile-retrieval-horizon":
        _cmd_retrieval_horizon(args)
    elif args.command == "profile-layer-opposition":
        _cmd_layer_opposition(args)
    elif args.command == "profile-cross-model":
        _cmd_cross_model(args)
    elif args.command == "mri-regrad":
        _cmd_mri_regrad(args)
    elif args.command == "profile-shart-anatomy":
        _cmd_shart_anatomy(args)
    elif args.command == "profile-bandwidth":
        _cmd_bandwidth(args)
    # === discover ===
    elif args.command == "discover-directions":
        _cmd_discover_directions(args)
    elif args.command == "discover-neurons":
        _cmd_discover_neurons(args)
    elif args.command == "discover-axes":
        _cmd_discover_axes(args)
    elif args.command == "discover-dimensionality":
        _cmd_discover_dimensionality(args)
    # === attack ===
    elif args.command == "attack-cliff":
        _cmd_attack_cliff(args)
    elif args.command == "attack-steer":
        _cmd_attack_steer(args)
    elif args.command == "attack-surface":
        _cmd_attack_surface(args)
    # === trace ===
    elif args.command == "trace-causal":
        _cmd_trace_causal(args)
    elif args.command == "trace-conversation":
        _cmd_trace_conversation(args)
    elif args.command == "trace-generation":
        _cmd_trace_generation(args)
    # === inspect ===
    elif args.command == "inspect-safetensors":
        _cmd_inspect_safetensors(args)
    elif args.command == "inspect-spectral":
        _cmd_inspect_spectral(args)
    elif args.command == "inspect-bundle":
        _cmd_inspect_bundle(args)
    # === embed ===
    elif args.command == "embed-direction":
        _cmd_embed_direction(args)
    # === probe ===
    elif args.command == "probe-battery":
        _cmd_probe_battery(args)
    elif args.command == "probe-safetybench":
        _cmd_probe_safetybench(args)
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


def _cmd_crystal_inspect(args: argparse.Namespace) -> None:
    """Scan an MRI for isolated-high-|z| tokens ("crystals") per layer."""
    import json
    from pathlib import Path
    import numpy as np

    mri_dir = Path(args.mri)
    decomp = mri_dir / "decomp"
    meta_path = decomp / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"No decomp at {decomp}")

    meta = json.loads(meta_path.read_text())
    n_layers = meta.get("n_real_layers", len(meta.get("layers", [])))

    tokens_path = mri_dir / "tokens.npz"
    if not tokens_path.exists():
        raise SystemExit(f"No tokens.npz at {mri_dir}")
    # Safe load: vocab_ids only (int array, no object decode).
    from heinrich.companion import _load_token_ids
    token_ids = _load_token_ids(str(mri_dir))
    texts = None  # text decoding not available here; use vocab_id → tokenizer separately

    layers = [args.layer] if args.layer is not None else list(range(n_layers))
    thresh = float(args.z_threshold)
    print(f"# Crystal scan: {mri_dir.name}  |z|>{thresh}  layers={layers[0]}..{layers[-1]}")
    print(f"{'layer':>6} {'n_crystals':>12} {'tok_idx':>10} {'vocab_id':>10} {'max_z':>8}")
    print("-" * 60)

    for li in layers:
        sp = decomp / f"L{li:02d}_scores.npy"
        if not sp.exists():
            continue
        scores = np.load(str(sp)).astype(np.float32)
        raw_std = scores.std(axis=0)
        std_floor = float(np.percentile(raw_std[raw_std > 0], 5)) if np.any(raw_std > 0) else 1e-8
        std = np.maximum(raw_std, std_floor) + 1e-8
        z_all = np.abs(scores.astype(np.float32)) / std[None, :]
        max_z = z_all.max(axis=1)
        outlier = max_z > thresh
        n_out = int(outlier.sum())
        if n_out == 0:
            print(f"{li:>6} {0:>12} {'—':>10} {'':>10} {'':>8}")
            continue
        idx_sorted = np.argsort(max_z)[::-1][:args.top]
        for rank, ti in enumerate(idx_sorted):
            if max_z[ti] <= thresh:
                break
            vocab = int(token_ids[ti]) if token_ids is not None and ti < len(token_ids) else -1
            head_col = f"{li:>6}" if rank == 0 else " " * 6
            count_col = f"{n_out:>12}" if rank == 0 else " " * 12
            print(f"{head_col} {count_col} {int(ti):>10} {vocab:>10} {max_z[ti]:>8.1f}")


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


def _cmd_audit_direction(args: argparse.Namespace) -> None:
    """Run the 5-test falsification pipeline on a mean-diff direction."""
    from .profile.audit import audit_direction

    truth_tokens = args.truth_tokens.split(",") if args.truth_tokens else None
    false_tokens = args.false_tokens.split(",") if args.false_tokens else None
    result = audit_direction(
        model_id=args.model,
        datasets=list(args.datasets),
        layer=args.layer,
        n_per_class=args.n_per_class,
        train_frac=args.train_frac,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        n_permutation=args.n_permutation,
        truth_tokens=truth_tokens,
        false_tokens=false_tokens,
        output_path=args.output,
    )
    # Terminal summary
    print(f"\n=== audit-direction: {args.model} L{args.layer} ===")
    print(f"train on: {result['train_dataset']}")
    print(f"in_domain_test_acc: {result['in_domain_test_acc']:.2%}")
    print(f"cohens_d: {result['cohens_d']:+.2f}  boot_p5: {result['boot_p5']:.2f}  perm_p95: {result['perm_p95']:.2f}  SNR: {result['snr']:.2f}")
    if result["transfer"]:
        print("transfer:")
        for k, v in result["transfer"].items():
            print(f"  {k}: {v:.2%}")
    v = result["vocab"]
    print(f"vocab_pass: {v['vocab_pass']} (pos_truth={v['pos_truth']} pos_false={v['pos_false']} neg_truth={v['neg_truth']} neg_false={v['neg_false']})")
    print(f"  pos_top: {v['pos_top'][:5]}")
    print(f"  neg_top: {v['neg_top'][:5]}")
    print(f"tests_passed: {result['tests_passed']}")
    print(f"VERDICT: {result['verdict']}")


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


def _cmd_within_script(args: argparse.Namespace) -> None:
    """Within-script dispersion and correlations."""
    from .profile.compare import within_script_analysis
    result = within_script_analysis(args.shrt, args.frt)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Within-Script Dispersion: {result['model']} (N={result['n_shared']}) ===")

    print(f"\n  {'script':<12} {'n':>5} {'mean_d':>8} {'std_d':>8} {'cv':>7} {'r(bytes)':>9} {'r(id)':>7}")
    print(f"  {'-'*60}")
    for s, data in result['scripts'].items():
        print(f"  {s:<12} {data['n']:>5} {data['mean_delta']:>8.2f} {data['std_delta']:>8.2f} "
              f"{data['cv']:>7.4f} {data['r_bytes_delta']:>9.4f} {data['r_id_delta']:>7.4f}")

    if result['r_script_mean_vs_cv'] is not None:
        print(f"\n  r(script_mean, script_cv) = {result['r_script_mean_vs_cv']}")

    print(f"\n  Byte-count bins:")
    print(f"\n  {'bytes':>5} {'n':>6} {'mean_d':>8} {'std_d':>8} {'cv':>7}")
    print(f"  {'-'*36}")
    for b, data in result['byte_bins'].items():
        flag = " *" if data.get('provisional') else ""
        print(f"  {b:>5} {data['n']:>6} {data['mean_delta']:>8.2f} {data['std_delta']:>8.2f} {data['cv']:>7.4f}{flag}")
    if any(d.get('provisional') for d in result['byte_bins'].values()):
        print(f"  (* n < 20)")

    print(f"\n  Within-script by byte count:")
    for s, data in result['scripts'].items():
        if not data.get('by_byte_count'):
            continue
        bins = data['by_byte_count']
        if len(bins) < 2:
            continue
        print(f"\n  {s}:")
        for b, bd in sorted(bins.items()):
            print(f"    {b} bytes: n={bd['n']:>4} mean_delta={bd['mean_delta']:>7.2f} std={bd['std_delta']:>7.2f}")


def _cmd_trd(args: argparse.Namespace) -> None:
    """Generate .trd per-head thread map."""
    from .backend.protocol import load_backend
    from .profile.trd import generate_trd

    backend = load_backend(args.model)

    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(',')]

    output = args.output or f"data/runs/{args.model.split('/')[-1]}.trd.npz"
    result = generate_trd(
        backend.model, args.shrt, args.frt,
        layers=layers, output=output,
    )

    print(f"\n=== .trd: {result['model']['name']} ({result['n_tokens']} tokens x "
          f"{len(result['layers'])} layers x {result['model']['n_heads']} heads) ===")

    for layer_str, summary in result['layer_summaries'].items():
        print(f"\n  L{layer_str}:")
        top = summary['top_heads']
        top_str = ', '.join('H{}={:.1f}'.format(h['head'], h['mean_contrib']) for h in top)
        print(f"    top heads: {top_str}")

        for s, sh in summary['script_heads'].items():
            top_h = sh['top_heads']
            h_str = ', '.join('H{}={:.3f}'.format(h['head'], h['fraction']) for h in top_h)
            print(f"    {s:<12} n={sh['n']:>4} entropy={sh['head_entropy']:.2f}  {h_str}")


def _cmd_layer_importance(args: argparse.Namespace) -> None:
    """Rank layers by contribution — no model needed."""
    from .profile.efficiency import layer_importance
    result = layer_importance(args.mri)
    print(f"\n=== Layer Importance: {result['model']} ({result['n_tokens']} tokens) ===\n")
    print(f"  {'L':>3} {'mean_delta':>10} {'std':>8} {'rank'}")
    ranked = sorted(result['contributions'], key=lambda x: x.get('mean_delta', 0))
    for c in result['contributions']:
        rank_pos = ranked.index(c) + 1
        prunable = '*' if c['layer'] in result['prunable_layers'] else ''
        print(f"  L{c['layer']:>2} {c.get('mean_delta',0):>10.2f} {c.get('std_delta',0):>8.2f} {rank_pos:>3}{prunable}")
    print(f"\n  Prunable (bottom 25%): {result['prunable_layers']}")
    print(f"  Threshold: {result['prune_threshold']}")


def _cmd_early_exit(args: argparse.Namespace) -> None:
    """Find tokens that can exit early — no model needed."""
    from .profile.efficiency import early_exit_analysis
    result = early_exit_analysis(args.mri, threshold_pct=args.threshold)
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    print(f"\n=== Early Exit: {result['model']} ({result['n_tokens']} tokens, {result['threshold_pct']}% threshold) ===\n")
    print(f"  Mean exit layer: {result['mean_exit_layer']} / {result['n_layers']-1}")
    print(f"  Tokens exiting early: {result['pct_early_exit']}%")
    if result.get('script_exit'):
        print(f"\n  {'script':<12} {'n':>6} {'mean_exit':>9} {'%early':>7}")
        for s in sorted(result['script_exit'], key=lambda x: result['script_exit'][x]['mean_exit_layer']):
            d = result['script_exit'][s]
            print(f"  {s:<12} {d['n']:>6} {d['mean_exit_layer']:>9.1f} {d['pct_early']:>6.1f}%")


def _cmd_template_overhead(args: argparse.Namespace) -> None:
    """Template vs content contribution — no model needed."""
    from .profile.efficiency import template_overhead
    result = template_overhead(args.template_mri, args.raw_mri)
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    print(f"\n=== Template Overhead: {result['model']} ({result['n_shared']} shared tokens) ===\n")
    print(f"  {'L':>3} {'template':>9} {'raw':>9} {'diff':>9} {'overhead%':>9}")
    for l in result['layers']:
        print(f"  L{l['layer']:>2} {l['template_mean_norm']:>9.1f} {l['raw_mean_norm']:>9.1f} "
              f"{l['diff_mean_norm']:>9.1f} {l['template_pct']:>8.1f}%")


def _cmd_backfill(args: argparse.Namespace) -> None:
    """Backfill missing data into existing MRIs."""
    from .backend.protocol import load_backend
    from .profile.mri import backfill_mri
    backend = load_backend(args.model)
    for mri_path in args.mri:
        print(f"\n=== Backfilling {mri_path} ===")
        result = backfill_mri(backend, mri_path)
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Filled: {result['filled'] or 'nothing missing'}")


def _cmd_mri(args: argparse.Namespace) -> None:
    """Complete model residual image."""
    from .backend.protocol import load_backend
    from .profile.mri import capture_mri
    backend_name = getattr(args, 'backend', 'auto')
    # Auto-detect decepticon checkpoints
    if backend_name == "auto" and args.model.endswith('.checkpoint.pt'):
        backend_name = "decepticon"
    backend_kwargs = {}
    if getattr(args, 'result_json', None):
        backend_kwargs['result_json'] = args.result_json
    if getattr(args, 'tokenizer_path', None):
        backend_kwargs['tokenizer_path'] = args.tokenizer_path
    backend = load_backend(args.model, backend=backend_name, **backend_kwargs)
    extra = {}
    if getattr(args, 'data', None):
        extra['val_data'] = args.data
    if getattr(args, 'n_seqs', None):
        extra['n_seqs'] = args.n_seqs
    if getattr(args, 'seq_len', None):
        extra['seq_len'] = args.seq_len
    if getattr(args, 'byte_level', False):
        extra['byte_level'] = True
    if getattr(args, 'warmup', 0) > 0:
        extra['warmup_bytes'] = args.warmup
    result = capture_mri(backend, mode=args.mode, n_index=args.n_index,
                          output=args.output, db_path=getattr(args, 'db', None),
                          **extra)
    meta = result.get('metadata', result)
    capture = meta.get('capture', meta)
    print(f"\n  {capture.get('n_tokens', '?')} tokens")
    print(f"  mode: {capture.get('mode', '?')}")
    print(f"  {meta.get('elapsed_s', '?')}s elapsed")


def _cmd_total_capture(args: argparse.Namespace) -> None:
    """Total residual capture — no interpretation, just data."""
    from .backend.protocol import load_backend
    from .profile.capture import total_capture

    backend = load_backend(args.model)
    result = total_capture(backend, n_index=args.n_index, output=args.output,
                           naked=getattr(args, 'naked', False),
                           raw=getattr(args, 'raw', False))
    print(f"\n  {result['capture']['n_tokens']} tokens x {result['capture']['n_layers']} layers x 2 positions")
    print(f"  {result['elapsed_s']}s elapsed")


def _cmd_basin(args: argparse.Namespace) -> None:
    """Map basin structure along a direction."""
    from .backend.protocol import load_backend
    from .profile.basin import map_basin
    import numpy as np

    backend = load_backend(args.model)
    direction = np.load(args.direction)

    result = map_basin(backend, direction, layer=args.layer,
                       mean_gap=args.mean_gap, n_prompts=args.n_prompts,
                       provenance={"direction_file": args.direction})

    print(f"\n=== Basin Map: {result['model']} (L{result['layer']}) ===")
    print(f"  {'alpha':>6} {'entropy':>8} {'degen%':>7} {'diversity':>9} {'top_tokens'}")
    print(f"  {'-'*55}")
    for r in result['alphas']:
        top = ', '.join(f'{k}={v}' for k, v in sorted(r['top_tokens'].items(),
                        key=lambda x: -x[1])[:3])
        print(f"  {r['alpha']:>6.1f} {r['mean_entropy']:>8.2f} {r['degenerate_pct']:>6.1f}% "
              f"{r['top_token_diversity']:>9.2f} {top}")

    if result['collapse_negative'] is not None:
        print(f"\n  Collapse (negative): alpha={result['collapse_negative']}")
    if result['collapse_positive'] is not None:
        print(f"  Collapse (positive): alpha={result['collapse_positive']}")


def _cmd_first_token(args: argparse.Namespace) -> None:
    """First-token logit gap for a direction."""
    from .backend.protocol import load_backend
    from .profile.basin import first_token_profile
    import numpy as np

    backend = load_backend(args.model)
    direction = np.load(args.direction)

    result = first_token_profile(backend, direction,
                                  provenance={"direction_file": args.direction})

    print(f"\n=== First-Token Profile: {result['model']} ===")
    if result['is_uniform_shift']:
        print(f"  WARNING: uniform shift detected (range={result['logit_range']:.2f})")
        print(f"  This direction has zero first-token specificity.")

    print(f"\n  Refuse tokens:")
    for tok, push in result['refuse_pushes'].items():
        print(f"    {tok:<15} {push:>+8.2f}")
    print(f"  Comply tokens:")
    for tok, push in result['comply_pushes'].items():
        print(f"    {tok:<15} {push:>+8.2f}")

    print(f"\n  Gap: {result['gap']:.2f} logits = {result['probability_ratio']:.0f}x")

    print(f"\n  Top amplified: {', '.join(t['token'] for t in result['top_amplified'][:5])}")
    print(f"  Top suppressed: {', '.join(t['token'] for t in result['top_suppressed'][:5])}")


def _cmd_lmhead(args: argparse.Namespace) -> None:
    """Output matrix geometry."""
    from .backend.protocol import load_backend
    from .profile.basin import lmhead_profile, direction_in_lmhead
    import numpy as np

    backend = load_backend(args.model)
    result = lmhead_profile(backend)

    print(f"\n=== lm_head Geometry: {result['model']} ===")
    print(f"  shape: {result['shape']}")
    print(f"  condition number: {result['condition_number']:.0f}x")
    svp = result['singular_value_profile']
    print(f"  S1={svp['S1']}  S10={svp['S10']}  S50={svp['S50']}  S100={svp['S100']}  Smin={svp['Smin']}")
    for pct, n in result['pcs_for_threshold'].items():
        print(f"  {float(pct)*100:.0f}% variance: {n} dims")

    for path in args.directions:
        d = np.load(path).astype(np.float32)
        d = d / np.linalg.norm(d)
        name = Path(path).stem
        dl = direction_in_lmhead(result, d, name=name)
        print(f"\n  {name}:")
        print(f"    top10={dl['loading_top10']:.3f}  top50={dl['loading_top50']:.3f}  "
              f"top100={dl['loading_top100']:.3f}  peak_sv={dl['peak_sv_index']}")


def _cmd_safety_rank(args: argparse.Namespace) -> None:
    """Project all tokens onto safety direction and rank."""
    from .profile.compare import safety_rank
    result = safety_rank(args.shrt, args.direction,
                         frt_path=args.frt, trd_path=args.trd)

    print(f"\n=== Safety Rank: {result['model']} ({result['n_tokens']} tokens) ===")
    print(f"  mean projection: {result['overall_mean_proj']}")
    print(f"  comply: {result['pct_comply']}%  refuse: {result['pct_refuse']}%")

    print(f"\n  Top tokens toward COMPLIANCE (negative projection):")
    print(f"  {'rank':>4} {'token':<20} {'proj':>8} {'delta':>6} {'script'}")
    print(f"  {'-'*55}")
    for t in result['top_comply'][:20]:
        print(f"  {t['rank']:>4} {t['token']:<20} {t['projection']:>8.2f} {t['delta']:>6.1f} {t['script']}")

    print(f"\n  Top tokens toward REFUSAL (positive projection):")
    print(f"  {'rank':>4} {'token':<20} {'proj':>8} {'delta':>6} {'script'}")
    print(f"  {'-'*55}")
    for t in result['top_refuse'][:20]:
        print(f"  {t['rank']:>4} {t['token']:<20} {t['projection']:>8.2f} {t['delta']:>6.1f} {t['script']}")

    print(f"\n  Per-script safety:")
    print(f"  {'script':<12} {'n':>6} {'mean_proj':>9} {'%comply':>7} {'%refuse':>7}")
    print(f"  {'-'*45}")
    for s in sorted(result['by_script'],
                    key=lambda x: result['by_script'][x]['mean_proj']):
        d = result['by_script'][s]
        print(f"  {s:<12} {d['n']:>6} {d['mean_proj']:>9.2f} {d['pct_comply']:>6.1f}% {d['pct_refuse']:>6.1f}%")

    if 'head_safety' in result:
        print(f"\n  Per-head safety correlation:")
        for layer_str, data in result['head_safety'].items():
            top = data['top_safety_heads']
            top_str = ', '.join('H{}={:.3f}'.format(h['head'], h['r_safety']) for h in top)
            print(f"    L{layer_str}: {top_str}")


def _cmd_discover_direction(args: argparse.Namespace) -> None:
    """Discover safety direction natively on a model."""
    from .backend.protocol import load_backend
    from .profile.compare import discover_safety_direction
    import numpy as np

    backend = load_backend(args.model)
    result = discover_safety_direction(
        backend, args.db,
        n_harmful=args.n_harmful, n_benign=args.n_benign)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Safety Direction: {result['model']} (L{result['layer']}) ===")
    print(f"  prompts: {result['n_harmful']} harmful + {result['n_benign']} benign")
    print(f"  accuracy: {result['accuracy']}")
    print(f"  effect size: {result['effect_size']}")
    print(f"  mean gap: {result['mean_gap']}")
    print(f"  stability: {result['stability']}")

    if args.output:
        np.save(args.output, result['direction'])
        print(f"  saved to {args.output}")
    else:
        default_out = f"data/runs/{result['model']}_safety_L{result['layer']}.npy"
        np.save(default_out, result['direction'])
        print(f"  saved to {default_out}")


def _cmd_silence(args: argparse.Namespace) -> None:
    """Measure the silence — where the baseline sits."""
    from .backend.protocol import load_backend
    from .profile.compare import silence_profile

    backend = load_backend(args.model)
    direction_override = None
    if getattr(args, 'direction', None):
        import numpy as np
        direction_override = np.load(args.direction)
    result = silence_profile(backend, args.shrt, args.frt,
                              db=args.db, direction_override=direction_override)

    print(f"\n=== Silence: {result['model']} (layer {result['layer']}) ===")
    print(f"  |silence|: {result['silence_norm']}")
    print(f"  entropy: {result['silence_entropy']}")
    print(f"  top token: {result['silence_top_token']}")
    print(f"  mean |displacement|: {result['displacement_mean_norm']}")
    print(f"  |silence - centroid|: {result['silence_to_centroid']}")

    print(f"\n  Silence in PCA space:")
    print(f"  {'PC':<6} {'projection':>10} {'n_stds':>8}")
    print(f"  {'-'*26}")
    for pc, data in list(result['silence_pc_projections'].items())[:10]:
        print(f"  {pc:<6} {data['projection']:>10.2f} {data['n_stds_from_center']:>8.1f}")

    safety = result.get('safety', {})
    if 'silence_projection' in safety:
        print(f"\n  Silence on safety axis:")
        print(f"    projection: {safety['silence_projection']:.4f}")
        print(f"    percentile among tokens: {safety['silence_as_pctile']:.1f}%")
        print(f"    displacement mean on safety: {safety['displacement_mean_proj']:.4f}")
        print(f"    absolute mean on safety: {safety['absolute_mean_proj']:.4f}")

        if 'by_script' in safety:
            print(f"\n  Absolute safety position per script:")
            print(f"  {'script':<12} {'n':>6} {'absolute':>9} {'std':>8}")
            print(f"  {'-'*38}")
            for s in sorted(safety['by_script'],
                           key=lambda x: -safety['by_script'][x]['absolute_mean']):
                d = safety['by_script'][s]
                print(f"  {s:<12} {d['n']:>6} {d['absolute_mean']:>9.2f} {d['absolute_std']:>8.2f}")
    elif safety.get('status'):
        print(f"\n  Safety: {safety['status']}")

    # Generate HTML visualization
    if 'by_script' in safety:
        from .profile.compare import silence_scatter_html
        model_short = result['model'].replace('/', '_')
        html_path = f"data/runs/silence_{model_short}.html"
        silence_scatter_html(result, html_path)
        print(f"\n  Visualization: {html_path}")


def _cmd_matrix(args: argparse.Namespace) -> None:
    """Data coverage matrix."""
    from .profile.compare import data_matrix
    result = data_matrix(args.data_dir)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Data Matrix: {result['data_dir']} ({result['n_models']} models) ===\n")
    print(f"  {'model':<18} {'h':>5} {'L':>3} {'frt':>4} {'shrt':>5} {'N':>6} "
          f"{'vecs':>5} {'dir':>4} {'allL':>5} {'sht':>4} {'trd':>4} {'dirs':>4}")
    print(f"  {'-'*72}")
    for name in sorted(result['coverage']):
        c = result['coverage'][name]
        def yn(v): return 'Y' if v else '-'
        dir_str = f"L{c['dir_layer']}" if c['direction'] else '-'
        print(f"  {name:<18} {c['hidden']:>5} {c['layers']:>3} "
              f"{yn(c['frt']):>4} {yn(c['shrt']):>5} {c['shrt_n']:>6} "
              f"{yn(c['vectors']):>5} {dir_str:>4} {yn(c['all_layers']):>5} "
              f"{yn(c['sht']):>4} {yn(c['trd']):>4} {c.get('n_directions',0):>4}")

    print(f"\n  h=hidden  L=layers  N=tokens  vecs=vectors  dir=safety direction")
    print(f"  allL=all-layers  trd=thread map  dirs=.npy direction files")


def _cmd_directions(args: argparse.Namespace) -> None:
    """Directional analysis of displacement vectors."""
    from .profile.compare import displacement_directions
    result = displacement_directions(args.shrt, args.frt,
                                     safety_shrt_path=getattr(args, 'safety_shrt', None))
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Displacement Directions: {result['model']} "
          f"(hidden={result['hidden_size']}, N={result['n_tokens']}) ===")

    print(f"\n  Within-script coherence (do tokens from the same script displace in the same direction?):")
    print(f"  {'script':<12} {'n':>5} {'coherence':>10} {'pairwise':>9} {'|mean_vec|':>10}")
    print(f"  {'-'*50}")
    for s in sorted(result['script_coherence'],
                    key=lambda x: -result['script_coherence'][x]['coherence_to_mean']):
        c = result['script_coherence'][s]
        print(f"  {s:<12} {c['n']:>5} {c['coherence_to_mean']:>10.4f} "
              f"{c['pairwise_cosine']:>9.4f} {c['mean_vec_norm']:>10.2f}")

    excluded = result.get('scripts_excluded', {})
    if excluded:
        print(f"\n  excluded (n < 5): {', '.join(f'{s}={n}' for s, n in excluded.items())}")

    print(f"\n  mean within-script coherence: {result['mean_within_coherence']}")
    print(f"  mean between-script cosine:   {result['mean_between_cosine']}")

    # Between-script separation matrix (top pairs)
    sep = result['between_script_cosine']
    if sep:
        sorted_pairs = sorted(sep.items(), key=lambda x: -abs(x[1]))
        print(f"\n  Between-script cosine (most aligned pairs):")
        for pair, cos in sorted_pairs[:10]:
            print(f"    {pair:<25} {cos:>7.4f}")
        print(f"  ...")
        print(f"  Between-script cosine (most orthogonal pairs):")
        for pair, cos in sorted_pairs[-5:]:
            print(f"    {pair:<25} {cos:>7.4f}")

    pca = result.get('pca', {})
    if 'pc1_variance_pct' in pca:
        print(f"\n  PCA on displacement vectors ({pca['n_vectors']} vectors):")
        print(f"    PC1: {pca['pc1_variance_pct']}%  PC2: {pca['pc2_variance_pct']}%  PC3: {pca['pc3_variance_pct']}%")
        print(f"    top 10: {pca['top_10_pct']}")
        for pct, n in pca['pcs_for_threshold'].items():
            print(f"    {float(pct)*100:.0f}% variance: {n} PCs")

        if pca.get('script_pc1_projections'):
            print(f"\n  Per-script PC1 projection:")
            print(f"  {'script':<12} {'n':>5} {'mean_pc1':>9} {'std_pc1':>8}")
            print(f"  {'-'*36}")
            for s in sorted(pca['script_pc1_projections'],
                           key=lambda x: -pca['script_pc1_projections'][x]['mean_pc1']):
                p = pca['script_pc1_projections'][s]
                print(f"  {s:<12} {p['n']:>5} {p['mean_pc1']:>9.2f} {p['std_pc1']:>8.2f}")

    if result.get('safety_layer') is not None:
        print(f"\n  Safety direction: layer {result['safety_layer']}")
        if result.get('safety_note'):
            print(f"  {result['safety_note']}")


def _cmd_pca_anatomy(args: argparse.Namespace) -> None:
    """Name the unnamed axes of displacement."""
    from .profile.compare import pca_anatomy

    # Parse direction arguments: name=path.npy
    direction_paths = {}
    if args.directions:
        for d in args.directions:
            if '=' in d:
                name, path = d.split('=', 1)
                direction_paths[name] = path
            else:
                # Use filename stem as name
                from pathlib import Path as P
                direction_paths[P(d).stem] = d

    result = pca_anatomy(args.shrt, args.frt,
                         direction_paths=direction_paths or None,
                         n_components=args.n_components)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== PCA Anatomy: {result['model']} "
          f"(N={result['n_tokens']}, hidden={result['hidden_size']}, "
          f"top {result['n_components']} PCs = {result['total_variance_explained']}%) ===")

    for comp in result['components']:
        k = comp['pc']
        print(f"\n{'='*70}")
        print(f"  PC{k}: {comp['variance_pct']:.2f}% "
              f"(cumulative {comp['cumulative_pct']:.1f}%)")

        # Script poles
        if comp['neg_pole_scripts'] or comp['pos_pole_scripts']:
            neg = ', '.join(comp['neg_pole_scripts'])
            pos = ', '.join(comp['pos_pole_scripts'])
            print(f"  (-) pole: {neg}")
            print(f"  (+) pole: {pos}")

        # Direction cosines
        if comp['direction_cosines']:
            cos_str = ', '.join(f"{n}={v:+.3f}" for n, v in comp['direction_cosines'].items())
            print(f"  direction cosines: {cos_str}")

        # Per-script table
        print(f"\n  {'script':<12} {'n':>6} {'mean':>8} {'std':>8}")
        print(f"  {'-'*36}")
        for s, v in comp['by_script'].items():
            print(f"  {s:<12} {v['n']:>6} {v['mean']:>8.2f} {v['std']:>8.2f}")

        # Extreme tokens
        print(f"\n  Top (+) tokens:")
        for t in comp['top_positive'][:10]:
            print(f"    {t['projection']:>8.2f}  {t['script']:<8} {t['delta']:>7.1f}  {t['token']}")
        print(f"  Top (-) tokens:")
        for t in comp['top_negative'][:10]:
            print(f"    {t['projection']:>8.2f}  {t['script']:<8} {t['delta']:>7.1f}  {t['token']}")


def _cmd_pca_survey(args: argparse.Namespace) -> None:
    """Compare PCA structure across models."""
    from .profile.compare import pca_survey

    pairs = []
    for p in args.pairs:
        parts = p.split(':')
        if len(parts) != 2:
            print(f"Error: pair must be shrt:frt, got '{p}'")
            return
        pairs.append((parts[0], parts[1]))

    result = pca_survey(pairs, n_components=args.n_components)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== PCA Survey: {result['n_models']} models ===\n")

    print(f"  {'model':<45} {'hidden':>6} {'tokens':>8} {'PCs@50%':>7} {'PC1%':>6} {'PC1 axis'}")
    print(f"  {'-'*95}")
    for s in result['summary']:
        name = s['name'].split('/')[-1][:42]
        print(f"  {name:<45} {s['hidden_size']:>6} {s['n_tokens']:>8} {s['pcs_for_50pct']:>7} {s['pc1_pct']:>5.1f}% {s['pc1_poles']}")

    if result['matches']:
        print(f"\n  Shared axes (Spearman rho >= 0.7):")
        print(f"  {'model_a':<22} {'PC':>3} {'%':>5}  {'model_b':<22} {'PC':>3} {'%':>5}  {'rho':>6} {'axis_a':<20} {'axis_b'}")
        print(f"  {'-'*110}")
        for m in result['matches'][:30]:
            na = m['model_a'].split('/')[-1][:20]
            nb = m['model_b'].split('/')[-1][:20]
            print(f"  {na:<22} {m['pc_a']:>3} {m['var_a']:>5.1f}  "
                  f"{nb:<22} {m['pc_b']:>3} {m['var_b']:>5.1f}  "
                  f"{m['spearman_rho']:>+6.3f} {m['pole_a']:<20} {m['pole_b']}")
    else:
        print(f"\n  No shared axes found (rho >= 0.7)")


def _cmd_pca_depth(args: argparse.Namespace) -> None:
    """PCA structure at every layer of an MRI."""
    from .profile.compare import pca_depth

    result = pca_depth(args.mri, n_sample=args.n_sample, layers=args.layers)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    def _fmt(r):
        print(f"\n=== PCA Depth: {r['model']} ({r['mode']}) "
              f"— {r['n_layers']} layers, {r['n_sampled']} tokens ===\n")

        print(f"  {'layer':>5} {'PC1%':>6} {'PC2%':>6} {'PC3%':>6} {'PCs@50%':>7} {'(-) pole':<12} {'(+) pole'}")
        print(f"  {'-'*60}")
        for lr in r['layers']:
            print(f"  L{lr['layer']:>3} {lr['pc1_pct']:>5.1f}% {lr['pc2_pct']:>5.1f}% "
                  f"{lr['pc3_pct']:>5.1f}% {lr['pcs_for_50pct']:>7} "
                  f"{lr['neg_pole']:<12} {lr['pos_pole']}")
    _json_or(args, result, _fmt, analysis_name="pca_depth")


def _cmd_mri_verify(args: argparse.Namespace) -> None:
    """Smoke-test a model: 5-token capture to verify compatibility."""
    import tempfile
    from .backend.protocol import load_backend
    from .profile.mri import capture_mri

    backend_name = getattr(args, 'backend', 'auto')
    if backend_name == "auto" and args.model.endswith('.checkpoint.pt'):
        backend_name = "decepticon"
    backend_kwargs = {}
    if getattr(args, 'result_json', None):
        backend_kwargs['result_json'] = args.result_json
    if getattr(args, 'tokenizer_path', None):
        backend_kwargs['tokenizer_path'] = args.tokenizer_path

    try:
        backend = load_backend(args.model, backend=backend_name, **backend_kwargs)
        cfg = backend.config
        print(f"Model: {cfg.model_type}, L={cfg.n_layers}, H={cfg.hidden_size}, V={cfg.vocab_size}")
    except Exception as e:
        print(f"FAIL: Cannot load model: {e}")
        return

    try:
        with tempfile.TemporaryDirectory() as tmp:
            result = capture_mri(backend, mode='raw', n_index=5, output=f'{tmp}/test.mri')
            meta = result.get('metadata', result)
            print(f"PASS: {meta.get('capture', {}).get('n_tokens', '?')} tokens captured successfully")
    except Exception as e:
        print(f"FAIL: Capture error: {e}")


def _cmd_mri_check_fixups(args: argparse.Namespace) -> None:
    """Scan MRIs in a directory and flag any with stale fix_level."""
    import json
    from pathlib import Path
    from .profile.mri import MRI_FIX_LEVEL

    min_fix = args.min_fix_level if args.min_fix_level is not None else MRI_FIX_LEVEL
    root = Path(args.dir).expanduser()
    if not root.exists():
        print(f"Error: directory {root} does not exist")
        return

    mris = sorted(set(root.rglob("*.mri")) | set(root.rglob("*.seq.mri")))
    if not mris:
        print(f"No .mri or .seq.mri directories found under {root}")
        return

    stale = []
    current = []
    unknown = []
    for mri_dir in mris:
        meta_path = mri_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            md = json.loads(meta_path.read_text())
        except Exception:
            unknown.append((mri_dir, "unreadable metadata"))
            continue
        fix_level = md.get("fix_level")
        if fix_level is None:
            stale.append((mri_dir, "no fix_level (pre-versioning)"))
        elif fix_level < min_fix:
            stale.append((mri_dir, f"fix_level={fix_level} < {min_fix}"))
        else:
            current.append(mri_dir)

    print(f"\n=== MRI fix-level check: {root} ===")
    print(f"  Minimum required fix_level: {min_fix} (heinrich current)")
    print(f"  Total MRIs: {len(mris)}")
    print(f"  Current: {len(current)}")
    print(f"  Stale (need recapture): {len(stale)}")
    if unknown:
        print(f"  Unreadable: {len(unknown)}")

    if stale:
        print(f"\n  Stale MRIs:")
        for mri_dir, reason in stale[:30]:
            rel = mri_dir.relative_to(root) if root in mri_dir.parents else mri_dir
            print(f"    {rel}  ({reason})")
        if len(stale) > 30:
            print(f"    ... and {len(stale) - 30} more")


def _resolve_recapture_source(mri_dir, md):
    """Given an MRI dir + its metadata, return (model_path, result_json,
    reason_or_None). Uses metadata.provenance first; falls back to naming
    convention for legacy MRIs without model_path. Only resolves the simple
    case — if the convention doesn't match a file on disk, returns
    (None, None, <reason>) rather than guessing.
    """
    from pathlib import Path as _P
    prov = md.get("provenance") or {}
    model_path = prov.get("model_path")
    result_json = prov.get("result_json")
    if model_path and _P(model_path).exists():
        return model_path, result_json, None
    # Legacy fallback: <mri_dir>/<base>.{seq.mri,mri} → <mri_dir>/<base>.checkpoint.pt
    name = mri_dir.name
    for suffix in (".seq.mri", ".mri"):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            break
    else:
        return None, None, f"unknown MRI suffix: {name}"
    guess = mri_dir.parent / f"{base}.checkpoint.pt"
    if guess.exists():
        # Probe for sibling result json (same-base or parent-run)
        rj_guess = mri_dir.parent / f"{base}.json"
        return str(guess), (str(rj_guess) if rj_guess.exists() else None), None
    # Step-checkpoint convention: `{run}-step{N}k.seq.mri` came from
    # `{run}-{totalk}k_step{N000}.checkpoint.pt` (session 11 naming). The
    # training run writes one JSON per run (`{run}-{totalk}k.json`).
    import re as _re
    step_match = _re.match(r"^(?P<run>.+)-step(?P<k>\d+)k$", base)
    if step_match:
        run = step_match.group("run")
        steps = int(step_match.group("k")) * 1000
        # Try each `{run}-{M}k.checkpoint.pt` whose _step{steps} file exists.
        for parent in mri_dir.parent.glob(f"{run}-*k.checkpoint.pt"):
            totalk_match = _re.match(
                rf"^{_re.escape(run)}-(?P<M>\d+)k\.checkpoint\.pt$",
                parent.name,
            )
            if not totalk_match:
                continue
            step_ckpt = parent.with_name(
                f"{run}-{totalk_match.group('M')}k_step{steps}.checkpoint.pt"
            )
            if step_ckpt.exists():
                # Path.with_suffix only swaps the last suffix (.pt→.json); we
                # need to strip the full `.checkpoint.pt` tail and re-suffix.
                rj = parent.with_name(parent.name.removesuffix(".checkpoint.pt") + ".json")
                return str(step_ckpt), (str(rj) if rj.exists() else None), None
    return None, None, f"no model_path in metadata; {guess.name} not found"


def _cmd_mri_recapture(args) -> None:
    """Re-run stale MRIs using metadata.provenance."""
    import json
    import subprocess
    import shutil
    from pathlib import Path
    from .profile.mri import MRI_FIX_LEVEL

    min_fix = args.min_fix_level if args.min_fix_level is not None else MRI_FIX_LEVEL
    root = Path(args.dir).expanduser()
    if not root.exists():
        print(f"Error: directory {root} does not exist")
        return

    mris = sorted(set(root.rglob("*.mri")) | set(root.rglob("*.seq.mri")))
    if not mris:
        print(f"No .mri or .seq.mri directories found under {root}")
        return

    # Build plan
    plan = []  # list of dicts: {mri, mode, cmd, reason}
    skipped = []  # list of (mri, reason)
    for mri_dir in mris:
        if args.only and not any(s in mri_dir.name for s in args.only):
            continue
        meta_path = mri_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            md = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            skipped.append((mri_dir, f"unreadable metadata: {e}"))
            continue
        fix_level = md.get("fix_level")
        if fix_level is not None and fix_level >= min_fix:
            continue  # current, skip

        arch = md.get("architecture")
        mode = (md.get("capture") or {}).get("mode")
        if arch != "causal_bank":
            skipped.append((mri_dir, f"unsupported architecture: {arch}"))
            continue
        if mode not in ("raw", "naked", "sequence"):
            skipped.append((mri_dir, f"unsupported mode: {mode}"))
            continue

        model_path, result_json, why = _resolve_recapture_source(mri_dir, md)
        if model_path is None:
            skipped.append((mri_dir, why))
            continue

        cmd = ["heinrich", "mri",
               "--model", model_path,
               "--mode", mode,
               "--output", str(mri_dir)]
        if result_json:
            cmd += ["--result-json", result_json]
        if mode == "sequence":
            val_data = (md.get("provenance") or {}).get("val_data")
            if not val_data or not Path(val_data).exists():
                skipped.append((mri_dir, f"sequence mode needs val_data; "
                                           f"recorded={val_data!r}"))
                continue
            # Byte-level model with non-byte val_data is a known session-11
            # failure mode: stale metadata points at fineweb_val_000000.bin
            # (sp-tokenized, vocab 1024/8192), which pre-fix heinrich silently
            # mangled via the byte-shard format bug. After the fix, byte models
            # reject it with IndexError. Prefer the _bytes.bin sibling when
            # we can prove the checkpoint is byte-level.
            if result_json and Path(result_json).exists():
                try:
                    rj = json.loads(Path(result_json).read_text())
                    tok = ((rj.get("dataset") or rj.get("config", {}).get("dataset") or {})
                           .get("tokenizer"))
                    if tok == "bytes" and "_bytes" not in Path(val_data).stem:
                        alt = Path(val_data).with_name(
                            Path(val_data).stem + "_bytes" + Path(val_data).suffix)
                        if alt.exists():
                            val_data = str(alt)
                except (json.JSONDecodeError, OSError):
                    pass
            cmd += ["--data", val_data]
            cap = md.get("capture") or {}
            if cap.get("n_seqs"):
                cmd += ["--n-seqs", str(cap["n_seqs"])]
            if cap.get("seq_len"):
                cmd += ["--seq-len", str(cap["seq_len"])]
        plan.append({
            "mri": mri_dir,
            "mode": mode,
            "cmd": cmd,
            "fix_level": fix_level,
        })

    print(f"\n=== MRI recapture plan: {root} ===")
    print(f"  Minimum required fix_level: {min_fix}")
    print(f"  To recapture: {len(plan)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Mode: {'execute' if args.execute else 'dry-run (use --execute to run)'}")

    if skipped:
        print(f"\n  Skipped (cannot resolve source):")
        for mri_dir, reason in skipped[:20]:
            rel = mri_dir.relative_to(root) if root in mri_dir.parents else mri_dir
            print(f"    {rel}  — {reason}")
        if len(skipped) > 20:
            print(f"    ... and {len(skipped) - 20} more")

    if not plan:
        return

    print(f"\n  Planned captures:")
    for item in plan[:20]:
        rel = item["mri"].relative_to(root) if root in item["mri"].parents else item["mri"]
        print(f"    {rel}  mode={item['mode']}  fix_level={item['fix_level']}")
    if len(plan) > 20:
        print(f"    ... and {len(plan) - 20} more")

    if not args.execute:
        print(f"\n  Dry-run: no changes made. Re-run with --execute to capture.")
        return

    # Execute: move old MRI to .bak-stale, run, if success rm .bak-stale else
    # restore.
    ok, failed = [], []
    for i, item in enumerate(plan, 1):
        mri_dir = item["mri"]
        bak = mri_dir.with_suffix(mri_dir.suffix + ".bak-stale")
        print(f"\n[{i}/{len(plan)}] recapture {mri_dir.name}")
        if bak.exists():
            shutil.rmtree(bak)
        mri_dir.rename(bak)
        try:
            rc = subprocess.call(item["cmd"])
        except OSError as e:
            rc = -1
            print(f"  subprocess failed: {e}")
        if rc == 0 and (mri_dir / "metadata.json").exists():
            shutil.rmtree(bak)
            ok.append(mri_dir)
        else:
            # Restore backup so we don't lose the original
            if mri_dir.exists():
                shutil.rmtree(mri_dir)
            bak.rename(mri_dir)
            failed.append((mri_dir, rc))

    print(f"\n=== Recapture done: {len(ok)} ok, {len(failed)} failed ===")
    if failed:
        for mri_dir, rc in failed:
            rel = mri_dir.relative_to(root) if root in mri_dir.parents else mri_dir
            print(f"  ! {rel}  rc={rc}")


def _cmd_mri_status(args: argparse.Namespace) -> None:
    """Show all MRIs: complete, incomplete, running."""
    import json
    import subprocess
    from pathlib import Path
    import numpy as np

    mri_dir = Path(args.dir)
    if not mri_dir.exists():
        print(f"MRI directory not found: {mri_dir}")
        return

    # Find all .mri directories
    mris = sorted(mri_dir.rglob("*.mri"))
    if not mris:
        print(f"No .mri directories in {mri_dir}")
        return

    # Check running captures
    try:
        ps = subprocess.run(["pgrep", "-af", "heinrich.cli mri"],
                            capture_output=True, text=True, timeout=5)
        running = ps.stdout.strip().split('\n') if ps.stdout.strip() else []
    except Exception:
        running = []

    complete = []
    incomplete = []
    for d in mris:
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            incomplete.append((d.name, "no metadata", {}))
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            incomplete.append((d.name, "bad metadata", {}))
            continue

        arch = meta.get("architecture", "transformer")
        model = meta.get("model", {})
        n_layers = model.get("n_layers", 0)

        if arch == "causal_bank":
            has_sub = (d / "substrate.npy").exists()
            has_hl = (d / "half_lives.npy").exists()
            has_embed = (d / "embedding.npy").exists()
            ok = has_sub and has_hl
            detail = f"modes={model.get('n_modes','?')} experts={model.get('n_experts','?')}"
            n_tok = meta.get("capture", {}).get("n_tokens", "?")
            if ok:
                health = ""
                try:
                    sub = np.load(d / "substrate.npy", mmap_mode='r')
                    if isinstance(n_tok, int) and sub.shape[0] != n_tok:
                        health = f" CORRUPT(tokens={sub.shape[0]} expected={n_tok})"
                    elif np.any(np.isnan(sub[:10].astype(np.float32))):
                        health = " CORRUPT(NaN)"
                except Exception as e:
                    health = f" CORRUPT({e})"
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                complete.append((d.name, f"{detail} {n_tok}tok {size/1e6:.0f}M{health}", meta))
            else:
                missing = []
                if not has_sub: missing.append("substrate")
                if not has_hl: missing.append("half_lives")
                if not has_embed: missing.append("embed")
                incomplete.append((d.name, f"{detail} missing={','.join(missing)}", meta))
        else:
            n_entry = len(list(d.glob("L*_entry.npy")))
            n_exit = len(list(d.glob("L*_exit.npy")))
            has_embed = (d / "embedding.npy").exists()
            has_lmhead = (d / "lmhead_raw.npy").exists()
            n_wt = len(list((d / "weights").glob("L*"))) if (d / "weights").exists() else 0

            has_entry = meta.get("capture", {}).get("has_entry", True)
            layers_ok = n_exit == n_layers and (not has_entry or n_entry == n_layers)
            detail = f"{n_layers}L h={model.get('hidden_size','?')}"
            mode = meta.get("capture", {}).get("mode", "?")
            n_tok = meta.get("capture", {}).get("n_tokens", "?")

            if layers_ok and has_embed and n_wt == n_layers:
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                # Health check: spot-check multiple layers and verify token count
                health = ""
                expected_h = model.get('hidden_size', 0)
                check_layers = [0, n_layers // 2, n_layers - 1] if n_layers > 1 else [0]
                for li in check_layers:
                    try:
                        test = np.load(d / f"L{li:02d}_exit.npy", mmap_mode='r')
                        if expected_h and test.shape[1] != expected_h:
                            health = f" CORRUPT(L{li:02d} shape={test.shape})"
                            break
                        if isinstance(n_tok, int) and test.shape[0] != n_tok:
                            health = f" CORRUPT(L{li:02d} tokens={test.shape[0]} expected={n_tok})"
                            break
                        if np.any(np.isnan(test[:10].astype(np.float32))):
                            health = f" CORRUPT(L{li:02d} NaN)"
                            break
                    except Exception as e:
                        health = f" CORRUPT(L{li:02d} {e})"
                        break
                complete.append((d.name, f"{detail} {mode} {n_tok}tok {size/1e9:.0f}G{health}", meta))
            else:
                parts = []
                if not layers_ok: parts.append(f"layers={n_entry}/{n_layers}")
                if not has_embed: parts.append("no embed")
                if not has_lmhead: parts.append("no lmhead")
                if n_wt < n_layers: parts.append(f"wt={n_wt}/{n_layers}")
                incomplete.append((d.name, f"{detail} {' '.join(parts)}", meta))

    legacy = list(Path("data/runs").glob("*.shrt.npz")) if Path("data/runs").exists() else []

    result = {
        "directory": str(mri_dir),
        "complete": [{"name": n, "detail": d} for n, d, _ in complete],
        "incomplete": [{"name": n, "detail": d} for n, d, _ in incomplete],
        "running": [r.strip() for r in running],
        "legacy_shrt": len(legacy),
    }

    def _fmt(r):
        print(f"\n=== MRI Library: {r['directory']} ===\n")
        print(f"Complete ({len(r['complete'])}):")
        for item in r['complete']:
            print(f"  \u2713 {item['name']:<35} {item['detail']}")

        if r['incomplete']:
            print(f"\nIncomplete ({len(r['incomplete'])}):")
            for item in r['incomplete']:
                print(f"  \u2717 {item['name']:<35} {item['detail']}")

        if r['running']:
            print(f"\nRunning ({len(r['running'])}):")
            for proc in r['running']:
                print(f"  \u25b6 {proc}")

        if r['legacy_shrt']:
            print(f"\nLegacy .shrt files ({r['legacy_shrt']} in data/runs/) \u2014 recapture as .mri")

        print()
    _json_or(args, result, _fmt)


def _cmd_mri_scan(args: argparse.Namespace) -> None:
    """Full MRI workup: all modes, health check, analysis."""
    import time as _time
    from pathlib import Path
    from .backend.protocol import load_backend
    from .profile.mri import capture_mri, verify_mri
    from .profile.compare import layer_deltas, logit_lens

    model_dir = Path(args.output)
    model_dir.mkdir(parents=True, exist_ok=True)

    backend_name = getattr(args, 'backend', 'auto')
    if backend_name == "auto" and args.model.endswith('.checkpoint.pt'):
        backend_name = "decepticon"
    backend_kwargs = {}
    if getattr(args, 'result_json', None):
        backend_kwargs['result_json'] = args.result_json
    if getattr(args, 'tokenizer_path', None):
        backend_kwargs['tokenizer_path'] = args.tokenizer_path

    print(f"\n{'='*60}")
    print(f"MRI SCAN: {args.model}")
    print(f"Output:   {model_dir}")
    print(f"{'='*60}\n")

    # Load model once
    backend = load_backend(args.model, backend=backend_name, **backend_kwargs)
    cfg = backend.config
    print(f"Model: {cfg.model_type}, L={cfg.n_layers}, H={cfg.hidden_size}, V={cfg.vocab_size}")

    is_cb = getattr(cfg, 'model_type', '') == 'causal_bank'
    if is_cb:
        modes = ["raw"]
        if getattr(args, 'data', None):
            modes.append("sequence")
        else:
            print("  (no --data provided, skipping sequence mode)")
    else:
        modes = ["raw", "naked", "template"]
    results = {}
    scan_start = _time.time()

    # Phase 0: Tokenizer profile (once per model, shared by all modes)
    tokenizer_path = model_dir / "tokenizer.npz"
    if not tokenizer_path.exists():
        print(f"\n--- Phase 0: Tokenizer Profile ---\n")
        from .profile.frt import generate_frt
        try:
            frt_meta = generate_frt(backend.tokenizer, output=str(tokenizer_path))
            print(f"  {frt_meta['tokenizer']['vocab_size']} tokens, "
                  f"{frt_meta['tokenizer']['n_real']} real, "
                  f"{frt_meta['tokenizer']['n_special']} special")
            if frt_meta.get('system_prompt', {}).get('injected'):
                print(f"  System prompt: {frt_meta['system_prompt']['default'][:60]}...")
        except Exception as e:
            print(f"  Tokenizer profile failed: {e}")
    else:
        print(f"\n--- Phase 0: Tokenizer profile exists ---\n")

    # Phase 1: Capture all modes
    print(f"\n--- Phase 1: Capture ---\n")
    for mode in modes:
        mri_path = str(model_dir / f"{mode}.mri")
        mri_dir = Path(mri_path)

        # Skip if already healthy
        if mri_dir.exists():
            check = verify_mri(mri_path)
            if check["healthy"]:
                print(f"  {mode}: already healthy, skipping capture")
                results[mode] = {"captured": False, "healthy": True, "path": mri_path}
                continue
            else:
                print(f"  {mode}: exists but unhealthy ({len(check['issues'])} issues), recapturing")
                import shutil
                shutil.rmtree(mri_path, ignore_errors=True)

        print(f"  {mode}: capturing...")
        t0 = _time.time()
        try:
            extra = {}
            if mode == "sequence" and getattr(args, 'data', None):
                extra['val_data'] = args.data
                extra['n_seqs'] = getattr(args, 'n_seqs', 50)
                extra['seq_len'] = getattr(args, 'seq_len', 512)
            capture_mri(backend, mode=mode, n_index=args.n_index, output=mri_path,
                        **extra)
            elapsed = _time.time() - t0
            results[mode] = {"captured": True, "elapsed_s": round(elapsed), "path": mri_path}
        except Exception as e:
            print(f"  {mode}: FAILED — {e}")
            # Clean up partial capture directory to avoid confusing future runs
            partial = Path(mri_path)
            if partial.exists():
                import shutil
                shutil.rmtree(partial, ignore_errors=True)
                print(f"  {mode}: cleaned up partial directory")
            results[mode] = {"captured": False, "error": str(e), "path": mri_path}

    # Phase 2: Health check all
    print(f"\n--- Phase 2: Health Check ---\n")
    all_healthy = True
    for mode in modes:
        mri_path = results[mode]["path"]
        if not Path(mri_path).exists():
            print(f"  {mode}: MISSING")
            results[mode]["healthy"] = False
            all_healthy = False
            continue
        check = verify_mri(mri_path)
        results[mode]["healthy"] = check["healthy"]
        results[mode]["summary"] = check["summary"]
        if check["healthy"]:
            s = check["summary"]
            print(f"  {mode}: HEALTHY  {s.get('n_tokens','?')}tok  {s.get('size_gb','?')}G")
        else:
            all_healthy = False
            print(f"  {mode}: ISSUES")
            for iss in check["issues"][:5]:
                print(f"    ! {iss}")

    if not all_healthy:
        print(f"\n  Some captures unhealthy. Stopping analysis.")
        return

    # Cross-mode token ordering validation + cache loaded MRIs for analysis
    from .profile.mri import load_mri as _load_mri
    import numpy as _np
    loaded_mris = {}  # mode -> LazyMRI, reused across analysis phases
    ref_mode = None
    ref_ids = None
    token_order_ok = True
    for mode in modes:
        mri_path = results[mode]["path"]
        if not Path(mri_path).exists():
            continue
        m = _load_mri(mri_path)
        loaded_mris[mode] = m
        ids = _np.array(m['token_ids'])
        if ref_ids is None:
            ref_mode = mode
            ref_ids = ids
        else:
            if len(ids) != len(ref_ids):
                print(f"  WARNING: {mode} has {len(ids)} tokens vs {ref_mode} has {len(ref_ids)}")
                token_order_ok = False
            else:
                n_mismatch = int(_np.sum(ids != ref_ids))
                if n_mismatch > 0:
                    print(f"  WARNING: {mode} vs {ref_mode}: {n_mismatch}/{len(ids)} token IDs differ")
                    token_order_ok = False
    if token_order_ok and ref_ids is not None:
        print(f"  Token ordering: consistent across all modes ({len(ref_ids)} tokens)")

    # Phase 3: Layer deltas (all modes)
    print(f"\n--- Phase 3: Layer Deltas ---\n")
    for mode in modes:
        mri_path = results[mode]["path"]
        print(f"  {mode}:")
        ld = layer_deltas(mri_path, n_sample=5000, _mri=loaded_mris.get(mode))
        if "error" in ld:
            print(f"    ERROR: {ld['error']}")
            continue
        # Find the crystallization layer (max amplification)
        max_amp = max(ld["layers"], key=lambda r: r["amplification"])
        # Find top 3 delta layers
        top_deltas = sorted(ld["layers"], key=lambda r: r["mean_delta_norm"], reverse=True)[:3]
        print(f"    Peak amplification: L{max_amp['layer']} ({max_amp['amplification']:.1f}x, "
              f"delta={max_amp['mean_delta_norm']:.0f})")
        print(f"    Largest deltas: " +
              ", ".join(f"L{r['layer']}({r['mean_delta_norm']:.0f})" for r in top_deltas))
        results[mode]["layer_deltas"] = ld

    # Phase 4: Logit lens (all modes)
    print(f"\n--- Phase 4: Logit Lens ---\n")
    n_layers = cfg.n_layers
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    for mode in modes:
        mri_path = results[mode]["path"]
        print(f"  {mode}:")
        ll = logit_lens(mri_path, n_sample=200, layers=sample_layers, _mri=loaded_mris.get(mode))
        if "error" in ll:
            print(f"    ERROR: {ll['error']}")
            continue
        for lr in ll["layers"]:
            from collections import Counter
            top1 = Counter(p['top_ids'][0] for p in lr['predictions'])
            mode_id, mode_count = top1.most_common(1)[0]
            concentration = mode_count / len(lr['predictions']) * 100
            print(f"    L{lr['layer']:>2}: top-1 id={mode_id} ({concentration:.0f}% of tokens)")
        results[mode]["logit_lens"] = ll

    # Phase 5: PCA depth (raw mode — most interpretable)
    # Phase 5: Gate analysis (all modes)
    print(f"\n--- Phase 5: Gate Analysis ---\n")
    from .profile.compare import gate_analysis
    for mode in modes:
        mri_path = results[mode]["path"]
        ga = gate_analysis(mri_path, n_sample=5000, _mri=loaded_mris.get(mode))
        if "error" in ga:
            print(f"  {mode}: {ga['error']}")
            continue
        print(f"  {mode}:")
        # Find most interesting layers: highest concentration and lowest
        by_conc = sorted(ga["layers"], key=lambda r: r["top1_concentration"], reverse=True)
        most_conc = by_conc[0]
        least_conc = by_conc[-1]
        print(f"    Most concentrated: L{most_conc['layer']} "
              f"({most_conc['top1_concentration']:.0%} share neuron {most_conc['top1_neuron']})")
        print(f"    Most diverse:      L{least_conc['layer']} "
              f"({least_conc['unique_neurons']} unique neurons)")
        # Per-script specialization at the most concentrated layer
        if most_conc.get("script_top1"):
            scripts_str = ", ".join(f"{s}→n{v['neuron']}" for s, v in most_conc["script_top1"].items())
            print(f"    Script routing at L{most_conc['layer']}: {scripts_str}")
        results[mode]["gate_analysis"] = ga

    # Phase 6: Attention analysis (template mode only)
    print(f"\n--- Phase 6: Attention Analysis (template) ---\n")
    from .profile.compare import attention_analysis
    tpl_path = results["template"]["path"]
    aa = attention_analysis(tpl_path, n_sample=5000, _mri=loaded_mris.get("template"))
    if "error" in aa:
        print(f"  {aa['error']}")
    else:
        print(f"  {'layer':>5} {'self':>6} {'prefix':>7} {'suffix':>7} {'entropy':>8}")
        print(f"  {'-'*40}")
        # Show sampled layers
        show_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        for r in aa["layers"]:
            if r["layer"] in show_layers:
                print(f"  L{r['layer']:>3} {r['self_weight']:>5.1%} {r['prefix_weight']:>6.1%} "
                      f"{r['suffix_weight']:>6.1%} {r['entropy']:>8.3f}")
        results["template"]["attention_analysis"] = aa

    # Phase 7: PCA depth (raw mode)
    print(f"\n--- Phase 7: PCA Depth (raw) ---\n")
    from .profile.compare import pca_depth
    raw_path = results["raw"]["path"]
    pd = pca_depth(raw_path, n_sample=5000)
    if "error" in pd:
        print(f"  ERROR: {pd['error']}")
    else:
        print(f"  {'layer':>5} {'PC1%':>6} {'PCs@50%':>7} {'axis'}")
        print(f"  {'-'*35}")
        for r in pd["layers"]:
            print(f"  L{r['layer']:>3} {r['pc1_pct']:>5.1f}% {r['pcs_for_50pct']:>7} "
                  f"{r['neg_pole']}→{r['pos_pole']}")
        results["raw"]["pca_depth"] = pd

    total_elapsed = _time.time() - scan_start
    total_gb = sum(r.get("summary", {}).get("size_gb", 0) for r in results.values())
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE: {args.model}")
    print(f"  3 modes, all healthy, {total_gb:.1f} GB total")
    print(f"  {total_elapsed:.0f}s elapsed")
    print(f"{'='*60}\n")


def _cmd_mri_health(args: argparse.Namespace) -> None:
    """Deep health check on MRI directories."""
    from pathlib import Path
    from .profile.mri import verify_mri

    if args.mri:
        mris = [Path(m) for m in args.mri]
    else:
        mri_dir = Path(args.dir)
        if not mri_dir.exists():
            print(f"MRI directory not found: {mri_dir}")
            return
        mris = sorted(mri_dir.rglob("*.mri"))

    if not mris:
        print("No .mri directories found.")
        return

    results = []
    for d in mris:
        r = verify_mri(str(d))
        r["path"] = str(d)
        r["name"] = d.name
        results.append(r)

    n_healthy = sum(1 for r in results if r["healthy"])
    n_issues = len(results) - n_healthy
    result = {"n_total": len(results), "n_healthy": n_healthy,
              "n_issues": n_issues, "mris": results}

    def _fmt(r):
        print(f"\n=== MRI Health Check: {r['n_total']} directories ===\n")
        for item in r['mris']:
            s = item["summary"]
            if item["healthy"]:
                size = s.get("size_gb", "?")
                print(f"  HEALTHY  {item['name']:<35} {s.get('model','?'):>8} {s.get('mode','?'):>8} "
                      f"{s.get('n_tokens','?'):>8}tok {s.get('n_layers','?'):>3}L {size}G")
            else:
                print(f"  ISSUES   {item['name']:<35} {s.get('model','?'):>8} {s.get('mode','?'):>8}")
                for iss in item["issues"]:
                    print(f"           ! {iss}")
        print(f"\n  {r['n_healthy']} healthy, {r['n_issues']} with issues, {r['n_total']} total\n")
    _json_or(args, result, _fmt)


def _cmd_mri_decompose(args: argparse.Namespace) -> None:
    """PCA decompose MRI for the companion viewer."""
    from .profile.compare import mri_decompose

    result = mri_decompose(args.mri, n_sample=args.n_sample,
                           n_components=args.n_components)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    def _fmt(r):
        print(f"\n=== MRI Decompose: {r['model']} ({r['mode']}) ===\n")
        print(f"  {r['n_tokens']} tokens × {r['n_components']} PCs × {r['n_layers']} layers")
        print(f"  Binary: {r['bin_size_mb']} MB → {r['mri_path']}/decomp/all_scores.bin\n")
        for lr in r['layers']:
            dim_bar = '#' * min(int(lr['intrinsic_dim']), 40)
            ln = lr['layer'] if isinstance(lr['layer'], str) else f"L{lr['layer']:02d}"
            print(f"  {ln:>5s}  PC1={lr['pc1_pct']:>5.1f}%  dim={lr['intrinsic_dim']:>4.0f}  "
                  f"nbr={lr['neighbor_stability']:.2f}  {dim_bar}")
        print()
    _json_or(args, result, _fmt)


def _cmd_mri_serve(args: argparse.Namespace) -> None:
    """Build query-shaped serve artifacts for the companion viewer."""
    from .companion_serve import build_serve_artifacts

    steps = tuple(
        int(part.strip())
        for part in str(args.steps).split(",")
        if part.strip()
    )
    result = build_serve_artifacts(args.mri, steps=steps, force=bool(args.force))
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    def _fmt(r):
        print(f"\n=== MRI Serve: {r['mri_path']} ===\n")
        print(f"  {r['n_tokens']} tokens × {r['full_k']} PCs × {r['n_layers']} layers")
        print(f"  Output: {r['serve_dir']}")
        for step_key, step_info in sorted(r["steps"].items(), key=lambda kv: int(kv[0])):
            print(f"  step={step_key:>3s}  {step_info['n_sample']:>8} sampled tokens  {step_info['pc_scores']}")
        print()

    _json_or(args, result, _fmt)


def _cmd_logit_lens(args: argparse.Namespace) -> None:
    """Logit lens: what would the model predict at each layer?"""
    from .profile.compare import logit_lens

    result = logit_lens(args.mri, top_k=args.top_k,
                        layers=args.layers, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    def _fmt(r):
        print(f"\n=== Logit Lens: {r['model']} — {r['n_tokens']} tokens, "
              f"{r['n_layers']} layers, top-{r['top_k']} ===\n")

        for lr in r['layers']:
            layer = lr['layer']
            preds = lr['predictions']
            # Summarize: most common top-1 prediction across sampled tokens
            from collections import Counter
            top1_counts = Counter(p['top_ids'][0] for p in preds)
            most_common = top1_counts.most_common(5)
            common_str = ', '.join(f"id={tid}({cnt})" for tid, cnt in most_common)
            print(f"  L{layer:>2}: top-1 mode: {common_str}")
    _json_or(args, result, _fmt, analysis_name="logit_lens")


def _cmd_layer_deltas(args: argparse.Namespace) -> None:
    """Layer deltas: what each layer actually computes."""
    from .profile.compare import layer_deltas

    result = layer_deltas(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    def _fmt(r):
        print(f"\n=== Layer Deltas: {r['model']} ({r['mode']}) — "
              f"{r['n_tokens']} tokens ===\n")
        print(f"  {'layer':>5} {'mean':>10} {'max':>10} {'std':>10} {'amplif':>8}")
        print(f"  {'-'*45}")
        for lr in r['layers']:
            print(f"  L{lr['layer']:>3} {lr['mean_delta_norm']:>10.2f} "
                  f"{lr['max_delta_norm']:>10.2f} {lr['std_delta_norm']:>10.2f} "
                  f"{lr['amplification']:>7.1f}x")
    _json_or(args, result, _fmt, analysis_name="layer_deltas")


def _cmd_gates(args: argparse.Namespace) -> None:
    """MLP gate analysis."""
    from .profile.compare import gate_analysis

    result = gate_analysis(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    def _fmt(r):
        print(f"\n=== Gate Analysis: {r['model']} ({r['mode']}) — "
              f"{r['n_tokens']} tokens, top-{r['gate_k']} ===\n")
        print(f"  {'layer':>5} {'unique':>7} {'top1%':>6} {'mean':>7} {'max':>7} {'top neuron'}")
        print(f"  {'-'*50}")
        for lr in r['layers']:
            print(f"  L{lr['layer']:>3} {lr['unique_neurons']:>7} {lr['top1_concentration']:>5.0%} "
                  f"{lr['mean_activation']:>7.2f} {lr['max_activation']:>7.2f} "
                  f"n{lr['top1_neuron']}")
    _json_or(args, result, _fmt, analysis_name="gate_analysis")


def _cmd_attention(args: argparse.Namespace) -> None:
    """Attention analysis."""
    from .profile.compare import attention_analysis

    result = attention_analysis(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    def _fmt(r):
        print(f"\n=== Attention Analysis: {r['model']} ({r['mode']}) — "
              f"{r['n_tokens']} tokens, {r['n_heads']} heads, "
              f"seq_len={r['seq_len']}, token_pos={r['token_pos']} ===\n")

        print(f"  {'layer':>5} {'self':>6} {'prefix':>7} {'suffix':>7} {'entropy':>8}")
        print(f"  {'-'*40}")
        for lr in r['layers']:
            print(f"  L{lr['layer']:>3} {lr['self_weight']:>5.1%} {lr['prefix_weight']:>6.1%} "
                  f"{lr['suffix_weight']:>6.1%} {lr['entropy']:>8.3f}")
    _json_or(args, result, _fmt, analysis_name="attention_analysis")


def _cmd_cb_manifold(args: argparse.Namespace) -> None:
    """Causal bank manifold analysis."""
    from .profile.compare import causal_bank_manifold

    result = causal_bank_manifold(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Causal Bank Manifold: {result['model']} — "
          f"{result['n_modes']} modes, {result['n_bands']} bands, "
          f"{result['n_experts']} experts, {result['n_tokens']} tokens ===\n")

    p = result['pca']
    print(f"  PCA: effective_dim={p['effective_dim']}, "
          f"PCs for 50%={p['pcs_for_50']}, 80%={p['pcs_for_80']}, 95%={p['pcs_for_95']}")
    print(f"  Top 10 PC variance: {p['top_10_variance']}")

    if result['band_loadings']:
        print(f"\n  Band loadings (% of L2 norm per band):")
        for bl in result['band_loadings'][:5]:
            pcts = ' '.join(f'B{i}={v:.0f}%' for i, v in enumerate(bl['band_pcts']))
            print(f"    PC{bl['pc']} ({bl['variance_pct']:.1f}%): {pcts}")

    if result['bands']:
        print(f"\n  {'Band':>6} {'HL range':>15} {'Modes':>6} {'Substrate L2':>13}")
        for b in result['bands']:
            print(f"  {b['band']:>6} {b['hl_min']:>6.1f}-{b['hl_max']:<6.1f} {b['n_modes']:>6} {b['substrate_l2']:>13.4f}")

    if result['readout']:
        r = result['readout']
        if 'bands' in r:
            print(f"\n  Readout weight by band:")
            for rb in r['bands']:
                print(f"    Band {rb['band']}: {rb['mean_weight']:.4f} ({rb['pct_of_total']:.1f}%)")
        if 'slow_fast_ratio' in r:
            print(f"  Slow/fast readout ratio: {r['slow_fast_ratio']:.4f}")
        print(f"  Dead modes: {r.get('dead_modes', '?')}")

    if result['routing']:
        print(f"\n  Router differentiation:")
        for key, info in result['routing'].items():
            print(f"    {key}: {info['n_experts']} experts, "
                  f"cos mean={info['router_cos_mean']:.4f} "
                  f"[{info['router_cos_min']:.3f}, {info['router_cos_max']:.3f}]")

    if result['gate']:
        print(f"\n  Gate analysis (gated delta):")
        for gate_name, info in result['gate'].items():
            print(f"    {gate_name}: default={info['default_opening']:.1%} "
                  f"[{info['opening_range'][0]:.1%}, {info['opening_range'][1]:.1%}]")

    if result['ssm']:
        ssm = result['ssm']
        print(f"\n  SSM scan: shape={ssm['A_shape']}, "
              f"hl range=[{ssm['scan_hl_range'][0]:.1f}, {ssm['scan_hl_range'][1]:.1f}]")


def _cmd_cb_compare(args: argparse.Namespace) -> None:
    """Compare two causal bank MRIs."""
    from .profile.compare import causal_bank_compare

    result = causal_bank_compare(args.a, args.b, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Causal Bank Comparison ({result['n_tokens']} tokens) ===\n")
    print(f"  A: {result['a']} ({result['a_info']['n_modes']} modes, {result['a_info']['n_bands']} bands)")
    print(f"  B: {result['b']} ({result['b_info']['n_modes']} modes, {result['b_info']['n_bands']} bands)")
    print(f"\n  Displacement correlation: {result['displacement_correlation']:.4f}")
    print(f"  CKA: {result['cka']:.4f}")
    if result['router_cosine']:
        print(f"  Router cosine: {result['router_cosine']}")


def _cmd_cb_health(args: argparse.Namespace) -> None:
    """Validate causal bank MRI."""
    from .profile.compare import causal_bank_health

    result = causal_bank_health(args.mri)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    status = "PASS" if result['ok'] else "FAIL"
    print(f"\n=== Causal Bank Health: {status} ===")
    print(f"  {result['n_modes']} modes, {result['n_bands']} bands, "
          f"{result['n_experts']} experts, {result['n_tokens']} tokens\n")
    for check in result['checks']:
        print(f"  {check}")


def _cmd_cb_loss(args: argparse.Namespace) -> None:
    """Causal bank loss decomposition."""
    from .profile.compare import causal_bank_loss

    result = causal_bank_loss(args.mri)
    if "error" in result:
        if getattr(args, 'json_output', False):
            json.dump(result, sys.stdout, default=str); print()
        else:
            print(f"Error: {result['error']}")
        return

    def _fmt(r):
        print(f"\n=== CB Loss: {r['model']} — "
              f"{r['n_seqs']} seqs x {r['seq_len']} positions ===\n")
        print(f"  Overall: {r['overall_bpb']:.4f} bpb\n")
        if r.get("warning"):
            print(f"  ⚠  WARNING: {r['warning']}\n")
        print(f"  {'Position':>12} {'Mean bpb':>10} {'Std':>8}")
        for p in r['by_position']:
            print(f"  {p['range']:>12} {p['mean_bpb']:>10.4f} {p['std_bpb']:>8.4f}")
        if r['by_band']:
            print(f"\n  Per-band loss:")
            for b in r['by_band']:
                print(f"    Band {b['band']}: {b['mean_bpb']:.4f} bpb")
        if r['autocorrelation']:
            print(f"\n  Loss autocorrelation:")
            for a in r['autocorrelation']:
                print(f"    lag {a['lag']:>3}: r={a['r']:.4f}")
    _json_or(args, result, _fmt)


def _cmd_cb_routing(args: argparse.Namespace) -> None:
    """Causal bank routing analysis."""
    from .profile.compare import causal_bank_routing

    result = causal_bank_routing(args.mri)
    if "error" in result:
        if getattr(args, 'json_output', False):
            json.dump(result, sys.stdout, default=str); print()
        else:
            print(f"Error: {result['error']}")
        return

    def _fmt(r):
        print(f"\n=== CB Routing: {r['model']} — "
              f"{r['n_experts']} experts, {r['n_seqs']} seqs ===\n")
        print(f"  Switch rate: {r['switch_rate']:.2f}%")
        print(f"  Routing margin: {r['routing_margin']:.4f}")
        print(f"  Routing entropy: {r['routing_entropy']:.4f}\n")
        print(f"  Overall distribution:")
        for d in r['overall_distribution']:
            bar = '#' * int(d['pct'] / 2)
            print(f"    E{d['expert']}: {d['pct']:>5.1f}% {bar}")
        if r['by_position']:
            print(f"\n  {'Position':>12} {'Switch%':>8}  Distribution")
            for p in r['by_position']:
                dist_str = ' '.join(f'{v:.0f}%' for v in p['distribution'])
                print(f"  {p['range']:>12} {p['switch_rate']:>7.1f}%  {dist_str}")
    _json_or(args, result, _fmt)


def _cmd_cb_temporal(args: argparse.Namespace) -> None:
    """Causal bank temporal attention forensics."""
    from .profile.compare import causal_bank_temporal

    result = causal_bank_temporal(args.mri)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== CB Temporal: {result['model']} — "
          f"{result['n_seqs']} seqs x {result['seq_len']} positions ===\n")

    print(f"  Output L2 by position:")
    for r in result['output_l2_by_position']:
        print(f"    {r['range']:>12}: mean={r['mean_l2']:.4f}  max={r['max_l2']:.4f}")

    cc = result['correlation_chain']
    print(f"\n  Correlation chain:")
    for k, v in cc.items():
        print(f"    {k}: r={v:.4f}")

    sp = result.get('snapshot_profile', {})
    if sp:
        print(f"\n  Snapshot profile: {sp['n_snapshots']} snapshots, "
              f"interval={sp['snapshot_interval']}, peak=#{sp['peak_snapshot']}")


def _cmd_cb_modes(args: argparse.Namespace) -> None:
    """Causal bank mode utilization."""
    from .profile.compare import causal_bank_modes

    result = causal_bank_modes(args.mri)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== CB Modes: {result['model']} — "
          f"{result['n_modes']} modes, {result['n_seqs']} seqs ===\n")
    print(f"  Dead modes: {result['dead_modes']}")

    if result['by_quartile']:
        print(f"\n  Mode activation by half-life quartile:")
        for q in result['by_quartile']:
            parts = ' '.join(f"{p['mean_abs']:.4f}" for p in q['by_position'])
            print(f"    Q{q['quartile']} ({q['hl_range']}, {q['n_modes']}m): "
                  f"ramp={q['ramp_ratio']:.1f}x  {parts}")

    if result['growth_curve']:
        print(f"\n  Substrate L2 growth:")
        for g in result['growth_curve']:
            print(f"    {g['range']:>12}: {g['mean_l2']:.4f}")

    if result['most_varying']:
        print(f"\n  Most position-varying modes:")
        for m in result['most_varying']:
            hl = f"hl={m['hl']:.1f}" if m['hl'] is not None else ""
            print(f"    mode {m['mode']}: std={m['std']:.4f} {hl}")


def _cmd_cb_decompose(args: argparse.Namespace) -> None:
    """Causal bank manifold decomposition."""
    from .profile.compare import causal_bank_decompose

    result = causal_bank_decompose(args.mri, n_sample=getattr(args, 'n_sample', None))
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== CB Decompose: {result['model']} — "
          f"{result['n_modes']} modes, {result['n_seqs']} seqs ===\n")

    p = result['pca']
    print(f"  PCA: eff_dim={p['effective_dim']}, "
          f"50%={p['pcs_for_50']}PCs, 80%={p['pcs_for_80']}PCs, 95%={p['pcs_for_95']}PCs")
    print(f"\n  Position R2: {result['position_r2']:.4f}")
    print(f"  Content R2:  {result['content_r2']:.4f}")
    print(f"\n  Variance breakdown (top 20 PCs):")
    print(f"    Position: {result['position_fraction']:.1f}%")
    print(f"    Content:  {result['content_fraction']:.1f}%")
    print(f"    Ghost:    {result['ghost_fraction']:.1f}%")

    print(f"\n  {'PC':>4} {'Var%':>6} {'Pos r':>7} {'Loss r':>7}")
    for i in range(min(10, len(result['top_variance_pct']))):
        print(f"  {i+1:>4} {result['top_variance_pct'][i]:>5.1f}% "
              f"{result['pc_position_r'][i]:>7.3f} {result['pc_loss_r'][i]:>7.3f}")


def _cmd_cb_omega_forensics(args: argparse.Namespace) -> None:
    """Omega projection weight forensics."""
    from .profile.compare import causal_bank_omega_forensics

    result = causal_bank_omega_forensics(args.checkpoint)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Omega Forensics: {result['n_modes']} modes x {result['n_input']} input ===\n")
    print(f"  Weight L2: {result['weight_l2']}")
    print(f"  Per-mode L2 range: {result['per_mode_l2_range']}")
    print(f"  Gini concentration: {result['gini_concentration']}")
    if result['fourier_corr'] is not None:
        print(f"  Fourier correlation: {result['fourier_corr']}")
        print(f"  Fourier drift: {result['fourier_drift']}")
    print(f"  Weight eff rank: {result['weight_eff_rank']} (50%={result['rank_for_50_pct']}, 90%={result['rank_for_90_pct']})")

    print(f"\n  Band allocation:")
    for name, info in result['band_allocation'].items():
        pct = info['frac_total'] * 100
        bar = '#' * int(pct / 2)
        print(f"    {name:>20}: {pct:>5.1f}% ({info['n_modes']} modes, L2={info['mean_l2']:.3f}) {bar}")


def _cmd_cb_additivity(args: argparse.Namespace) -> None:
    """Verify the additivity law on bpb and/or geometry metrics:
    metric(A+B+...) ≈ metric(baseline) + Σ Δ(Mᵢ)."""
    from .profile.compare import causal_bank_additivity

    noise_floors = {
        "bpb":         args.noise_floor,
        "eff_dim":     args.eff_dim_floor,
        "pos_r2":      args.pos_r2_floor,
        "cont_r2":     args.cont_r2_floor,
        "active_frac": args.active_frac_floor,
    }
    try:
        result = causal_bank_additivity(
            args.baseline, args.mutations, args.combination,
            noise_floor=args.noise_floor,
            metrics=tuple(args.metrics),
            noise_floors=noise_floors,
            svd_samples=args.svd_samples,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    def _fmt(r):
        print(f"\n=== CB Additivity Check ===\n")
        print(f"  Baseline:   {r['baseline']['mri'].split('/')[-1]}")
        for m in r['mutations']:
            print(f"  Mutation:   {m['mri'].split('/')[-1]}")
        print(f"  Combination: {r['combination']['mri'].split('/')[-1]}\n")

        for metric in r['metrics']:
            info = r['per_metric'][metric]
            tag = "⚠ NON-ADDITIVE" if info['is_non_additive'] else "✓ additive"
            # Choose precision per metric: bpb/R² → 4dp; eff_dim → 1dp
            fmt = ".1f" if metric == "eff_dim" else ".4f"
            deltas = "  ".join(
                f"Δ{i+1}={format(m.get(f'delta_{metric}', 0), '+'+fmt)}"
                for i, m in enumerate(r['mutations'])
            )
            print(f"  [{metric}] baseline={format(info['baseline'], fmt)}  "
                  f"{deltas}")
            print(f"    predicted={format(info['predicted'], fmt)}  "
                  f"actual={format(info['actual'], fmt)}  "
                  f"gap={format(info['gap'], '+'+fmt)}  "
                  f"(floor ±{info['noise_floor']})")
            print(f"    {tag} — {info['verdict']}\n")
    _json_or(args, result, _fmt)


def _cmd_cb_rotation_forensics(args: argparse.Namespace) -> None:
    """Rotation / gate-weight forensics for non-adaptive substrates."""
    from .profile.compare import causal_bank_rotation_forensics

    result = causal_bank_rotation_forensics(args.checkpoint)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    fams = ", ".join(result["substrate_families"])
    print(f"\n=== CB Rotation Forensics ({fams}) — {result['n_modules_total']} modules ===\n")
    print(f"  Trained (non-zero weight): {result['n_modules_trained']}")
    print(f"  Frozen at zero (untrained): {result['n_modules_frozen_at_zero']}")
    if result['frozen_modules']:
        print(f"  ⚠ Frozen modules: {', '.join(result['frozen_modules'])}")

    print(f"\n  {'Module':>42} {'shape':>14} {'L2':>8} {'bias L2':>8} {'eff rank':>9} {'top SV':>8}")
    for mod, info in result['modules'].items():
        shape = 'x'.join(str(s) for s in info.get('weight_shape', ()))
        print(f"  {mod:>42} {shape:>14} "
              f"{info.get('weight_l2', 0):>8.3f} "
              f"{info.get('bias_l2', 0):>8.3f} "
              f"{info.get('effective_rank', 0):>9.2f} "
              f"{info.get('top_sv', 0):>8.3f}")


def _cmd_cb_pc_bands(args: argparse.Namespace) -> None:
    """Per-PC-band content / position decomposition for one or more MRIs."""
    from .profile.compare import causal_bank_pc_bands

    reports = []
    for mri in args.mris:
        r = causal_bank_pc_bands(mri, n_bootstrap=args.n_bootstrap)
        if "error" in r:
            print(f"Error on {mri}: {r['error']}")
            continue
        reports.append(r)

    result = {"reports": reports}

    def _fmt(res):
        print(f'\n=== CB PC-band decomposition: {len(res["reports"])} MRI(s) ===\n')
        # Per-band table across MRIs
        # Header
        band_labels = [b["range"] for b in res["reports"][0]["bands"]] if res["reports"] else []
        band_header = "  ".join(f'PC{l:>8s}'.rjust(26) for l in band_labels)
        print(f'  {"mri":<55} {"EffDim":>7} {"score":>6} {"verdict":>12}  '
              + "  ".join(f'{"PC " + l:>24}' for l in band_labels))
        sub_header = (" " * (55 + 1)) + (" " * (7 + 6 + 12 + 2)) + \
                      "  ".join(f'{"var% pos_r2 byte_r2":>24}' for _ in band_labels)
        print(sub_header)
        print('-' * max(132, 77 + 26 * len(band_labels)))
        for r in res["reports"]:
            name = r["mri"].split("/")[-1].replace(".seq.mri", "").replace(".mri", "")[-55:]
            score_str = (f'{r["partition_score"]:>6.1f}'
                         if r["partition_score"] < 1000 else f'{">1000":>6}')
            row = [f'  {name:<55}',
                   f'{r["eff_dim"]:>7.1f}',
                   score_str,
                   f'{r["partition_verdict"]:>12}']
            for b in r["bands"]:
                row.append(f' {b["var_pct"]:>5.1f}% {b["pos_r2"]:>6.3f} {b["byte_r2"]:>6.3f}'.rjust(24))
            print("  ".join(row))

        print(f'\n  Cumulative position R² vs top-k PCs:')
        print(f'  {"mri":<55} ' + "  ".join(f'{"k=" + str(e["top_k"]):>8}' for e in res["reports"][0]["cumulative_pos_r2"]))
        for r in res["reports"]:
            name = r["mri"].split("/")[-1].replace(".seq.mri", "").replace(".mri", "")[-55:]
            vals = "  ".join(f'{e["pos_r2"]:>8.3f}' for e in r["cumulative_pos_r2"])
            print(f'  {name:<55} {vals}')

        # Summary by verdict
        from collections import Counter
        verdicts = Counter(r["partition_verdict"] for r in res["reports"])
        if verdicts:
            print(f'\n  Partition summary: '
                  + ", ".join(f'{v}={c}' for v, c in sorted(verdicts.items())))
        flags = [r for r in res["reports"] if r["two_band_partition"]]
        if flags:
            print(f'  ⚠ {len(flags)} MRI(s) have a two-band partition '
                  f'(partial or partitioned). EffDim undercounts working dim.')

    _json_or(args, result, _fmt)


def _cmd_cb_trajectory(args: argparse.Namespace) -> None:
    """Trajectory analysis across checkpoints."""
    from .profile.compare import causal_bank_trajectory

    result = causal_bank_trajectory(args.mris)

    def _fmt(r):
        print(f"\n=== Trajectory: {r['n_checkpoints']} checkpoints ===\n")
        print(f"  EffDim trend: {r['eff_dim_trend']}\n")
        print(f"  {'Checkpoint':>40} {'EffDim':>8} {'PosR²':>8} {'ConR²':>8} {'Active':>8} {'Dead':>6}")
        for p in r['points']:
            name = p['path'].split('/')[-1] if '/' in p['path'] else p['path']
            print(f"  {name:>40} {p['eff_dim']:>8.1f} {p['pos_r2']:>8.4f} {p['cont_r2']:>8.4f} "
                  f"{p['active_frac']:>8.4f} {p['dead_modes']:>6}")
        if r['derivatives']:
            print(f"\n  Derivatives:")
            for d in r['derivatives']:
                print(f"    dEffDim={d['d_eff_dim']:+.1f}  dPosR²={d['d_pos_r2']:+.6f}  "
                      f"dConR²={d['d_cont_r2']:+.4f}  dActive={d['d_active_frac']:+.4f}")
    _json_or(args, result, _fmt)


def _cmd_cb_invertibility(args: argparse.Namespace) -> None:
    """Substrate invertibility — how far back can it reconstruct?"""
    from .profile.compare import causal_bank_invertibility

    result = causal_bank_invertibility(args.mri, max_lookback=args.max_lookback)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Invertibility: {result['model']} — {result['n_modes']} modes ===\n")
    print(f"  Memory type: {result['memory_type']}")
    print(f"  Memory horizon: {result['memory_horizon']} positions")
    print(f"  Peak R²: {result['peak_r2']:.4f}  Plateau R²: {result['plateau_r2']:.4f}\n")

    print(f"  {'Lookback':>10} {'R²':>8} {'Acc':>8}")
    for lb in result['lookbacks']:
        bar = '#' * int(max(0, lb['r2']) * 40)
        print(f"  T-{lb['lookback']:>7} {lb['r2']:>8.4f} {lb['accuracy']:>8.4f} {bar}")


def _cmd_cb_rotation_probe(args: argparse.Namespace) -> None:
    """Nonlinear + rotational information probes."""
    from .profile.compare import causal_bank_rotation_probe

    result = causal_bank_rotation_probe(args.mri, n_sample=getattr(args, 'n_sample', None))
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== CB Rotation Probe: {result['model']} — "
          f"{result['n_modes']} modes, {result['n_seqs']} seqs ===\n")

    p = result['probes']
    print(f"  Position probes (test R²):")
    print(f"    Linear (PCA50):    {p['position']['linear_pca50']:.6f}")
    print(f"    MLP (PCA50):       {p['position']['mlp_pca50']:.6f}")
    print(f"    MLP (raw256):      {p['position']['mlp_raw256']:.6f}")
    print(f"    Angle (linear):    {p['position']['angle_linear']:.6f}")
    print(f"    Angle (MLP):       {p['position']['angle_mlp']:.6f}")

    boost = p['position']['mlp_pca50'] - p['position']['linear_pca50']
    print(f"    MLP boost:         {boost:+.6f} {'*** NONLINEAR SIGNAL' if boost > 0.01 else ''}")

    print(f"\n  Loss probes (test R²):")
    print(f"    Linear (PCA50):    {p['loss']['linear_pca50']:.6f}")
    print(f"    MLP (PCA50):       {p['loss']['mlp_pca50']:.6f}")
    loss_boost = p['loss']['mlp_pca50'] - p['loss']['linear_pca50']
    print(f"    MLP boost:         {loss_boost:+.6f}")

    a = result['angular']
    print(f"\n  Angular analysis ({a['n_pairs_analyzed']} mode pairs):")
    print(f"    Max pair-position |r|: {a['max_pair_position_r']:.4f}")
    print(f"    Mean pair-position |r|: {a['mean_pair_position_r']:.4f}")
    if a['velocity_difficulty_corr'] is not None:
        print(f"    Velocity-difficulty r:  {a['velocity_difficulty_corr']:.4f}")

    print(f"\n  Phase velocity by position:")
    for v in a['velocity_by_position']:
        print(f"    {v['range']:>12}: |v|={v['mean_abs_velocity']:.4f}  std={v['velocity_std']:.4f}")

    print(f"\n  Top position-correlated pairs:")
    for tp in a['top_position_pairs']:
        print(f"    modes {tp['modes']}: r={tp['position_r']:.4f}")


def _cmd_cb_gate_forensics(args: argparse.Namespace) -> None:
    """Causal bank write gate forensics."""
    from .profile.compare import causal_bank_gate_forensics

    result = causal_bank_gate_forensics(args.mri)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== CB Gate Forensics: {result['model']} ({result['gate_type']}) — "
          f"{result['n_gate_dims']} dims, {result['n_seqs']} seqs ===\n")

    print(f"  VERDICT: {result['verdict']}\n")

    print(f"  Position correlation (mean gate vs position): {result['position_correlation']:.4f}")
    if result['difficulty_correlation'] is not None:
        print(f"  Difficulty correlation (gate vs embed norm): {result['difficulty_correlation']:.4f}")
    if result['loss_correlation'] is not None:
        print(f"  Loss correlation (gate vs loss):             {result['loss_correlation']:.4f}")

    print(f"\n  Effective rank: {result['effective_rank']:.1f}")
    rk = result['effective_rank_pcs']
    print(f"    PCs for 50%={rk['pcs_for_50']}, 80%={rk['pcs_for_80']}, 95%={rk['pcs_for_95']}")
    print(f"  Entropy ratio: {result['entropy_ratio']:.4f} "
          f"(mean={result['mean_mode_entropy']:.4f}, max={result['max_possible_entropy']:.4f})")

    print(f"\n  Gate activation by position:")
    print(f"  {'Position':>12} {'Mean':>8} {'Std':>8} {'Overwrite%':>11}")
    for r in result['by_position']:
        print(f"  {r['range']:>12} {r['mean_gate']:>8.4f} {r['std_gate']:>8.4f} "
              f"{r['overwrite_fraction'] * 100:>10.1f}%")

    if result['gate_by_half_life']:
        print(f"\n  Gate by half-life quartile:")
        for q in result['gate_by_half_life']:
            print(f"    Q{q['quartile']} ({q['hl_range']}, {q['n_modes']}m): "
                  f"mean={q['mean_gate']:.4f}, overwrite={q['overwrite_fraction']:.1%}, "
                  f"|pos_r|={q['mean_pos_r']:.4f}")

    print(f"\n  Most position-dependent modes:")
    for m in result['position_dependent_modes']:
        hl = f"hl={m['half_life']:.1f}" if 'half_life' in m else ""
        print(f"    mode {m['mode']:>3}: r={m['position_r']:+.4f} {hl}")

    if result.get('extra_gates'):
        print(f"\n  Extra gates:")
        for gname, ginfo in result['extra_gates'].items():
            print(f"    {gname}: mean={ginfo['mean']:.4f}, position_r={ginfo['position_correlation']:.4f}")

    print(f"\n  Gate PCA (top 10 variance %): {result['top_gate_variance_pct']}")


def _cmd_cb_substrate_local(args: argparse.Namespace) -> None:
    """Causal bank substrate vs local balance."""
    from .profile.compare import causal_bank_substrate_local

    result = causal_bank_substrate_local(args.mri)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== CB Substrate vs Local: {result['model']} — "
          f"{result['n_seqs']} seqs x {result['seq_len']} positions ===\n")
    print(f"  Has local path: {result['has_local']}")
    if result['crossover_position'] is not None:
        print(f"  Crossover position: {result['crossover_position']}")

    header = f"  {'Position':>12} {'Sub L2':>10}"
    if result['has_local']:
        header += f" {'Local L2':>10} {'Ratio':>8}"
    print(header)
    for r in result['by_position']:
        line = f"  {r['range']:>12} {r['substrate_l2']:>10.4f}"
        if 'local_l2' in r:
            line += f" {r['local_l2']:>10.4f} {r['substrate_local_ratio']:>7.1f}x"
        print(line)


def _cmd_tokenizer_difficulty(args: argparse.Namespace) -> None:
    """Per-token difficulty from embedding norms."""
    from .profile.compare import tokenizer_difficulty

    result = tokenizer_difficulty(args.mri)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Tokenizer Difficulty: {result['model']} — "
          f"{result['n_tokens']} tokens, {result['embed_dim']}d ===\n")
    print(f"  Effective dim: {result['effective_dim']}")
    if 'embed_substrate_r' in result:
        print(f"  Embed-substrate correlation: r={result['embed_substrate_r']:.4f}")
    print(f"  Embed norm range: [{result['embed_norm_range'][0]:.4f}, {result['embed_norm_range'][1]:.4f}]")
    if result['near_duplicates'] >= 0:
        print(f"  Near-duplicate pairs (cos>0.9): {result['near_duplicates']}")
    print(f"\n  Difficulty quartiles:")
    for q in result['difficulty_quartiles']:
        print(f"    {q['label']:>12}: {q['n_tokens']:>5} tokens, mean_norm={q['mean_norm']:.4f}")


def _cmd_tokenizer_compare(args: argparse.Namespace) -> None:
    """Compare sentencepiece tokenizers."""
    from .profile.compare import tokenizer_compare

    sample_text = None
    if args.text:
        from pathlib import Path
        sample_text = Path(args.text).read_text()

    result = tokenizer_compare(args.tokenizers, sample_text=sample_text)

    print(f"\n=== Tokenizer Comparison ===\n")
    for t in result['tokenizers']:
        print(f"  {t['path']}:")
        print(f"    Vocab: {t['vocab_size']}, byte tokens: {t['byte_tokens']}, "
              f"byte fallback: {t['byte_fallback_pct']:.2f}%")
        print(f"    Mean token bytes: {t['mean_token_bytes']}")
        if t['bytes_per_token'] is not None:
            print(f"    Compression: {t['bytes_per_token']:.2f} B/tok, "
                  f"{t['tokens_per_byte']:.4f} tok/B")
        lens = t['length_distribution']
        print(f"    Length dist: " +
              ' '.join(f"{k}={v}" for k, v in lens.items() if v > 0))

    if result['overlap']:
        print(f"\n  Overlap:")
        for key, val in result['overlap'].items():
            print(f"    {key}: {val['common']} common, jaccard={val['jaccard']:.4f}")


def _cmd_cb_causality(args: argparse.Namespace) -> None:
    """Causality verification for causal bank models."""
    from .backend.protocol import load_backend
    from .profile.compare import causal_bank_causality

    backend_kwargs = {}
    if getattr(args, 'result_json', None):
        backend_kwargs['result_json'] = args.result_json
    if getattr(args, 'tokenizer_path', None):
        backend_kwargs['tokenizer_path'] = args.tokenizer_path
    backend = load_backend(args.model, backend="decepticon", **backend_kwargs)
    result = causal_bank_causality(backend, seq_len=args.seq_len, n_tests=args.n_tests)

    print(f"\n=== Causality Check ===\n")
    print(f"  {result['verdict']}")
    print(f"  Tested {result['n_tests']} positions in seq_len={result['seq_len']}")
    if result['violations']:
        print(f"\n  Violations:")
        for v in result['violations']:
            print(f"    Position {v['position']}: max_diff={v['max_logit_diff']:.6f}")


def _cmd_cb_reproduce(args: argparse.Namespace) -> None:
    """Determinism check for causal bank models."""
    from .backend.protocol import load_backend
    from .profile.compare import causal_bank_reproduce

    backend_kwargs = {}
    if getattr(args, 'result_json', None):
        backend_kwargs['result_json'] = args.result_json
    if getattr(args, 'tokenizer_path', None):
        backend_kwargs['tokenizer_path'] = args.tokenizer_path
    backend = load_backend(args.model, backend="decepticon", **backend_kwargs)
    result = causal_bank_reproduce(backend, seq_len=args.seq_len)

    print(f"\n=== Reproducibility Check ===\n")
    print(f"  {result['verdict']}")
    print(f"  max_diff={result['max_diff']}, mean_diff={result['mean_diff']}")


def _cmd_cb_effective_context(args: argparse.Namespace) -> None:
    """Per-position bpb curve; identify the effective-context knee."""
    from .profile.compare import _cb_effective_context

    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]
    result = _cb_effective_context(
        model_path=args.model,
        val=args.val,
        seqlen=args.seqlen,
        n_trials=args.n_trials,
        buckets=buckets,
        knee_threshold=args.knee_threshold,
        result_json=getattr(args, "result_json", None),
        tokenizer_path=getattr(args, "tokenizer_path", None),
    )

    def _fmt(r: dict) -> None:
        print(f"\n=== Effective-context for {r['model']} ===\n")
        print(f"  val_data: {r['val_data']}")
        print(f"  n_trials={r['n_trials']}  seqlen={r['seqlen']}  "
              f"threshold={r['knee_threshold']}\n")
        print(f"  {'bucket':<16} {'n':>6}  {'bpb_mean':>8}  {'bpb_sem':>8}")
        for b in r["buckets"]:
            label = f"[{b['min']},{b['max']})"
            print(f"  {label:<16} {b['n']:>6}  {b['bpb_mean']:>8.4f}  "
                  f"{b['bpb_sem']:>8.4f}")
        print()
        if r["knee_bucket_min"] is None:
            print(f"  knee: NOT DETECTED (curve still decreasing)")
        else:
            print(f"  knee: [{r['knee_bucket_min']},{r['knee_bucket_max']})")
        print(f"  saturation_bpb: {r['saturation_bpb']:.4f}\n")

    _json_or(args, result, _fmt)


def _cmd_cb_ablations(args: argparse.Namespace) -> None:
    """Ablation forensics: substrate/local/truncate path contributions to bpb."""
    from .profile.compare import _cb_ablations

    result = _cb_ablations(
        model_path=args.model,
        ablate=args.ablate,
        val=args.val,
        n_tokens=args.n_tokens,
        result_json=getattr(args, "result_json", None),
        tokenizer_path=getattr(args, "tokenizer_path", None),
    )

    def _fmt(r: dict) -> None:
        print(f"\n=== Ablation forensics for {r['model']} ===\n")
        print(f"  ablation: {r['ablation']}")
        print(f"  n_tokens: {r['n_tokens']}\n")
        print(f"  baseline:   {r['baseline_bpb']:.4f} bpb")
        print(f"  ablated:    {r['ablated_bpb']:.4f} bpb")
        sign = "+" if r["delta_bpb"] >= 0 else ""
        print(f"  delta:     {sign}{r['delta_bpb']:.4f} bpb  "
              f"({r['multiplier']:.3f}×)\n")

    _json_or(args, result, _fmt)


def _cmd_lookup_fraction(args: argparse.Namespace) -> None:
    """How much is table lookup vs computation?"""
    from .profile.compare import lookup_fraction

    result = lookup_fraction(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    def _fmt(r):
        lf = r['lookup_fraction']
        print(f"\n=== Lookup Fraction: {r['model']} ({r['mode']}) — "
              f"{r['n_tokens']} tokens ===\n")
        print(f"  Lookup-solvable: {r['lookup_solvable']} ({lf:.1%})")
        print(f"  Compute-needed:  {r['compute_needed']} ({1-lf:.1%})")

        print(f"\n  By script:")
        print(f"  {'script':<12} {'n':>6} {'lookup':>7} {'fraction':>9}")
        print(f"  {'-'*36}")
        for s, v in sorted(r['by_script'].items(), key=lambda x: -x[1]['fraction']):
            print(f"  {s:<12} {v['n']:>6} {v['lookup']:>7} {v['fraction']:>8.1%}")

        print(f"\n  By layer (fraction still matching embedding prediction):")
        print(f"  {'layer':>5} {'match%':>7}")
        print(f"  {'-'*14}")
        for lr in r['by_layer']:
            print(f"  L{lr['layer']:>3} {lr['fraction']:>6.1%}")
    _json_or(args, result, _fmt, analysis_name="lookup_fraction")


def _cmd_distribution_drift(args: argparse.Namespace) -> None:
    """Distribution drift per layer."""
    from .profile.compare import distribution_drift

    result = distribution_drift(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    def _fmt(r):
        print(f"\n=== Distribution Drift: {r['model']} ({r['mode']}) — {r['n_tokens']} tokens ===\n")
        print(f"  {'layer':>5} {'top1Δ':>6} {'KL':>10} {'TVD':>7} {'entropy':>8}")
        print(f"  {'-'*40}")
        for lr in r['layers']:
            print(f"  L{lr['layer']:>3} {lr['top1_changed']:>5.1%} {lr['mean_kl']:>10.6f} "
                  f"{lr['mean_tvd']:>7.4f} {lr['mean_entropy']:>8.2f}")
    _json_or(args, result, _fmt, analysis_name="distribution_drift")


def _cmd_retrieval_horizon(args: argparse.Namespace) -> None:
    """Retrieval horizon per layer."""
    from .profile.compare import retrieval_horizon

    result = retrieval_horizon(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    def _fmt(r):
        print(f"\n=== Retrieval Horizon: {r['model']} ({r['mode']}) — "
              f"{r['n_tokens']} tokens, {r['n_heads']} heads, "
              f"seq_len={r['seq_len']}, token_pos={r['token_pos']} ===\n")
        print(f"  {'layer':>5} {'dist':>6} {'self%':>6} {'head peaks'}")
        print(f"  {'-'*50}")
        for lr in r['layers']:
            peaks = ', '.join(str(p) for p in lr['head_peak_positions'])
            print(f"  L{lr['layer']:>3} {lr['mean_retrieval_distance']:>+5.1f} "
                  f"{lr['self_attention']:>5.1%} [{peaks}]")
    _json_or(args, result, _fmt, analysis_name="retrieval_horizon")


def _cmd_layer_opposition(args: argparse.Namespace) -> None:
    """MLP vs attention opposition per layer."""
    from .profile.compare import layer_opposition

    result = layer_opposition(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    def _fmt(r):
        print(f"\n=== Layer Opposition: {r['model']} ({r['mode']}) — {r['n_tokens']} tokens ===\n")
        print(f"  {'layer':>5} {'cos(M,A)':>9} {'mlp':>8} {'attn':>8} {'delta':>8} {'cancel':>7} {'rel_err':>8}")
        print(f"  {'-'*57}")
        for lr in r['layers']:
            print(f"  L{lr['layer']:>3} {lr['cos_mlp_attn']:>+8.4f} {lr['mlp_norm']:>8.2f} "
                  f"{lr['attn_norm']:>8.2f} {lr['delta_norm']:>8.2f} {lr['cancellation']:>6.0%} "
                  f"{lr['relative_error']:>7.1%}")
    _json_or(args, result, _fmt, analysis_name="layer_opposition")


def _cmd_cross_model(args: argparse.Namespace) -> None:
    """Compare models on shared vocabulary."""
    from .profile.compare import cross_model

    result = cross_model(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    def _fmt(r):
        print(f"\n=== Cross-Model Comparison: {r['n_shared']} shared tokens ===\n")

        print(f"  Per-model stats:")
        for m in r['models']:
            grad = f"  grad={m.get('mean_grad', '?')}" if 'mean_grad' in m else ""
            print(f"    {m['name']:<15} disp={m['mean_disp']:>8.1f} ±{m['std_disp']:<6.1f}{grad}")

        print(f"\n  Pairwise comparisons:")
        print(f"  {'A':<15} {'B':<15} {'disp_rho':>9} {'grad_rho':>9} {'overlap':>8}")
        print(f"  {'-'*58}")
        for c in r['comparisons']:
            grad_r = f"{c.get('gradient_rho', ''):>9}" if 'gradient_rho' in c else "        —"
            print(f"  {c['model_a']:<15} {c['model_b']:<15} {c['displacement_rho']:>+8.4f} "
                  f"{grad_r} {c['top_overlap']:>3}/{c['top_n']}")

        print(f"\n  Top shared sharts (highest mean displacement):")
        if r['shared_sharts']:
            models = r['models']
            header = "  " + f"{'#':>3} {'token':<20}"
            for m in models:
                header += f" {m['name']:>10}"
            print(header)
            print(f"  {'-'*(25 + 11*len(models))}")
            for s in r['shared_sharts'][:15]:
                line = f"  {s['rank']:>3} {s['token']:<20}"
                for m in models:
                    key = f"disp_{m['name']}"
                    line += f" {s.get(key, 0):>10.1f}"
                print(line)
    _json_or(args, result, _fmt, analysis_name="cross_model")


def _cmd_mri_regrad(args: argparse.Namespace) -> None:
    """Recompute embedding gradients for existing MRIs."""
    import time as _time
    from pathlib import Path
    from .backend.protocol import load_backend
    from .profile.mri import _framework_ops, load_mri
    import numpy as np

    backend_name = 'auto'
    if args.model.endswith('.checkpoint.pt'):
        backend_name = 'decepticon'
    backend = load_backend(args.model, backend=backend_name)
    ops = _framework_ops(backend)
    cfg = backend.config
    hidden = cfg.hidden_size

    for mri_path in args.mri:
        p = Path(mri_path)
        if not p.is_dir():
            print(f"  SKIP: {mri_path} not a directory")
            continue

        m = load_mri(mri_path)
        meta = m['metadata']
        n_tok = meta['capture']['n_tokens']
        mode = meta['capture']['mode']
        token_pos = meta['capture'].get('token_pos', 0)
        token_ids = m['token_ids']
        T_seq = meta['capture'].get('seq_len', 1)

        print(f"\n  Recomputing gradients: {p.name} ({mode}, {n_tok} tokens)")

        # Build mask
        mask = ops.triu_mask(T_seq) if T_seq > 1 else None

        # Build input sequences (same logic as capture)
        if mode == "template":
            from .profile.shrt import _extract_template_parts
            prefix_ids, suffix_ids = _extract_template_parts(backend.tokenizer)
        else:
            prefix_ids, suffix_ids = [], []

        batch_size = 32
        emb_grad = np.zeros((n_tok, hidden), dtype=np.float16)
        t0 = _time.time()

        # Rebuild token sample (same order as original capture)
        vocab_size = backend.tokenizer.vocab_size
        real_tokens = []
        seen = set()
        for tid in range(vocab_size):
            tok = backend.tokenizer.decode([tid], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
            if tok.strip() and tok not in seen:
                seen.add(tok)
                real_tokens.append((tid, tok))

        n_index = meta.get('provenance', {}).get('n_index')
        if n_index and n_index < len(real_tokens):
            import numpy as _np
            rng = _np.random.RandomState(meta.get('provenance', {}).get('seed', 42))
            idx = rng.choice(len(real_tokens), n_index, replace=False)
            sample = [real_tokens[i] for i in sorted(idx)]
        else:
            sample = real_tokens

        # Sanity check: rebuilt sample must match stored token IDs
        rebuilt_ids = np.array([tid for tid, _ in sample[:n_tok]])
        stored_ids = np.array(token_ids[:n_tok])
        if not np.array_equal(rebuilt_ids, stored_ids):
            n_mismatch = int(np.sum(rebuilt_ids != stored_ids))
            print(f"  ERROR: rebuilt token IDs differ from stored ({n_mismatch}/{n_tok} mismatch)")
            print(f"    first 5 rebuilt: {rebuilt_ids[:5].tolist()}")
            print(f"    first 5 stored:  {stored_ids[:5].tolist()}")
            print(f"    SKIPPING — tokenizer or _detect_script may have changed since capture")
            continue

        for batch_start in range(0, n_tok, batch_size):
            batch_end = min(batch_start + batch_size, n_tok)
            batch = sample[batch_start:batch_end]

            if mode in ("raw", "naked"):
                inp = ops.array([[tid] for tid, _ in batch])
            else:
                inp = ops.array([prefix_ids + [tid] + suffix_ids for tid, _ in batch])

            emb = ops.embed(inp)
            eg = ops.embedding_grad(emb, mask)
            if len(eg.shape) == 3:
                emb_grad[batch_start:batch_end] = eg[:, token_pos, :].astype(np.float16)
            else:
                emb_grad[batch_start:batch_end] = eg.astype(np.float16)

            if (batch_end) % 1000 < batch_size or batch_end == n_tok:
                elapsed = _time.time() - t0
                rate = batch_end / max(elapsed, 0.01)
                print(f"    {batch_end}/{n_tok} ({rate:.0f} tok/s)", end="\r")

        np.save(p / "embedding_grad.npy", emb_grad)
        elapsed = _time.time() - t0
        print(f"    {n_tok} tokens, {elapsed:.0f}s — saved")


def _cmd_shart_anatomy(args: argparse.Namespace) -> None:
    """What makes a shart a shart?"""
    from .profile.compare import shart_anatomy

    result = shart_anatomy(args.mri, n_sample=args.n_sample, top_n=args.top_n)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    def _fmt(r):
        d = r['displacement']
        c = r['crystal']
        g = r['gradient']
        f = r['frozen_zone']
        b = r['bandwidth']

        print(f"\n=== Shart Anatomy: {r['model']} ({r['mode']}) — {r['n_tokens']} tokens ===\n")

        print(f"  Displacement: [{d['min']}, {d['max']}], median={d['median']}, ratio={d['ratio_max_median']}x")
        print(f"  Crystal: L{c['layer']} neuron {c['neuron']} ({c['amplification']}x amplification)")
        print(f"    r(crystal_activation, displacement) = {c['corr_with_displacement']}")
        print(f"    Energy in crystal neuron: {c['energy_fraction']:.0%}")
        if g['corr_with_displacement'] is not None:
            print(f"  Gradient: r(embedding_grad, displacement) = {g['corr_with_displacement']:.4f}")
            print(f"    Top sharts mean grad: {g['top_sharts_mean']}  Bottom mean: {g['bottom_mean']}")
        print(f"  Frozen zone: gate cosine = {f['gate_cosine']} ({f['interpretation']})")
        print(f"  Bandwidth: r(active_neurons, displacement) = {b['active_neuron_disp_corr']} ({b['interpretation']})")

        print(f"\n  Top {args.top_n} sharts:")
        print(f"  {'#':>3} {'disp':>8} {'grad':>7} {'crystal':>8} {'active':>6} {'script':<8} token")
        print(f"  {'-'*55}")
        for t in r['top_sharts'][:args.top_n]:
            grad = f"{t.get('grad_norm', 0):>7.1f}" if 'grad_norm' in t else "     —"
            crys = f"{t.get('crystal_activation', 0):>8.0f}" if 'crystal_activation' in t else "      —"
            act = f"{t.get('active_neurons', 0):>6}" if 'active_neurons' in t else "    —"
            print(f"  {t['rank']:>3} {t['displacement']:>8.0f} {grad} {crys} {act} {t['script']:<8} {t['token']}")

        print(f"\n  Bottom {args.top_n}:")
        print(f"  {'#':>3} {'disp':>8} {'grad':>7} {'script':<8} token")
        print(f"  {'-'*40}")
        for t in r['bottom_tokens'][:args.top_n]:
            grad = f"{t.get('grad_norm', 0):>7.1f}" if 'grad_norm' in t else "     —"
            print(f"  {t['rank']:>3} {t['displacement']:>8.0f} {grad} {t['script']:<8} {t['token']}")
    _json_or(args, result, _fmt, analysis_name="shart_anatomy")


def _cmd_bandwidth(args: argparse.Namespace) -> None:
    """Bandwidth efficiency per layer."""
    from .profile.compare import bandwidth_efficiency

    result = bandwidth_efficiency(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    def _fmt(r):
        eff = r['bandwidth_efficiency']
        print(f"\n=== Bandwidth Efficiency: {r['model']} ({r['mode']}) — "
              f"{r['n_tokens']} tokens ===\n")
        print(f"  Model size:   {r['total_model_bytes']/1e6:.1f} MB")
        print(f"  Active bytes: {r['total_active_bytes']/1e6:.1f} MB ({eff:.1%})")
        print(f"  Wasted:       {r['wasted_fraction']:.1%}")
        print(f"  MLP neurons:  {r['gate_k']} captured / {r['intermediate_size']} total")

        print(f"\n  {'layer':>5} {'skip%':>6} {'MLP active':>11} {'efficiency':>10}")
        print(f"  {'-'*35}")
        for lr in r['layers']:
            print(f"  L{lr['layer']:>3} {lr['skip_fraction']:>5.0%} "
                  f"{lr['mlp_active_neurons']:>5.0f}/{lr['mlp_total_neurons']:<5} "
                  f"{lr['efficiency']:>9.1%}")
    _json_or(args, result, _fmt, analysis_name="bandwidth")


def _cmd_code_anatomy(args: argparse.Namespace) -> None:
    """Code token subcategory displacement and layer trajectories."""
    from .profile.compare import code_anatomy
    result = code_anatomy(args.shrt, args.frt,
                          all_layers_shrt_path=getattr(args, 'all_layers', None))
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Code Anatomy: {result['model']} ({result['n_code_measured']} measured / {result['n_code_vocab']} in vocab) ===")
    print(f"  Grand mean delta: {result['grand_mean_delta']}")

    print(f"\n  Code subcategories:")
    print(f"  {'category':<14} {'measured':>8} {'vocab':>6} {'cover%':>7} {'mean_d':>8} {'relative':>9} {'examples'}")
    print(f"  {'-'*80}")
    for cat, data in result['code_subcategories'].items():
        examples = ' '.join(data['examples'][:4])
        mean_d = f"{data['mean_delta']:>8.2f}" if data.get('mean_delta') is not None else f"{'---':>8}"
        rel = f"{data['relative']:>9.3f}" if data.get('relative') is not None else f"{'---':>9}"
        print(f"  {cat:<14} {data['n_measured']:>8} {data['n_vocab']:>6} {data['coverage_pct']:>6.1f}% {mean_d} {rel} {examples}")

    print(f"\n  Other scripts:")
    print(f"  {'script':<14} {'n':>5} {'mean_d':>8} {'relative':>9}")
    print(f"  {'-'*40}")
    for s, data in result['context_scripts'].items():
        print(f"  {s:<14} {data['n']:>5} {data['mean_delta']:>8.2f} {data['relative']:>9.3f}")

    if 'code_trajectories' in result:
        print(f"\n  Layer trajectories (code):")
        print(f"  {'category':<14} {'n':>5} {'early':>6} {'mid':>6} {'late':>6} {'range':>6} {'late<early'}")
        print(f"  {'-'*60}")
        for cat, data in result['code_trajectories'].items():
            falls = "yes" if data['falls'] else "no"
            print(f"  {cat:<14} {data['n_tokens']:>5} {data['early']:>6.3f} {data['mid']:>6.3f} "
                  f"{data['late']:>6.3f} {data['range']:>6.3f} {falls}")

        if 'script_trajectories' in result:
            print(f"\n  Layer trajectories (other scripts):")
            print(f"  {'script':<14} {'n':>5} {'early':>6} {'late':>6} {'range':>6} {'late<early'}")
            print(f"  {'-'*50}")
            for s, data in result['script_trajectories'].items():
                falls = "yes" if data['falls'] else "no"
                print(f"  {s:<14} {data['n_tokens']:>5} {data['early']:>6.3f} "
                      f"{data['late']:>6.3f} {data['range']:>6.3f} {falls}")


def _cmd_decompose(args: argparse.Namespace) -> None:
    """Decompose displacement into attention and MLP fractions per layer per script."""
    from .backend.protocol import load_backend
    from .cartography.runtime import forward_pass
    from .profile.shrt import _extract_clean_baseline, _extract_template_parts
    from .profile.frt import load_frt, _detect_script
    import numpy as np
    import json
    from collections import defaultdict

    backend = load_backend(args.model)
    cfg = backend.config

    if args.layers == 'all':
        layers = list(range(cfg.n_layers))
    else:
        layers = [int(x) for x in args.layers.split(',')]

    frt = load_frt(args.frt)
    fl = {int(frt['token_ids'][i]): str(frt['scripts'][i])
          for i in range(len(frt['token_ids']))}

    baseline = _extract_clean_baseline(backend.tokenizer)
    prefix_ids, suffix_ids = _extract_template_parts(backend.tokenizer)

    # Build token sample (same logic as .shrt)
    real_tokens = []
    seen_texts = set()
    for tid in range(backend.tokenizer.vocab_size):
        tok = backend.tokenizer.decode([tid])
        if tok.strip() and len(tok) > 0 and not tok.startswith('[control') and not tok.startswith('<'):
            if tok not in seen_texts:
                seen_texts.add(tok)
                real_tokens.append((tid, tok))

    rng = np.random.RandomState(42)
    n_sample = min(args.n_index, len(real_tokens))
    sample_idx = set(rng.choice(len(real_tokens), n_sample, replace=False))

    # Small-script exhaustion
    script_pools = defaultdict(list)
    for i, (tid, tok) in enumerate(real_tokens):
        script_pools[_detect_script(tok)].append(i)
    for script, indices in script_pools.items():
        if 0 < len(indices) < 100:
            sample_idx.update(indices)

    sample = [real_tokens[i] for i in sorted(sample_idx)]
    print(f"Decomposing {len(sample)} tokens x {len(layers)} layers...")

    # For each layer: get baseline residuals under each ablation mode
    # Pre-compute baselines per layer per mode
    layer_baselines = {}
    for layer in layers:
        full = forward_pass(backend.model, backend.tokenizer, baseline,
                           return_residual=True, residual_layer=layer)
        skip = forward_pass(backend.model, backend.tokenizer, baseline,
                           ablate_layers={layer}, ablate_mode="zero",
                           return_residual=True, residual_layer=layer)
        zero_attn = forward_pass(backend.model, backend.tokenizer, baseline,
                                ablate_layers={layer}, ablate_mode="zero_attn",
                                return_residual=True, residual_layer=layer)
        zero_mlp = forward_pass(backend.model, backend.tokenizer, baseline,
                               ablate_layers={layer}, ablate_mode="zero_mlp",
                               return_residual=True, residual_layer=layer)
        layer_baselines[layer] = {
            "full": full["residual"],
            "skip": skip["residual"],
            "zero_attn": zero_attn["residual"],
            "zero_mlp": zero_mlp["residual"],
        }

    # Per-token, per-layer decomposition
    # Store: attn_frac[layer][script] = list of fractions
    attn_fracs = {l: defaultdict(list) for l in layers}
    mlp_fracs = {l: defaultdict(list) for l in layers}
    attn_norms = {l: defaultdict(list) for l in layers}
    mlp_norms = {l: defaultdict(list) for l in layers}

    for tid, tok in sample:
        input_ids = prefix_ids + [tid] + suffix_ids
        script = fl.get(tid, 'unknown')
        if script in ('special', 'unknown'):
            continue

        for layer in layers:
            try:
                bl = layer_baselines[layer]

                full = forward_pass(backend.model, backend.tokenizer, "",
                                   token_ids=input_ids,
                                   return_residual=True, residual_layer=layer)
                zero_a = forward_pass(backend.model, backend.tokenizer, "",
                                    token_ids=input_ids,
                                    ablate_layers={layer}, ablate_mode="zero_attn",
                                    return_residual=True, residual_layer=layer)
                zero_m = forward_pass(backend.model, backend.tokenizer, "",
                                    token_ids=input_ids,
                                    ablate_layers={layer}, ablate_mode="zero_mlp",
                                    return_residual=True, residual_layer=layer)
                skip = forward_pass(backend.model, backend.tokenizer, "",
                                   token_ids=input_ids,
                                   ablate_layers={layer}, ablate_mode="zero",
                                   return_residual=True, residual_layer=layer)

                # Token's contribution at this layer
                full_delta = full["residual"] - bl["full"]
                attn_delta = zero_m["residual"] - bl["zero_mlp"]
                mlp_delta = zero_a["residual"] - bl["zero_attn"]

                full_norm = float(np.linalg.norm(full_delta))
                attn_norm = float(np.linalg.norm(attn_delta))
                mlp_norm = float(np.linalg.norm(mlp_delta))

                total = attn_norm + mlp_norm
                if total > 1e-8:
                    attn_fracs[layer][script].append(attn_norm / total)
                    mlp_fracs[layer][script].append(mlp_norm / total)
                    attn_norms[layer][script].append(attn_norm)
                    mlp_norms[layer][script].append(mlp_norm)
            except Exception:
                pass

    # Report
    all_scripts = set()
    for l in layers:
        all_scripts.update(attn_fracs[l].keys())
    all_scripts = sorted(s for s in all_scripts if any(len(attn_fracs[l].get(s, [])) >= 5 for l in layers))

    print(f"\n=== Attention / MLP Decomposition ({len(sample)} tokens) ===\n")
    header = f"{'layer':>5}"
    for s in all_scripts:
        header += f"  {s[:6]:>7}"
    print(header + "  (values = MLP fraction)")

    for layer in layers:
        row = f"L{layer:>3}"
        for s in all_scripts:
            vals = mlp_fracs[layer].get(s, [])
            if len(vals) >= 5:
                row += f"  {np.mean(vals):>7.1%}"
            else:
                row += f"  {'---':>7}"
        print(row)

    # Absolute norms: where the magnitude lives
    print(f"\n{'layer':>5}", end='')
    for s in all_scripts:
        print(f"  {s[:6]:>7}", end='')
    print("  (values = mean |attn| / mean |mlp|)")

    for layer in layers:
        row = f"L{layer:>3}"
        for s in all_scripts:
            a_vals = attn_norms[layer].get(s, [])
            m_vals = mlp_norms[layer].get(s, [])
            if len(a_vals) >= 5:
                row += f"  {np.mean(a_vals):>3.1f}/{np.mean(m_vals):>3.1f}"
            else:
                row += f"  {'---':>7}"
        print(row)

    # Save if output specified
    if args.output:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save_data = {"layers": np.array(layers)}
        for s in all_scripts:
            attn_arr = np.array([[np.mean(attn_fracs[l].get(s, [0])) for l in layers]])
            mlp_arr = np.array([[np.mean(mlp_fracs[l].get(s, [0])) for l in layers]])
            save_data[f"attn_{s}"] = attn_arr.astype(np.float32)
            save_data[f"mlp_{s}"] = mlp_arr.astype(np.float32)
        np.savez_compressed(args.output, **save_data)
        print(f"\n  Saved to {args.output}")


def _cmd_validate_ablation(args: argparse.Namespace) -> None:
    """Validate that zero_attn + zero_mlp = full layer output."""
    from .backend.protocol import load_backend
    import numpy as np

    backend = load_backend(args.model)
    layer = args.layer

    # Use the clean baseline template with a test token
    from .profile.shrt import _extract_clean_baseline
    baseline = _extract_clean_baseline(backend.tokenizer)

    # Full forward
    fwd_full = backend.forward(baseline, return_residual=True, residual_layer=layer)
    h_full = fwd_full.residual

    # Zero attention (MLP only)
    from .cartography.runtime import forward_pass
    result_zero_attn = forward_pass(
        backend.model, backend.tokenizer, baseline,
        ablate_layers={layer}, ablate_mode="zero_attn",
        return_residual=True, residual_layer=layer)
    h_zero_attn = result_zero_attn["residual"]

    # Zero MLP (attention only)
    result_zero_mlp = forward_pass(
        backend.model, backend.tokenizer, baseline,
        ablate_layers={layer}, ablate_mode="zero_mlp",
        return_residual=True, residual_layer=layer)
    h_zero_mlp = result_zero_mlp["residual"]

    # No ablation at this layer — get pre-layer state
    result_skip = forward_pass(
        backend.model, backend.tokenizer, baseline,
        ablate_layers={layer}, ablate_mode="zero",
        return_residual=True, residual_layer=layer)
    h_skip = result_skip["residual"]

    # Decompose
    attn_contrib = h_zero_mlp - h_skip   # what attention added
    mlp_contrib = h_zero_attn - h_skip   # what MLP added
    full_contrib = h_full - h_skip       # what the full layer added
    reconstructed = attn_contrib + mlp_contrib

    error = float(np.linalg.norm(full_contrib - reconstructed))
    full_norm = float(np.linalg.norm(full_contrib))
    rel_error = error / max(full_norm, 1e-8)

    print(f"\n=== Ablation Validation: Layer {layer} ===")
    print(f"  |full_contrib|:    {full_norm:.4f}")
    print(f"  |attn_contrib|:    {float(np.linalg.norm(attn_contrib)):.4f}")
    print(f"  |mlp_contrib|:     {float(np.linalg.norm(mlp_contrib)):.4f}")
    print(f"  |reconstruction error|: {error:.6f}")
    print(f"  relative error:    {rel_error:.6f}")
    if rel_error < 0.01:
        print(f"  PASS: decomposition is valid (error < 1%)")
    elif rel_error < 0.05:
        print(f"  WARN: decomposition has {rel_error*100:.1f}% error")
    else:
        print(f"  FAIL: decomposition error {rel_error*100:.1f}% — ablation modes are broken")


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
    result = layer_scripts(args.shrt, args.frt,
                           safety_shrt_path=getattr(args, 'safety_shrt', None))
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

    if result.get('crossings'):
        print(f"\n  Crossings (where one script overtakes another):")
        for c in result['crossings']:
            s1, s2 = c['scripts']
            print(f"    L{c['cross_layer']:>2}: {s1} crosses {s2} ({s1}:{c['s1_before']:.3f}→{c['s1_after']:.3f}, {s2}:{c['s2_before']:.3f}→{c['s2_after']:.3f})")

    sig = result.get('crossing_significance', {})
    if sig.get('n_crossings', 0) >= 2:
        print(f"\n  Crossing significance:")
        print(f"    {sig['n_crossings']} crossings across {sig['n_script_pairs']} script pairs, {sig['n_layers']} layers")
        print(f"    crossing rate: {sig['crossing_rate']} (fraction of pairs that cross)")
        for ws in [2, 3, 4]:
            key = f"window_{ws}"
            if key in sig:
                w = sig[key]
                print(f"    best {ws}-layer window: L{w['best_window'][0]}-L{w['best_window'][1]}, "
                      f"{w['crossings_in_window']} crossings (expected {w['expected_if_uniform']} if uniform, "
                      f"ratio {w['ratio']}x)")
        if 'by_layer' in sig:
            layers_with = [(l, n) for l, n in sig['by_layer'].items() if n > 0]
            if layers_with:
                print(f"    per-layer: {', '.join(f'L{l}={n}' for l, n in layers_with)}")

        sl = sig.get('safety_layer', {})
        if sl.get('layer') is not None:
            print(f"\n  Safety layer overlap (discovered L{sl['layer']}, accuracy {sl['accuracy']}):")
            print(f"    crossings at safety layer: {sl['crossings_at_layer']}")
            print(f"    crossings in ±1 window {sl['window']}: {sl['crossings_in_window']} "
                  f"(expected {sl['expected_if_uniform']} if uniform, ratio {sl['ratio']}x)")
            if sl['scripts_crossing']:
                for sc in sl['scripts_crossing']:
                    s1, s2 = sc['scripts']
                    print(f"      L{sc['layer']}: {s1} crosses {s2}")
            else:
                print(f"    no crossings at or near the safety layer")
        elif sl.get('status') == 'not discovered':
            print(f"\n  Safety layer: {sl['note']}")


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
    has_dim = any(p.get('dimensionality') for p in result['profiles'])
    if has_dim:
        print(f"{'model':<10} {'hidden':>6} {'sens':>7} {'bl_ent':>7} {'PC1%':>5} {'PCs@50%':>7} {'PCs@80%':>7}")
        for p in sorted(result['profiles'], key=lambda x: -x['sensitivity']):
            d = p.get('dimensionality', {})
            pc1 = f"{d['pc1_pct']:>5.1f}" if d.get('pc1_pct') else f"{'---':>5}"
            p50 = f"{d['pcs_50']:>7}" if d.get('pcs_50') else f"{'---':>7}"
            p80 = f"{d['pcs_80']:>7}" if d.get('pcs_80') else f"{'---':>7}"
            print(f"{p['model']:<10} {p['hidden_size']:>6} {p['sensitivity']:>7.4f} "
                  f"{p['baseline_entropy']:>7.2f} {pc1} {p50} {p80}")
    else:
        print(f"{'model':<10} {'hidden':>6} {'sensitivity':>11} {'baseline_ent':>12} {'top':>8}")
        for p in sorted(result['profiles'], key=lambda x: -x['sensitivity']):
            print(f"{p['model']:<10} {p['hidden_size']:>6} {p['sensitivity']:>11.4f} {p['baseline_entropy']:>12.2f} {p['baseline_top']:>8}")

    if result['concordance']:
        print(f"\n{'script':<12} {'n':>3} {'relative':>8} {'std':>5} {'range':>12} {'status':>10}")
        for sc in sorted(result['concordance'], key=lambda x: -result['concordance'][x]['mean_relative']):
            c = result['concordance'][sc]
            status = "std<0.15" if c['std_below_0.15'] else "std>=0.15"
            rng = f"{c['min']:.2f}-{c['max']:.2f}"
            print(f"{sc:<12} {c['n_models']:>3} {c['mean_relative']:>8.3f} {c['std_relative']:>5.3f} {rng:>12} {status:>10}")

    if result['kendalls_w'] is not None:
        print(f"\nKendall's W: {result['kendalls_w']:.4f}  (1.0=perfect agreement, 0.0=none)")


# === Unified backend helper ===

def _load(args):
    """Load a model backend from args.model. Handles MLX, HF, decepticon."""
    from .backend.protocol import load_backend
    backend_name = getattr(args, 'backend', 'auto')
    if backend_name == 'auto' and args.model.endswith('.checkpoint.pt'):
        backend_name = 'decepticon'
    return load_backend(args.model, backend=backend_name)


def _to_dict(obj):
    """Convert dataclass/namedtuple to dict, numpy arrays to lists."""
    import dataclasses
    import numpy as _np
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = {}
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            d[f.name] = _to_dict(v)
        return d
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.floating, _np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [_to_dict(v) for v in sorted(obj, key=str)]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


# === discover commands ===

def _cmd_discover_directions(args):
    """Find contrastive directions from DB prompts."""
    from .discover.directions import find_direction_suite
    from .core.db import SignalDB

    backend = _load(args)
    db = SignalDB(getattr(args, 'db_path', None))
    prompts = db.get_prompts(limit=99999)
    pos = [p['text'] for p in prompts if not p.get('is_benign', True)][:args.n_sample]
    neg = [p['text'] for p in prompts if p.get('is_benign', True)][:args.n_sample]
    if len(pos) < 10 or len(neg) < 10:
        print("Need ≥10 harmful + ≥10 benign prompts in DB. Run: heinrich eval --prompts simple_safety first.")
        return
    cfg = backend.config
    layer = args.layer or (cfg.n_layers - cfg.n_layers // 4)
    result = find_direction_suite(
        backend.model, backend.tokenizer, pos, neg,
        name=args.name, layers=[layer], backend=backend)
    result_dict = _to_dict(result)
    def _fmt(r):
        for d in r.get('directions', []):
            print(f"  L{d.get('layer', '?')}: accuracy={d.get('accuracy', 0):.1%} "
                  f"effect_size={d.get('effect_size', 0):.2f}")
    _json_or(args, result_dict, _fmt)
    db.close()


def _cmd_discover_neurons(args):
    """Scan MLP neurons at a layer."""
    from .discover.neurons import scan_neurons
    from .core.db import SignalDB

    backend = _load(args)
    db = SignalDB(getattr(args, 'db_path', None))
    prompts = db.get_prompts(limit=99999)
    pos = [p['text'] for p in prompts if not p.get('is_benign', True)][:100]
    neg = [p['text'] for p in prompts if p.get('is_benign', True)][:100]
    result = scan_neurons(
        backend.model, backend.tokenizer, pos, neg,
        args.layer, top_k=args.top_k, backend=backend)
    result_dict = _to_dict(result)
    def _fmt(r):
        for n in r.get('selective_neurons', r.get('neurons', []))[:args.top_k]:
            sel = n.get('selectivity', n.get('max_z', 0))
            print(f"  neuron {n.get('neuron', n.get('index', '?'))}: "
                  f"selectivity={sel:.2f}  "
                  f"pos={n.get('mean_pos_activation', 0):.2f}  "
                  f"neg={n.get('mean_neg_activation', 0):.2f}")
    _json_or(args, result_dict, _fmt)
    db.close()


def _cmd_discover_axes(args):
    """Discover orthogonal behavioral axes."""
    from .cartography.axes import discover_axes

    backend = _load(args)
    cfg = backend.config
    layer = args.layer or (cfg.n_layers - cfg.n_layers // 4)
    result = discover_axes(backend.model, backend.tokenizer, layer=layer)
    result_dict = _to_dict(result)
    def _fmt(r):
        for ax in r.get('axes', []):
            print(f"  {ax.get('name', '?'):>15}: variance={ax.get('variance_explained', 0):.1%}")
    _json_or(args, result_dict, _fmt)


def _cmd_discover_dimensionality(args):
    """Estimate behavioral manifold dimensionality."""
    from .cartography.space import estimate_dimensionality
    from .core.db import SignalDB

    backend = _load(args)
    db = SignalDB(getattr(args, 'db_path', None))
    prompts = db.get_prompts(limit=args.n_prompts)
    texts = [p['text'] for p in prompts]
    cfg = backend.config
    layer = args.layer or (cfg.n_layers - cfg.n_layers // 4)
    result = estimate_dimensionality(backend.model, backend.tokenizer, texts, layer=layer)
    result_dict = _to_dict(result)
    def _fmt(r):
        print(f"  Intrinsic dimensionality: {r.get('intrinsic_dim_estimate', '?')}")
        print(f"  Dims for 90%: {r.get('dims_for_90', '?')}")
        print(f"  Dims for 99%: {r.get('dims_for_99', '?')}")
    _json_or(args, result_dict, _fmt)
    db.close()


# === attack commands ===

def _cmd_attack_cliff(args):
    """Find steering cliff at a layer."""
    from .attack.cliff import find_cliff
    import numpy as _np

    backend = _load(args)
    direction = _np.load(args.direction)
    result = find_cliff(
        backend.model, backend.tokenizer, args.prompt,
        direction, "direction", layer=args.layer, backend=backend)
    result_dict = _to_dict(result)
    def _fmt(r):
        print(f"  Dead zone edge: {r.get('dead_zone_edge', '?')}")
        print(f"  Cliff magnitude: {r.get('cliff_magnitude', '?')}")
        print(f"  Stiffness: {r.get('stiffness', '?')}")
    _json_or(args, result_dict, _fmt)


def _cmd_attack_steer(args):
    """Generate with a direction vector applied."""
    from .attack.steer import generate_steered
    import numpy as _np

    backend = _load(args)
    direction = _np.load(args.direction)
    modifications = {(args.layer, 0): args.alpha}
    result = generate_steered(
        backend.model, backend.tokenizer, args.prompt,
        modifications, args.max_tokens, backend=backend)
    def _fmt(r):
        print(f"  Prompt: {r.get('prompt', '')}")
        print(f"  Generated: {r.get('generated', '')}")
    _json_or(args, result, _fmt)


def _cmd_attack_surface(args):
    """Map vulnerability surface."""
    from .attack.cliff import map_vulnerability_surface
    import numpy as _np

    backend = _load(args)
    dir_path = Path(args.directions)
    directions = {}
    for f in dir_path.glob("*.npy"):
        directions[f.stem] = _np.load(f)
    with open(args.prompts) as fh:
        prompts = [line.strip() for line in fh if line.strip()]
    result = map_vulnerability_surface(
        backend.model, backend.tokenizer, directions, prompts,
        layer=args.layer, backend=backend)
    result_dict = _to_dict(result)
    def _fmt(r):
        print(f"  Directions: {len(r.get('directions', []))}")
        print(f"  Prompts: {len(r.get('prompts', []))}")
        for cp in r.get('cliff_points', [])[:10]:
            print(f"    {cp.get('direction', '?')} × '{cp.get('prompt', '')[:40]}': "
                  f"cliff={cp.get('cliff_magnitude', '?')}")
    _json_or(args, result_dict, _fmt)


# === trace commands ===

def _cmd_trace_causal(args):
    """Causal tracing: layer × position heatmap."""
    from .cartography.trace import causal_trace

    backend = _load(args)
    result = causal_trace(
        backend.model, backend.tokenizer,
        args.clean, args.corrupt, backend=backend)
    result_dict = _to_dict(result)
    def _fmt(r):
        print(f"  Layers: {r.get('n_layers', '?')}, Positions: {r.get('n_positions', '?')}")
        for site in r.get('top_sites', [])[:10]:
            print(f"    L{site.get('layer', '?')} pos={site.get('position', '?')}: "
                  f"recovery={site.get('recovery', 0):.3f}")
    _json_or(args, result_dict, _fmt)


def _cmd_trace_conversation(args):
    """Track safety across multi-turn conversation."""
    from .cartography.conversation import trace_conversation
    import numpy as _np

    backend = _load(args)
    with open(args.turns) as fh:
        turns = json.load(fh)
    direction = _np.load(args.direction) if args.direction else None
    result = trace_conversation(
        backend, turns,
        safety_direction=direction, safety_layer=args.layer)
    result_dict = _to_dict(result)
    def _fmt(r):
        for t in r.get('turns', []):
            print(f"  Turn {t.get('turn', '?')}: refusal={t.get('refusal_prob', 0):.3f} "
                  f"compliance={t.get('compliance_prob', 0):.3f}")
    _json_or(args, result_dict, _fmt)


def _cmd_trace_generation(args):
    """Monitor residual stream during generation."""
    from .cartography.flow import generation_trace
    import numpy as _np

    backend = _load(args)
    directions = None
    if args.directions:
        dir_path = Path(args.directions)
        directions = {f.stem: _np.load(f) for f in dir_path.glob("*.npy")}
    result = generation_trace(
        backend.model, backend.tokenizer, args.prompt,
        max_tokens=args.max_tokens, directions=directions, backend=backend)
    result_dict = _to_dict(result)
    def _fmt(r):
        print(f"  Generated: {r.get('generated_text', '')[:200]}")
        for snap in r.get('snapshots', [])[:5]:
            print(f"    step {snap.get('step', '?')}: entropy={snap.get('entropy', 0):.3f} "
                  f"token={snap.get('token', '?')}")
    _json_or(args, result_dict, _fmt)


# === inspect commands (no model needed) ===

def _cmd_inspect_safetensors(args):
    """Catalog a .safetensors file."""
    from .inspect.tensor import inspect_safetensors_file

    result = inspect_safetensors_file(Path(args.source), name_regex=args.name_regex)
    def _fmt(r):
        print(f"  {r.get('tensor_count', 0)} tensors, {r.get('file_bytes', 0) / 1e6:.1f} MB")
        for t in r.get('tensors', [])[:20]:
            print(f"    {t.get('name', '?'):>40} {str(t.get('shape', '?')):>20} {t.get('dtype', '?')}")
    _json_or(args, result, _fmt)


def _cmd_inspect_spectral(args):
    """Spectral stats of a matrix."""
    from .inspect.spectral import spectral_stats
    import numpy as _np

    matrix = _np.load(args.source)
    result = spectral_stats(matrix, args.topk)
    def _fmt(r):
        for k, v in r.items():
            print(f"  {k}: {v}")
    _json_or(args, result, _fmt)


def _cmd_inspect_bundle(args):
    """Audit a weight bundle."""
    from .inspect.tensor import audit_bundle

    result = audit_bundle(Path(args.source), topk=args.topk, only_square=args.only_square)
    def _fmt(r):
        print(f"  {r.get('tensor_count', 0)} tensors audited")
        for t in r.get('tensors', [])[:10]:
            print(f"    {t.get('name', '?'):>40}: σ1={t.get('sigma1', 0):.2f}")
    _json_or(args, result, _fmt)


# === embed commands ===

def _cmd_embed_direction(args):
    """Find tokens aligned with a direction in embedding/unembedding space."""
    import numpy as _np

    backend = _load(args)
    direction = _np.load(args.direction).astype(_np.float32)

    # Get embedding or unembedding matrix from backend
    model_inner = getattr(backend.model, 'model', backend.model)
    import mlx.core as mx
    if args.space == "unembedding":
        from .cartography.runtime import _lm_head
        # Use lm_head weight matrix
        h_probe = _np.eye(backend.config.hidden_size, dtype=_np.float32)
        # Direct: project direction through lm_head
        matrix = None
        for attr in ['lm_head', 'output']:
            w = getattr(backend.model, attr, None)
            if w is not None:
                matrix = _np.array(w.weight.astype(mx.float32))
                break
        if matrix is None:
            # Tied weights: use embedding
            matrix = _np.array(model_inner.embed_tokens.weight.astype(mx.float32))
    else:
        matrix = _np.array(model_inner.embed_tokens.weight.astype(mx.float32))

    # Project all tokens onto direction
    scores = matrix @ direction
    top_pos = _np.argsort(-scores)[:args.top_k]
    top_neg = _np.argsort(scores)[:args.top_k]

    pos_tokens = []
    for idx in top_pos:
        tok = backend.tokenizer.decode([int(idx)], skip_special_tokens=True)
        pos_tokens.append({"token": tok, "token_id": int(idx), "score": float(scores[idx])})
    neg_tokens = []
    for idx in top_neg:
        tok = backend.tokenizer.decode([int(idx)], skip_special_tokens=True)
        neg_tokens.append({"token": tok, "token_id": int(idx), "score": float(scores[idx])})

    result = {
        "direction": args.direction,
        "space": args.space,
        "positive": pos_tokens,
        "negative": neg_tokens,
    }
    def _fmt(r):
        print(f"  Most aligned (+):")
        for t in r['positive'][:15]:
            print(f"    {t.get('score', 0):>+8.3f}  {t.get('token', '?')}")
        print(f"  Most opposed (-):")
        for t in r['negative'][:15]:
            print(f"    {t.get('score', 0):>+8.3f}  {t.get('token', '?')}")
    _json_or(args, result, _fmt)


# === probe commands ===

def _cmd_probe_battery(args):
    """Full behavioral probe battery."""
    from .cartography.probes import full_probe_battery

    backend = _load(args)
    results = full_probe_battery(
        backend.model, backend.tokenizer, max_tokens=args.max_tokens)
    result = {"probes": [_to_dict(r) for r in results]}
    def _fmt(r):
        for p in r['probes']:
            print(f"  {p.get('name', '?'):>20}: engaged={p.get('engaged', False)} "
                  f"entropy={p.get('entropy', 0):.3f}")
    _json_or(args, result, _fmt)


def _cmd_probe_safetybench(args):
    """Safety benchmark evaluation."""
    from .cartography.safetybench import evaluate_model

    backend = _load(args)
    result = evaluate_model(
        backend.model, backend.tokenizer, args.dataset,
        alpha=args.alpha, max_tokens=args.max_tokens)
    result_dict = _to_dict(result)
    def _fmt(r):
        print(f"  Refusal rate: {r.get('refusal_rate', 0):.1%}")
        print(f"  Compliance rate: {r.get('compliance_rate', 0):.1%}")
        for cat, stats in r.get('by_category', {}).items():
            print(f"    {cat:>20}: refused={stats.get('refused', 0)}/{stats.get('total', 0)}")
    _json_or(args, result_dict, _fmt)


if __name__ == "__main__":
    main()
