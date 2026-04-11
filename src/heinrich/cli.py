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

    p_mriscan = sub.add_parser("mri-scan", help="Full MRI workup: capture all modes, health check, layer deltas, logit lens, PCA depth")
    p_mriscan.add_argument("--model", required=True, help="Model ID or checkpoint path")
    p_mriscan.add_argument("--output", "-o", required=True, help="Model output directory (e.g. /Volumes/sharts/qwen-0.5b)")
    p_mriscan.add_argument("--backend", choices=["auto", "mlx", "hf", "decepticon"], default="auto")
    p_mriscan.add_argument("--n-index", type=int, default=None, help="Number of tokens (default: full vocabulary)")
    p_mriscan.add_argument("--result-json", default=None, help="Decepticon: path to result.json")
    p_mriscan.add_argument("--tokenizer-path", default=None, help="Decepticon: path to tokenizer model")

    p_mrihealth = sub.add_parser("mri-health", help="Deep health check: verify shapes, NaN, weights, consistency for every MRI")
    p_mrihealth.add_argument("--dir", default="/Volumes/sharts", help="MRI directory (default: /Volumes/sharts)")
    p_mrihealth.add_argument("--mri", nargs="*", default=None, help="Specific .mri directories to check (default: all)")

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
    p_mri.add_argument("--mode", choices=["template", "naked", "raw"], default="template")
    p_mri.add_argument("--n-index", type=int, default=None, help="Number of tokens (default: full vocabulary)")
    p_mri.add_argument("--output", "-o", required=True, help="Output .mri directory")
    p_mri.add_argument("--db", default=None, help="Database path for direction discovery")
    p_mri.add_argument("--result-json", default=None, help="Decepticon: path to result.json for model config")
    p_mri.add_argument("--tokenizer-path", default=None, help="Decepticon: path to tokenizer model file")

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
    elif args.command == "mri-scan":
        _cmd_mri_scan(args)
    elif args.command == "mri-health":
        _cmd_mri_health(args)
    elif args.command == "profile-logit-lens":
        _cmd_logit_lens(args)
    elif args.command == "profile-layer-deltas":
        _cmd_layer_deltas(args)
    elif args.command == "profile-gates":
        _cmd_gates(args)
    elif args.command == "profile-attention":
        _cmd_attention(args)
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
    result = capture_mri(backend, mode=args.mode, n_index=args.n_index,
                          output=args.output, db_path=getattr(args, 'db', None))
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

    print(f"\n=== PCA Depth: {result['model']} ({result['mode']}) "
          f"— {result['n_layers']} layers, {result['n_sampled']} tokens ===\n")

    print(f"  {'layer':>5} {'PC1%':>6} {'PC2%':>6} {'PC3%':>6} {'PCs@50%':>7} {'(-) pole':<12} {'(+) pole'}")
    print(f"  {'-'*60}")
    for r in result['layers']:
        print(f"  L{r['layer']:>3} {r['pc1_pct']:>5.1f}% {r['pc2_pct']:>5.1f}% "
              f"{r['pc3_pct']:>5.1f}% {r['pcs_for_50pct']:>7} "
              f"{r['neg_pole']:<12} {r['pos_pole']}")


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

    print(f"\n=== MRI Library: {mri_dir} ===\n")
    print(f"Complete ({len(complete)}):")
    for name, detail, _ in complete:
        print(f"  ✓ {name:<35} {detail}")

    if incomplete:
        print(f"\nIncomplete ({len(incomplete)}):")
        for name, detail, _ in incomplete:
            print(f"  ✗ {name:<35} {detail}")

    if running:
        print(f"\nRunning ({len(running)}):")
        for r in running:
            print(f"  ▶ {r.strip()}")

    # Also check for legacy .shrt files
    legacy = list(Path("data/runs").glob("*.shrt.npz")) if Path("data/runs").exists() else []
    if legacy:
        print(f"\nLegacy .shrt files ({len(legacy)} in data/runs/) — recapture as .mri")

    print()


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
            capture_mri(backend, mode=mode, n_index=args.n_index, output=mri_path)
            elapsed = _time.time() - t0
            results[mode] = {"captured": True, "elapsed_s": round(elapsed), "path": mri_path}
        except Exception as e:
            print(f"  {mode}: FAILED — {e}")
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

    # Phase 3: Layer deltas (all modes)
    print(f"\n--- Phase 3: Layer Deltas ---\n")
    for mode in modes:
        mri_path = results[mode]["path"]
        print(f"  {mode}:")
        ld = layer_deltas(mri_path, n_sample=5000)
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
        ll = logit_lens(mri_path, n_sample=200, layers=sample_layers)
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
        ga = gate_analysis(mri_path, n_sample=5000)
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
    aa = attention_analysis(tpl_path, n_sample=5000)
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

    n_healthy = 0
    n_issues = 0

    print(f"\n=== MRI Health Check: {len(mris)} directories ===\n")
    for d in mris:
        result = verify_mri(str(d))
        s = result["summary"]
        if result["healthy"]:
            n_healthy += 1
            size = s.get("size_gb", "?")
            print(f"  HEALTHY  {d.name:<35} {s.get('model','?'):>8} {s.get('mode','?'):>8} "
                  f"{s.get('n_tokens','?'):>8}tok {s.get('n_layers','?'):>3}L {size}G")
        else:
            n_issues += 1
            print(f"  ISSUES   {d.name:<35} {s.get('model','?'):>8} {s.get('mode','?'):>8}")
            for iss in result["issues"]:
                print(f"           ! {iss}")

    print(f"\n  {n_healthy} healthy, {n_issues} with issues, {len(mris)} total\n")


def _cmd_logit_lens(args: argparse.Namespace) -> None:
    """Logit lens: what would the model predict at each layer?"""
    from .profile.compare import logit_lens

    result = logit_lens(args.mri, top_k=args.top_k,
                        layers=args.layers, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Logit Lens: {result['model']} — {result['n_tokens']} tokens, "
          f"{result['n_layers']} layers, top-{result['top_k']} ===\n")

    for lr in result['layers']:
        layer = lr['layer']
        preds = lr['predictions']
        # Summarize: most common top-1 prediction across sampled tokens
        from collections import Counter
        top1_counts = Counter(p['top_ids'][0] for p in preds)
        most_common = top1_counts.most_common(5)
        common_str = ', '.join(f"id={tid}({cnt})" for tid, cnt in most_common)
        print(f"  L{layer:>2}: top-1 mode: {common_str}")


def _cmd_layer_deltas(args: argparse.Namespace) -> None:
    """Layer deltas: what each layer actually computes."""
    from .profile.compare import layer_deltas

    result = layer_deltas(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Layer Deltas: {result['model']} ({result['mode']}) — "
          f"{result['n_tokens']} tokens ===\n")

    print(f"  {'layer':>5} {'mean':>10} {'max':>10} {'std':>10} {'amplif':>8}")
    print(f"  {'-'*45}")
    for r in result['layers']:
        print(f"  L{r['layer']:>3} {r['mean_delta_norm']:>10.2f} "
              f"{r['max_delta_norm']:>10.2f} {r['std_delta_norm']:>10.2f} "
              f"{r['amplification']:>7.1f}x")


def _cmd_gates(args: argparse.Namespace) -> None:
    """MLP gate analysis."""
    from .profile.compare import gate_analysis

    result = gate_analysis(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Gate Analysis: {result['model']} ({result['mode']}) — "
          f"{result['n_tokens']} tokens, top-{result['gate_k']} ===\n")

    print(f"  {'layer':>5} {'unique':>7} {'top1%':>6} {'mean':>7} {'max':>7} {'top neuron'}")
    print(f"  {'-'*50}")
    for r in result['layers']:
        print(f"  L{r['layer']:>3} {r['unique_neurons']:>7} {r['top1_concentration']:>5.0%} "
              f"{r['mean_activation']:>7.2f} {r['max_activation']:>7.2f} "
              f"n{r['top1_neuron']}")


def _cmd_attention(args: argparse.Namespace) -> None:
    """Attention analysis."""
    from .profile.compare import attention_analysis

    result = attention_analysis(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Attention Analysis: {result['model']} ({result['mode']}) — "
          f"{result['n_tokens']} tokens, {result['n_heads']} heads, "
          f"seq_len={result['seq_len']}, token_pos={result['token_pos']} ===\n")

    print(f"  {'layer':>5} {'self':>6} {'prefix':>7} {'suffix':>7} {'entropy':>8}")
    print(f"  {'-'*40}")
    for r in result['layers']:
        print(f"  L{r['layer']:>3} {r['self_weight']:>5.1%} {r['prefix_weight']:>6.1%} "
              f"{r['suffix_weight']:>6.1%} {r['entropy']:>8.3f}")


def _cmd_lookup_fraction(args: argparse.Namespace) -> None:
    """How much is table lookup vs computation?"""
    from .profile.compare import lookup_fraction

    result = lookup_fraction(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    lf = result['lookup_fraction']
    print(f"\n=== Lookup Fraction: {result['model']} ({result['mode']}) — "
          f"{result['n_tokens']} tokens ===\n")
    print(f"  Lookup-solvable: {result['lookup_solvable']} ({lf:.1%})")
    print(f"  Compute-needed:  {result['compute_needed']} ({1-lf:.1%})")

    print(f"\n  By script:")
    print(f"  {'script':<12} {'n':>6} {'lookup':>7} {'fraction':>9}")
    print(f"  {'-'*36}")
    for s, v in sorted(result['by_script'].items(), key=lambda x: -x[1]['fraction']):
        print(f"  {s:<12} {v['n']:>6} {v['lookup']:>7} {v['fraction']:>8.1%}")

    print(f"\n  By layer (fraction still matching embedding prediction):")
    print(f"  {'layer':>5} {'match%':>7}")
    print(f"  {'-'*14}")
    for r in result['by_layer']:
        print(f"  L{r['layer']:>3} {r['fraction']:>6.1%}")


def _cmd_distribution_drift(args: argparse.Namespace) -> None:
    """Distribution drift per layer."""
    from .profile.compare import distribution_drift

    result = distribution_drift(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Distribution Drift: {result['model']} ({result['mode']}) — {result['n_tokens']} tokens ===\n")
    print(f"  {'layer':>5} {'top1Δ':>6} {'KL':>10} {'TVD':>7} {'entropy':>8}")
    print(f"  {'-'*40}")
    for r in result['layers']:
        print(f"  L{r['layer']:>3} {r['top1_changed']:>5.1%} {r['mean_kl']:>10.6f} "
              f"{r['mean_tvd']:>7.4f} {r['mean_entropy']:>8.2f}")


def _cmd_retrieval_horizon(args: argparse.Namespace) -> None:
    """Retrieval horizon per layer."""
    from .profile.compare import retrieval_horizon

    result = retrieval_horizon(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Retrieval Horizon: {result['model']} ({result['mode']}) — "
          f"{result['n_tokens']} tokens, {result['n_heads']} heads, "
          f"seq_len={result['seq_len']}, token_pos={result['token_pos']} ===\n")
    print(f"  {'layer':>5} {'dist':>6} {'self%':>6} {'head peaks'}")
    print(f"  {'-'*50}")
    for r in result['layers']:
        peaks = ', '.join(str(p) for p in r['head_peak_positions'])
        print(f"  L{r['layer']:>3} {r['mean_retrieval_distance']:>+5.1f} "
              f"{r['self_attention']:>5.1%} [{peaks}]")


def _cmd_layer_opposition(args: argparse.Namespace) -> None:
    """MLP vs attention opposition per layer."""
    from .profile.compare import layer_opposition

    result = layer_opposition(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Layer Opposition: {result['model']} ({result['mode']}) — {result['n_tokens']} tokens ===\n")
    print(f"  {'layer':>5} {'cos(M,A)':>9} {'mlp':>8} {'attn':>8} {'delta':>8} {'cancel':>7} {'rel_err':>8}")
    print(f"  {'-'*57}")
    for r in result['layers']:
        print(f"  L{r['layer']:>3} {r['cos_mlp_attn']:>+8.4f} {r['mlp_norm']:>8.2f} "
              f"{r['attn_norm']:>8.2f} {r['delta_norm']:>8.2f} {r['cancellation']:>6.0%} "
              f"{r['relative_error']:>7.1%}")


def _cmd_cross_model(args: argparse.Namespace) -> None:
    """Compare models on shared vocabulary."""
    from .profile.compare import cross_model

    result = cross_model(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== Cross-Model Comparison: {result['n_shared']} shared tokens ===\n")

    print(f"  Per-model stats:")
    for m in result['models']:
        grad = f"  grad={m.get('mean_grad', '?')}" if 'mean_grad' in m else ""
        print(f"    {m['name']:<15} disp={m['mean_disp']:>8.1f} ±{m['std_disp']:<6.1f}{grad}")

    print(f"\n  Pairwise comparisons:")
    print(f"  {'A':<15} {'B':<15} {'disp_rho':>9} {'grad_rho':>9} {'overlap':>8}")
    print(f"  {'-'*58}")
    for c in result['comparisons']:
        grad_r = f"{c.get('gradient_rho', ''):>9}" if 'gradient_rho' in c else "        —"
        print(f"  {c['model_a']:<15} {c['model_b']:<15} {c['displacement_rho']:>+8.4f} "
              f"{grad_r} {c['top_overlap']:>3}/{c['top_n']}")

    print(f"\n  Top shared sharts (highest mean displacement):")
    if result['shared_sharts']:
        models = result['models']
        header = "  " + f"{'#':>3} {'token':<20}"
        for m in models:
            header += f" {m['name']:>10}"
        print(header)
        print(f"  {'-'*(25 + 11*len(models))}")
        for s in result['shared_sharts'][:15]:
            line = f"  {s['rank']:>3} {s['token']:<20}"
            for m in models:
                key = f"disp_{m['name']}"
                line += f" {s.get(key, 0):>10.1f}"
            print(line)


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

    d = result['displacement']
    c = result['crystal']
    g = result['gradient']
    f = result['frozen_zone']
    b = result['bandwidth']

    print(f"\n=== Shart Anatomy: {result['model']} ({result['mode']}) — {result['n_tokens']} tokens ===\n")

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
    for t in result['top_sharts'][:args.top_n]:
        grad = f"{t.get('grad_norm', 0):>7.1f}" if 'grad_norm' in t else "     —"
        crys = f"{t.get('crystal_activation', 0):>8.0f}" if 'crystal_activation' in t else "      —"
        act = f"{t.get('active_neurons', 0):>6}" if 'active_neurons' in t else "    —"
        print(f"  {t['rank']:>3} {t['displacement']:>8.0f} {grad} {crys} {act} {t['script']:<8} {t['token']}")

    print(f"\n  Bottom {args.top_n}:")
    print(f"  {'#':>3} {'disp':>8} {'grad':>7} {'script':<8} token")
    print(f"  {'-'*40}")
    for t in result['bottom_tokens'][:args.top_n]:
        grad = f"{t.get('grad_norm', 0):>7.1f}" if 'grad_norm' in t else "     —"
        print(f"  {t['rank']:>3} {t['displacement']:>8.0f} {grad} {t['script']:<8} {t['token']}")


def _cmd_bandwidth(args: argparse.Namespace) -> None:
    """Bandwidth efficiency per layer."""
    from .profile.compare import bandwidth_efficiency

    result = bandwidth_efficiency(args.mri, n_sample=args.n_sample)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    eff = result['bandwidth_efficiency']
    print(f"\n=== Bandwidth Efficiency: {result['model']} ({result['mode']}) — "
          f"{result['n_tokens']} tokens ===\n")
    print(f"  Model size:   {result['total_model_bytes']/1e6:.1f} MB")
    print(f"  Active bytes: {result['total_active_bytes']/1e6:.1f} MB ({eff:.1%})")
    print(f"  Wasted:       {result['wasted_fraction']:.1%}")
    print(f"  MLP neurons:  {result['gate_k']} captured / {result['intermediate_size']} total")

    print(f"\n  {'layer':>5} {'skip%':>6} {'MLP active':>11} {'efficiency':>10}")
    print(f"  {'-'*35}")
    for r in result['layers']:
        print(f"  L{r['layer']:>3} {r['skip_fraction']:>5.0%} "
              f"{r['mlp_active_neurons']:>5.0f}/{r['mlp_total_neurons']:<5} "
              f"{r['efficiency']:>9.1%}")


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


if __name__ == "__main__":
    main()
