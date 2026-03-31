"""Fetch files from GitHub PRs and repos via the gh CLI."""
from __future__ import annotations
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from ..signal import Signal, SignalStore


def fetch_github_pr(
    store: SignalStore,
    pr_url: str,
    *,
    label: str | None = None,
    checkout_dir: Path | str | None = None,
) -> Path:
    """Fetch PR diff metadata and changed files. Returns path to temp checkout or checkout_dir."""
    # Parse: https://github.com/owner/repo/pull/123
    parts = pr_url.rstrip("/").split("/")
    owner_repo = f"{parts[-4]}/{parts[-3]}"
    pr_number = parts[-1]
    model_label = label or f"{owner_repo}#{pr_number}"

    # Get PR metadata via gh CLI
    try:
        pr_json = subprocess.run(
            ["gh", "pr", "view", pr_number, "--repo", owner_repo, "--json",
             "title,body,files,additions,deletions,changedFiles,state,baseRefName"],
            capture_output=True, text=True, timeout=30,
        )
        if pr_json.returncode == 0:
            pr_data = json.loads(pr_json.stdout)
            store.add(Signal("pr_title", "fetch", model_label, "title", 0.0, {"title": pr_data.get("title", "")}))
            store.add(Signal("pr_state", "fetch", model_label, "state", 0.0, {"state": pr_data.get("state", "")}))
            store.add(Signal("pr_additions", "fetch", model_label, "additions", float(pr_data.get("additions", 0)), {}))
            store.add(Signal("pr_deletions", "fetch", model_label, "deletions", float(pr_data.get("deletions", 0)), {}))
            store.add(Signal("pr_changed_files", "fetch", model_label, "changed_files", float(pr_data.get("changedFiles", 0)), {}))

            for f in pr_data.get("files", []):
                path = f.get("path", "")
                adds = f.get("additions", 0)
                dels = f.get("deletions", 0)
                store.add(Signal("pr_file", "fetch", model_label, path, float(adds + dels),
                                 {"additions": adds, "deletions": dels}))
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        store.add(Signal("fetch_error", "fetch", model_label, "gh_cli", 0.0, {"error": "gh CLI failed"}))

    # Download the changed files
    out_dir = Path(checkout_dir) if checkout_dir else Path(tempfile.mkdtemp(prefix="heinrich_"))
    try:
        subprocess.run(
            ["gh", "pr", "diff", pr_number, "--repo", owner_repo],
            capture_output=True, text=True, timeout=30,
        )
        # Clone just the PR branch
        subprocess.run(
            ["gh", "pr", "checkout", pr_number, "--repo", owner_repo, "--detach"],
            capture_output=True, text=True, timeout=60, cwd=str(out_dir),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return out_dir


def fetch_github_path(
    store: SignalStore,
    url: str,
    *,
    label: str | None = None,
) -> None:
    """Fetch metadata from any GitHub URL (repo, PR, tree, blob)."""
    model_label = label or url.split("/")[-1]

    if "/pull/" in url:
        fetch_github_pr(store, url, label=model_label)
    else:
        # Repo or path — just emit the URL as a signal
        store.add(Signal("github_url", "fetch", model_label, url, 0.0, {}))


def is_github_url(source: str) -> bool:
    """Check if a source string looks like a GitHub URL."""
    return source.startswith("https://github.com/") or source.startswith("github.com/")
