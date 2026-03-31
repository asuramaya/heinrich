"""Tests for GitHub PR / path fetching."""
from unittest.mock import patch, MagicMock
import json
from heinrich.fetch.github import fetch_github_pr, fetch_github_path, is_github_url
from heinrich.signal import SignalStore


def test_is_github_url():
    assert is_github_url("https://github.com/owner/repo/pull/123")
    assert is_github_url("https://github.com/owner/repo")
    assert not is_github_url("/local/path")
    assert not is_github_url("owner/repo")


@patch("heinrich.fetch.github.subprocess")
def test_fetch_github_pr_emits_signals(mock_subprocess):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({
        "title": "Add new model",
        "state": "OPEN",
        "additions": 100,
        "deletions": 20,
        "changedFiles": 5,
        "files": [
            {"path": "records/run/submission.json", "additions": 10, "deletions": 0},
            {"path": "records/run/results.json", "additions": 15, "deletions": 0},
        ],
    })
    mock_subprocess.run.return_value = mock_result
    mock_subprocess.TimeoutExpired = TimeoutError

    store = SignalStore()
    fetch_github_pr(store, "https://github.com/owner/repo/pull/42", label="test-pr")

    titles = store.filter(kind="pr_title")
    assert len(titles) == 1
    assert titles[0].metadata["title"] == "Add new model"

    files = store.filter(kind="pr_file")
    assert len(files) == 2

    additions = store.filter(kind="pr_additions")
    assert additions[0].value == 100.0


@patch("heinrich.fetch.github.subprocess")
def test_fetch_github_pr_handles_failure(mock_subprocess):
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_subprocess.run.return_value = mock_result
    mock_subprocess.TimeoutExpired = TimeoutError

    store = SignalStore()
    fetch_github_pr(store, "https://github.com/owner/repo/pull/999", label="bad-pr")
    # Should not crash — may emit 0 signals or an error signal


def test_fetch_github_path_pr():
    """Just verify it doesn't crash for a PR URL with mocked subprocess."""
    store = SignalStore()
    with patch("heinrich.fetch.github.subprocess") as mock:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock.run.return_value = mock_result
        mock.TimeoutExpired = TimeoutError
        fetch_github_path(store, "https://github.com/owner/repo/pull/1")
