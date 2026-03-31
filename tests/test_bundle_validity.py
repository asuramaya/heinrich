import json
import tempfile
from pathlib import Path
from heinrich.bundle.validity import write_validity_bundle


def test_write_bundle_creates_files():
    out = Path(tempfile.mkdtemp()) / "bundle"
    manifest = {
        "bundle_id": "test-001",
        "claim": {"name": "test claim"},
        "metrics": {"bridge_bpb": 0.5},
        "audits": {},
    }
    result = write_validity_bundle(manifest, out)
    assert result["bundle_id"] == "test-001"
    assert result["claim_level"] == 1
    assert (out / "evidence" / "claim.json").exists()
    assert (out / "evidence" / "metrics.json").exists()
    assert (out / "bundle_manifest.json").exists()
    assert (out / "README.md").exists()


def test_write_bundle_claim_level():
    out = Path(tempfile.mkdtemp()) / "bundle"
    manifest = {
        "bundle_id": "test-002",
        "metrics": {"bridge_bpb": 0.5, "packed_artifact_bpb": 0.52},
        "audits": {"tier2": {"status": "pass"}},
    }
    result = write_validity_bundle(manifest, out)
    assert result["claim_level"] == 4


def test_write_bundle_readme_content():
    out = Path(tempfile.mkdtemp()) / "bundle"
    manifest = {"bundle_id": "test-003", "metrics": {"bpb": 0.5}}
    write_validity_bundle(manifest, out)
    readme = (out / "README.md").read_text()
    assert "test-003" in readme
    assert "heinrich" in readme


def test_write_bundle_with_attachment():
    src_dir = Path(tempfile.mkdtemp())
    (src_dir / "report.json").write_text('{"status": "pass"}')
    out = Path(tempfile.mkdtemp()) / "bundle"
    manifest = {
        "bundle_id": "test-004",
        "base_dir": str(src_dir),
        "attachments": [{"source": "report.json", "dest": "audits/report.json"}],
    }
    result = write_validity_bundle(manifest, out)
    assert result["attachment_count"] == 1
    assert (out / "audits" / "report.json").exists()


def test_write_bundle_empty():
    out = Path(tempfile.mkdtemp()) / "bundle"
    result = write_validity_bundle({"bundle_id": "empty"}, out)
    assert result["claim_level"] == 0
