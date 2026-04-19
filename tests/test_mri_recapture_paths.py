"""Unit tests for _resolve_recapture_source path-relative-to-sharts fallback (T5)."""
from __future__ import annotations

from pathlib import Path


def test_resolves_absolute_path_when_it_exists(tmp_path):
    from heinrich.cli import _resolve_recapture_source

    ckpt = tmp_path / "model.checkpoint.pt"
    ckpt.write_bytes(b"FAKE")
    mri_dir = tmp_path / "out.seq.mri"
    mri_dir.mkdir()
    md = {"provenance": {"model_path": str(ckpt)}}

    model_path, _, why = _resolve_recapture_source(mri_dir, md)
    assert model_path == str(ckpt)
    assert why is None


def test_path_relative_to_sharts_fallback(tmp_path):
    """Only the sharts-relative fallback can resolve this: the MRI name
    doesn't match the checkpoint name, so the legacy naming-convention
    fallback finds nothing, but the recorded path with its sharts prefix
    swapped for the current mount is correct.
    """
    from heinrich.cli import _resolve_recapture_source

    new_sharts = tmp_path / "sharts"
    (new_sharts / "heinrich" / "session11").mkdir(parents=True)
    # Real checkpoint at the new mount point.
    ckpt = new_sharts / "heinrich" / "session11" / "actual-ckpt.checkpoint.pt"
    ckpt.write_bytes(b"FAKE")

    # MRI directory has a DIFFERENT name than the checkpoint — renamed
    # after capture, so the legacy `<mri-base>.checkpoint.pt` guess can't
    # find it.
    mri_dir = new_sharts / "heinrich" / "session11" / "renamed-mri.seq.mri"
    mri_dir.mkdir()
    md = {"provenance": {
        "model_path": "/Volumes/OldSharts/heinrich/session11/actual-ckpt.checkpoint.pt",
    }}

    model_path, _, why = _resolve_recapture_source(mri_dir, md)
    assert model_path == str(ckpt), (
        f"expected sharts-relative fallback to find ckpt; "
        f"got {model_path} ({why})")


def test_path_relative_to_sharts_survives_legacy_name_match_noise(tmp_path):
    """Sanity check: the sharts-relative fallback doesn't break the already-
    working case where the legacy naming convention also matches (both
    should succeed and pick the same checkpoint).
    """
    from heinrich.cli import _resolve_recapture_source

    new_sharts = tmp_path / "sharts"
    (new_sharts / "heinrich" / "session11").mkdir(parents=True)
    ckpt = new_sharts / "heinrich" / "session11" / "foo.checkpoint.pt"
    ckpt.write_bytes(b"FAKE")

    mri_dir = new_sharts / "heinrich" / "session11" / "foo.seq.mri"
    mri_dir.mkdir()
    md = {"provenance": {
        "model_path": "/Volumes/OldSharts/heinrich/session11/foo.checkpoint.pt",
    }}

    model_path, _, why = _resolve_recapture_source(mri_dir, md)
    assert model_path == str(ckpt)


def test_path_relative_fallback_noop_when_mri_has_no_sharts_root(tmp_path):
    """If the MRI isn't under a 'sharts' directory, path-relative fallback
    can't help. The function falls through to the legacy naming convention
    and returns None with a reason.
    """
    from heinrich.cli import _resolve_recapture_source

    mri_dir = tmp_path / "nope.seq.mri"
    mri_dir.mkdir()
    md = {"provenance": {
        "model_path": "/Volumes/OldSharts/nope.checkpoint.pt",
    }}

    model_path, _, why = _resolve_recapture_source(mri_dir, md)
    assert model_path is None
    assert why is not None
