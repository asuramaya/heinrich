"""Observatory — the consumer-side publish surface for Heinrich.

Packages a decomposed `.mri` into the worker-native artifact (sidecars + the
lean consumer file set) and publishes it to an R2 bucket over the S3 API.
The viewer (web/) reads exactly what `publish` uploads; see
`web/ARTIFACT_FORMAT.md` for the contract.
"""
from __future__ import annotations

from .publish import publish, write_sidecars, consumer_files, manifest_entry

__all__ = ["publish", "write_sidecars", "consumer_files", "manifest_entry"]
