#!/usr/bin/env python3
"""Upload the two oversized neuron indexes to R2, restoring the Neurons lane.

`token_neurons.bin` for smollm2-360m (327 MB) and qwen2.5-0.5b-instruct (466 MB)
exceed wrangler's 300 MiB single-PUT CLI limit, so they were skipped in the
original publish. R2's S3 API has no such limit (boto3's upload_file does
automatic multipart), so this uploads just those two files to the exact keys
the Worker reads (`{model}/raw.mri/decomp/token_neurons.bin`).

Create an R2 API token first (Cloudflare dashboard -> R2 -> Manage R2 API Tokens
-> Create, Object Read & Write on the heinrich-mri bucket), then:

  R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
    .venv/bin/python web/scripts/publish_neuron_indexes.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ACCOUNT = os.environ.get("R2_ACCOUNT_ID", "c3c6733068f1c3ea1a7255743e35d95c")
BUCKET = os.environ.get("R2_BUCKET", "heinrich-mri")
ROOT = Path(__file__).resolve().parents[1] / ".r2-export"
REL = "raw.mri/decomp/token_neurons.bin"
TARGETS = ["smollm2-360m", "qwen2.5-0.5b-instruct"]


def main() -> int:
    ak = os.environ.get("R2_ACCESS_KEY_ID")
    sk = os.environ.get("R2_SECRET_ACCESS_KEY")
    if not (ak and sk):
        print(
            "Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY. Create an R2 API token\n"
            "with Object Read & Write on the heinrich-mri bucket in the Cloudflare\n"
            "dashboard (R2 -> Manage R2 API Tokens).",
            file=sys.stderr,
        )
        return 2

    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{ACCOUNT}.r2.cloudflarestorage.com",
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name="auto",
    )

    for model in TARGETS:
        src = ROOT / model / REL
        key = f"{model}/{REL}"
        if not src.exists():
            print(f"SKIP {model}: {src} not found", file=sys.stderr)
            continue
        mb = src.stat().st_size / 1e6
        print(f"-> {key}  ({mb:.0f} MB) uploading ...", flush=True)
        client.upload_file(str(src), BUCKET, key)  # auto-multipart for large files
        head = client.head_object(Bucket=BUCKET, Key=key)
        print(f"   done: {head['ContentLength'] / 1e6:.0f} MB in R2")

    print(
        "\nUploaded. Verify (expects HTTP 200, not 404):\n"
        "  curl -s -o /dev/null -w '%{http_code}\\n' "
        "'https://hcirnieh.com/api/neuron-field/smollm2-360m/raw?token=0'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
