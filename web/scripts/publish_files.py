#!/usr/bin/env python3
"""Upload arbitrary artifact files to R2 over the S3 API.

Generalizes publish_neuron_indexes.py: pass file paths under web/.data and
each uploads to the R2 key matching its path relative to web/.data (the same
keys the Worker range-reads). No wrangler 300 MiB limit — boto3's upload_file
does automatic multipart. Idempotent: objects already present at the right
size are skipped.

  set -a; source web/.env; set +a
  .venv/bin/python web/scripts/publish_files.py web/.data/smollm2-135m/raw.mri/decomp/vocab_*.{bin,json,npy}
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ACCOUNT = os.environ.get("R2_ACCOUNT_ID", "c3c6733068f1c3ea1a7255743e35d95c")
BUCKET = os.environ.get("R2_BUCKET", "heinrich-mri")
DATA_ROOT = Path(__file__).resolve().parents[1] / ".data"


def main(paths: list[str]) -> int:
    ak = os.environ.get("R2_ACCESS_KEY_ID")
    sk = os.environ.get("R2_SECRET_ACCESS_KEY")
    if not (ak and sk):
        print("Set R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY (see web/.env)", file=sys.stderr)
        return 2
    if not paths:
        print(__doc__, file=sys.stderr)
        return 2

    import boto3
    from botocore.exceptions import ClientError

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{ACCOUNT}.r2.cloudflarestorage.com",
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name="auto",
    )

    rc = 0
    for p in paths:
        src = Path(p).resolve()
        if not src.exists():
            print(f"SKIP missing: {src}", file=sys.stderr)
            rc = 1
            continue
        try:
            key = str(src.relative_to(DATA_ROOT))
        except ValueError:
            print(f"SKIP outside web/.data: {src}", file=sys.stderr)
            rc = 1
            continue
        size = src.stat().st_size
        try:
            head = client.head_object(Bucket=BUCKET, Key=key)
            if head["ContentLength"] == size:
                print(f"OK (exists) {key} ({size / 1e6:.1f} MB)")
                continue
        except ClientError:
            pass
        print(f"UPLOAD {key} ({size / 1e6:.1f} MB)...", flush=True)
        client.upload_file(str(src), BUCKET, key)
        print(f"  done: {key}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
