#!/usr/bin/env bash
# Push the synthetic MRI library (web/.data) into R2, preserving key layout.
#   ./upload.sh local    → local wrangler dev simulation (.wrangler/state)
#   ./upload.sh remote   → real R2 bucket (needs `wrangler r2 bucket create heinrich-mri`)
set -euo pipefail
MODE="${1:-local}"
BUCKET="heinrich-mri"
FLAG="--local"; [ "$MODE" = "remote" ] && FLAG="--remote"
cd "$(dirname "$0")"
[ -d .data ] || { echo "no .data — run scripts/cf_synth_mri.py first"; exit 1; }
count=0
while IFS= read -r f; do
  key="${f#.data/}"
  wrangler r2 object put "$BUCKET/$key" --file "$f" $FLAG >/dev/null 2>&1
  count=$((count+1))
done < <(find .data -type f)
echo "uploaded $count objects to $BUCKET ($MODE)"
