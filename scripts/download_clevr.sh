#!/usr/bin/env bash
# CLEVR test subset — instructions, not auto-downloader.
#
# CLEVR_v1.0 is 18GB. We deliberately do NOT auto-fetch it; the user should be
# in control of large downloads. This script prints the canonical recipe for
# extracting a 10-scene subset to data/clevr_test_subset/.
#
# Layout produced:
#   data/clevr_test_subset/
#   ├── scenes.json          ← first 10 scenes from CLEVR_val_scenes.json
#   └── images/
#       ├── CLEVR_val_000000.png
#       ├── … 10 PNGs total
set -euo pipefail

cat <<'EOF'
CLEVR test subset extraction
============================

CLEVR_v1.0 ships as a single 18GB zip. We only need 10 val images plus the
matching slice of scenes.json. Run the following from the repo root:

  mkdir -p data/clevr_test_subset/images
  curl -L -o /tmp/CLEVR_v1.0.zip https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip

  # 10-scene slice of the val scenes JSON
  unzip -p /tmp/CLEVR_v1.0.zip CLEVR_v1.0/scenes/CLEVR_val_scenes.json \
    | python -c 'import json,sys; d=json.load(sys.stdin); d["scenes"]=d["scenes"][:10]; json.dump(d,sys.stdout)' \
    > data/clevr_test_subset/scenes.json

  # Matching 10 val PNGs
  for i in 0 1 2 3 4 5 6 7 8 9; do
    fn=$(printf "CLEVR_val_%06d.png" "$i")
    unzip -p /tmp/CLEVR_v1.0.zip "CLEVR_v1.0/images/val/$fn" \
      > "data/clevr_test_subset/images/$fn"
  done

  rm /tmp/CLEVR_v1.0.zip

Verify with:
  python -c "from evaluation.clevr_subset import iter_cases; print(len(iter_cases()))"
  # → 50

This recipe is also documented in scripts/download_clevr.sh and PHASE_4_CLEANUP.md.
EOF
