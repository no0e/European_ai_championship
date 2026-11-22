#!/bin/bash
# Official watermark models (once). The data zips must be unpacked into $T1_DATA/release (see README).
source "$(dirname "$0")/lib.sh"; set -euo pipefail
cd "$S/starter_kit" && $PY download_models.py --models_dir "$T1_MODELS"
for p in "$T1_DATA/release/train/metadata.json" "$T1_DATA/release/test/metadata.json" "$T1_DATA/release/train/masks_128.npz" \
         "$T1_DATA/clean/DIV2K_valid_HR" "$CONVNEXT_WEIGHTS"; do
  [ -e "$p" ] && echo "OK  $p" || echo "MISSING  $p"
done
