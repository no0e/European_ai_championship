#!/bin/bash
# Anchor images: 5,000 COCO val2017 images, short side 256, centre crop 224 -> $T3_WORK/anchors.npz
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
$PY "$D/src/prepare_anchors.py" "$COCO_DIR" "$T3_WORK/anchors.npz" 5000
