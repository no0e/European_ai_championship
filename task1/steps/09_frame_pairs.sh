#!/bin/bash
# Key finding of 22/11: the published image is a crop of a LARGER watermarked frame (side s x 512, s = 1.02-1.17).
# Pairs are now watermarked on a frame of side up to 1.18 x 512, completed by reflection, then cropped back.
source "$(dirname "$0")/lib.sh"; set -euo pipefail
i=0
for m in pixelseal trustmark stegastamp; do
  [ -f pairs_frame/$m/index.json ] || CUDA_VISIBLE_DEVICES=$(gpu $i) $PY "$DN/gen_pairs.py" --method $m --reps 2 --rects 2 \
    --with_full --holdout 24 --frame 0.84 --frame_max 1.18 --out pairs_frame > logs/09_gen_$m.log 2>&1 &
  i=$((i + 1))
done; wait
