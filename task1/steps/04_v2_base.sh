#!/bin/bash
# v2: exact synthetic pairs (x, embed(x, m)) with the official embedders, then one dense detector per method,
# trained on composites only (6,000 steps). Base images = non-full images of the two OTHER methods + DIV2K crops.
source "$(dirname "$0")/lib.sh"; set -euo pipefail
i=0
for m in pixelseal trustmark stegastamp; do
  [ -f pairs/$m/index.json ] || CUDA_VISIBLE_DEVICES=$(gpu $i) $PY "$DN/gen_pairs.py" --method $m --reps 2 --rects 2 \
    --out pairs > logs/04_gen_$m.log 2>&1 &
  i=$((i + 1))
done; wait
C="--run v2 --steps 6000 --workers $WORKERS --p_rect 0.12 --pairs pairs"
train $(gpu 0) --method pixelseal  $C > logs/04_v2_pixelseal.log 2>&1 &
train $(gpu 1) --method trustmark  $C > logs/04_v2_trustmark.log 2>&1 &
train $(gpu 2) --method stegastamp $C --timm_weights "$CONVNEXT_WEIGHTS" > logs/04_v2_stegastamp.log 2>&1 &
wait; grep -h "VAL RÉEL" logs/04_v2_*.log | tail -3
