#!/bin/bash
# Second scale (A3: fine 128 map instead of 64) + realistic holes (A1a, built into common.py) + watermark strength
# 0.8-1.2 (A1b) + doubled pool with the FULL images of the other methods and 24 held-out images (A2), long training.
source "$(dirname "$0")/lib.sh"; set -euo pipefail
i=0
for m in pixelseal trustmark; do
  [ -f pairs_long/$m/index.json ] || CUDA_VISIBLE_DEVICES=$(gpu $i) $PY "$DN/gen_pairs.py" --method $m --reps 2 --rects 2 \
    --with_full --holdout 24 --out pairs_long > logs/07_gen_$m.log 2>&1 &
  i=$((i + 1))
done; wait
B=$LONG_BUDGET_MIN
C="--run long --scale2 --bs 8 --pairs pairs_long --steps 40000 --budget_min $B --max_min $((B + 15)) --eval_every 2000 \
 --workers $WORKERS --synth_n 48 --synth_floor 0.75 --synth_floor_step 20000 --alpha 0.8 1.2 --p_rect 0.12"
train $(gpu 0) --method pixelseal --init runs/v2/pixelseal.pt $C > logs/07_long_pixelseal.log 2>&1 &
train $(gpu 1) --method trustmark --init runs/v2/trustmark.pt $C > logs/07_long_trustmark.log 2>&1 &
wait; grep -h "VAL RÉEL" logs/07_long_*.log | tail -2
