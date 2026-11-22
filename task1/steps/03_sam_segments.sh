#!/bin/bash
# SAM segments of every image -> $T1_DATA/sam/out/{train,test}/<image>.npz. The composite generator uses them as
# object-shaped masks (MaskSampler); without them it falls back on its other shapes.
source "$(dirname "$0")/lib.sh"; set -euo pipefail
N=${#G[@]}
for split in train test; do
  for i in $(seq 0 $((N - 1))); do
    CUDA_VISIBLE_DEVICES=${G[$i]} $PY "$S/sam/sam_feats.py" --split $split --shard $i/$N --weights "$SAM_WEIGHTS" \
      --out_dir "$T1_DATA/sam/out" > logs/03_sam_${split}_$i.log 2>&1 &
  done; wait
done
