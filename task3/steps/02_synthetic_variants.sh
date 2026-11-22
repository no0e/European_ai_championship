#!/bin/bash
# Synthetic variants of the forget images (sd-turbo image-to-image, strength 0.5, 2 per image), used only by model y1.
# Needs diffusers + transformers; during the hackathon they were installed in $T3_WORK/pylib:
#   uv pip install --target $T3_WORK/pylib --no-deps diffusers transformers tokenizers regex
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
cd "$T3_WORK"; mkdir -p k5
G=($GPUS); N=4; i=0
for s in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=${G[$((s % ${#G[@]}))]} $PY "$D/src/gen_variants.py" --strength 0.5 --k 2 --shard $s/$N \
    --out k5/var_s0.5_$s.npz > k5/gen_s0.5_$s.log 2>&1 &
  i=$((i + 1)); [ $((i % ${#G[@]})) -eq 0 ] && wait
done; wait
$PY "$D/src/merge_variants.py" k5/var_s0.5_{0,1,2,3}.npz --out k5/syn_s0.5.npz
