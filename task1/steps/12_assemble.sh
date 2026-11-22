#!/bin/bash
# Ensemble m5: mean of the probability maps of the runs below (a folder listed twice weighs double), guided filter and
# decision rule chosen out-of-fold per (method, predicted regime), TrustMark rectangles -> out/sub_m5.pt + CV report.
source "$(dirname "$0")/lib.sh"; set -euo pipefail
# models trained WITHOUT folds must be split into a per-fold view, otherwise their weight differs between train and test
$PY "$DN/mk_folds.py" runs/k5b k5b && $PY "$DN/mk_folds.py" runs/k6b k6b
mkdir -p out cv; R=""
for f in 0 1 2 3 4; do
  R="$R runs/k6/f$f runs/k6/f$f mix/k6b/f$f mix/k6b/f$f mix/k5b/f$f"                     # PixelSeal / TrustMark
  for r in v6 v6p v7 v7p v8; do [ -d runs/$r/f$f ] && R="$R runs/$r/f$f"; done              # second scale (agent A)
  R="$R runs/v3/f$f runs/v4/f$f runs/k5/f$f runs/k5/f$f"                                   # StegaStamp (+ k5b above)
done
mkdir -p mix/k5b_ss && for f in 0 1 2 3 4; do mkdir -p mix/k5b_ss/f$f; for s in train test; do
  ln -sf "$(cd mix/k5b/f$f && pwd)/stegastamp_$s.npz" mix/k5b_ss/f$f/stegastamp_$s.npz; done; R="$R mix/k5b_ss/f$f"; done   # SS k5b x2
OMP_NUM_THREADS=${OMP_ASM:-8} $PY "$DN/assemble.py" --runs $R --rects "rects/rect_tm_test_s*.json" \
  --post none c:8:0.001 c:8:0.01 c:16:0.01 c:16:0.001 c:16:0.0001 c:24:0.001 c:8:0.0001 \
  --out out/sub_m5.pt --save_cv cv/cv_m5.npz 2>&1 | tee logs/12_assemble.log | tail -16
