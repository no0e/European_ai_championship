#!/bin/bash
# Second-scale models fine-tuned in CV on the real train images: v6 (v3 recipe, 1,500 steps), then v6p (v4 recipe:
# pseudo-labels of the test from v6 of the same fold, p_fill / p_blur, 3,000 steps). PixelSeal and TrustMark.
source "$(dirname "$0")/lib.sh"; set -euo pipefail
C="--pairs pairs_long --bs 8 --workers $WORKERS --keep_last --lr 3e-4 --lr_backbone 2e-5 --p_rect 0.12 --alpha 0.8 1.2 --synth_n 0 --scale2"
folds $(gpu 0) pixelseal v6 runs/long/pixelseal.pt none 1500 $C &
folds $(gpu 1) trustmark v6 runs/long/trustmark.pt none 1500 $C &
wait
folds $(gpu 0) pixelseal v6p "runs/v6/f{f}/pixelseal.pt" "runs/v6/f{f}/pixelseal_test.npz" 3000 $C --p_fill 0.5 --p_blur 0.1 &
folds $(gpu 1) trustmark v6p "runs/v6/f{f}/trustmark.pt" "runs/v6/f{f}/trustmark_test.npz" 3000 $C --p_fill 0.5 --p_blur 0.1 &
wait
# v7, v7p, v8 are NOT reproduced here: this very recipe, but started from two other second-scale base trainings
# (v7 from one on the step-04 pairs, v8 from one whose options we could not recover). README section 7 has the
# details and the job numbers; step 12 assembles without them and says so.
