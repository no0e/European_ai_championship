#!/bin/bash
# CV fine-tuning on the real train images: k6 (second scale + frame, PixelSeal / TrustMark), k5 (StegaStamp + frame).
source "$(dirname "$0")/lib.sh"; set -euo pipefail
C="--workers 10 --keep_last --lr 3e-4 --lr_backbone 2e-5 --p_rect 0.12 --alpha 0.8 1.2 --synth_n 0 --bs 16 --pairs pairs_frame"
# k6: batch 16 (in the hackathon launcher the generic options came last and overrode --bs 8)
folds $(gpu 0) pixelseal k6 runs/k6b/pixelseal.pt none 1500 --scale2 $C &
folds $(gpu 1) trustmark k6 runs/k6b/trustmark.pt none 1500 --scale2 $C &
folds $(gpu 2) stegastamp k5 runs/k5b/stegastamp.pt none 1500 $C --timm_weights "$CONVNEXT_WEIGHTS" &
wait
