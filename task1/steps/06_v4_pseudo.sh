#!/bin/bash
# v4 (StegaStamp): from v3 of the SAME fold, + test images pseudo-labelled by that v3 model (which never saw the fold),
# + composites whose unwatermarked area comes from another image (p_fill) or is blurred (p_blur), 3,000 steps.
source "$(dirname "$0")/lib.sh"; set -euo pipefail
folds $(gpu 0) stegastamp v4 "runs/v3/f{f}/stegastamp.pt" "runs/v3/f{f}/stegastamp_test.npz" 3000 --workers $WORKERS --keep_last \
  --lr 3e-4 --lr_backbone 2e-5 --p_rect 0.12 --p_fill 0.5 --p_blur 0.1 --pairs pairs --timm_weights "$CONVNEXT_WEIGHTS"
