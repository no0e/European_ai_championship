#!/bin/bash
# v3 (StegaStamp): v2 fine-tuned in 5-fold CV with the real train images (half synthetic, half real images of the 4
# other folds); out-of-fold predictions on train, test = mean of the 5 folds. Used by the final ensemble for StegaStamp.
source "$(dirname "$0")/lib.sh"; set -euo pipefail
folds $(gpu 0) stegastamp v3 runs/v2/stegastamp.pt none 1500 --workers $WORKERS --keep_last --lr 3e-4 --lr_backbone 2e-5 \
  --p_rect 0.12 --pairs pairs --timm_weights "$CONVNEXT_WEIGHTS"
