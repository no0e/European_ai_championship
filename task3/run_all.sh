#!/bin/bash
# Full pipeline of task 3, from the official files to plat6.pt. Submission (step 05) is left manual.
set -euo pipefail; D=$(cd "$(dirname "$0")" && pwd)
bash "$D/steps/01_prepare_anchors.sh"
bash "$D/steps/02_synthetic_variants.sh"
bash "$D/steps/03_train_models.sh"
bash "$D/steps/04_weight_average.sh"
echo "plat6.pt written. Submit with: bash $D/steps/05_submit.sh"
