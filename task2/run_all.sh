#!/bin/bash
# Full pipeline of task 2, from the collected answers to submission.csv. Submission (step 10) is left manual.
# Steps 07-08 need a GPU and the Hugging Face checkpoints; with SKIP_WHITEBOX=1 the other steps run on CPU and
# step 09 rebuilds the submitted CSV from the evidence table.
set -euo pipefail; D=$(cd "$(dirname "$0")" && pwd)
for s in 00_init_workdir 01_collect_reference 02_collect_blackbox 03_families 04_blackbox_split 05_collect_long 06_structure_tests; do
  echo "== $s"; bash "$D/steps/$s.sh"
done
if [ -z "${SKIP_WHITEBOX:-}" ]; then
  echo "== 07_whitebox_score"; bash "$D/steps/07_whitebox_score.sh"
  echo "== 08_whitebox_signals"; bash "$D/steps/08_whitebox_signals.sh"
else
  echo "== 08 (repetition rate only, CPU)"; (source "$D/config.env"; cd "$T2_WORK" && $PY "$D/src/whitebox/repetition.py" | tail -4)
fi
echo "== 09_decide_partition"; bash "$D/steps/09_decide_partition.sh"
