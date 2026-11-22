#!/bin/bash
# Final partition from results/endpoint_evidence.csv (+ the black-box QWEN_A split of step 04) -> submission.csv
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
cd "$T2_WORK"
$PY "$D/src/decide_partition.py" --evidence "$D/results/endpoint_evidence.csv" --blackbox submission_blackbox.csv \
  --out submission.csv --check "$D/results/submission_final_0.85.csv"
