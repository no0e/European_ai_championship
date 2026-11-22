#!/bin/bash
# Black-box split inside each family (solve.py --taille 6: the most coherent group of 6 cosmetic variants).
# Only its QWEN_A split is kept in the final answer (confirmed by the 01:19 diagnostic submission).
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
cd "$T2_WORK/blackbox_966"
PYTHONPATH="$D" $PY "$D/src/blackbox/solve.py" --taille 6 --out ../submission_blackbox.csv > ../logs/04_solve.log 2>&1
cmp -s ../submission_blackbox.csv "$D/results/submissions/20261121-233444_score0500.csv" \
  && echo "black-box split identical to the 0.50 submission" || echo "WARNING: differs from the 0.50 submission"
