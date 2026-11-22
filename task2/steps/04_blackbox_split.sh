#!/bin/bash
# Black-box split inside each family (solve.py --taille 6: the most coherent group of 6 cosmetic variants).
# Only its QWEN_A split is kept in the final answer (confirmed by the 01:19 diagnostic submission).
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
cd "$T2_WORK/blackbox_966"
PYTHONPATH="$D" $PY "$D/src/blackbox/solve.py" --taille 6 --out ../submission_blackbox.csv > ../logs/04_solve.log 2>&1
# The comparison below is informative only. --strip-trailing-cr because a Windows checkout of the reference CSV
# is CRLF while the file just written is LF. A difference in family C (QWEN_B) happens and is harmless: only the
# QWEN_A split of this step reaches the answer, and step 09 re-checks the final CSV byte for byte. See the README.
REF=$D/results/submissions/20261121-233444_score0500.csv
if diff -q --strip-trailing-cr ../submission_blackbox.csv "$REF" > /dev/null; then
  echo "black-box split identical to the 0.50 submission"
else
  echo "black-box split differs from the 0.50 submission (expected in family C, see README section 4)"
fi
