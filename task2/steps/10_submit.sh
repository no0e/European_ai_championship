#!/bin/bash
# Submit a CSV (default: $T2_WORK/submission.csv); one submission per 30 min. Archived in $T2_WORK/best/.
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
cd "$T2_WORK" && $PY "$D/src/structure/submit.py" "${1:-$T2_WORK/submission.csv}" "${2:-}"
