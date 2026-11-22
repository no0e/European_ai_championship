#!/bin/bash
# Free answers of the 9 training endpoints ref-01..ref-09 (official script). Skipped unless COLLECT=1.
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
[ "$COLLECT" = 1 ] || { echo "COLLECT=0: using data/reference_answers.jsonl"; exit 0; }
cd "$T2_WORK" && $PY "$D/src/official/query_reference.py"
