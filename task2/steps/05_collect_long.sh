#!/bin/bash
# Paid long collection (44 requests x 256 tokens per endpoint: fragile questions, equivalent tool choices, arbitrary
# choices, random sequences). Skipped unless COLLECT=1: data/answers_long_ep.jsonl holds the 1,453 answers (22/11).
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
[ "$COLLECT" = 1 ] || { echo "COLLECT=0: using data/answers_long_ep.jsonl"; exit 0; }
cd "$D/src/structure" && $PY plan2.py && $PY plan2.py pilot --go && $PY plan2.py full --go
