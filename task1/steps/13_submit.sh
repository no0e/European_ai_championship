#!/bin/bash
# Submit a .pt (default: out/sub_m5.pt). Key in WMLOC_API_KEY. Journal: $T1_WORK/out/submissions.csv
source "$(dirname "$0")/lib.sh"; set -euo pipefail
$PY "$DN/submit_file.py" "${1:-$T1_WORK/out/sub_m5.pt}" --note "${2:-m5}"
