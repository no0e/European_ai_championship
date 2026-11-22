#!/bin/bash
# Submit a state_dict (default: plat6.pt). Waits for the server rate limit.
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
$PY "$D/src/submit.py" "${1:-$T3_WORK/models/plat6.pt}"
