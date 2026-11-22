#!/bin/bash
# plat6 = parameter-wise mean of the six state_dicts -> $T3_WORK/models/plat6.pt (the submitted file)
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
cd "$T3_WORK"
M=$T3_WORK/models
$PY "$D/src/k3.py" avg --env real --inputs "$M/ps1_a1.0.pt,$M/ps10_a1.0.pt,$M/st1_a1.0.pt,$M/bt1_a1.0.pt,$M/bt2_a1.0.pt,$M/y1_a1.0.pt" \
  --out "$M/plat6.pt" 2>&1 | tee logs/avg_plat6.log | tail -3
