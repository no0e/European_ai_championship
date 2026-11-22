#!/bin/bash
# Families (3 x 11) read on habits of the base model, on the first 966 answers (the data available at 22:45 on 21/11,
# when the 0.50 submission was built). Output: $T2_WORK/ep_families2.json (MCC_family = 1.00 on the leaderboard).
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
mkdir -p "$T2_WORK/blackbox_966" && cd "$T2_WORK/blackbox_966"
head -966 ../answers_ep.jsonl > answers_ep.jsonl; cp ../ep_features.json .
PYTHONPATH="$D" $PY "$D/src/blackbox/families2.py" > ../logs/03_families.log 2>&1
cp ep_families2.json ..
# --strip-trailing-cr: json.dump writes CRLF on Windows, the committed file keeps LF.
if diff -q --strip-trailing-cr ep_families2.json "$D/results/ep_families2.json" > /dev/null; then
  echo "families identical to the submitted ones"
else
  echo "WARNING: families differ from results/ep_families2.json"
fi
