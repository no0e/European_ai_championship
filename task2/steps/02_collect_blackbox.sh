#!/bin/bash
# Paid black-box collection on ep-01..ep-33 (probes of src/blackbox/probes.py, lists of probe ids in *.txt).
# Skipped unless COLLECT=1: data/answers_ep.jsonl holds the 1,672 answers we paid for (21/11).
# collect.py caches every answer and never pays twice for the same (endpoint, probe, repeat); --dry-run prints the cost.
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
[ "$COLLECT" = 1 ] || { echo "COLLECT=0: using data/answers_ep.jsonl"; exit 0; }
cd "$T2_WORK"
for p in ep_ident ep_pass1 ep_pass2 ep_pass3 ep_pass4 ep_pass5a ep_pass5b; do
  $PY "$D/src/blackbox/collect.py" --endpoints ep --only-file $p.txt --repeats 4 --out answers_ep.jsonl
done
$PY "$D/src/blackbox/collect.py" --endpoints ep --only-file ep_pass5c.txt --repeats 3 --out answers_ep.jsonl
