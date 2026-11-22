#!/bin/bash
# Prepare $T2_WORK: answers we collected (data/), submitted CSVs and their public scores (results/submissions/).
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
mkdir -p "$T2_WORK/best" "$T2_WORK/logs" "$T2_WORK/whitebox/out"
cd "$T2_WORK"
for f in answers_ep.jsonl answers_long_ep.jsonl reference_answers.jsonl snapshot_answers_ref2.jsonl practice_answers.csv \
         reference_prompts.txt ep_features.json; do
  [ -e "$f" ] || cp "$D/data/$f" .
done
cp -n "$D"/data/probe_passes/*.txt . 2>/dev/null || true
cp "$D"/results/submissions/*.csv "$D"/results/submissions/index.jsonl best/
grep -v 0850 "$D/results/submissions/obs.txt" > best/obs.txt        # public scores known BEFORE the final submission
ln -sfn "$T2_WORK" whitebox/data; ln -sfn "$T2_MODELS" whitebox/models
echo "work dir ready: $T2_WORK"
