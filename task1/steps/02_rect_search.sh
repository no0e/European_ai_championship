#!/bin/bash
# TrustMark rectangles of the test set: on the TrustMark test images predicted "not full", search the crop that
# re-reads the message. assemble.py keeps a rectangle when its bit accuracy is >= 0.90 (75 images during the hackathon).
source "$(dirname "$0")/lib.sh"; set -euo pipefail
mkdir -p rects
$PY - > rects/names_tm_test.txt <<PYEOF
import sys; sys.path.insert(0, "$DN")
from common import global_ll, load_meta, regime_thresholds
thr = regime_thresholds()["trustmark"]; ll = global_ll("test")
print("\n".join(m["image_name"] for m in load_meta("test") if m["method"] == "trustmark" and ll[m["image_name"]] <= thr))
PYEOF
echo "$(wc -l < rects/names_tm_test.txt) TrustMark test images predicted not full"
N=${#G[@]}
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=${G[$i]} $PY "$S/rect_search.py" --split test --method trustmark --names rects/names_tm_test.txt \
    --data "$T1_DATA/release" --models "$T1_MODELS" --amp --shard $i/$N --out rects/rect_tm_test_s$i.json > logs/02_rect_$i.log 2>&1 &
done; wait
