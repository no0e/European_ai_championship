#!/bin/bash
# Full pipeline of task 1, from the official data to out/sub_m5.pt (about 6-8 hours on one node with 4 x A100).
# Submission (step 13) is left manual. Every step can be re-run alone; finished outputs are skipped where possible.
set -euo pipefail; D=$(cd "$(dirname "$0")" && pwd)
for s in 00_setup 01_signals 02_rect_search 03_sam_segments 04_v2_base 05_v3_cv 06_v4_pseudo 07_long_scale2 08_long_cv \
         09_frame_pairs 10_frame_bases 11_frame_cv 12_assemble; do
  echo "== $s $(date +%H:%M)"; bash "$D/steps/$s.sh"
done
