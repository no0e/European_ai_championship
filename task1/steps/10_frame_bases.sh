#!/bin/bash
# Bases trained on the frame pairs only (no real image): k5b (from v2, three methods) and k6b (second scale, from long).
source "$(dirname "$0")/lib.sh"; set -euo pipefail
B=$K5B_BUDGET_MIN; C="--run k5b --pairs pairs_frame --steps 24000 --keep_last --alpha 0.8 1.2 --p_rect 0.12 --bs 16"
train $(gpu 0) --method pixelseal --init runs/v2/pixelseal.pt $C --budget_min $B --max_min $((B + 10)) --eval_every 4000 \
  --workers $WORKERS --synth_n 48 > logs/10_k5b_pixelseal.log 2>&1 &
train $(gpu 1) --method trustmark --init runs/v2/trustmark.pt $C --budget_min $B --max_min $((B + 10)) --eval_every 4000 \
  --workers $WORKERS --synth_n 48 > logs/10_k5b_trustmark.log 2>&1 &
B=$K5B_SS_BUDGET_MIN
train $(gpu 2) --method stegastamp --init runs/v2/stegastamp.pt --timm_weights "$CONVNEXT_WEIGHTS" $C --budget_min $B \
  --max_min 22 --eval_every 24000 --workers 16 --synth_n 0 > logs/10_k5b_stegastamp.log 2>&1 &
wait
B=$K6B_BUDGET_MIN; C="--run k6b --scale2 --pairs pairs_frame --steps 24000 --budget_min $B --max_min 20 --eval_every 24000 \
 --keep_last --workers 16 --synth_n 0 --alpha 0.8 1.2 --p_rect 0.12 --bs 8"
train $(gpu 0) --method pixelseal --init runs/long/pixelseal.pt $C > logs/10_k6b_pixelseal.log 2>&1 &
train $(gpu 1) --method trustmark --init runs/long/trustmark.pt $C > logs/10_k6b_trustmark.log 2>&1 &
wait; grep -h "VAL RÉEL" logs/10_*.log | cut -c1-200
