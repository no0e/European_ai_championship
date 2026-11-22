#!/bin/bash
# The six models averaged in plat6 (exact options of the hackathon runs, see README section 4).
# Each model: targets (tg.py) then distillation (k3.py distill), ~6 min on one A100.
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
cd "$T3_WORK"; mkdir -p targets models logs
TG="--rank mv --sigma 7 --tb 1.4 --pos_scale 0.5"
DI="--lr 1e-4 --w_ce 1 --alphas 1.0 --epochs 30 --view_delta 4.5"
run() {  # $1 gpu, $2 name, $3 tg.py options ("-" = reuse the targets of ps1), $4 distill options
  local gpu=$1 n=$2 tgo=$3 dio=$4
  if [ "$tgo" = "-" ]; then
    until [ -s targets/ps1.pt ]; do sleep 10; done; cp targets/ps1.pt targets/$n.pt
  else
    CUDA_VISIBLE_DEVICES=$gpu $PY "$D/src/tg.py" --env real $tgo --out targets/$n.pt > logs/tg_$n.log 2>&1
  fi
  CUDA_VISIBLE_DEVICES=$gpu $PY "$D/src/k3.py" distill --env real --targets "$T3_WORK/targets/$n.pt" $dio \
    --out "$T3_WORK/models/$n.pt" > logs/distill_$n.log 2>&1
  echo "$n done: $(grep -h 'alpha=' logs/distill_$n.log | tail -1 | cut -c1-200)"
}
JOBS=(
  "ps1|$TG --mu -2.5|$DI --strong_frac 0.2 --clean_frac 0.3"
  "ps10|$TG --mu -2.5|$DI --strong_frac 0.2 --clean_frac 0.3 --seed 1"
  "bt1|$TG --mu -1.1 --beta 2|$DI --strong_frac 0.2 --clean_frac 0.3"
  "bt2|$TG --mu -3.9 --beta -2|$DI --strong_frac 0.2 --clean_frac 0.3"
  "st1|$TG --mu -2.5|$DI --clean_frac 0.25 --strong_frac 0.35"
  "y1|-|$DI --strong_frac 0.2 --clean_frac 0.3 --syn_frac 0.3"
)
G=($GPUS); k=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r n tgo dio <<< "$j"
  [ -s models/${n}_a1.0.pt ] && { echo "$n already trained"; continue; }
  run "${G[$((k % ${#G[@]}))]}" "$n" "$tgo" "$dio" &
  k=$((k + 1)); [ $((k % ${#G[@]})) -eq 0 ] && wait
done; wait
