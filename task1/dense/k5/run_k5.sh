#!/bin/bash
# k5 — quatre chaînes indépendantes, une par GPU (lu au DÉMARRAGE du job : modifiable tant qu'il attend).
T=/p/scratch/training2666/diette1/Sevel_Labs_ENSAI/Hackathon_Watermark_Localisation_Task
cd $T/dense/k5 || exit 1
source $T/.venv/bin/activate
export PYTHONPATH=$T/sam/pylib:$PYTHONPATH OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
A=$T/agentA; PF=pairs_frame; PO=$T/dense/pairs; L=out; BUD=${BUD:-18}; S1=1500; S2=3000
FR=all/frames_trustmark.json
mkdir -p $L runs
exec >> $L/run_k5_${SLURM_JOB_ID:-x}.log 2>&1
echo "début=$(date) host=$(hostname)"; nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

CVO="--workers 20 --keep_last --lr 3e-4 --lr_backbone 2e-5 --p_rect 0.12 --alpha 0.8 1.2 --synth_n 0 --bs 16"
plis() {  # $1 GPU, $2 méthode, $3 run, $4 départ ({f}), $5 pseudo ({f} ou none), $6 pas, $7 options, $8.. plis
  gpu=$1; m=$2; run=$3; init=$4; pseudo=$5; st=$6; ex=$7; shift 7
  for f in "$@"; do
    [ -f runs/$run/f$f/${m}_test.npz ] && continue                      # déjà fait (autre job)
    mkdir -p runs/$run; mkdir runs/$run/.verrou_${m}_f$f 2>/dev/null || continue   # pris par un autre job
    ini=${init//\{f\}/$f}; k=0
    while [ ! -f $ini ] && [ $k -lt 90 ]; do sleep 20; k=$((k + 1)); done   # départ produit par une autre chaîne
    [ -f $ini ] || { echo "départ absent : $ini"; rmdir runs/$run/.verrou_${m}_f$f; continue; }
    ps=""; [ "$pseudo" != none ] && ps="--pseudo ${pseudo//\{f\}/$f} --pseudo_frac 0.2"
    CUDA_VISIBLE_DEVICES=$gpu python -u train_dense.py --method $m --run $run --fold $f --init ${init//\{f\}/$f} $ps $ex $CVO \
      --steps $st --eval_every $st >> $L/cv_${m}_${run}.log 2>&1
  done
  echo "[$(date +%H:%M)] $m $run fini : $(grep -c 'VAL RÉEL' $L/cv_${m}_${run}.log) plis"
}
base() {  # $1 GPU, $2 méthode : paires avec cadre puis base depuis v2
  mkdir -p runs/k5b; mkdir runs/k5b/.verrou_$2 2>/dev/null || { echo "base $2 prise par un autre job"; return; }
  [ -f $PF/$2/index.json ] || CUDA_VISIBLE_DEVICES=$1 python -u gen_pairs.py --method $2 --reps 2 --rects 2 --with_full --holdout 24 \
      --frame 0.84 --frame_max 1.18 --out $PF > $L/gen_$2.log 2>&1
  tail -n1 $L/gen_$2.log
  CUDA_VISIBLE_DEVICES=$1 python -u train_dense.py --method $2 --run k5b --init $T/dense/runs/v2/$2.pt --pairs $PF --steps 24000 \
      --budget_min $BUD --max_min $((BUD + 10)) --eval_every 4000 --keep_last --workers 20 --synth_n 48 --alpha 0.8 1.2 --p_rect 0.12 --bs 16 \
      > $L/base_$2.log 2>&1
  echo "[$(date +%H:%M)] base $2 finie"; grep -h "VAL SYNTH" $L/base_$2.log | tail -1; grep -h "VAL RÉEL" $L/base_$2.log | tail -1 | cut -c1-230
}

# GPU 0 : PixelSeal — base avec cadre, affinage en CV, pseudo-étiquettes
( base 0 pixelseal
  plis 0 pixelseal k5 "runs/k5b/pixelseal.pt" none $S1 "--pairs $PF" 0 1 2 3 4
  plis 0 pixelseal k5p "runs/k5/f{f}/pixelseal.pt" "runs/k5/f{f}/pixelseal_test.npz" $S2 "--pairs $PF --p_fill 0.5 --p_blur 0.1" 0 1 2 3 4 ) &
# GPU 1 : TrustMark — base avec cadre, affinage en CV (vraies images brutes), puis le même avec mise au cadre
( base 1 trustmark
  plis 1 trustmark k5 "runs/k5b/trustmark.pt" none $S1 "--pairs $PF" 0 1 2 3 4
  plis 1 trustmark k5p "runs/k5/f{f}/trustmark.pt" "runs/k5/f{f}/trustmark_test.npz" $S2 "--pairs $PF --p_fill 0.5 --p_blur 0.1" 0 1 2 3 4 ) &
# GPU 2 : TrustMark mise au cadre — depuis v2 (synthétique aligné), puis pseudo-étiquettes
( plis 2 trustmark k5c "$T/dense/runs/v2/trustmark.pt" none $S1 "--pairs $PO --frames $FR" 0 1 2 3 4
  plis 2 trustmark k5cp "runs/k5c/f{f}/trustmark.pt" "runs/k5c/f{f}/trustmark_test.npz" $S2 "--pairs $PO --frames $FR --p_fill 0.5 --p_blur 0.1" 0 1 2 3 4
  plis 2 trustmark k5c_b "$T/dense/runs/v2/trustmark.pt" none $S1 "--pairs $PO --frames $FR --seed 1" 0 1 2 3 4 ) &
# GPU 3 : TrustMark mise au cadre depuis v3 du pli ; puis PixelSeal, bras rapide (v3 + paires avec cadre)
( plis 3 trustmark k5c3 "$A/runs/v3/f{f}/trustmark.pt" none $S1 "--pairs $PO --frames $FR" 0 1 2 3 4
  while [ ! -f $PF/pixelseal/index.json ]; do sleep 20; done; sleep 20
  plis 3 pixelseal k5q "$A/runs/v3/f{f}/pixelseal.pt" none $S1 "--pairs $PF" 0 1 2 3 4
  [ -f run_extra.sh ] && bash run_extra.sh 3 ) &
sleep 300; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; tail -qn1 $L/base_*.log $L/cv_*.log | cut -c1-160
wait
echo "fin=$(date)"
