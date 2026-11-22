#!/bin/bash
# k5 — chaînes supplémentaires lancées par srun --overlap dans un job en cours : extra.sh <chaîne> <GPU>
T=/p/scratch/training2666/diette1/Sevel_Labs_ENSAI/Hackathon_Watermark_Localisation_Task
cd $T/dense/k5 || exit 1
source $T/.venv/bin/activate
export PYTHONPATH=$T/sam/pylib:$PYTHONPATH OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
A=$T/agentA; PF=pairs_frame; PO=$T/dense/pairs; L=out; S1=1500; S2=3000; FR=all/frames_trustmark.json
CVO="--workers 10 --keep_last --lr 3e-4 --lr_backbone 2e-5 --p_rect 0.12 --alpha 0.8 1.2 --synth_n 0 --bs 16"
plis() {  # $1 GPU, $2 méthode, $3 run, $4 départ ({f}), $5 pseudo ({f} ou none), $6 pas, $7 options, $8.. plis
  gpu=$1; m=$2; run=$3; init=$4; pseudo=$5; st=$6; ex=$7; shift 7
  for f in "$@"; do
    [ -f runs/$run/f$f/${m}_test.npz ] && continue
    mkdir -p runs/$run; mkdir runs/$run/.verrou_${m}_f$f 2>/dev/null || continue
    ini=${init//\{f\}/$f}; k=0
    while [ ! -f $ini ] && [ $k -lt 90 ]; do sleep 20; k=$((k + 1)); done
    [ -f $ini ] || { echo "départ absent : $ini"; rmdir runs/$run/.verrou_${m}_f$f; continue; }
    ps=""; [ "$pseudo" != none ] && ps="--pseudo ${pseudo//\{f\}/$f} --pseudo_frac 0.2"
    CUDA_VISIBLE_DEVICES=$gpu python -u train_dense.py --method $m --run $run --fold $f --init $ini $ps $ex $CVO \
      --steps $st --eval_every $st >> $L/cv_${m}_${run}.log 2>&1
  done
  echo "[$(date +%H:%M)] $m $run fini : $(grep -c 'VAL RÉEL' $L/cv_${m}_${run}.log) plis"
}
g=$2
case $1 in
  ps_b)   # PixelSeal : 2e graine du bras rapide (v3_b + paires avec cadre), puis pseudo-étiquettes sur k5q
    plis $g pixelseal k5q_b "$A/runs/v3_b/f{f}/pixelseal.pt" none $S1 "--pairs $PF --seed 1" 0 1 2 3 4
    plis $g pixelseal k5qp "runs/k5q/f{f}/pixelseal.pt" "runs/k5q/f{f}/pixelseal_test.npz" $S2 "--pairs $PF --p_fill 0.5 --p_blur 0.1" 0 1 2 3 4 ;;
  tm_q)   # TrustMark : v3 + paires avec cadre (vraies images brutes), puis la même chose avec mise au cadre des vraies images
    plis $g trustmark k5q "$A/runs/v3/f{f}/trustmark.pt" none $S1 "--pairs $PF" 0 1 2 3 4
    plis $g trustmark k5qc "$A/runs/v3/f{f}/trustmark.pt" none $S1 "--pairs $PF --frames $FR" 0 1 2 3 4
    plis $g trustmark k5q_b "$A/runs/v3_b/f{f}/trustmark.pt" none $S1 "--pairs $PF --seed 1" 0 1 2 3 4 ;;
  ps_p)   # PixelSeal : pseudo-étiquettes du test sur k5q (recette v4)
    plis $g pixelseal k5qp "runs/k5q/f{f}/pixelseal.pt" "runs/k5q/f{f}/pixelseal_test.npz" $S2 "--pairs $PF --p_fill 0.5 --p_blur 0.1" 0 1 2 3 4 ;;
  ps_4)   # PixelSeal : v4 / v4_b du pli + paires avec cadre
    plis $g pixelseal k5q4 "$A/runs/v4/f{f}/pixelseal.pt" none $S1 "--pairs $PF --seed 2" 0 1 2 3 4
    plis $g pixelseal k5q4_b "$A/runs/v4_b/f{f}/pixelseal.pt" none $S1 "--pairs $PF --seed 3" 0 1 2 3 4 ;;
  a3b)    # base deuxième échelle (A3) : poids « long » de l'agent A, repris sur les paires avec cadre, puis affinage en CV
    m=$3
    if mkdir runs/k6b_verrou_$m 2>/dev/null; then
      mkdir -p runs/k6b
      CUDA_VISIBLE_DEVICES=$g python -u train_dense.py --method $m --run k6b --init $A/runs/long/$m.pt --scale2 --pairs $PF --steps 24000 \
        --budget_min ${BUD6:-13} --max_min 20 --eval_every 24000 --keep_last --workers 16 --synth_n 0 --alpha 0.8 1.2 --p_rect 0.12 --bs 8 > $L/base6_$m.log 2>&1
      echo "[$(date +%H:%M)] base A3 $m finie"; grep -h "VAL RÉEL" $L/base6_$m.log | tail -1 | cut -c1-230
    fi
    plis $g $m k6 "runs/k6b/$m.pt" none $S1 "--pairs $PF --scale2 --bs 8" 0 1 2 3 4 ;;
  a3f)    # affinage en CV de la base A3, plis dans l'ordre inverse (autre GPU)
    m=$3; plis $g $m k6 "runs/k6b/$m.pt" none $S1 "--pairs $PF --scale2 --bs 8" 4 3 2 1 0 ;;
  a3g)    # affinage en CV de la base A3, plis dans un troisième ordre
    m=$3; plis $g $m k6 "runs/k6b/$m.pt" none $S1 "--pairs $PF --scale2 --bs 8" 2 3 1 4 0 ;;
  ssb)    # StegaStamp : base sur paires avec cadre (depuis v2), puis affinage en CV
    TW="--timm_weights $T/dense/weights/convnext_tiny.safetensors"
    if mkdir runs/k5b/.verrou_stegastamp 2>/dev/null; then
      while [ ! -f $PF/stegastamp/index.json ]; do sleep 15; done; sleep 10
      CUDA_VISIBLE_DEVICES=$g python -u train_dense.py --method stegastamp --run k5b --init $T/dense/runs/v2/stegastamp.pt $TW --pairs $PF --steps 24000 \
        --budget_min ${BUDS:-14} --max_min 22 --eval_every 24000 --keep_last --workers 16 --synth_n 0 --alpha 0.8 1.2 --p_rect 0.12 --bs 16 > $L/base_stegastamp.log 2>&1
      echo "[$(date +%H:%M)] base stegastamp finie"; grep -h "VAL RÉEL" $L/base_stegastamp.log | tail -1 | cut -c1-230
    fi
    plis $g stegastamp k5 "runs/k5b/stegastamp.pt" none $S1 "--pairs $PF $TW" 0 1 2 3 4 ;;
  psb2)   # PixelSeal : la base k5b n'a eu que 4 500 pas — on la prolonge (k7b), puis affinage en CV (k7)
    if mkdir runs/k7b_verrou 2>/dev/null; then
      CUDA_VISIBLE_DEVICES=$g python -u train_dense.py --method pixelseal --run k7b --init runs/k5b/pixelseal.pt --pairs $PF --steps 24000 \
        --budget_min ${BUD7:-16} --max_min 24 --eval_every 24000 --keep_last --workers 16 --synth_n 0 --alpha 0.8 1.2 --p_rect 0.12 --bs 16 --lr 5e-4 --lr_backbone 3e-5 > $L/base7_pixelseal.log 2>&1
      echo "[$(date +%H:%M)] base k7b pixelseal finie"; grep -h "VAL RÉEL" $L/base7_pixelseal.log | tail -1 | cut -c1-230
    fi
    plis $g pixelseal k7 "runs/k7b/pixelseal.pt" none $S1 "--pairs $PF" 0 1 2 3 4 ;;
  ss)     # StegaStamp : paires avec cadre puis v3 + ces paires
    [ -f $PF/stegastamp/index.json ] || CUDA_VISIBLE_DEVICES=$g python -u gen_pairs.py --method stegastamp --reps 2 --rects 2 --with_full --holdout 24 \
        --frame 0.84 --frame_max 1.18 --out $PF > $L/gen_stegastamp.log 2>&1
    tail -n1 $L/gen_stegastamp.log
    plis $g stegastamp k5q "$A/runs/v3/f{f}/stegastamp.pt" none $S1 "--pairs $PF --timm_weights $T/dense/weights/convnext_tiny.safetensors" 0 1 2 3 4 ;;
esac
