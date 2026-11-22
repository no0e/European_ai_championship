#!/bin/bash
# Tâche 1, détecteur dense — v4 : affinage en CV (comme v3) en partant des poids v3 DU MÊME PLI, avec
#   - images du test pseudo-étiquetées par le modèle v3 du même pli (il n'a pas vu ce pli : CV honnête),
#     décision top-k, bande de 2 cellules autour de la frontière ignorée ;
#   - composites plus réalistes : zone non tatouée prise dans une autre image (p_fill) ou lissée (p_blur) ;
#   - 3000 pas au lieu de 1500.
# GPU3 : 2e graine TrustMark, puis ablation PixelSeal SANS pseudo-étiquettes (effet propre des pseudo-étiquettes).
# Code dans $T/final (copie de la branche task1-final) ; runs/v3* = liens vers ../dense/runs (lecture seule).
#   cd $T/final && sbatch --dependency=afterany:<job d'équipe en cours> job_v4.sh
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=01:10:00
#SBATCH --job-name=t1_v4
#SBATCH --output=out/job_v4.out
#SBATCH --error=out/job_v4.err
#SBATCH --open-mode=append

T=/p/scratch/training2666/diette1/Sevel_Labs_ENSAI/Hackathon_Watermark_Localisation_Task
cd $T/final || exit 1
source $T/.venv/bin/activate
export PYTHONPATH=$T/sam/pylib:$PYTHONPATH OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
RUN=${RUN:-v4}; STEPS=${STEPS:-3000}
L=out/$RUN
for k in $(seq 1 60); do mkdir -p $L runs/$RUN 2>/dev/null && touch $L/.ok 2>/dev/null && break; echo "écriture refusée, attente ($k)"; sleep 10; done
echo "RUN=$RUN STEPS=$STEPS host=$(hostname) début=$(date)"
COMMON="--steps $STEPS --workers 20 --keep_last --eval_every $STEPS --lr 3e-4 --lr_backbone 2e-5 --p_rect 0.12 --p_fill 0.5 --p_blur 0.1"

seq_run() {  # $1 GPU, $2 méthode, $3 run, $4 poids de départ ({f} = pli), $5 pseudo-étiquettes ({f} = pli, ou none), $6.. args
  gpu=$1; m=$2; run=$3; init=$4; pseudo=$5; shift 5
  for f in 0 1 2 3 4; do
    ps=""; [ "$pseudo" != none ] && ps="--pseudo ${pseudo//\{f\}/$f} --pseudo_frac 0.2"
    CUDA_VISIBLE_DEVICES=$gpu python -u train_dense.py --method $m --run $run --fold $f --init ${init//\{f\}/$f} \
      $COMMON $ps "$@" >> $L/cv_${m}_$(basename $run).log 2>&1
  done
}
TMPS="runs/v3/f{f}/trustmark_test.npz runs/v3_b/f{f}/trustmark_test.npz"
seq_run 0 pixelseal  $RUN "runs/v3/f{f}/pixelseal.pt"  "runs/v3/f{f}/pixelseal_test.npz" &
seq_run 1 trustmark  $RUN "runs/v3/f{f}/trustmark.pt"  "$TMPS" &
seq_run 2 stegastamp $RUN "runs/v3/f{f}/stegastamp.pt" "runs/v3/f{f}/stegastamp_test.npz" --timm_weights weights/convnext_tiny.safetensors &
( seq_run 3 trustmark ${RUN}_b  "runs/v3_b/f{f}/trustmark.pt" "$TMPS" --seed 1
  seq_run 3 pixelseal ${RUN}_np "runs/v3/f{f}/pixelseal.pt" none ) &
sleep 300; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv; tail -qn1 $L/cv_*.log
wait
grep -h "VAL RÉEL" $L/cv_*.log | cut -c1-200 | tail -25
echo "fin=$(date)"
