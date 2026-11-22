#!/bin/bash
# Tâche 1, détecteur dense — étape 2 : affinage en CV 5 plis avec les vraies images du train (un seul nœud).
#   cd $T/dense && sbatch --export=ALL,RUN=v3,INIT=v2 job_cv.sh
# Par méthode et par pli : départ des poids runs/$INIT, lots moitié synthétiques / moitié vraies images des 4
# autres plis ; validation sur le pli (hors-pli honnête, dernier pas gardé) ; test = moyenne des 5 plis.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00
#SBATCH --job-name=t1_cv
#SBATCH --output=out/job_cv.out
#SBATCH --error=out/job_cv.err
#SBATCH --open-mode=append

T=/p/scratch/training2666/diette1/Sevel_Labs_ENSAI/Hackathon_Watermark_Localisation_Task
cd $T/dense || exit 1
source $T/.venv/bin/activate
export PYTHONPATH=$T/sam/pylib:$PYTHONPATH OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
RUN=${RUN:-v3}; INIT=${INIT:-v2}; STEPS=${STEPS:-1500}; EXTRA=${EXTRA:---p_rect 0.12}
L=out/$RUN
for k in $(seq 1 60); do mkdir -p $L runs/$RUN 2>/dev/null && touch $L/.ok 2>/dev/null && break; echo "écriture refusée, attente ($k)"; sleep 10; done
echo "RUN=$RUN INIT=$INIT STEPS=$STEPS host=$(hostname) début=$(date)"
COMMON="--steps $STEPS --workers 20 --keep_last --eval_every $STEPS --lr 3e-4 --lr_backbone 2e-5 $EXTRA"

seq_run() {  # $1 = GPU, $2 = méthode, $3 = dossier de run, $4 = poids de départ, $5.. = args en plus
  gpu=$1; m=$2; run=$3; init=$4; shift 4
  for f in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$gpu python -u train_dense.py --method $m --run $run --fold $f --init $init $COMMON "$@" >> $L/cv_${m}_$(basename $run).log 2>&1
  done
}
seq_run 0 pixelseal  $RUN   runs/$INIT/pixelseal.pt &
seq_run 1 trustmark  $RUN   runs/$INIT/trustmark.pt &
seq_run 2 stegastamp $RUN   runs/$INIT/stegastamp.pt --timm_weights weights/convnext_tiny.safetensors &
seq_run 3 trustmark  ${RUN}_b runs/${INIT}_b/trustmark.pt --seed 1 &
wait
grep -h "VAL RÉEL" $L/cv_*.log | cut -c1-200 | tail -20
echo "fin=$(date)"
