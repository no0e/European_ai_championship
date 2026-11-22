#!/bin/bash
# Tâche 1, détecteur dense. UN SEUL NŒUD (règle de l'administration) : 4 processus, un par GPU.
#   cd $T/dense && mkdir -p out && sbatch job_dense.sh
# Variables : RUN (v1), STEPS (6000), GEN (1 = générer les paires si absentes), REPS (2), EXTRA (args train).
# Avant de soumettre : squeue -u diette1,guillet1,dahan1,lamirand1 doit être vide.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=01:30:00
#SBATCH --job-name=t1_dense
#SBATCH --output=out/job_dense.out
#SBATCH --error=out/job_dense.err
#SBATCH --open-mode=append

T=/p/scratch/training2666/diette1/Sevel_Labs_ENSAI/Hackathon_Watermark_Localisation_Task
cd $T/dense || exit 1
source $T/.venv/bin/activate
export PYTHONPATH=$T/sam/pylib:$PYTHONPATH OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
RUN=${RUN:-v1}; STEPS=${STEPS:-6000}; REPS=${REPS:-2}; EXTRA=${EXTRA:---p_rect 0.12}
L=out/$RUN
# Le quota de fichiers du projet partagé sature par moments : attendre que l'écriture fonctionne (10 min max).
for k in $(seq 1 60); do mkdir -p $L runs/$RUN runs/${RUN}_b 2>/dev/null && touch $L/.ok 2>/dev/null && break; echo "écriture refusée, attente ($k)"; sleep 10; done
echo "RUN=$RUN STEPS=$STEPS host=$(hostname) début=$(date)"

if [ "${GEN:-1}" = 1 ] && [ ! -f pairs/stegastamp/index.json ]; then
  i=0
  for m in pixelseal trustmark stegastamp; do
    CUDA_VISIBLE_DEVICES=$i python -u gen_pairs.py --method $m --reps $REPS --rects 2 > $L/gen_$m.log 2>&1 &
    i=$((i + 1))
  done
  wait
  tail -qn1 $L/gen_*.log; du -sh pairs; date
fi

CUDA_VISIBLE_DEVICES=0 python -u train_dense.py --method pixelseal  --run $RUN --steps $STEPS --workers 20 $EXTRA > $L/train_pixelseal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u train_dense.py --method trustmark  --run $RUN --steps $STEPS --workers 20 $EXTRA > $L/train_trustmark.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u train_dense.py --method stegastamp --run $RUN --steps $STEPS --workers 20 --timm_weights weights/convnext_tiny.safetensors $EXTRA > $L/train_stegastamp.log 2>&1 &
# 4e GPU : deuxième graine TrustMark (la méthode la plus difficile), pour moyenner les deux modèles
CUDA_VISIBLE_DEVICES=3 python -u train_dense.py --method trustmark --seed 1 --run ${RUN}_b --steps $STEPS --workers 20 $EXTRA > $L/train_trustmark_b.log 2>&1 &
sleep 240; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv; tail -qn1 $L/train_*.log
wait
grep -h "VAL RÉEL" $L/train_*.log | tail -8
echo "fin=$(date)"
