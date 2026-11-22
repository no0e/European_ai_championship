#!/bin/bash
# Tâche 1, méthode « propositions d'objets » (SAM). UN SEUL NŒUD (règle de l'administration du hackathon) :
# 4 processus en parallèle sur les 4 GPU du nœud, puis l'évaluation CPU sur le même nœud.
#   cd $T/sam && mkdir -p out && sbatch job_sam.sh                 # train : caractéristiques + 3 tests
#   cd $T/sam && sbatch --export=ALL,SPLIT=test job_sam.sh        # test : caractéristiques seulement
# Avant de soumettre : vérifier qu'aucun autre job de l'équipe ne tourne (squeue -u <logins de l'équipe>).
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=t1_sam
#SBATCH --output=out/job_%j.out
#SBATCH --error=out/job_%j.err

T=/p/scratch/training2666/diette1/Sevel_Labs_ENSAI/Hackathon_Watermark_Localisation_Task
cd $T/sam || exit 1
source $T/.venv/bin/activate
export PYTHONPATH=$T/sam/pylib:$PYTHONPATH OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
SPLIT=${SPLIT:-train}
echo "SPLIT=$SPLIT host=$(hostname) début=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k python -u sam_feats.py --split $SPLIT --shard $k/4 > out/feats_${SPLIT}_$k.log 2>&1 &
done
sleep 120; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv; tail -n 2 out/feats_${SPLIT}_*.log
wait
echo "caractéristiques finies : $(ls out/$SPLIT | wc -l) images ; $(date)"
grep -h -E "Traceback|Error" out/feats_${SPLIT}_*.log | head -5

if [ "$SPLIT" = train ]; then
  python -u sam_eval.py 2>&1 | tee out/report_train.txt
fi
echo "fin=$(date)"
