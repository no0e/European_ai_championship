#!/bin/bash
# Tâche 1, agent A — étape A7 : finition du modèle long. Deux phases dans un seul job.
#   phase 1 (recette v3) : affinage en CV 5 plis sur les vraies images, à partir des poids du job_long ;
#   phase 2 (recette v4) : un tour de pseudo-étiquettes du test, depuis les poids de la phase 1 DU MÊME PLI.
# Pas de deuxième tour (v5 mesuré à −0,0007 la nuit du 21 au 22).
#
# Seules PixelSeal et TrustMark sont refaites : StegaStamp garde ses v3 / v4 (déjà à 0,90). Les 5 plis sont
# donc répartis sur 4 GPU (2 par méthode) au lieu de 1, ce qui divise la durée par deux.
#
#   cd <dossier du script> && squeue -u diette1,guillet1,dahan1,lamirand1
#   sbatch --dependency=afterany:<jobs listés> --export=ALL,SRC_PS=long,SRC_TM=long_w job_cv_long.sh
# Variables : SRC (long ou long_w) et, pour un bras différent par méthode, SRC_PS / SRC_TM.
#             `--scale2` est DÉDUIT de la source (un dossier en `_w` = témoin sans A3) : on ne lit
#             jamais des poids avec une architecture qui n'est pas celle de leur entraînement.
#             RUN (v6), PAIRS (pairs_long), S1 (1500), S2 (3000), DEADLINE (09:00), PHASE2 (1).
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=02:30:00
#SBATCH --job-name=t1_cvlong
#SBATCH --output=out/job_cv_long.out
#SBATCH --error=out/job_cv_long.err
#SBATCH --open-mode=append

T=/p/scratch/training2666/diette1/Sevel_Labs_ENSAI/Hackathon_Watermark_Localisation_Task
# Le code de l'agent A ne vit pas forcément dans $T/final (qui appartient à guillet1) : on se place
# dans le dossier du script lui-même, d'où qu'il soit lancé.
# Slurm recopie le script dans son spool : $0 n'y designe plus le dossier d'origine. C'est
# $SLURM_SUBMIT_DIR qui donne le dossier d'ou sbatch a ete lance ; $0 ne sert que hors Slurm.
D=${D:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}}
cd "$D" || exit 1
source $T/.venv/bin/activate
export PYTHONPATH=$T/sam/pylib:$PYTHONPATH OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
SRC=${SRC:-long}; SRC_PS=${SRC_PS:-$SRC}; SRC_TM=${SRC_TM:-$SRC}
RUN=${RUN:-v6}; PAIRS=${PAIRS:-pairs_long}
S1=${S1:-1500}; S2=${S2:-3000}; DEADLINE=${DEADLINE:-09:00}; BS=${BS:-8}
# un dossier qui finit par _w est le témoin, entraîné SANS la deuxième échelle
sc_de() { case "$1" in *_w) echo "" ;; *) echo "--scale2" ;; esac; }
SC_PS=$(sc_de "$SRC_PS"); SC_TM=$(sc_de "$SRC_TM")
L=out/$RUN
for k in $(seq 1 60); do mkdir -p $L runs/$RUN runs/${RUN}p 2>/dev/null && touch $L/.ok 2>/dev/null && break; echo "écriture refusée, attente ($k)"; sleep 10; done
echo "PixelSeal : runs/$SRC_PS ${SC_PS:-(sans A3)} | TrustMark : runs/$SRC_TM ${SC_TM:-(sans A3)}"
echo "RUN=$RUN PAIRS=$PAIRS host=$(hostname) début=$(date)"
[ -f runs/$SRC_PS/pixelseal.pt ] || { echo "ABANDON : runs/$SRC_PS/pixelseal.pt absent"; exit 1; }
[ -f runs/$SRC_TM/trustmark.pt ] || { echo "ABANDON : runs/$SRC_TM/trustmark.pt absent"; exit 1; }

reste() {  # minutes jusqu'à l'échéance
  NOW=$(date +%s); END=$(date -d "today $DEADLINE" +%s)
  [ $END -le $NOW ] && END=$(date -d "tomorrow $DEADLINE" +%s)
  echo $(( (END - NOW) / 60 ))
}

COMMON="--pairs $PAIRS --bs $BS --workers 20 --keep_last --lr 3e-4 --lr_backbone 2e-5 \
 --p_rect 0.12 --alpha 0.8 1.2 --synth_n 0"

# ---------------------------------------------------------------- phase 1 : recette v3
plis() {  # $1 GPU, $2 méthode, $3 run, $4 départ ({f} = pli), $5 pseudo ({f} ou none), $6 archi, $7.. plis
  gpu=$1; m=$2; run=$3; init=$4; pseudo=$5; sc=$6; shift 6
  for f in "$@"; do
    ps=""; [ "$pseudo" != none ] && ps="--pseudo ${pseudo//\{f\}/$f} --pseudo_frac 0.2"
    CUDA_VISIBLE_DEVICES=$gpu python -u train_dense.py --method $m --run $run --fold $f \
      --init ${init//\{f\}/$f} $ps $sc $PH_EXTRA $COMMON --steps $PH_STEPS --eval_every $PH_STEPS \
      >> $L/cv_${m}_${run}_g$gpu.log 2>&1
  done
}

PH_STEPS=$S1; PH_EXTRA=""
echo "phase 1 (recette v3, $S1 pas) — $(reste) min avant $DEADLINE — $(date +%H:%M)"
plis 0 pixelseal $RUN "runs/$SRC_PS/pixelseal.pt" none "$SC_PS" 0 1 2 &
plis 1 pixelseal $RUN "runs/$SRC_PS/pixelseal.pt" none "$SC_PS" 3 4   &
plis 2 trustmark $RUN "runs/$SRC_TM/trustmark.pt" none "$SC_TM" 0 1 2 &
plis 3 trustmark $RUN "runs/$SRC_TM/trustmark.pt" none "$SC_TM" 3 4   &
sleep 300; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv; tail -qn1 $L/cv_*.log
wait
echo "--- phase 1 terminée $(date +%H:%M) ---"; grep -h "VAL RÉEL" $L/cv_*_${RUN}_*.log | cut -c1-200 | tail -10
for f in 0 1 2 3 4; do for m in pixelseal trustmark; do
  [ -f runs/$RUN/f$f/${m}_test.npz ] || echo "MANQUE runs/$RUN/f$f/${m}_test.npz"
done; done

# ---------------------------------------------------------------- phase 2 : recette v4 (pseudo-étiquettes)
R=$(reste)
if [ "${PHASE2:-1}" != 1 ]; then echo "phase 2 désactivée"; exit 0; fi
if [ "$R" -lt 45 ]; then echo "phase 2 abandonnée : $R min avant $DEADLINE, il en faut ~45"; exit 0; fi
PH_STEPS=$S2; PH_EXTRA="--p_fill 0.5 --p_blur 0.1"
echo "phase 2 (recette v4, $S2 pas) — $R min avant $DEADLINE — $(date +%H:%M)"
plis 0 pixelseal ${RUN}p "runs/$RUN/f{f}/pixelseal.pt" "runs/$RUN/f{f}/pixelseal_test.npz" "$SC_PS" 0 1 2 &
plis 1 pixelseal ${RUN}p "runs/$RUN/f{f}/pixelseal.pt" "runs/$RUN/f{f}/pixelseal_test.npz" "$SC_PS" 3 4   &
plis 2 trustmark ${RUN}p "runs/$RUN/f{f}/trustmark.pt" "runs/$RUN/f{f}/trustmark_test.npz" "$SC_TM" 0 1 2 &
plis 3 trustmark ${RUN}p "runs/$RUN/f{f}/trustmark.pt" "runs/$RUN/f{f}/trustmark_test.npz" "$SC_TM" 3 4   &
wait
echo "--- phase 2 terminée ---"; grep -h "VAL RÉEL" $L/cv_*_${RUN}p_*.log | cut -c1-200 | tail -10
ls runs/$RUN/f*/ runs/${RUN}p/f*/
echo "fin=$(date)"
