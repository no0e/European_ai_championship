#!/bin/bash
# Tâche 1, agent A — entraînement long PixelSeal / TrustMark (A1 + A2 + A3), avec TÉMOIN sans A3.
#
# Les quatre processus reçoivent le MÊME budget en minutes ; chacun mesure son coût par pas sur les 200
# premiers pas et ramène son nombre de pas (et le recuit du pas d'apprentissage) à ce qui tient dedans.
# La comparaison est donc « le meilleur modèle qu'on obtient en un temps donné, avec et sans A3 »,
# pas « à nombre de pas égal » : c'est la comparaison utile pour décider ce qu'on livre à 05 h 30.
#
#   cd <dossier du script> && mkdir -p out && squeue -u diette1,guillet1,dahan1,lamirand1
#   sbatch --dependency=afterany:<TOUS les jobs listés> job_long.sh
# Variables : RUN (long), PAIRS (pairs_long), REPS (2), GEN (1), HOLD (24), DEADLINE (05:30),
#             STEPS (40000, plafond), BS_A3 (8), BS_W (16), EXTRA.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=03:30:00
#SBATCH --job-name=t1_long
#SBATCH --output=out/job_long.out
#SBATCH --error=out/job_long.err
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
RUN=${RUN:-long}; PAIRS=${PAIRS:-pairs_long}; REPS=${REPS:-2}; HOLD=${HOLD:-24}
STEPS=${STEPS:-40000}; DEADLINE=${DEADLINE:-05:30}; BS_A3=${BS_A3:-8}; BS_W=${BS_W:-16}; EXTRA=${EXTRA:-}
L=out/$RUN
# Le quota de fichiers du projet partagé sature par moments : attendre que l'écriture fonctionne (10 min max).
for k in $(seq 1 60); do mkdir -p $L runs/$RUN runs/${RUN}_w 2>/dev/null && touch $L/.ok 2>/dev/null && break; echo "écriture refusée, attente ($k)"; sleep 10; done
echo "RUN=$RUN PAIRS=$PAIRS STEPS=$STEPS échéance=$DEADLINE host=$(hostname) début=$(date)"
df -h /p/scratch/training2666 | tail -1

# --- Paires : réservoir doublé (images full des deux autres méthodes) + images tenues à l'écart.
# Compte ≈ 1020 images de base par méthode × REPS ; à REPS=2, x.npy + w.npy ≈ 3,2 Go par méthode.
# On NE touche pas à pairs/ (lien en lecture seule vers les paires de la nuit) : nouveau dossier.
if [ "${GEN:-1}" = 1 ] && [ ! -f $PAIRS/trustmark/index.json ]; then
  CUDA_VISIBLE_DEVICES=0 python -u gen_pairs.py --method pixelseal --reps $REPS --rects 2 --with_full --holdout $HOLD --out $PAIRS > $L/gen_pixelseal.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python -u gen_pairs.py --method trustmark --reps $REPS --rects 2 --with_full --holdout $HOLD --out $PAIRS > $L/gen_trustmark.log 2>&1 &
  wait
  head -qn1 $L/gen_*.log; tail -qn1 $L/gen_*.log; du -sh $PAIRS; df -h /p/scratch/training2666 | tail -1; date
fi
for m in pixelseal trustmark; do
  [ -f $PAIRS/$m/index.json ] || { echo "ABANDON : $PAIRS/$m/index.json absent (voir $L/gen_$m.log)"; exit 1; }
done

# --- Budget : ce qui reste jusqu'à l'échéance, moins 25 min pour les prédictions test et la marge.
NOW=$(date +%s); END=$(date -d "today $DEADLINE" +%s)
[ $END -le $NOW ] && END=$(date -d "tomorrow $DEADLINE" +%s)
BUD=$(( (END - NOW) / 60 - 25 ))
[ $BUD -lt 20 ] && BUD=20
echo "budget d'entraînement : $BUD min par processus (échéance $DEADLINE, il est $(date +%H:%M))"

# A1a (formes et nombre de trous tirés du réel) est dans common.py, sans option. A1b = --alpha.
# A1c (--p_wdeg) est volontairement ABSENT : hypothèse non tranchée, à n'ajouter que si A0 le demande.
COMMON="--pairs $PAIRS --steps $STEPS --budget_min $BUD --max_min $((BUD + 15)) --eval_every 2000 \
 --workers 20 --synth_n 48 --synth_floor 0.75 --synth_floor_step 20000 --alpha 0.8 1.2 --p_rect 0.12 $EXTRA"

# GPU 0-1 : A1 + A2 + A3 (deuxième échelle, carte fine 128 au lieu de 64), lot réduit par précaution mémoire.
# GPU 2-3 : les mêmes SANS A3 — témoin, et repli sûr si la deuxième échelle dégrade.
CUDA_VISIBLE_DEVICES=0 python -u train_dense.py --method pixelseal --run $RUN     --init runs/v2/pixelseal.pt --scale2 --bs $BS_A3 $COMMON > $L/pixelseal_a3.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u train_dense.py --method trustmark --run $RUN     --init runs/v2/trustmark.pt --scale2 --bs $BS_A3 $COMMON > $L/trustmark_a3.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u train_dense.py --method pixelseal --run ${RUN}_w --init runs/v2/pixelseal.pt          --bs $BS_W  $COMMON > $L/pixelseal_w.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u train_dense.py --method trustmark --run ${RUN}_w --init runs/v2/trustmark.pt          --bs $BS_W  $COMMON > $L/trustmark_w.log 2>&1 &
sleep 420; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv; tail -qn3 $L/*_a3.log $L/*_w.log
wait
echo "--- validation synthétique (images de base jamais vues) ---"; grep -h "VAL SYNTH" $L/*_a3.log $L/*_w.log | tail -12
echo "--- validation réelle ---";                                   grep -h "VAL RÉEL"  $L/*_a3.log $L/*_w.log | cut -c1-200 | tail -12
echo "--- budget mesuré ---";                                       grep -h "budget "   $L/*_a3.log $L/*_w.log
ls -la runs/$RUN runs/${RUN}_w
echo "fin=$(date)"
