#!/bin/bash
# asm.sh <nom> <dossiers de runs...> : assemblage (8 filtres guidés, rectangles TM), CV sauvée, comparaison à la référence n° 2170.
T=/p/scratch/training2666/diette1/Sevel_Labs_ENSAI/Hackathon_Watermark_Localisation_Task
cd $T/dense/k5; source $T/.venv/bin/activate
export PYTHONPATH=$T/sam/pylib HF_HUB_OFFLINE=1 OMP_NUM_THREADS=${OMP:-4}
n=$1; shift
nice -n 5 python -u assemble.py --runs "$@" --rects "../../paulin/out/rect_tm_test_s*.json" \
  --post none c:8:0.001 c:8:0.01 c:16:0.01 c:16:0.001 c:16:0.0001 c:24:0.001 c:8:0.0001 --out out/sub_$n.pt --save_cv cv/cv_$n.npz 2>&1 | tail -16
python -u compare_cv.py $T/agentA/cv/cv_ref.npz cv/cv_$n.npz --gate 2>&1 | tail -18
echo "asm $n fini $(date +%H:%M)"
