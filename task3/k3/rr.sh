#!/bin/bash
# Distillation réelle : rr.sh NOM "options tg.py" "options distill"   (réessaie si le quota de fichiers du projet sature)
cd /p/scratch/training2666/diette1/Sevel_Labs_ENSAI/unlearning/k3
for try in 1 2 3 4 5 6; do
  python tg.py --env real $2 --out tgt_real/$1.pt > logs/real_tg_$1.log 2>&1 && python k3.py distill --env real --targets k3/tgt_real/$1.pt $3 --out sub/$1.pt > logs/real_$1.log 2>&1
  [ -s sub/$1_a1.0.pt ] && break
  echo "$(date +%H:%M:%S) $1 : échec (essai $try), nouvel essai dans 60 s" >> logs/retry.log; sleep 60
done
