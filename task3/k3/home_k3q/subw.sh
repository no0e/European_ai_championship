#!/bin/bash
# attend que les fichiers existent puis les soumet dans l ordre : subw.sh f1 f2 ...
cd /p/scratch/training2666/diette1/Sevel_Labs_ENSAI/unlearning; source .venv/bin/activate
for f in "$@"; do until [ -s "$f" ] && [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 20 ]; do sleep 5; done; python k3/submit.py "$f"; done
