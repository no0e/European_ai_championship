#!/bin/bash
# sub.sh <fichier .pt> <note> : envoi avec la clé d équipe (lue dans unlearning/.submit_t3.py, jamais affichée)
T=/p/scratch/training2666/diette1/Sevel_Labs_ENSAI/Hackathon_Watermark_Localisation_Task
cd $T/dense/k5; source $T/.venv/bin/activate; export PYTHONPATH=$T/sam/pylib HF_HUB_OFFLINE=1
export WMLOC_API_KEY=$(python -c "
import re
s=open(\"$T/../unlearning/.submit_t3.py\").read()
m=re.search(r\"API_KEY\s*=\s*[\x22\x27]([^\x22\x27]+)[\x22\x27]\", s)
print(m.group(1) if m else \"\")")
[ -n "$WMLOC_API_KEY" ] || { echo "clé absente"; exit 1; }
python submit_file.py "$1" --note "$2" 2>&1 | tail -2
