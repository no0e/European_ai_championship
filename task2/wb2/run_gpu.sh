#!/bin/bash
# Lance une commande python comme étape « overlap » d'un job déjà alloué de l'équipe (aucun nouveau job).
# usage : [J=jobid] run_gpu.sh <gpu> <args de python...>
J=${J:-15905401}
G=$1; shift
srun --jobid=$J --overlap -n1 --cpus-per-task=48 --gres=gpu:0 env CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$HOME/t2wb/pylib:$HOME/t2wb/code PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 T2W=$HOME/t2wb /p/scratch/training2666/diette1/Sevel_Labs_ENSAI/Hackathon_Watermark_Localisation_Task/.venv/bin/python "$@"
