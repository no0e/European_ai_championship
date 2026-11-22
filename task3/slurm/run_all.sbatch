#!/bin/bash
# Example for a Slurm cluster with 4 GPUs per node (JURECA-DC during the hackathon). Adapt account/partition.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=t3_plat6
#SBATCH --output=t3_plat6_%j.out
# Run from the task3/ folder:  sbatch slurm/run_all.sbatch   (anchors need internet: run step 01 on a login node first)
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} PYTHONUNBUFFERED=1
bash steps/02_synthetic_variants.sh && bash steps/03_train_models.sh && bash steps/04_weight_average.sh
