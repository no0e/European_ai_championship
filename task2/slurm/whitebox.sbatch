#!/bin/bash
# Example: white-box steps on one GPU of a Slurm cluster. Adapt account/partition. Run from the task2/ folder.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=t2_whitebox
#SBATCH --output=t2_whitebox_%j.out
export T2_GPU=0
bash steps/07_whitebox_score.sh && bash steps/08_whitebox_signals.sh
