#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=01:40:00
#SBATCH --job-name=t1_k6
#SBATCH --output=out/job_k6.out
#SBATCH --error=out/job_k6.err
#SBATCH --open-mode=append
cd /p/scratch/training2666/diette1/Sevel_Labs_ENSAI/Hackathon_Watermark_Localisation_Task/dense/k5 && bash run_k6.sh
