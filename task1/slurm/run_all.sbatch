#!/bin/bash
# Example for a Slurm cluster with 4 GPUs per node (JURECA-DC during the hackathon). Adapt account/partition/time.
# Run from the task1/ folder: sbatch slurm/run_all.sbatch   (step 00 needs internet: run it on a login node first)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=10:00:00
#SBATCH --job-name=t1_m5
#SBATCH --output=t1_m5_%j.out
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
for s in 01_signals 02_rect_search 03_sam_segments 04_v2_base 05_v3_cv 06_v4_pseudo 07_long_scale2 08_long_cv \
         09_frame_pairs 10_frame_bases 11_frame_cv 12_assemble; do bash steps/$s.sh; done
