#!/bin/bash
# Generic 1-GPU job. Usage: sbatch -J <name> [--gres=gpu:v100:1] gpu.sh <script.py> [args...]
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/share/gpu0/jjwhit/crps_review/paper/mri/logs/%x_%j.out
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
export WANDB_MODE=offline
set -x
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
cd /home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mri
python -u "$@"
