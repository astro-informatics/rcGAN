#!/bin/bash
# Generic 1xA100 job for the CRPS review. Usage: sbatch -J <name> review_gpu.sh <script.py> [args...]
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/share/gpu0/jjwhit/crps_review/logs/%x_%j.out
#SBATCH --error=/share/gpu0/jjwhit/crps_review/logs/%x_%j.err

source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
export WANDB_MODE=disabled
set -x
nvidia-smi --query-gpu=name,memory.total --format=csv
cd /home/jjwhit/.herdr/worktrees/rcGAN/crps-review/mass_map_utils/scripts/review
python -u "$@"
