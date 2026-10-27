#!/bin/bash
# Usage: sbatch -J mri_eval_<name> mri_crps/jobs/eval.sh <name> <ckpt> [N]
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/share/gpu0/jjwhit/crps_review/paper/mri/logs/%x_%j.out
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
nvidia-smi --query-gpu=name,memory.total --format=csv
cd /home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mri
set -ex
python -u mri_crps/sample.py --name $1 --ckpt $2 --n ${3:-128}
python -u mri_crps/analyse.py --name $1
