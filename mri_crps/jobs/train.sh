#!/bin/bash
# Usage: sbatch -J mri_<name> mri_crps/jobs/train.sh <exp-name> <reg> [extra train.py args]
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=23:30:00
#SBATCH --requeue
#SBATCH --output=/share/gpu0/jjwhit/crps_review/paper/mri/logs/%x_%j.out
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
nvidia-smi --query-gpu=name,memory.total --format=csv
cd /home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mri
git rev-parse HEAD
set -x
exp=$1; reg=$2; shift 2
python -u mri_crps/train.py --config configs/mri_crps.yml --exp-name $exp --reg $reg --max-time 00:23:00:00 "$@"
