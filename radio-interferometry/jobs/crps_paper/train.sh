#!/bin/bash
# Usage: sbatch -J tr_radio_<M> radio-interferometry/jobs/crps_paper/train.sh <M> [--resume --resume-epoch E]
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/share/gpu0/jjwhit/crps_review/paper/radio/logs/%x_%j.out
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
export WANDB_MODE=offline WANDB_DIR=/share/gpu0/jjwhit/crps_review/paper/radio/wandb
mkdir -p $WANDB_DIR
M=$1; shift
nvidia-smi --query-gpu=name,memory.total --format=csv
cd /home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-radio
git rev-parse HEAD
python -u train.py --config configs/crps_paper/radio_sim_$M.yaml --exp-name radio_sim_$M --num-gpus 1 "$@"
