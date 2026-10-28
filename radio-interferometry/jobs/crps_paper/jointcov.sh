#!/bin/bash
# Usage: sbatch -J jc_radio_<M> jointcov.sh <M> [extra args]   (M in R F MS MSsig Fnoadv)
#SBATCH -p GPU
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/share/gpu0/jjwhit/crps_review/paper/radio/logs/%x_%j.out
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
export WANDB_MODE=offline
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
# Slurm does not isolate GPUs on these nodes: if the assigned GPU already holds > 1 GB of someone else's memory,
# switch to the highest-index GPU with < 1 GB in use (an invalid index counts as busy).
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  used=$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [[ "$used" =~ ^[0-9]+$ ]] || used=99999
  if [ "$used" -gt 1000 ]; then
    free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1000 {print $1}' | sort -n | tail -1)
    [ -n "$free" ] && echo "assigned GPU $CUDA_VISIBLE_DEVICES busy (${used} MiB); using GPU $free" && export CUDA_VISIBLE_DEVICES=$free
  fi
fi
cd /home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-radio/radio-interferometry/scripts/crps_paper
git log -1 --format=%H
M=$1; shift
python -u jointcov_radio.py --method $M "$@"
