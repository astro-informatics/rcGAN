#!/bin/bash
# Usage: sbatch -J <name> --gres=gpu:1 job_eval.sh <stageA|joint|py> <script.py> [args...]
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/share/gpu0/jjwhit/crps_review/paper/mm/logs/%x_%j.out
#SBATCH --error=/share/gpu0/jjwhit/crps_review/paper/mm/logs/%x_%j.err
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
WT=/home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mm
export PYTHONPATH=$WT
set -x
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
cd $WT/mass_map_utils/scripts/paper_mm
git log -1 --format=%H
# Slurm does not isolate GPUs on these nodes: if the assigned GPU already holds > 1 GB of someone else's memory,
# switch to the highest-index GPU with < 1 GB in use.
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  used=$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${used:-0}" -gt 1000 ]; then
    free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1000 {print $1}' | sort -n | tail -1)
    [ -n "$free" ] && echo "assigned GPU $CUDA_VISIBLE_DEVICES busy (${used} MiB); using GPU $free" && export CUDA_VISIBLE_DEVICES=$free
  fi
fi
if [ "$1" = py ]; then shift; python -u "$@"; else python -u mm_run.py "$@"; fi
