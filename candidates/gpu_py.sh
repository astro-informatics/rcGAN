#!/bin/bash
# Usage: sbatch -J cand_<name> candidates/gpu_py.sh <script.py> [args]
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=compute-gpu-0-5
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/share/gpu0/jjwhit/crps_review/candidates/logs/%x_%j.out
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
# busy-GPU workaround: if the assigned GPU already has > 1 GB in use, switch to the highest-index GPU with < 1 GB in use
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  used=$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [[ "$used" =~ ^[0-9]+$ ]] || used=99999
  if [ "$used" -gt 1000 ]; then
    free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1000 {print $1}' | sort -n | tail -1)
    [ -n "$free" ] && echo "assigned GPU $CUDA_VISIBLE_DEVICES busy (${used} MiB); using GPU $free" && export CUDA_VISIBLE_DEVICES=$free
  fi
fi
nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=name,memory.total,memory.used --format=csv
cd /home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mri
git rev-parse HEAD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
set -x

python -u "$@"
