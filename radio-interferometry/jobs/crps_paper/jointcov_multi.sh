#!/bin/bash
# Usage: sbatch -J jc_radio_<x> --cpus-per-task=<n procs> jointcov_multi.sh M:variant [M:variant ...]
# Runs one jointcov_radio.py process per spec concurrently on one GPU (sampling is ~1% of the time; jointcov is single-core numpy).
#SBATCH -p GPU
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/share/gpu0/jjwhit/crps_review/paper/radio/logs/%x_%j.out
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
export WANDB_MODE=offline OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
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
nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=name --format=csv,noheader
cd /home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-radio/radio-interferometry/scripts/crps_paper
git log -1 --format=%H
L=/share/gpu0/jjwhit/crps_review/paper/radio/logs
for spec in "$@"; do
  M=${spec%%:*}; V=${spec#*:}
  python -u jointcov_radio.py --method $M --variant $V > $L/jc_${M}_${V}_${SLURM_JOB_ID}.log 2>&1 &
  sleep 20  # stagger CUDA/cuDNN initialisation
done
wait
tail -n 2 $L/jc_*_${SLURM_JOB_ID}.log
nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.used --format=csv
