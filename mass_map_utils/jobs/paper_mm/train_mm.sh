#!/bin/bash
# paper-mm training job: identical to crps_sweep/train_alpha_100.sh (F100) except the output dir,
# offline wandb and the extra --crps-mode / --adv-scale arguments.
# Usage: sbatch -J tr_<name> --gres=gpu:<type>:4 train_mm.sh <exp-name> <extra train.py args...>
#SBATCH -p GPU
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=/share/gpu0/jjwhit/crps_review/paper/mm/logs/%x_%j.out
#SBATCH --error=/share/gpu0/jjwhit/crps_review/paper/mm/logs/%x_%j.err
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
export WANDB_MODE=offline
export WANDB_DIR=/share/gpu0/jjwhit/crps_review/paper/mm/wandb
mkdir -p $WANDB_DIR
WT=/home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mm
export PYTHONPATH=$WT
set -x
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
cd $WT
git log -1 --format=%H
# log GPU memory/utilisation every 10 min (diagnostics; GPUs on these nodes are not cgroup-isolated)
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 600 > /share/gpu0/jjwhit/crps_review/paper/mm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv &
echo "slurm CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# Slurm does not isolate GPU devices on these nodes (other users' processes sit on the "allocated" indices).
# With PICK_FREE_GPUS=1, use the 4 highest-index GPUs that have < 1 GB in use instead.
if [ "$PICK_FREE_GPUS" = 1 ]; then
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1000 {print $1}' | sort -n | tail -4 | sort -n | paste -sd,)
  echo "picked CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
EXP=$1; shift
srun python -u train.py --config ./configs/paper_mm.yml --exp-name $EXP --num-gpus 4 "$@"
