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
#SBATCH --open-mode=append
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
# Slurm's own assignment is kept whenever all 4 of its GPUs are free (avoids two jobs that start together picking the same GPUs).
if [ "$PICK_FREE_GPUS" = 1 ] && [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  busy=$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '!($1 < 1000)' | wc -l)
  nok=$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | grep -c .)
  [ "$busy" = 0 ] && [ "$nok" = 4 ] && { echo "Slurm's GPUs $CUDA_VISIBLE_DEVICES are free; keeping them"; PICK_FREE_GPUS=0; }
fi
if [ "$PICK_FREE_GPUS" = 1 ]; then
  # gpu-0-5 advertises 10 GPUs but has 9, so Slurm can hand out a 4-GPU slot with only 3 real free GPUs (and a non-existent index):
  # wait up to 20 min for 4 free GPUs instead of falling back to Slurm's assignment, which then fails at start-up; then requeue
  # (same job id, so the eval chain's afterok dependency survives) rather than hold an idle 4-GPU allocation
  for i in $(seq 10); do
    export PICKED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1000 {print $1}' | sort -n | tail -4 | sort -n | paste -sd,)
    [ $(echo $PICKED | tr , '\n' | grep -c .) -ge 4 ] && break
    echo "$(date '+%T') fewer than 4 free GPUs ($PICKED); waiting"; PICKED=; sleep 120
  done
  if [ -z "$PICKED" ]; then
    [ "${SLURM_RESTART_COUNT:-0}" -lt 24 ] && { echo "no 4 free GPUs after 20 min; requeueing (restart ${SLURM_RESTART_COUNT:-0})"; kill %1 2>/dev/null; scontrol requeue $SLURM_JOB_ID; sleep 300; }
    echo "no 4 free GPUs; giving up"; exit 1
  fi
  echo "picked GPUs $PICKED"  # srun re-sets CUDA_VISIBLE_DEVICES per task, so it is set again inside the step
fi
EXP=$1; shift
if [ -n "$PICKED" ]; then
  srun bash -c 'export CUDA_VISIBLE_DEVICES=$PICKED; echo "task $SLURM_PROCID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; exec python -u train.py "$@"' _ --config ./configs/paper_mm.yml --exp-name $EXP --num-gpus 4 "$@"
else
  srun python -u train.py --config ./configs/paper_mm.yml --exp-name $EXP --num-gpus 4 "$@"
fi
