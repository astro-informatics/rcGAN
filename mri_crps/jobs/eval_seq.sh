#!/bin/bash
# Usage: sbatch -J mri_eval_<tag> mri_crps/jobs/eval_seq.sh <ckpt_dir> <suffix> name1 name2 ...
#   evaluates <ckpt_dir>/<name>.ckpt as samples/<name><suffix> sequentially on one GPU
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/share/gpu0/jjwhit/crps_review/paper/mri/logs/%x_%j.out
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
nvidia-smi --query-gpu=name,memory.total --format=csv
cd /home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mri
set -ex
d=$1; suf=$2; shift 2
for k in "$@"; do
  python -u mri_crps/sample.py --name $k$suf --ckpt $d/$k.ckpt --n 128
  python -u mri_crps/analyse.py --name $k$suf
done
