#!/bin/bash
# Usage: sbatch -J ev_radio radio-interferometry/jobs/crps_paper/eval.sh <epoch> <tag> [<tag> ...]
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/share/gpu0/jjwhit/crps_review/paper/radio/logs/%x_%j.out
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
cd /home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-radio/radio-interferometry/scripts/crps_paper
git rev-parse HEAD
E=$1; shift
for T in "$@"; do
  python -u eval_sample.py --ckpt "/share/gpu0/jjwhit/crps_review/paper/radio/models/radio_sim_$T/checkpoint-epoch=$E.ckpt" --tag $T --n-test 1000 --n-samp 128
done
