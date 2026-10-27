#!/bin/bash
# Usage: sbatch -J ev_radio radio-interferometry/jobs/crps_paper/eval.sh <tag>:<epoch>:<n_test> [...]
# The final epoch (69) is stored as stats_<tag>.npz, other epochs as stats_<tag>_e<epoch>.npz.
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
for spec in "$@"; do
  IFS=: read T E NT <<< "$spec"
  TAG=$T; [ "$E" != 69 ] && TAG=${T}_e$E
  python -u eval_sample.py --ckpt "/share/gpu0/jjwhit/crps_review/paper/radio/models/radio_sim_$T/checkpoint-epoch=$E.ckpt" --tag $TAG --n-test $NT --n-samp 128
done
