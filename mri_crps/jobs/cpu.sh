#!/bin/bash
# Usage: sbatch -J mri_<name> mri_crps/jobs/cpu.sh <script.py> [args...]
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/share/gpu0/jjwhit/crps_review/paper/mri/logs/%x_%j.out
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
cd /home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mri
set -ex
for cmd in "$@"; do eval python -u $cmd; done
