#!/bin/bash
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/share/gpu0/jjwhit/crps_review/paper/mri/logs/%x_%j.out
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd /home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mri
for s in "$@"; do python -u mri_crps/preprocess.py $s --workers 8; done
