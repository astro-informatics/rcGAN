#!/bin/bash
#SBATCH -J mm_sigma
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o /share/gpu0/jjwhit/crps_review/paper/mm/logs/%x_%j.out
# CPU only (no --gres): computes sigma_s for MS-sigma from the training truths.
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh; conda activate cGAN
WT=/home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mm
export PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=""
cd $WT
python -u mass_map_utils/scripts/paper_mm/mm_sigma.py configs/paper_mm.yml /share/gpu0/jjwhit/crps_review/paper/mm/ms_sigma_levels.json
