#!/bin/bash
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/share/gpu0/jjwhit/crps_review/paper/mri/logs/%x_%j.out
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
nvidia-smi --query-gpu=name,memory.total --format=csv
cd /home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mri
set -ex
rm -rf /share/gpu0/jjwhit/crps_review/paper/mri/ckpt_smoke
python -u mri_crps/train.py --config configs/mri_crps_smoke.yml --exp-name smoke_R --reg rcgan --max-steps 400 --limit-val-batches 20
python -u mri_crps/train.py --config configs/mri_crps_smoke.yml --exp-name smoke_F --reg fair --max-steps 60 --limit-val-batches 4
python -u mri_crps/train.py --config configs/mri_crps_smoke.yml --exp-name smoke_MS --reg ms --max-steps 60 --limit-val-batches 4
python -u mri_crps/sample.py --ckpt /share/gpu0/jjwhit/crps_review/paper/mri/ckpt_smoke/smoke_R/last.ckpt --name smoke_R --n 16 --max-items 4
python -u mri_crps/analyse.py --name smoke_R
