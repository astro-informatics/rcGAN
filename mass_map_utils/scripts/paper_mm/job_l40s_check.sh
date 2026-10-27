#!/bin/bash
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH -w compute-gpu-0-5
#SBATCH --nice=0
#SBATCH -o /share/gpu0/jjwhit/crps_review/paper/mm/logs/%x_%j.out
#SBATCH -e /share/gpu0/jjwhit/crps_review/paper/mm/logs/%x_%j.err
# jt_sample for F100 on 4 maps on the L40S node (CPU FFT patch in mm_run.py), compared with the original run.
D=/home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mm/mass_map_utils/scripts/paper_mm
bash $D/job_eval.sh joint jt_sample.py --model F100 --n-samp 256 --n-maps 4 --chunk 64 --out-sub l40s_check || exit 1
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh; conda activate cGAN
python $D/l40s_check.py /share/gpu0/jjwhit/crps_review/paper/mm/eval/joint/l40s_check/F100 /share/gpu0/jjwhit/crps_review/joint/main/F100
