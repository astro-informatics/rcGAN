#!/bin/bash
# CPU analysis over the models whose outputs exist. Usage: sbatch job_analyse.sh "R F100 MS ..."
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=compute-gpu-0-5
#SBATCH --output=/share/gpu0/jjwhit/crps_review/paper/mm/logs/%x_%j.out
#SBATCH --error=/share/gpu0/jjwhit/crps_review/paper/mm/logs/%x_%j.err
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
WT=/home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mm
export PYTHONPATH=$WT
cd $WT/mass_map_utils/scripts/paper_mm
set -x
M="$1"
python -u mm_analyse.py --models $M --B 2000
python -u mm_extra.py $M
SA=""; for k in $M; do [ -e /share/gpu0/jjwhit/crps_review/paper/mm/eval/stageA/stats/$k/maps.npz ] && SA="$SA $k"; done
python -u mm_run.py stageA a_analyse.py --models $SA --ref F100
python -u mm_recon_summary.py $M
python -u mm_paper_figs.py --models $M
python -u mm_loss_curves.py
