#!/bin/bash
# Seeds follow-up: the main CPU analysis over every finished run (main six + ES/seed runs), written to separate seeds outputs
# (joint/analysis_seeds, stageA_seeds/analysis, recon/recon_summary_seeds.json, figures/seeds/), then mm_seeds.py tables/figures.
# Usage: sbatch job_analyse_seeds.sh
#SBATCH -J an_seeds
#SBATCH -p GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --exclude=compute-gpu-0-5
#SBATCH --output=/share/gpu0/jjwhit/crps_review/paper/mm/logs/%x_%j.out
#SBATCH --error=/share/gpu0/jjwhit/crps_review/paper/mm/logs/%x_%j.err
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN
WT=/home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mm
E=/share/gpu0/jjwhit/crps_review/paper/mm/eval
export PYTHONPATH=$WT
cd $WT/mass_map_utils/scripts/paper_mm
set -x
ALL="R F100 MS MSS FNA MSNA ES0 ES1 ES2 F1 F2 R0 R1 R2 MS1 MS2 MSS1 MSS2"
# runs with a complete joint test (300 maps), stage-A stats and recon
M=""; SA=""
for k in $ALL; do
  grep -q '"n_done": 300' $E/joint/main/$k/meta.json 2>/dev/null && M="$M $k"
  [ -e $E/stageA/stats/$k/pix_rank.npy ] && SA="$SA $k"
done
python -u mm_analyse.py --models $M --B 2000 --tag seeds
MM_STAGEA_ROOT=$E/stageA_seeds/ python -u mm_run.py stageA a_analyse.py --models $SA --ref F100
python -u mm_recon_summary.py --tag seeds $ALL
python -u mm_paper_figs.py --models $M --out seeds
python -u mm_seeds.py
