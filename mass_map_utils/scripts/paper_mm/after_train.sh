#!/bin/bash
# Submit the post-training chain for one paper-mm run:
#   validate.py (best val RMSE over epochs 80-99 -> checkpoint_best.ckpt, as for F100)
#   -> jt_sample (300 maps, N=256), a_sample_stats (1000 maps, N=256), mm_recon (1000 maps, P=1/8/32)
# Usage: after_train.sh <exp-name> <key> <dependency job id or 0> [extra sbatch args for eval jobs]
set -e
EXP=$1; KEY=$2; DEP=$3; shift 3
cd /home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mm/mass_map_utils/scripts/paper_mm
WT=/home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mm
D=""; [ "$DEP" != 0 ] && D="--dependency=afterok:$DEP"
O="--gres=gpu:1 --exclude=compute-gpu-0-5 $*"
V=$(sbatch --parsable -J val_$KEY $D $O --time=06:00:00 job_eval.sh py $WT/scripts/mass_map/validate.py --config $WT/configs/paper_mm.yml --exp-name $EXP)
J=$(sbatch --parsable -J jt_$KEY --dependency=afterok:$V $O --time=10:00:00 job_eval.sh joint jt_sample.py --model $KEY --n-samp 256 --n-maps 300 --chunk 64)
A=$(sbatch --parsable -J sA_$KEY --dependency=afterok:$V $O --time=16:00:00 job_eval.sh stageA a_sample_stats.py --model $KEY --n-samp 256 --n-maps 1000 --chunk 64)
R=$(sbatch --parsable -J rec_$KEY --dependency=afterok:$V $O --time=06:00:00 job_eval.sh stageA $PWD/mm_recon.py --model $KEY --n-maps 1000)
# storage: once checkpoint_best exists, delete the remaining per-epoch checkpoints (keep best + csv logs)
C=$(sbatch --parsable -J rm_$KEY --dependency=afterok:$V -p GPU --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:10:00 \
  -o /share/gpu0/jjwhit/crps_review/paper/mm/logs/rm_${KEY}_%j.out \
  --wrap "D=/share/gpu0/jjwhit/crps_review/paper/mm/ckpt/$EXP; test -s \$D/checkpoint_best.ckpt && rm -v \$D/checkpoint-epoch=*.ckpt")
echo "$KEY: validate $V -> jt $J, stageA $A, recon $R, cleanup $C"
