#!/bin/bash
# Seeds follow-up (BRIEF_mm_seeds.md): train one run and queue its evaluation.
# Usage: submit_seed.sh <KEY> <SEED> [train dependency, e.g. afterany:123]
#   KEY in ES / R / F / MS / MSS; exp name and key as in mm_run.py SEED_RUNS (e.g. ES0 -> mmgan_es_s0).
# Training: train_mm.sh on 4 GPUs, same config as every other paper-mm run, per-epoch checkpoints only for epochs 80-99.
# Evaluation is one serial stream over all seed runs (at most 1 eval GPU next to the 8 training GPUs):
#   validate -> jt (300 maps) -> recon (1000) -> stage-A (1000) -> jointcov (BRIEF_jointcov.md), each afterany the previous; the stream's last job id
#   is kept in $LAST so the next run's chain queues behind it. Epoch checkpoints are deleted once checkpoint_best exists.
set -e
KEY=$1; SEED=$2; TDEP=$3
WT=/home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mm; S=$WT/mass_map_utils/scripts/paper_mm; L=/share/gpu0/jjwhit/crps_review/paper/mm/logs
LAST=$L/.seeds_last_eval
case $KEY in
  ES) EXP=mmgan_es_s$SEED; A="--crps-mode es" ;;
  R) EXP=mmgan_rcgan_s$SEED; A="--crps-mode rcgan" ;;
  F) EXP=mmgan_crps_fair_s$SEED; A="--crps-mode alpha_fair --crps-alpha 1.0" ;;
  MS) EXP=mmgan_crps_ms_s$SEED; A="--crps-mode ms" ;;
  MSS) EXP=mmgan_crps_ms_sigma_s$SEED; A="--crps-mode ms_sigma --ms-sigma-file /share/gpu0/jjwhit/crps_review/paper/mm/ms_sigma_levels.json" ;;
  *) echo "unknown key $KEY"; exit 1 ;;
esac
K=$KEY$SEED
NOV100=--exclude=compute-gpu-0-0,compute-gpu-0-1  # V100 16 GB: batch 9 OOMs
# TR_EXCLUDE (env) overrides the training exclude list, e.g. to keep a run off a node another seed run is using
EXC=${TR_EXCLUDE:+--exclude=$TR_EXCLUDE}; EXC=${EXC:-$NOV100}
T=$(PICK_FREE_GPUS=1 sbatch --parsable -J tr_$K --gres=gpu:4 $EXC ${TDEP:+--dependency=$TDEP} $WT/mass_map_utils/jobs/paper_mm/train_mm.sh \
    $EXP $A --seed $SEED --ckpt-from-epoch 80)
sub(){ name=$1; dep=$2; shift 2; sbatch --parsable -J $name --gres=gpu:1 $NOV100 -o $L/${name}_%j.out -e $L/${name}_%j.err --dependency=$dep "$@"; }
P=$(cat $LAST 2>/dev/null || true)
V=$(sub val_$K afterok:$T${P:+,afterany:$P} --time=06:00:00 $S/job_eval.sh py $WT/scripts/mass_map/validate.py --config $WT/configs/paper_mm.yml --exp-name $EXP)
J=$(sub jt_$K afterany:$V --time=10:00:00 $S/job_eval.sh joint jt_sample.py --model $K --n-samp 256 --n-maps 300 --chunk 64)
R=$(sub rec_$K afterany:$J --time=06:00:00 $S/job_eval.sh stageA $S/mm_recon.py --model $K --n-maps 1000)
G=$(sub sA_$K afterany:$R --time=16:00:00 $S/job_eval.sh stageA a_sample_stats.py --model $K --n-samp 256 --n-maps 1000 --chunk 64)
JC=$(sub jc_$K afterany:$G --time=03:00:00 --cpus-per-task=16 --mem=64G $S/job_eval.sh stageA $S/mm_jointcov.py --models $K)
C=$(sbatch --parsable -J rm_$K --dependency=afterok:$V -p GPU --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:10:00 -o $L/rm_${K}_%j.out \
    --wrap "D=/share/gpu0/jjwhit/crps_review/paper/mm/ckpt/$EXP; test -s \$D/checkpoint_best.ckpt && rm -v \$D/checkpoint-epoch=*.ckpt")
echo $JC > $LAST
echo "$(date '+%F %T') $K ($EXP, seed $SEED): train $T${TDEP:+ ($TDEP)} -> val $V -> jt $J -> rec $R -> sA $G -> jc $JC; cleanup $C" | tee -a $L/seeds_jobs.log
