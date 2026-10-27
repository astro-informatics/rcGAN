#!/bin/bash
# (Re)submit the paper-mm evaluation chains (27 Oct 21:55). Stream A runs one job at a time (1 eval GPU while 8 GPUs train);
# the MS-sigma / MSNA streams start only when their training job has released 4 GPUs, then jt / recon / stage-A run in parallel.
set -e
WT=/home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mm; S=$WT/mass_map_utils/scripts/paper_mm; L=/share/gpu0/jjwhit/crps_review/paper/mm/logs
TR_MSS=$1; TR_MSNA=$2
sub(){ name=$1; dep=$2; shift 2; sbatch --parsable -J $name --gres=gpu:1 -o $L/${name}_%j.out -e $L/${name}_%j.err ${dep:+--dependency=$dep} $S/job_eval.sh "$@"; }
val(){ sub val_$1 "$2" py $WT/scripts/mass_map/validate.py --config $WT/configs/paper_mm.yml --exp-name $3; }
jt(){ sub jt_$1 "$2" joint jt_sample.py --model $1 --n-samp 256 --n-maps 300 --chunk 64; }
rec(){ sub rec_$1 "$2" stageA $S/mm_recon.py --model $1 --n-maps 1000; }
sA(){ sub sA_$1 "$2" stageA a_sample_stats.py --model $1 --n-samp 256 --n-maps 1000 --chunk 64; }
# stream A
c=$(sbatch --parsable -J l40s_check $S/job_l40s_check.sh)
a=$(jt MS afterany:$c); b=$(val FNA afterany:$a mmgan_crps_fair_noadv); d=$(jt FNA afterok:$b,afterany:$b)
e=$(rec MS afterany:$d); f=$(rec FNA afterany:$e); g=$(sA MS afterany:$f); h=$(sA FNA afterany:$g)
echo "A: check $c jt_MS $a val_FNA $b jt_FNA $d rec_MS $e rec_FNA $f sA_MS $g sA_FNA $h"
# MS-sigma
v=$(val MSS afterok:$TR_MSS mmgan_crps_ms_sigma); echo "MSS: val $v jt $(jt MSS afterok:$v) rec $(rec MSS afterok:$v) sA $(sA MSS afterok:$v)"
# MSNA
v=$(val MSNA afterok:$TR_MSNA mmgan_crps_ms_noadv); echo "MSNA: val $v jt $(jt MSNA afterok:$v) rec $(rec MSNA afterok:$v) sA $(sA MSNA afterok:$v)"
