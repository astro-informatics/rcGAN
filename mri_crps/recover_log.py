"""Recover per-step d_loss / g_loss from a Slurm training log (progress-bar postfix, 3 significant figures).

Needed for MS epochs 0-30: the resumed job (CSVLogger version 0) overwrote metrics.csv. Validation PSNR and beta
are not in the progress bar and cannot be recovered. CSV step = epoch * batches_per_epoch + n - 1 (checked on R).

Usage: python mri_crps/recover_log.py <slurm.out> <out.csv> [--batches 1440] [--max-epoch 30]
"""
import argparse
import csv
import re

ap = argparse.ArgumentParser()
ap.add_argument('log')
ap.add_argument('out')
ap.add_argument('--batches', type=int, default=1440)
ap.add_argument('--max-epoch', type=int, default=30)
args = ap.parse_args()

pat = re.compile(r'Epoch (\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[[^\]]*?d_loss=([-\d.e+na]+), g_loss=([-\d.e+na]+)')
txt = open(args.log, errors='ignore').read().replace('\r', '\n')
rows = {}
for m in pat.finditer(txt):
    e, n = int(m[1]), int(m[2])
    if e <= args.max_epoch and 1 <= n <= args.batches:
        step = e * args.batches + n - 1
        if step % 10 == 9:  # log_every_n_steps = 10, as in metrics.csv
            rows[step] = (m[4], m[5], e)
with open(args.out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['d_loss', 'g_loss', 'epoch', 'step'])
    for s in sorted(rows):
        d, g, e = rows[s]
        w.writerow([d, g, e, s])
print(len(rows), 'rows ->', args.out)
