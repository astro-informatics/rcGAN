"""Example reconstructions: per method, 32-sample mean, |error|, pixel std and two samples (plus a zoom).

Usage: python mri_crps/examples.py R F MS [--items 0 40 100] [--tag main]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mri_crps.data import OUT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('names', nargs='+')
ap.add_argument('--items', type=int, nargs='+', default=[0, 40, 100])
ap.add_argument('--tag', default='main')
args = ap.parse_args()

S = {k: np.load(OUT / 'samples' / k / 'samples.npy', mmap_mode='r') for k in args.names}
GT = np.load(OUT / 'samples' / args.names[0] / 'gt.npy', mmap_mode='r')
ZF = np.load(OUT / 'samples' / args.names[0] / 'zf.npy', mmap_mode='r')
crop = (slice(32, 352), slice(32, 352))
for i in args.items:
    gt = GT[i][crop]
    cols = ['32-avg', '|error| x5', 'std x10', 'sample 1', 'sample 2', 'zoom: sample 1']
    fig, axs = plt.subplots(len(args.names) + 1, len(cols), figsize=(2.2 * len(cols), 2.3 * (len(args.names) + 1)))
    zy, zx = slice(120, 200), slice(120, 200)
    axs[0, 0].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axs[0, 0].set_title('ground truth', fontsize=8)
    axs[0, 1].imshow(ZF[i][crop], cmap='gray', vmin=0, vmax=1)
    axs[0, 1].set_title('zero-filled', fontsize=8)
    axs[0, 5].imshow(gt[zy, zx], cmap='gray', vmin=0, vmax=1)
    axs[0, 5].set_title('zoom: truth', fontsize=8)
    for r, k in enumerate(args.names, start=1):
        s = np.asarray(S[k][i, :32])[:, crop[0], crop[1]]
        m = s.mean(0)
        panels = [m, np.abs(m - gt) * 5, s.std(0) * 10, s[0], s[1], s[0][zy, zx]]
        for c, (im, t) in enumerate(zip(panels, cols)):
            axs[r, c].imshow(im, cmap='gray' if c not in (1, 2) else 'magma', vmin=0, vmax=1)
            axs[r, c].set_title(f'{k}: {t}', fontsize=8)
    for a in axs.flat:
        a.axis('off')
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / 'figures' / f'examples_item{i}_{args.tag}.{ext}', dpi=150)
    plt.close(fig)
print('ok')
