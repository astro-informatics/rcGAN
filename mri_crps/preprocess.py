"""Cache coil-compressed 8-coil 384x384 complex images for the selected volumes.

Reproduces the first half of data/lightning/MRIDataModule.DataTransform exactly
(ifft of the raw k-space, centre crop to 384x384, SVD coil compression to 8 virtual coils),
so that training only has to apply the GRO mask and the instance normalisation.
Output: <cache>/<split>/<fname>.npy, complex64 array of shape (n_slices, 8, 384, 384).

Usage: python preprocess.py <split> [--workers N]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.mri.espirit import ifft  # noqa: E402
from data.lightning.MRIDataModule import ImageCropandKspaceCompression  # noqa: E402

ROOT = Path('/share/gpu0/tl3/fastMRI_dataset')
OUT = Path('/share/gpu0/jjwhit/crps_review/paper/mri')


def process(job):
    src, dst, n_slices = job
    if os.path.exists(dst):
        return dst, 0.0
    t = time.time()
    out = np.zeros((n_slices, 8, 384, 384), dtype=np.complex64)
    with h5py.File(src, 'r') as f:
        ksp_all = f['kspace'][:n_slices]
    for s in range(n_slices):
        kspace = ksp_all[s].transpose(1, 2, 0)
        x = ifft(kspace, (0, 1))
        out[s] = ImageCropandKspaceCompression(x, None).transpose(2, 0, 1)
    tmp = str(dst) + '.tmp.npy'
    np.save(tmp, out)
    os.replace(tmp, dst)
    return dst, time.time() - t


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('split', choices=['train', 'val', 'test'])
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    splits = json.loads((OUT / 'manifest' / 'splits.json').read_text())
    entries = splits[args.split]
    (OUT / 'cache' / args.split).mkdir(parents=True, exist_ok=True)
    jobs = [(str(ROOT / e['dir'] / e['name']), str(OUT / 'cache' / args.split / (e['name'] + '.npy')), e['n_slices'])
            for e in entries]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (dst, dt) in enumerate(pool.imap_unordered(process, jobs)):
            print(f'{i + 1}/{len(jobs)} {os.path.basename(dst)} {dt:.1f}s (elapsed {time.time() - t0:.0f}s)', flush=True)
