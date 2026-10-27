"""Draw N posterior samples per test slice and store coil-combined magnitude images.

Paired across methods: the test list (manifest/splits.json 'test', 216 slices) and the latent noise
(torch.manual_seed(SEED0 + i) before the N draws of slice i, drawn in fixed chunks) are identical
for every checkpoint. Eval-mode BN, as in the repo's scripts/mri/test.py.
Magnitude images: |S^H x| with the ESPIRiT maps S (as in the repo's test.py, sigpy Multiply().H).

All images are divided by the max of the ground-truth magnitude of their slice (scale.npy), which leaves
every metric used (PSNR/SSIM with data_range = gt.max(), min-max LPIPS/CFID, coverage, ratios) unchanged.
Output dir <out>/samples/<name>/: samples.npy (n_items, N, 384, 384) float32, gt.npy, zf.npy (n_items, 384, 384),
support.npy (n_items, 384, 384) bool (ESPIRiT map support), scale.npy, meta.json.

Usage: python mri_crps/sample.py --ckpt <path> --name R [--n 128] [--max-items K]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.lightning.rcGAN import rcGAN  # noqa: E402
from mri_crps.data import OUT, CachedSliceData  # noqa: E402

SEED0 = 12345
CHUNK = 16


def coil_combine(t, std, mean, maps):
    """t: (B, 16, H, W) normalised coil images; maps: (8, H, W) complex -> (B, H, W) magnitude."""
    t = t * std + mean
    c = torch.complex(t[:, 0:8], t[:, 8:16])
    return (maps.conj()[None] * c).sum(1).abs()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--max-items', type=int, default=None)
    args = p.parse_args()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False

    ds = CachedSliceData('test', with_maps=True)
    n_items = len(ds) if args.max_items is None else min(args.max_items, len(ds))
    out = OUT / 'samples' / args.name
    out.mkdir(parents=True, exist_ok=True)

    model = rcGAN.load_from_checkpoint(args.ckpt).cuda().eval()
    S = np.lib.format.open_memmap(out / 'samples.npy', mode='w+', dtype=np.float32, shape=(n_items, args.n, 384, 384))
    GT = np.zeros((n_items, 384, 384), np.float32)
    ZF = np.zeros((n_items, 384, 384), np.float32)
    SUP = np.zeros((n_items, 384, 384), bool)
    SCALE = np.zeros(n_items)
    names = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_items):
            y, x, mask, mean, std, maps, name, sl = ds[i]
            names.append([name, int(sl)])
            y, x, mask = y.cuda()[None], x.cuda()[None], mask.cuda()[None]
            mean, std = float(mean), float(std)
            maps = torch.complex(maps[..., 0], maps[..., 1]).cuda()
            gt = coil_combine(x, std, mean, maps)[0]
            scale = float(gt.max())
            SCALE[i] = scale
            GT[i] = (gt / scale).cpu().numpy()
            ZF[i] = (coil_combine(y, std, mean, maps)[0] / scale).cpu().numpy()
            SUP[i] = (maps.abs() ** 2).sum(0).cpu().numpy() > 0
            torch.manual_seed(SEED0 + i)
            for c in range(0, args.n, CHUNK):
                b = min(CHUNK, args.n - c)
                g = model(y.expand(b, -1, -1, -1), mask.expand(b, -1, -1, -1, -1))
                S[i, c:c + b] = (coil_combine(g, std, mean, maps) / scale).cpu().numpy()
            if i % 10 == 0:
                print(f'{i + 1}/{n_items} {time.time() - t0:.0f}s', flush=True)
    S.flush()
    np.save(out / 'gt.npy', GT)
    np.save(out / 'zf.npy', ZF)
    np.save(out / 'support.npy', SUP)
    np.save(out / 'scale.npy', SCALE)
    json.dump(dict(ckpt=args.ckpt, n=args.n, seed0=SEED0, chunk=CHUNK, items=names,
                   seconds=time.time() - t0, beta_std=float(model.std_mult), reg=getattr(model, 'reg', 'rcgan')),
              open(out / 'meta.json', 'w'), indent=1)
    print('done', time.time() - t0)
