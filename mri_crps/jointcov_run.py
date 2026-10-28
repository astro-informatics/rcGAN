"""Marginal + joint coverage (shared ~/crps_review/paper/common/jointcov.py) for one MRI checkpoint.

Samples are generated on the fly exactly as in sample.py (same 216 test slices, N = 128, latent seed SEED0 + i, chunks of
16, strict fp32, magnitude |S^H x| / GT max), so they are the same draws as the main evaluation; nothing is stored.
A producer thread samples on the GPU while jointcov.run() ranks the previous slice on the CPU.
Mask: fg = ESPIRiT support AND (5x5 box-smoothed GT > 0.05), as in analyse.py.
The method's run() output is merged into <out>/jointcov.json under a file lock (several jobs may finish together).

Usage: python mri_crps/jointcov_run.py --name R [--ckpt-dir .../ckpt/final] [--n 128] [--max-items K] [--json path]
"""
import argparse
import fcntl
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / 'crps_review/paper/common'))
import jointcov  # noqa: E402
from models.lightning.rcGAN import rcGAN  # noqa: E402
from mri_crps.data import OUT, CachedSliceData  # noqa: E402
from mri_crps.sample import CHUNK, SEED0, coil_combine  # noqa: E402

FG_THR = 0.05
KS = (1, 2, 4, 8, 16, 32, 64, 128, None)  # powers of two <= 384 / 2, plus the whole image
LABELS = {'R': 'rcGAN', 'F': 'fair CRPS', 'MS': 'multiscale fair CRPS', 'MSsig': 'multiscale fair CRPS (1/sigma_s weights)'}
NOTES = ('fastMRI brain multicoil, R = 8 GRO mask, 8 virtual coils; image = coil-combined magnitude |S^H x| (ESPIRiT maps), '
         '384 x 384, divided by the GT max of each slice. 216 test slices (72 rcGAN test slices + 144 from held-out val '
         'volumes), N = 128 samples per slice, latent seeds identical across methods (same draws as the main evaluation). '
         'Mask = per-slice tissue foreground: ESPIRiT support AND 5x5 box-smoothed GT > 0.05 of the slice max; outside the '
         'support all samples and the truth are exactly 0. Epoch-40 checkpoints; all methods 40 epochs, 360 training volumes, '
         'batch 2. No no-critic (noadv) or ES runs exist for MRI.')


def merge(path, name, res, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.load(open(path)) if path.exists() else {}
        d.update({k: meta[k] for k in ['domain', 'image_side', 'N', 'n_cases', 'mask', 'notes']})
        d.setdefault('labels', {})[name] = LABELS.get(name, name)
        d.setdefault('methods', {})[name] = res
        # keep the schema's key order
        order = ['domain', 'image_side', 'N', 'n_cases', 'mask', 'labels', 'methods', 'notes']
        d = {k: d[k] for k in order}
        tmp = Path(str(path) + f'.tmp{os.getpid()}')
        json.dump(d, open(tmp, 'w'))
        tmp.replace(path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--ckpt-dir', default=str(OUT / 'ckpt' / 'final'))
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--max-items', type=int, default=None)
    p.add_argument('--json', default=str(OUT / 'jointcov.json'))
    args = p.parse_args()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    ds = CachedSliceData('test', with_maps=True)
    n_items = len(ds) if args.max_items is None else min(args.max_items, len(ds))
    model = rcGAN.load_from_checkpoint(os.path.join(args.ckpt_dir, f'{args.name}.ckpt')).cuda().eval()
    box = torch.ones(1, 1, 5, 5, device='cuda') / 25
    q = queue.Queue(maxsize=3)
    masks = {}

    def producer():
        with torch.no_grad():
            for i in range(n_items):
                y, x, mask, mean, std, maps, name, sl = ds[i]
                y, x, mask = y.cuda()[None], x.cuda()[None], mask.cuda()[None]
                mean, std = float(mean), float(std)
                maps = torch.complex(maps[..., 0], maps[..., 1]).cuda()
                gt = coil_combine(x, std, mean, maps)[0]
                scale = float(gt.max())
                gt = gt / scale
                sup = (maps.abs() ** 2).sum(0) > 0
                fg = sup & (F.conv2d(gt[None, None], box, padding=2)[0, 0] > FG_THR)
                torch.manual_seed(SEED0 + i)
                s = []
                for c in range(0, args.n, CHUNK):
                    b = min(CHUNK, args.n - c)
                    g = model(y.expand(b, -1, -1, -1), mask.expand(b, -1, -1, -1, -1))
                    s.append((coil_combine(g, std, mean, maps) / scale).cpu().numpy())
                q.put((i, np.concatenate(s), gt.cpu().numpy(), fg.cpu().numpy()))

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    t0 = time.time()

    def sample_fn(c):
        i, xs, y, fg = q.get()
        assert i == c
        masks[c] = fg
        if c % 10 == 0:
            print(f'{c + 1}/{n_items} {time.time() - t0:.0f}s', flush=True)
        return xs, y

    res = jointcov.run(sample_fn, list(range(n_items)), mask=lambda c: masks.pop(c), ks=KS, seed=0, null_check=True)
    i90 = res['model']['levels'].index(0.9)
    print('pixel 90% coverage: model', res['model']['stats']['pixel']['1']['cov'][i90],
          'null', res['null']['stats']['pixel']['1']['cov'][i90])
    meta = dict(domain='mri', image_side=384, N=args.n, n_cases=n_items,
                mask='per-slice tissue foreground: ESPIRiT support AND (5x5 box-smoothed GT > 0.05 x slice max)', notes=NOTES)
    if args.max_items is not None:
        meta['notes'] = NOTES + f' SMOKE TEST: {n_items} slices only.'
    merge(Path(args.json), args.name, res, meta)
    print('done', time.time() - t0, '->', args.json)
