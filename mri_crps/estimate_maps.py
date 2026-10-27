"""ESPIRiT sensitivity maps for cached val/test slices (GPU job).

As scripts/mri/estimate_maps.py: maps are estimated from the R=8 GRO-undersampled k-space of the
8 virtual coils, calib_width=16, crop=0.70, kernel_width=6 (sigpy EspiritCalib).
Output: <out>/maps/<split>/<fname>.npy, complex64 (n_slices, 8, 384, 384).

Usage: python estimate_maps.py <split> [<split> ...]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import sigpy as sp
import sigpy.mri as mr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mri_crps.data import OUT, make_item  # noqa: E402
from utils.mri.fftc import fft2c_new  # noqa: E402
from utils.mri.math import tensor_to_complex_np  # noqa: E402

if __name__ == '__main__':
    splits = json.loads((OUT / 'manifest' / 'splits.json').read_text())
    for split in sys.argv[1:]:
        (OUT / 'maps' / split).mkdir(parents=True, exist_ok=True)
        for i, e in enumerate(splits[split]):
            dst = OUT / 'maps' / split / (e['name'] + '.npy')
            if dst.exists():
                continue
            t = time.time()
            ims = np.load(OUT / 'cache' / split / (e['name'] + '.npy'))
            maps = np.zeros_like(ims)
            for s in range(ims.shape[0]):
                y, x, mask, mean, std = make_item(ims[s])
                yc = np.zeros((8, 384, 384, 2), dtype=np.float32)
                yc[..., 0] = y[0:8].numpy()
                yc[..., 1] = y[8:16].numpy()
                ksp = fft2c_new(torch.from_numpy(yc) * std.float() + mean.float())
                maps[s] = mr.app.EspiritCalib(tensor_to_complex_np(ksp), calib_width=16, device=sp.Device(0),
                                              crop=0.70, kernel_width=6, show_pbar=False).run().get()
            tmp = str(dst) + '.tmp.npy'
            np.save(tmp, maps)
            os.replace(tmp, dst)
            print(f'{split} {i + 1}/{len(splits[split])} {e["name"]} {time.time() - t:.1f}s', flush=True)
