"""Reconstruction metrics of the P-sample posterior mean, P in {1, 8, 32} (rcGAN-paper style).
Run through mm_run.py (stage-A TestSet: sorted kappa_test, noise seed 1_000_000 + i,
latents torch.manual_seed(i), as stage A).
Metrics over the COSMOS footprint, native orientation, kappa units, using the repo's
mass_map_utils/scripts/ks_utils definitions (psnr = 10 log10(max(truth) / mse), snr, rmse, pearson)
plus SSIM (skimage, full 300x300 map, data_range = truth range in the footprint, SSIM map averaged
over the footprint).

Usage: python mm_run.py stageA $PWD/mm_recon.py --model MS [--n-maps 1000]
Writes EVAL/recon/<key>.npz with arrays (n_maps, len(P)) per metric.
"""
import argparse
import os
import time

import numpy as np
import torch
from skimage.metrics import structural_similarity

import common
from common import TestSet, load_cfg, sample
from mass_map_utils.scripts.ks_utils import pearsoncoeff, psnr, rmse, snr

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, choices=list(common.MODELS))
p.add_argument("--n-maps", type=int, default=1000)
p.add_argument("--chunk", type=int, default=32)
args = p.parse_args()
PS = [1, 8, 32]
torch.set_float32_matmul_precision("highest")
cfg = load_cfg()
ts = TestSet(cfg)
mask = ts.mask
n = min(args.n_maps, len(ts))
model = common.load_model(common.MODELS[args.model])
out = {k: np.zeros((n, len(PS))) for k in ["psnr", "snr", "rmse", "pearson", "ssim"]}
t0 = time.time()
for i in range(n):
    y, xn = ts.get(i)
    S = sample(model, y, max(PS), chunk=args.chunk, seed=i).double().cpu().numpy()
    x = (xn[0].double() * cfg.kappa_std + cfg.kappa_mean).numpy()
    rng = x[mask].max() - x[mask].min()
    for j, P in enumerate(PS):
        m = S[:P].mean(0)
        out["psnr"][i, j] = psnr(x.copy(), m.copy(), mask)
        out["snr"][i, j] = snr(x.copy(), m.copy(), mask)
        out["rmse"][i, j] = rmse(x.copy(), m.copy(), mask)
        out["pearson"][i, j] = pearsoncoeff(x.copy(), m.copy(), mask)
        _, smap = structural_similarity(x, m, data_range=rng, full=True)
        out["ssim"][i, j] = smap[mask].mean()
    if i % 100 == 0:
        print(f"[{args.model}] {i + 1}/{n} {time.time() - t0:.0f}s psnr32={out['psnr'][i, 2]:.2f}", flush=True)
d = common.OUT_ROOT.replace("stageA/", "recon/")
os.makedirs(d, exist_ok=True)
np.savez(os.path.join(d, f"{args.model}.npz"), P=np.array(PS), n_maps=n, **out)
print("done", time.time() - t0)
