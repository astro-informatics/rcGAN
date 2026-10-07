"""Effect of the mask orientation on the existing 32-sample ECP (read-only, CPU).

The saved maps under /share/gpu0/jjwhit/samples/<eval_dir>/ are rot180(kappa.T)
(mmGAN.reformat transposes H/W, plot.py rotates by 180 deg). calibration.py applies the
untransformed COSMOS mask; the correct footprint for these arrays is rot90(mask.T, 2).
ECP is computed exactly as calibration.ecp with lam = 1 (np.percentile central intervals).

Usage: python a_mask_ecp.py eval_real_output eval_crps_fair3
"""
import json
import os
import sys

import numpy as np

from common import OUT_ROOT

LEVELS = np.round(np.arange(0.05, 1.0, 0.05), 2)
mask = np.load("/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_mask.npy", allow_pickle=True).astype(bool)
masks = {"old_mask": mask, "correct_rot180_maskT": np.rot90(mask.T, 2)}
masks["only_in_old"] = masks["old_mask"] & ~masks["correct_rot180_maskT"]
masks["only_in_correct"] = masks["correct_rot180_maskT"] & ~masks["old_mask"]

res = {}
for d in sys.argv[1:]:
    base = f"/share/gpu0/jjwhit/samples/{d}/"
    n = len([f for f in os.listdir(base + "kappa") if f.startswith("np_gt_")])
    cov = {k: np.zeros((n, len(LEVELS))) for k in masks}
    for i in range(n):
        gt = np.load(base + f"kappa/np_gt_{i + 1:04d}.npy").real
        s = np.load(base + f"recon/np_samps_{i + 1:04d}.npy").real
        for j, L in enumerate(LEVELS):
            lo, hi = np.percentile(s, [100 * (1 - L) / 2, 100 - 100 * (1 - L) / 2], axis=0)
            inside = (gt >= lo) & (gt <= hi)
            for k, m in masks.items():
                cov[k][i, j] = inside[m].mean()
    res[d] = {"n_maps": n, "levels": LEVELS.tolist(), "ref_n32": (LEVELS * 31 / 33).tolist(),
              **{k: v.mean(0).tolist() for k, v in cov.items()},
              "pixels": {k: int(m.sum()) for k, m in masks.items()}}
    print(d, json.dumps({k: np.round(v.mean(0)[[9, 17, 18]], 4).tolist() for k, v in cov.items()}), flush=True)

os.makedirs(OUT_ROOT + "analysis", exist_ok=True)
json.dump(res, open(OUT_ROOT + "analysis/mask_ecp32.json", "w"), indent=1)
