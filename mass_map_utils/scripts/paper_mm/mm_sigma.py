"""sigma_s for MS-sigma (coordinator 2026-10-27): std of the level-s orthonormal Haar scaling coefficients of the
TRAINING truths, in exactly the units the mmGAN CRPS loss sees: normalised kappa (kappa - kappa_mean) / kappa_std,
full 300x300 field (incl. outside the survey mask). Uses all training maps (sorted). Writes JSON with sigma and
weights = 1/sigma, which train.py reads via --ms-sigma-file.

Usage: python mm_sigma.py <config.yml> <out.json>
"""
import json
import pathlib
import sys

import numpy as np
import torch
import yaml

from models.lightning.proper_scores import default_levels, haar_level_std

cfg = yaml.safe_load(open(sys.argv[1]))
files = sorted((pathlib.Path(cfg["data_path"]) / "kappa_train").iterdir())
n_levels = default_levels(cfg["im_size"], cfg["im_size"])


def batches(bs=100):
    for i in range(0, len(files), bs):
        k = np.stack([np.load(f, allow_pickle=True).astype(np.float64) for f in files[i:i + bs]])
        # same as MMDataTransform: transforms.normalize(pt_gt, kappa_mean, kappa_std), shape (N, 1, H, W)
        yield (torch.from_numpy(k)[:, None] - cfg["kappa_mean"]) / cfg["kappa_std"]


sigma = haar_level_std(batches(), n_levels)
out = dict(n_train_maps=len(files), n_levels=n_levels, units="normalised kappa, full 300x300 field",
           kappa_mean=cfg["kappa_mean"], kappa_std=cfg["kappa_std"],
           sigma=sigma, weights=[1.0 / s for s in sigma])
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out, indent=1))
