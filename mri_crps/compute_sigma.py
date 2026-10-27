"""sigma_s for MS-sigma: std of the level-s orthonormal Haar scaling coefficients of the training truths.

Uses exactly the tensor the regulariser sees: the normalised 16-channel (8 coils x re/im) ground truth x of every
training slice of the run's training set (configs/mri_crps.yml n_train_volumes; 2880 slices), 7 levels at 384x384.
Output: manifest/ms_sigma.json.
"""
import json
import sys
import types
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.lightning.proper_scores import default_levels, haar_level_std  # noqa: E402
from mri_crps.data import OUT, CachedSliceData  # noqa: E402

cfg = types.SimpleNamespace(**yaml.safe_load(open('configs/mri_crps.yml')))
ds = CachedSliceData('train', with_maps=False, max_volumes=cfg.n_train_volumes)
loader = DataLoader(ds, batch_size=16, num_workers=8)
n_levels = default_levels(384, 384)
sigma = haar_level_std((b[1] for b in loader), n_levels)
res = dict(sigma_s=sigma, n_levels=n_levels, n_items=len(ds), tensor='normalised 16-channel coil-image truth x (as l1_std_p)',
           n_train_volumes=cfg.n_train_volumes)
print(res)
json.dump(res, open(OUT / 'manifest' / 'ms_sigma.json', 'w'), indent=1)
